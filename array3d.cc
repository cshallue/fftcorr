#include "array3d.h"

#include "fft_utils.h"

Array3D::Array3D(int ngrid[3]) {
  for (int j = 0; j < 3; j++) {
    ngrid_[j] = ngrid[j];
    assert(ngrid_[j] > 0 && ngrid_[j] < 1e4);
  }

  // Setup ngrid2_.
#ifdef FFTSLAB
// That said, the rest of the code should work even if extra space is used.
// Some operations will blindly apply to the pad cells, but that's ok.
// In particular, we might consider having ngrid2_ be evenly divisible by
// the critical alignment stride (32 bytes for AVX, but might be more for cache
// lines) or even by a full PAGE for NUMA memory.  Doing this *will* force a
// more complicated FFT, but at least for the NUMA case this is desired: we want
// to force the 2D FFT to run on its socket, and only have the last 1D FFT
// crossing sockets.  Re-using FFTW plans requires the consistent memory
// alignment.
#define FFT_ALIGN 16
  // This is in units of Floats.  16 doubles is 1024 bits.
  ngrid2_ = FFT_ALIGN * (ngrid2_ / FFT_ALIGN + 1);
#else
  // ngrid2_ pads out the array for the in-place FFT.
  // The default 3d FFTW format must have the following:
  ngrid2_ = (ngrid_[2] / 2 + 1) * 2;  // For the in-place FFT
#endif
  assert(ngrid2_ % 2 == 0);
  fprintf(stdout, "# Using ngrid2_=%d for FFT r2c padding\n", ngrid2_);

  ngrid3_ = (uint64)ngrid_[0] * ngrid_[1] * ngrid2_;

  // Allocate data_ array.
  int err =
      posix_memalign((void **)&data_, PAGE, sizeof(Float) * ngrid3_ + PAGE);
  assert(err == 0);
  assert(data_ != NULL);

  // NULL is a valid fftw_plan value; the planner will return NULL if it fails.
  fft_ = NULL;
  fftYZ_ = NULL;
  fftX_ = NULL;
  ifft_ = NULL;
  ifftYZ_ = NULL;
  ifftX_ = NULL;
}

Array3D::~Array3D() {
  if (data_ != NULL) free(data_);
    // Destroy the FFT plans.
#ifndef FFTSLAB
  if (fft_ != NULL) fftw_destroy_plan(fft_);
  if (ifft_ != NULL) fftw_destroy_plan(ifft_);
#else
  if (fftX_ != NULL) fftw_destroy_plan(fftX_);
  if (fftYZ_ != NULL) fftw_destroy_plan(fftYZ_);
  if (ifftX_ != NULL) fftw_destroy_plan(ifftX_);
  if (ifftYZ_ != NULL) fftw_destroy_plan(ifftTZ_);
#endif
#ifdef OPENMP
#ifndef FFTSLAB
  fftw_cleanup_threads();
#endif
#endif
}

void Array3D::setup_fft() {
  // TODO: move this function into this class.
  setup_FFTW(fft_, fftYZ_, fftX_, ifft_, ifftYZ_, ifftX_, ngrid_, ngrid2_,
             data_);
}

void Array3D::execute_fft() {
  // TODO: assert that setup has been called. Decide best way to crash with
  // informative message. Same with execute_ifft
  // FFTonly.Start();
#ifndef FFTSLAB
  fftw_execute(fft_);
#else
  // FFTyz.Start();
// Then need to call this for every slab.  Can OMP these lines
#pragma omp parallel for MY_SCHEDULE
  for (uint64 x = 0; x < ngrid_[0]; x++)
    fftw_execute_dft_r2c(fftYZ_, data_ + x * ngrid_[1] * ngrid2_,
                         (fftw_complex *)data_ + x * ngrid_[1] * ngrid2_ / 2);
    // FFTyz.Stop();
    // FFTx.Start();
#pragma omp parallel for schedule(dynamic, 1)
  for (uint64 y = 0; y < ngrid_[1]; y++)
    fftw_execute_dft(fftX_, (fftw_complex *)data_ + y * ngrid2_ / 2,
                     (fftw_complex *)data_ + y * ngrid2_ / 2);
    // FFTx.Stop();
#endif
  // FFTonly.Stop();
}

void Array3D::execute_ifft() {
  // TODO: class knows whether it's Fourier transformed or not?
  // FFTonly.Start();
#ifndef FFTSLAB
  fftw_execute(ifft_);
#else
  // FFTx.Start();
// Then need to call this for every slab.  Can OMP these lines
#pragma omp parallel for schedule(dynamic, 1)
  for (uint64 y = 0; y < ngrid_[1]; y++)
    fftw_execute_dft(ifftX_, (fftw_complex *)data_ + y * ngrid2_ / 2,
                     (fftw_complex *)data_ + y * ngrid2_ / 2);
    // FFTx.Stop();
    // FFTyz.Start();
#pragma omp parallel for MY_SCHEDULE
  for (uint64 x = 0; x < ngrid_[0]; x++)
    fftw_execute_dft_c2r(ifftYZ_,
                         (fftw_complex *)data_ + x * ngrid_[1] * ngrid2_ / 2,
                         data_ + x * ngrid_[1] * ngrid2_);
    // FFTyz.Stop();
#endif
  // FFTonly.Stop();
}

void Array3D::set_value(Float value) {
  // Initialize data_ by setting each element.
  // We want to touch the whole matrix, because in NUMA this defines the
  // association of logical memory into the physical banks.
  // Init.Start();
#ifdef SLAB
  int nx = ngrid_[0];
  const uint64 nyz = ngrid3_ / nx;
#pragma omp parallel for MY_SCHEDULE
  for (int x = 0; x < nx; ++x) {
    Float *slab = data_ + x * nyz;
    for (uint64 i = 0; i < nyz; ++i) {
      slab[i] = value;
    }
  }
#else
#pragma omp parallel for MY_SCHEDULE
  for (uint64 i = 0; i < ngrid3_; i++) {
    data_[i] = value;
  }
#endif
  // Init.Stop();
}

void Array3D::copy_from(const Float *other) {
  // Init.Start();
#ifdef SLAB
  int nx = ngrid_[0];
  const uint64 nyz = ngrid3_ / nx;
#pragma omp parallel for MY_SCHEDULE
  for (int x = 0; x < nx; ++x) {
    Float *slab = data_ + x * nyz;
    for (uint64 i = 0; i < nyz; ++i) {
      slab[i] = other[i];
    }
  }
#else
#pragma omp parallel for MY_SCHEDULE
  for (uint64 i = 0; i < ngrid3_; i++) {
    data_[i] = other[i];
  }
#endif
  // Init.Stop();
}

void Array3D::add_scalar(Float s) {
#ifdef SLAB
  int nx = ngrid_[0];
  const uint64 nyz = ngrid3_ / nx;
#pragma omp parallel for MY_SCHEDULE
  for (int x = 0; x < nx; ++x) {
    Float *slab = data_ + x * nyz;
    for (uint64 i = 0; i < nyz; ++i) {
      slab[i] += s;
    }
  }
#else
#pragma omp parallel for MY_SCHEDULE
  for (uint64 i = 0; i < ngrid3_; ++i) {
    data_[i] += s;
  }
#endif
}

Float Array3D::sum() const {
  Float tot = 0.0;
#ifdef SLAB
  int nx = ngrid_[0];
  const uint64 nyz = ngrid3_ / nx;
#pragma omp parallel for MY_SCHEDULE reduction(+ : tot)
  for (int x = 0; x < nx; ++x) {
    Float *slab = data_ + x * nyz;
    for (uint64 i = 0; i < nyz; ++i) {
      tot += slab[i];
    }
  }
#else
#pragma omp parallel for MY_SCHEDULE reduction(+ : tot)
  for (uint64 i = 0; i < ngrid3_; ++i) {
    tot += data_[i];
  }
#endif
  return tot;
}

// TODO: come up with a way to template these parallelizable ops
Float Array3D::sumsq() const {
  Float tot = 0.0;
#ifdef SLAB
  int nx = ngrid_[0];
  const uint64 nyz = ngrid3_ / nx;
#pragma omp parallel for MY_SCHEDULE reduction(+ : tot)
  for (int x = 0; x < nx; ++x) {
    Float *slab = data_ + x * nyz;
    for (uint64 i = 0; i < nyz; ++i) {
      tot += slab[i];
    }
  }
#else
#pragma omp parallel for MY_SCHEDULE reduction(+ : tot)
  for (uint64 i = 0; i < ngrid3_; ++i) {
    tot += data_[i] * data_[i];
  }
#endif
  return tot;
}
