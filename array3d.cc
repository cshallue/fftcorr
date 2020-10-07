#include "array3d.h"

Array3D::Array3D(int ngrid[3]) {
  for (int j = 0; j < 3; j++) {
    ngrid_[j] = ngrid[j];
    assert(ngrid_[j] > 0 && ngrid_[j] < 1e4);
  }
  is_fourier_space_ = false;

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
#ifndef FFTSLAB
  fft_ = NULL;
  ifft_ = NULL;
#else
  fftX_ = NULL;
  fftYZ_ = NULL;
  ifftYZ_ = NULL;
  ifftX_ = NULL;
#endif
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
  // Setup the FFTW plans, possibly from disk, and save the wisdom
  fprintf(stdout, "# Planning the FFTs...");
  fflush(NULL);
  // FFTW.Start();
  FILE *fp = NULL;
#ifdef OPENMP
#ifndef FFTSLAB
  {
    int errval = fftw_init_threads();
    assert(errval);
  }
  fftw_plan_with_nthreads(omp_get_max_threads());
#endif
#define WISDOMFILE "wisdom_fftw_omp"
#else
#define WISDOMFILE "wisdom_fftw"
#endif
#ifdef FFTSLAB
#undef WISDOMFILE
#define WISDOMFILE "wisdom_fftw"
#endif
  fp = fopen(WISDOMFILE, "r");
  if (fp != NULL) {
    fprintf(stdout, "Reading %s...", WISDOMFILE);
    fflush(NULL);
    fftw_import_wisdom_from_file(fp);
    fclose(fp);
  }

  // Interpret data_ as complex.
  fftw_complex *cdata = (fftw_complex *)data_;

#ifndef FFTSLAB
  // The following interface should work even if ngrid2 was 'non-minimal',
  // as might be desired by padding.
  int nfft[3], nfftc[3];
  nfft[0] = nfftc[0] = ngrid_[0];
  nfft[1] = nfftc[1] = ngrid_[1];
  // Since ngrid2 is always even, this will trick
  // FFTW to assume ngrid2/2 Complex numbers in the result, while
  // fulfilling that nfft[2]>=ngrid[2].
  nfft[2] = ngrid2_;
  nfftc[2] = nfft[2] / 2;
  int howmany = 1;  // Only one forward and inverse FFT.
  int dist = 0;     // Unused because howmany = 1.
  int stride = 1;   // Array is continuous in memory.
  fft_ = fftw_plan_many_dft_r2c(3, ngrid_, howmany, data_, nfft, stride, dist,
                                cdata, nfftc, stride, dist, FFTW_MEASURE);
  ifft_ = fftw_plan_many_dft_c2r(3, ngrid_, howmany, cdata, nfftc, stride, dist,
                                 data_, nfft, stride, dist, FFTW_MEASURE);

  /*	// The original interface, which only works if ngrid2 is tightly packed.
  fft_ = fftw_plan_dft_r2c_3d(ngrid[0], ngrid[1], ngrid[2],
                  data_, cdata, FFTW_MEASURE);
  ifft_ = fftw_plan_dft_c2r_3d(ngrid[0], ngrid[1], ngrid[2],
                  cdata, data_, FFTW_MEASURE);
*/

#else
  // If we wanted to split into 2D and 1D by hand (and therefore handle the OMP
  // aspects ourselves), then we need to have two plans each.
  int nfft2[2], nfft2c[2];
  nfft2[0] = nfft2c[0] = ngrid_[1];
  nfft2[1] = ngrid2_;  // Since ngrid2 is always even, this will trick
  nfft2c[1] = nfft2[1] / 2;
  int ngridYZ[2];
  ngridYZ[0] = ngrid_[1];
  ngridYZ[1] = ngrid_[2];
  fftYZ_ = fftw_plan_many_dft_r2c(2, ngridYZ, 1, data_, nfft2, 1, 0, cdata,
                                  nfft2c, 1, 0, FFTW_MEASURE);
  ifftYZ_ = fftw_plan_many_dft_c2r(2, ngridYZ, 1, cdata, nfft2c, 1, 0, data_,
                                   nfft2, 1, 0, FFTW_MEASURE);

  // After we've done the 2D r2c FFT, we have to do the 1D c2c transform.
  // We'll plan to parallelize over Y, so that we're doing (ngrid[2]/2+1)
  // 1D FFTs at a time.
  // Elements in the X direction are separated by ngrid[1]*ngrid2/2 complex
  // numbers.
  int ngridX = ngrid_[0];
  fftX_ = fftw_plan_many_dft(1, &ngridX, (ngrid_[2] / 2 + 1), cdata, NULL,
                             ngrid_[1] * ngrid2_ / 2, 1, cdata, NULL,
                             ngrid_[1] * ngrid2_ / 2, 1, -1, FFTW_MEASURE);
  ifftX_ = fftw_plan_many_dft(1, &ngridX, (ngrid_[2] / 2 + 1), cdata, NULL,
                              ngrid_[1] * ngrid2_ / 2, 1, cdata, NULL,
                              ngrid_[1] * ngrid2_ / 2, 1, +1, FFTW_MEASURE);
#endif

  fp = fopen(WISDOMFILE, "w");
  assert(fp != NULL);
  fftw_export_wisdom_to_file(fp);
  fclose(fp);
  fprintf(stdout, "Done!\n");
  fflush(NULL);
  // FFTW.Stop();
}

void Array3D::execute_fft() {
  assert(!is_fourier_space_);
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
  assert(is_fourier_space_);
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
  assert(!is_fourier_space_);
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

void Array3D::multiply_with_conjugation(const Array3D &other) {
  assert(is_fourier_space_);
  // Element-wise multiply by conjugate of other
  // TODO: check same dimensions.
  Complex *data = (Complex *)data_;
  const Complex *other_data = (Complex *)other.data();
  uint64 size = ngrid3_ / 2;  // Note that size refers to the Complex size.
#ifdef SLAB
  int nx = ngrid_[0];
  const uint64 nyz = size / nx;
#pragma omp parallel for MY_SCHEDULE
  for (int x = 0; x < nx; ++x) {
    Complex *slab = data + x * nyz;
    Complex *other_slab = other_data + x * nyz;
    for (uint64 i = 0; i < nyz; ++i) {
      slab[i] *= std::conj(other_slab[i]);
    }
  }
#else
#pragma omp parallel for MY_SCHEDULE
  for (uint64 i = 0; i < size; ++i) {
    data[i] *= std::conj(other_data[i]);
  }
#endif
}

void Array3D::copy_from(const Array3D &other) {
  // TODO: check same dimensions.
  // Init.Start();
  is_fourier_space_ = other.is_fourier_space_;
  const Float *other_data = other.data_;
#ifdef SLAB
  int nx = ngrid_[0];
  const uint64 nyz = ngrid3_ / nx;
#pragma omp parallel for MY_SCHEDULE
  for (int x = 0; x < nx; ++x) {
    Float *slab = data_ + x * nyz;
    for (uint64 i = 0; i < nyz; ++i) {
      slab[i] = other_data[i];
    }
  }
#else
#pragma omp parallel for MY_SCHEDULE
  for (uint64 i = 0; i < ngrid3_; i++) {
    data_[i] = other_data[i];
  }
#endif
  // Init.Stop();
}

void Array3D::restore_from(const Array3D &other) {
  assert(is_fourier_space_ == other.is_fourier_space_);
  if (other.data_[1] != data_[1] ||
      other.data_[1 + ngrid_[2]] != data_[1 + ngrid_[2]] ||
      other.data_[ngrid3_ - 1] != data_[ngrid3_ - 1]) {
    // Init.Start();
    copy_from(other);
    // Init.Stop();
  }
}

void Array3D::add_scalar(Float s) {
  assert(!is_fourier_space_);
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
  assert(!is_fourier_space_);
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
  assert(!is_fourier_space_);
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
