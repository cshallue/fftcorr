#include "array3d.h"

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

  // Allocate dens_ to [ngrid3_] and set it to zero
  // TODO: do we always want to initialize on construction?
  data_ = NULL;
  initialize(data_, ngrid3_, ngrid_[0]);
}

Array3D::~Array3D() {
  if (data_ != NULL) free(data_);
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

void Array3D::initialize(Float *&m, const uint64 size, const int nx) {
  // Initialize a matrix m and set it to zero.
  // We want to touch the whole matrix, because in NUMA this defines the
  // association of logical memory into the physical banks. nx will be our slab
  // decomposition; it must divide into size evenly Warning: This will only
  // allocate new space if m==NULL.  This allows one to reuse space.  But(!!)
  // there is no check that you've not changed the size of the matrix -- you
  // could overflow the previously allocated space.
  fprintf(stderr, "size: %llu, nx: %d\n", size, nx);
  assert(size % nx == 0);
  // Init.Start();
  if (m == NULL) {
    int err = posix_memalign((void **)&m, PAGE, sizeof(Float) * size + PAGE);
    assert(err == 0);
  }
  assert(m != NULL);
#ifdef SLAB
  const uint64 nyz = size / nx;
#pragma omp parallel for MY_SCHEDULE
  for (int x = 0; x < nx; x++) {
    Float *mslab = m + x * nyz;
    for (uint64 j = 0; j < nyz; j++) mslab[j] = 0.0;
  }
#else
#pragma omp parallel for MY_SCHEDULE
  for (uint64 j = 0; j < size; j++) m[j] = 0.0;
#endif
  // Init.Stop();
  return;
}