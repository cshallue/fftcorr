#include "discrete_field.h"

DiscreteField::DiscreteField(std::array<int, 3> shape) {
  rshape_ = shape;
  cshape_ = std::array<int, 3>({shape[0], shape[1], rshape_[2] / 2 + 1});
  rsize_ = (uint64)rshape_[0] * rshape_[1] * rshape_[2];
  csize_ = (uint64)cshape_[0] * cshape_[1] * cshape_[2];

  int dsize_z;  // dsize_z pads out the array for the in-place FFT.
#ifdef FFTSLAB
// The rest of the code should work even if extra space is used.
// Some operations will blindly apply to the pad cells, but that's ok.
// In particular, we might consider having dsize_z be evenly divisible by
// the critical alignment stride (32 bytes for AVX, but might be more for cache
// lines) or even by a full PAGE for NUMA memory.  Doing this *will* force a
// more complicated FFT, but at least for the NUMA case this is desired: we want
// to force the 2D FFT to run on its socket, and only have the last 1D FFT
// crossing sockets.  Re-using FFTW plans requires the consistent memory
// alignment.
#define FFT_ALIGN 16
  // This is in units of Floats.  16 doubles is 1024 bits.
  dsize_z = FFT_ALIGN * (rshape_[2] / FFT_ALIGN + 1);
#else
  // The default 3d FFTW format must have the following:
  dsize_z = 2 * (rshape_[2] / 2 + 1);  // For the in-place FFT
#endif
  assert(dsize_z % 2 == 0);
  fprintf(stdout, "# Using dsize_z_=%d for FFT r2c padding\n", dsize_z);

  std::array<int, 3> dshape = {rshape_[0], rshape_[1], dsize_z};
  arr_.initialize(dshape);

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

DiscreteField::~DiscreteField() {
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

void DiscreteField::setup_fft() {
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

  // Interpret arr_ as complex.
  Float *data = arr_.data();
  fftw_complex *cdata = (fftw_complex *)data;

#ifndef FFTSLAB
  // The following interface should work even if dsize_z was 'non-minimal',
  // as might be desired by padding.
  int nfft[3], nfftc[3];
  nfft[0] = nfftc[0] = rshape_[0];
  nfft[1] = nfftc[1] = rshape_[1];
  // Since dsize_z is always even, this will trick
  // FFTW to assume dsize_z/2 Complex numbers in the result, while
  // fulfilling that nfft[2]>=ngrid[2].
  nfft[2] = arr_.shape(2);
  nfftc[2] = nfft[2] / 2;
  int howmany = 1;  // Only one forward and inverse FFT.
  int dist = 0;     // Unused because howmany = 1.
  int stride = 1;   // Array is continuous in memory.
  fft_ = fftw_plan_many_dft_r2c(3, rshape_.data(), howmany, data, nfft, stride,
                                dist, cdata, nfftc, stride, dist, FFTW_MEASURE);
  ifft_ =
      fftw_plan_many_dft_c2r(3, rshape_.data(), howmany, cdata, nfftc, stride,
                             dist, data, nfft, stride, dist, FFTW_MEASURE);

#else
  // If we wanted to split into 2D and 1D by hand (and therefore handle the OMP
  // aspects ourselves), then we need to have two plans each.
  int dsize_z = arr_.shape(2);
  int nfft2[2], nfft2c[2];
  nfft2[0] = nfft2c[0] = rshape_[1];
  nfft2[1] = dsize_z;  // Since dsize_z is always even, this will trick
  nfft2c[1] = nfft2[1] / 2;
  int nYZ[2];
  nYZ[0] = rshape_[1];
  nYZ[1] = rshape_[2];
  fftYZ_ = fftw_plan_many_dft_r2c(2, nYZ, 1, data, nfft2, 1, 0, cdata, nfft2c,
                                  1, 0, FFTW_MEASURE);
  ifftYZ_ = fftw_plan_many_dft_c2r(2, nYZ, 1, cdata, nfft2c, 1, 0, data, nfft2,
                                   1, 0, FFTW_MEASURE);

  // After we've done the 2D r2c FFT, we have to do the 1D c2c transform.
  // We'll plan to parallelize over Y, so that we're doing (ngrid[2]/2+1)
  // 1D FFTs at a time.
  // Elements in the X direction are separated by ngrid[1]*dsize_z/2 complex
  // numbers.
  int nX = rshape_[0];
  fftX_ = fftw_plan_many_dft(1, &nX, (rshape_[2] / 2 + 1), cdata, NULL,
                             rshape_[1] * dsize_z / 2, 1, cdata, NULL,
                             rshape_[1] * dsize_z / 2, 1, -1, FFTW_MEASURE);
  ifftX_ = fftw_plan_many_dft(1, &nX, (rshape_[2] / 2 + 1), cdata, NULL,
                              rshape_[1] * dsize_z / 2, 1, cdata, NULL,
                              rshape_[1] * dsize_z / 2, 1, +1, FFTW_MEASURE);
#endif

  fp = fopen(WISDOMFILE, "w");
  assert(fp != NULL);
  fftw_export_wisdom_to_file(fp);
  fclose(fp);
  fprintf(stdout, "Done!\n");
  fflush(NULL);
  // FFTW.Stop();
}

void DiscreteField::execute_fft() {
  // TODO: assert that setup has been called. Decide best way to crash with
  // informative message. Same with execute_ifft
  // FFTonly.Start();
#ifndef FFTSLAB
  fftw_execute(fft_);
#else
  // FFTyz.Start();
  // Then need to call this for every slab.  Can OMP these lines
  int dsize_z = arr_.shape(2);
#pragma omp parallel for MY_SCHEDULE
  for (uint64 x = 0; x < rshape_[0]; x++)
    fftw_execute_dft_r2c(fftYZ_, data + x * rshape_[1] * dsize_z,
                         (fftw_complex *)data + x * rshape_[1] * dsize_z / 2);
    // FFTyz.Stop();
    // FFTx.Start();
#pragma omp parallel for schedule(dynamic, 1)
  for (uint64 y = 0; y < rshape_[1]; y++)
    fftw_execute_dft(fftX_, (fftw_complex *)data + y * dsize_z / 2,
                     (fftw_complex *)data + y * dsize_z / 2);
    // FFTx.Stop();
#endif
  // FFTonly.Stop();
}

void DiscreteField::execute_ifft() {
  // FFTonly.Start();
#ifndef FFTSLAB
  fftw_execute(ifft_);
#else
  // FFTx.Start();
  // Then need to call this for every slab.  Can OMP these lines
  int dsize_z = arr_.shape(2);
#pragma omp parallel for schedule(dynamic, 1)
  for (uint64 y = 0; y < rshape_[1]; y++)
    fftw_execute_dft(ifftX_, (fftw_complex *)data + y * dsize_z / 2,
                     (fftw_complex *)data + y * dsize_z / 2);
    // FFTx.Stop();
    // FFTyz.Start();
#pragma omp parallel for MY_SCHEDULE
  for (uint64 x = 0; x < rshape_[0]; x++)
    fftw_execute_dft_c2r(ifftYZ_,
                         (fftw_complex *)data + x * rshape_[1] * dsize_z / 2,
                         data + x * rshape_[1] * dsize_z);
    // FFTyz.Stop();
#endif
  // FFTonly.Stop();
}

void DiscreteField::copy_from(const DiscreteField &other) {
  // TODO: check same dimensions.
  arr_.copy_from(other.arr_);
}

void DiscreteField::restore_from(const DiscreteField &other) {
  // TODO: check same dimensions.
  Float *this_data = arr_.data();
  const Float *other_data = other.arr_.data();
  if (other_data[1] != this_data[1] ||
      other_data[1 + dshape()[2]] != this_data[1 + dshape()[2]] ||
      other_data[dsize() - 1] != this_data[dsize() - 1]) {
    // Init.Start();
    arr_.copy_from(other.arr_);
    // Init.Stop();
  }
}

void DiscreteField::add_scalar(Float s) { arr_.add_scalar(s); }

Float DiscreteField::sum() const { return arr_.sum(); }

Float DiscreteField::sumsq() const { return arr_.sumsq(); }

void DiscreteField::multiply_with_conjugation(const DiscreteField &other) {
  arr_.multiply_with_conjugation(other.arr_);
}

void DiscreteField::extract_submatrix(Array3D *out) const {
  // Extract out a submatrix, centered on [0,0,0] of this array
  // Extract.Start();
  const std::array<int, 3> &ngrid = rshape_;
  const std::array<int, 3> &oshape = out->shape();
  int cx = oshape[0] / 2;  // This is the middle of the submatrix
  int cy = oshape[1] / 2;  // This is the middle of the submatrix
  int cz = oshape[2] / 2;  // This is the middle of the submatrix
#pragma omp parallel for schedule(dynamic, 1)
  for (uint64 i = 0; i < oshape[0]; ++i) {
    uint64 ii = (ngrid[0] - cx + i) % ngrid[0];
    for (int j = 0; j < oshape[1]; ++j) {
      uint64 jj = (ngrid[1] - cy + j) % ngrid[1];
      for (int k = 0; k < oshape[2]; ++k) {
        uint64 kk = (ngrid[2] - cz + k) % ngrid[2];
        out->at(i, j, k) += arr_.at(ii, jj, kk);
      }
    }
  }
  // Extract.Stop();
}

// TODO: could unify this with the above with probably minimal overhead.
void DiscreteField::extract_submatrix(const Array3D &mult, Array3D *out) const {
  // Extract out a submatrix, centered on [0,0,0] of this array
  // Multiply elementwise by mult.
  // Extract.Start();
  const std::array<int, 3> &oshape = out->shape();
  int cx = oshape[0] / 2;  // This is the middle of the submatrix
  int cy = oshape[1] / 2;  // This is the middle of the submatrix
  int cz = oshape[2] / 2;  // This is the middle of the submatrix
#pragma omp parallel for schedule(dynamic, 1)
  for (uint64 i = 0; i < oshape[0]; ++i) {
    uint64 ii = (rshape_[0] - cx + i) % rshape_[0];
    for (int j = 0; j < oshape[1]; ++j) {
      uint64 jj = (rshape_[1] - cy + j) % rshape_[1];
      for (int k = 0; k < oshape[2]; ++k) {
        uint64 kk = (rshape_[2] - cz + k) % rshape_[2];
        out->at(i, j, k) += mult.at(i, j, k) * arr_.at(ii, jj, kk);
      }
    }
  }
  // Extract.Stop();
}

// TODO: simplify this like the above.
void DiscreteField::extract_submatrix_C2R(const Array3D &corr,
                                          Array3D *total) const {
  // Given a large matrix work[ngrid^3/2],
  // extract out a submatrix of size csize^3, centered on work[0,0,0].
  // The input matrix is Complex * with the half-domain Fourier convention.
  // We are only summing the real part; the imaginary part always sums to zero.
  // Need to reflect the -z part around the origin, which also means reflecting
  // x & y. ngrid[2] and ngrid2 are given as their Float values, not yet divided
  // by two. Multiply the result by corr[csize^3] and add it onto total[csize^3]
  // Again, zero lag is mapping to corr(csize/2, csize/2, csize/2),
  // but it is at (0,0,0) in the FFT grid.
  // Extract.Start();
  const std::array<int, 3> &ngrid = rshape_;
  int ngrid2 = arr_.shape(2);
  const std::array<int, 3> &csize = corr.shape();
  int cx = csize[0] / 2;  // This is the middle of the submatrix
  int cy = csize[1] / 2;  // This is the middle of the submatrix
  int cz = csize[2] / 2;  // This is the middle of the submatrix
  const Complex *work = arr_.cdata();
  Float *tdata = total->data();
  const Float *cdata = corr.data();
#pragma omp parallel for schedule(dynamic, 1)
  for (uint64 i = 0; i < csize[0]; i++) {
    uint64 ii = (ngrid[0] - cx + i) % ngrid[0];
    uint64 iin = (ngrid[0] - ii) % ngrid[0];  // The reflected coord
    for (int j = 0; j < csize[1]; j++) {
      uint64 jj = (ngrid[1] - cy + j) % ngrid[1];
      uint64 jjn = (ngrid[1] - jj) % ngrid[1];           // The reflected coord
      Float *t = tdata + (i * csize[1] + j) * csize[2];  //  (i,j,0)
      const Float *cc = cdata + (i * csize[1] + j) * csize[2];  //  (i,j,0)
      // The positive half-plane (inclusize)
      const Complex *Y = work + (ii * ngrid[1] + jj) * ngrid2 / 2 - cz;
      // This is (ii,jj,-cz)
      for (int k = cz; k < csize[2]; k++) t[k] += cc[k] * std::real(Y[k]);
      // The negative half-plane (inclusize), reflected.
      // k=cz-1 should be +1, k=0 should be +cz
      Y = work + (iin * ngrid[1] + jjn) * ngrid2 / 2 + cz;
      // This is (iin,jjn,+cz)
      for (int k = 0; k < cz; k++) t[k] += cc[k] * std::real(Y[-k]);
    }
  }
  // Extract.Stop();
}