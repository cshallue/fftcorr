#include "fft_grid.h"

#include <assert.h>
#include <math.h>

#include "../array/array_ops.h"
#include "../array/row_major_array.h"
#include "../multithreading.h"

FftGrid::FftGrid(std::array<int, 3> shape) {
  setup_time_.start();
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
  arr_.allocate(dshape);
  data_ = arr_.data();
  array_ops::set_all(0.0, arr_);  // Very Important. Touch the whole array.
  // carr_ is a complex view.
  carr_.set_data({dshape[0], dshape[1], dshape[2] / 2}, (Complex *)data_);
  cdata_ = carr_.data();

// NULL is a valid fftw_plan value; the planner will return NULL if it fails.
#ifndef FFTSLAB
  fft_ = NULL;
  ifft_ = NULL;
#else
  fftx_ = NULL;
  fftyz_ = NULL;
  ifftyz_ = NULL;
  ifftx_ = NULL;
#endif
  setup_time_.stop();
  plan_fft();
}

FftGrid::~FftGrid() {
  // Destroy the FFT plans.
#ifndef FFTSLAB
  if (fft_ != NULL) fftw_destroy_plan(fft_);
  if (ifft_ != NULL) fftw_destroy_plan(ifft_);
#else
  if (fftx_ != NULL) fftw_destroy_plan(fftx_);
  if (fftyz_ != NULL) fftw_destroy_plan(fftyz_);
  if (ifftx_ != NULL) fftw_destroy_plan(ifftx_);
  if (ifftyz_ != NULL) fftw_destroy_plan(ifftyz_);
#endif
#ifdef OPENMP
#ifndef FFTSLAB
  fftw_cleanup_threads();
#endif
#endif
}

void FftGrid::plan_fft() {
  plan_time_.start();
  // Setup the FFTW plans, possibly from disk, and save the wisdom
  fprintf(stdout, "# Planning the FFTs...");
  fflush(NULL);
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
  Float *data = data_;
  fftw_complex *cdata = (fftw_complex *)data;

#ifndef FFTSLAB
  // The following interface should work even if dsize_z was 'non-minimal',
  // as might be desired by padding.
  int nfft[3], nfftc[3];
  nfft[0] = nfftc[0] = rshape_[0];
  nfft[1] = nfftc[1] = rshape_[1];
  // Since dsize_z is always even, this will trick
  // FFTW to assume dsize_z/2 Complex numbers in the result, while
  // fulfilling that nfft[2]>=shape[2].
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
  fftyz_ = fftw_plan_many_dft_r2c(2, nYZ, 1, data, nfft2, 1, 0, cdata, nfft2c,
                                  1, 0, FFTW_MEASURE);
  ifftyz_ = fftw_plan_many_dft_c2r(2, nYZ, 1, cdata, nfft2c, 1, 0, data, nfft2,
                                   1, 0, FFTW_MEASURE);

  // After we've done the 2D r2c FFT, we have to do the 1D c2c transform.
  // We'll plan to parallelize over Y, so that we're doing (shape[2]/2+1)
  // 1D FFTs at a time.
  // Elements in the X direction are separated by shape[1]*dsize_z/2 complex
  // numbers.
  int nX = rshape_[0];
  fftx_ = fftw_plan_many_dft(1, &nX, (rshape_[2] / 2 + 1), cdata, NULL,
                             rshape_[1] * dsize_z / 2, 1, cdata, NULL,
                             rshape_[1] * dsize_z / 2, 1, -1, FFTW_MEASURE);
  ifftx_ = fftw_plan_many_dft(1, &nX, (rshape_[2] / 2 + 1), cdata, NULL,
                              rshape_[1] * dsize_z / 2, 1, cdata, NULL,
                              rshape_[1] * dsize_z / 2, 1, +1, FFTW_MEASURE);
#endif

  fp = fopen(WISDOMFILE, "w");
  assert(fp != NULL);
  fftw_export_wisdom_to_file(fp);
  fclose(fp);
  fprintf(stdout, "Done!\n");
  fflush(NULL);
  plan_time_.stop();
}

void FftGrid::execute_fft() {
  // TODO: assert that setup has been called. Decide best way to crash with
  // informative message. Same with execute_ifft
  fft_time_.start();
#ifndef FFTSLAB
  fftw_execute(fft_);
#else
  fftyz_time_.start();
  // Then need to call this for every slab.  Can OMP these lines
  int dsize_z = arr_.shape(2);
#pragma omp parallel for MY_SCHEDULE
  for (int x = 0; x < rshape_[0]; x++) {
    fftw_execute_dft_r2c(fftyz_, data_ + x * rshape_[1] * dsize_z,
                         (fftw_complex *)data_ + x * rshape_[1] * dsize_z / 2);
  }
  fftyz_time_.stop();
  fftx_time_.start();
#pragma omp parallel for schedule(dynamic, 1)
  for (int y = 0; y < rshape_[1]; y++) {
    fftw_execute_dft(fftx_, (fftw_complex *)data_ + y * dsize_z / 2,
                     (fftw_complex *)data_ + y * dsize_z / 2);
  }
  fftx_time_.stop();
#endif
  fft_time_.stop();
}

void FftGrid::execute_ifft() {
  fft_time_.start();
#ifndef FFTSLAB
  fftw_execute(ifft_);
#else
  fftx_time_.start();
  // Then need to call this for every slab.  Can OMP these lines
  int dsize_z = arr_.shape(2);
#pragma omp parallel for schedule(dynamic, 1)
  for (int y = 0; y < rshape_[1]; y++) {
    fftw_execute_dft(ifftx_, (fftw_complex *)data_ + y * dsize_z / 2,
                     (fftw_complex *)data_ + y * dsize_z / 2);
  }
  fftx_time_.stop();
  fftyz_time_.start();
#pragma omp parallel for MY_SCHEDULE
  for (int x = 0; x < rshape_[0]; x++) {
    fftw_execute_dft_c2r(ifftyz_,
                         (fftw_complex *)data_ + x * rshape_[1] * dsize_z / 2,
                         data_ + x * rshape_[1] * dsize_z);
  }
  fftyz_time_.stop();
#endif
  fft_time_.stop();
}

// void FftGrid::restore_from(const RowMajorArrayPtr<Float, 3> &other) {
//   // TODO: check same dimensions.
//   if (other.at(0, 0, 1) != arr_.at(0, 0, 1) ||
//       other.at(0, 1, 1) != arr_.at(0, 1, 1) ||
//       other.at(1, 1, 1) != arr_.at(1, 1, 1)) {
//     setup_time_.start();
//     arr_.copy_from(other);
//     setup_time_.stop();
//   }
// }

void FftGrid::extract_submatrix(RowMajorArrayPtr<Float, 3> *out) const {
  extract_submatrix(out, NULL);
}

void FftGrid::extract_submatrix(RowMajorArrayPtr<Float, 3> *out,
                                const RowMajorArrayPtr<Float, 3> *mult) const {
  extract_time_.start();
  // TODO: check dimensions.
  // Extract out a submatrix, centered on [0,0,0] of this array
  // Multiply elementwise by mult.
  const std::array<int, 3> &oshape = out->shape();
  int ox = oshape[0] / 2;  // This is the middle of the submatrix
  int oy = oshape[1] / 2;  // This is the middle of the submatrix
  int oz = oshape[2] / 2;  // This is the middle of the submatrix

  // TODO: if mult is null, create an array of 1s. Currently, the only time we
  // DON'T have mult is in the isotropic case, we only call this function once
  // in that case. Meanwhile, we call this function many times in the
  // anisotropic case, where mult is not null.
#pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < oshape[0]; ++i) {
    int ii = (rshape_[0] - ox + i) % rshape_[0];
    for (int j = 0; j < oshape[1]; ++j) {
      int jj = (rshape_[1] - oy + j) % rshape_[1];
      for (int k = 0; k < oshape[2]; ++k) {
        int kk = (rshape_[2] - oz + k) % rshape_[2];
        Float *out_data = out->get_row(i, j);
        const Float *m = mult ? mult->get_row(i, j) : NULL;
        const Float *arr_data = arr_.get_row(ii, jj);
        if (mult) {
          out_data[k] += m[k] * arr_data[kk];
        } else {
          out_data[k] += arr_data[kk];
        }
      }
    }
  }
  extract_time_.stop();
}

void FftGrid::extract_submatrix_C2R(RowMajorArrayPtr<Float, 3> *out) const {
  extract_submatrix_C2R(out, NULL);
}

void FftGrid::extract_submatrix_C2R(
    RowMajorArrayPtr<Float, 3> *out,
    const RowMajorArrayPtr<Float, 3> *mult) const {
  extract_time_.start();
  // Given a large matrix work[shape^3/2],
  // extract out a submatrix of size csize^3, centered on work[0,0,0].
  // The input matrix is Complex * with the half-domain Fourier convention.
  // We are only summing the real part; the imaginary part always sums to zero.
  // Need to reflect the -z part around the origin, which also means reflecting
  // x & y. shape[2] and shape2 are given as their Float values, not yet divided
  // by two. Multiply the result by corr[csize^3] and add it onto total[csize^3]
  // Again, zero lag is mapping to corr(csize/2, csize/2, csize/2),
  // but it is at (0,0,0) in the FFT grid.
  const std::array<int, 3> &oshape = out->shape();
  int ox = oshape[0] / 2;  // This is the middle of the submatrix
  int oy = oshape[1] / 2;  // This is the middle of the submatrix
  int oz = oshape[2] / 2;  // This is the middle of the submatrix
#pragma omp parallel for schedule(dynamic, 1)
  for (int i = 0; i < oshape[0]; ++i) {
    int ii = (cshape_[0] - ox + i) % cshape_[0];
    int iin = (cshape_[0] - ii) % cshape_[0];  // The reflected coord
    for (int j = 0; j < oshape[1]; ++j) {
      int jj = (cshape_[1] - oy + j) % cshape_[1];
      int jjn = (cshape_[1] - jj) % cshape_[1];  // The reflected coord
      // The negative half-plane (inclusive), reflected.
      // k=oz-1 should be +1, k=0 should be +oz
      // This is (iin,jjn,+oz)
      Float *out_data = out->get_row(i, j);
      const Float *m = mult ? mult->get_row(i, j) : NULL;
      const Complex *carr_data = carr_.get_row(iin, jjn);
      for (int k = 0; k < oz; ++k) {
        if (mult) {
          out_data[k] += m[k] * std::real(carr_data[oz - k]);
        } else {
          out_data[k] += std::real(carr_data[oz - k]);
        }
      }
      // The positive half-plane (inclusive)
      // This is (ii,jj,-oz)
      carr_data = carr_.get_row(ii, jj);
      for (int k = oz; k < oshape[2]; ++k) {
        if (mult) {
          out_data[k] += m[k] * std::real(carr_data[k - oz]);
        } else {
          out_data[k] += std::real(carr_data[k - oz]);
        }
      }
    }
  }
  extract_time_.stop();
}

// Returns the ith frequency of the DFT in grid units.
Float fftfreq(int i, int n) {
  return i < n / 2 ? i / ((Float)n) : (i - n) / ((Float)n);
}

// Returns exp(- 2 pi^2 sigma^2 k^2)
Array1D<Float> kgaussian(int n, Float sigma) {
  Array1D<Float> out(n);
  Float two_pi2_sigma2 = 2 * M_PI * M_PI * sigma * sigma;
  Float k;
  for (int i = 0; i < n; ++i) {
    k = fftfreq(i, n);
    out[i] = exp(-two_pi2_sigma2 * k * k);
  }
  return out;
}

void FftGrid::convolve_with_gaussian(Float sigma) {
  convolve_time_.start();
  execute_fft();

  int nx = rshape_[0];
  int ny = rshape_[1];
  int nz = rshape_[2];
  Array1D<Float> expkx = kgaussian(nx, sigma);
  Array1D<Float> expky = kgaussian(ny, sigma);
  Array1D<Float> expkz = kgaussian(nz, sigma);

#pragma omp parallel for schedule(dynamic, 1)
  for (int ix = 0; ix < cshape_[0]; ++ix) {
    for (int iy = 0; iy < cshape_[1]; ++iy) {
      // We're including the factor of (nx * ny * nz) that FFTW doesn't include
      // in the inverse FFT.
      Float expkxky = expkx[ix] * expky[iy] / nx / ny / nz;
      Complex *row = carr_.get_row(ix, iy);
      for (int iz = 0; iz < cshape_[2]; ++iz) {
        row[iz] *= expkxky * expkz[iz];
      }
    }
  }

  execute_ifft();
  convolve_time_.stop();
}