#include "fft_grid.h"

#include <assert.h>
#include <math.h>

#include "../array/array_ops.h"
#include "../array/row_major_array.h"
#include "../multithreading.h"

FftGrid::FftGrid(std::array<int, 3> shape)
    : rshape_(shape), cshape_({shape[0], shape[1], rshape_[2] / 2 + 1}) {
  setup_time_.start();
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
  grid_.allocate(dshape);
  // We want to touch the whole array before FFT planning.
  array_ops::set_all(0.0, grid_);
  // cgrid_ is a complex view.
  cgrid_.set_data({dshape[0], dshape[1], dshape[2] / 2},
                  (Complex *)grid_.data());

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

void FftGrid::plan_fft(unsigned flags) {
  plan_time_.start();
  Float *data = grid_.data();
  fftw_complex *cdata = (fftw_complex *)data;  // Interpret data as complex.
  const int *dshape = grid_.shape().data();

#ifndef FFTSLAB
  const int nfft[3] = {dshape[0], dshape[1], dshape[2]};
  const int nfftc[3] = {dshape[0], dshape[1], dshape[2] / 2};
  int howmany = 1;  // Only one forward and inverse FFT.
  int dist = 0;     // Unused because howmany = 1.
  int stride = 1;   // Array is continuous in memory.
  fft_ = fftw_plan_many_dft_r2c(3, rshape_.data(), howmany, data, nfft, stride,
                                dist, cdata, nfftc, stride, dist, flags);
  ifft_ = fftw_plan_many_dft_c2r(3, rshape_.data(), howmany, cdata, nfftc,
                                 stride, dist, data, nfft, stride, dist, flags);
#else
  // Split into 2D and 1D by hand.
  {
    const int nfft2[2] = {dshape[1], dshape[2]};
    const int nfft2c[2] = {dshape[1], dshape[2] / 2};
    const int nYZ[2] = {rshape_[1], rshape_[2]};
    int howmany = 1;
    int dist = 0;
    int stride = 1;
    fftyz_ = fftw_plan_many_dft_r2c(2, nYZ, howmany, data, nfft2, stride, dist,
                                    cdata, nfft2c, stride, dist, flags);
    ifftyz_ = fftw_plan_many_dft_c2r(2, nYZ, howmany, cdata, nfft2c, stride,
                                     dist, data, nfft2, stride, dist, flags);
  }
  // After the 2D r2c FFT, we have to do the 1D c2c transform.
  {
    const int nX[1] = {rshape_[0]};
    // Parallelize over Y, so we're doing this many 1D FFTs at a time.
    int howmany = rshape_[2] / 2 + 1;
    int dist = 1;
    // Elements in the X direction are separated by this many complex numbers.
    int stride = dshape[1] * dshape[2] / 2;
    const int *embed = NULL;
    fftx_ = fftw_plan_many_dft(1, nX, howmany, cdata, embed, stride, dist,
                               cdata, embed, stride, dist, -1, flags);
    ifftx_ = fftw_plan_many_dft(1, nX, howmany, cdata, embed, stride, dist,
                                cdata, embed, stride, dist, +1, flags);
  }
#endif
  plan_time_.stop();
}

bool FftGrid::fft_ready() {
#ifndef FFTSLAB
  return fft_ != NULL;
#else
  return fftx_ != NULL;
#endif
}

void FftGrid::execute_fft() {
  assert(fft_ready());
  fft_time_.start();
#ifndef FFTSLAB
  fftw_execute(fft_);
#else
  fftyz_time_.start();
  Float *data = grid_.data();
  // Then need to call this for every slab.  Can OMP these lines
  int dsize_z = grid_.shape(2);
#pragma omp parallel for MY_SCHEDULE
  for (int x = 0; x < rshape_[0]; x++) {
    fftw_execute_dft_r2c(fftyz_, data + x * rshape_[1] * dsize_z,
                         (fftw_complex *)data + x * rshape_[1] * dsize_z / 2);
  }
  fftyz_time_.stop();
  fftx_time_.start();
#pragma omp parallel for schedule(dynamic, 1)
  for (int y = 0; y < rshape_[1]; y++) {
    fftw_execute_dft(fftx_, (fftw_complex *)data + y * dsize_z / 2,
                     (fftw_complex *)data + y * dsize_z / 2);
  }
  fftx_time_.stop();
#endif
  fft_time_.stop();
}

void FftGrid::execute_ifft() {
  assert(fft_ready());
  fft_time_.start();
#ifndef FFTSLAB
  fftw_execute(ifft_);
#else
  fftx_time_.start();
  Float *data = grid_.data();
  // Then need to call this for every slab.  Can OMP these lines
  int dsize_z = grid_.shape(2);
#pragma omp parallel for schedule(dynamic, 1)
  for (int y = 0; y < rshape_[1]; y++) {
    fftw_execute_dft(ifftx_, (fftw_complex *)data + y * dsize_z / 2,
                     (fftw_complex *)data + y * dsize_z / 2);
  }
  fftx_time_.stop();
  fftyz_time_.start();
#pragma omp parallel for MY_SCHEDULE
  for (int x = 0; x < rshape_[0]; x++) {
    fftw_execute_dft_c2r(ifftyz_,
                         (fftw_complex *)data + x * rshape_[1] * dsize_z / 2,
                         data + x * rshape_[1] * dsize_z);
  }
  fftyz_time_.stop();
#endif
  fft_time_.stop();
}

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
        const Float *grid_data = grid_.get_row(ii, jj);
        if (mult) {
          out_data[k] += m[k] * grid_data[kk];
        } else {
          out_data[k] += grid_data[kk];
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
      const Complex *cgrid_data = cgrid_.get_row(iin, jjn);
      for (int k = 0; k < oz; ++k) {
        if (mult) {
          out_data[k] += m[k] * std::real(cgrid_data[oz - k]);
        } else {
          out_data[k] += std::real(cgrid_data[oz - k]);
        }
      }
      // The positive half-plane (inclusive)
      // This is (ii,jj,-oz)
      cgrid_data = cgrid_.get_row(ii, jj);
      for (int k = oz; k < oshape[2]; ++k) {
        if (mult) {
          out_data[k] += m[k] * std::real(cgrid_data[k - oz]);
        } else {
          out_data[k] += std::real(cgrid_data[k - oz]);
        }
      }
    }
  }
  extract_time_.stop();
}