#ifndef CORRELATE_H
#define CORRELATE_H

#include <assert.h>

#include <array>
#include <cmath>
#include <vector>

#include "../array/array_ops.h"
#include "../array/row_major_array.h"
#include "../grid/config_space_grid.h"
#include "../grid/fft_grid.h"
#include "../histogram/histogram_list.h"
#include "../particle_mesh/window_functions.h"
#include "../profiling/timer.h"
#include "../types.h"
#include "spherical_harmonics.h"

// TODO: locate somewhere else.
Array1D<Float> sequence(Float start, Float step, int size) {
  Array1D<Float> seq(size);
  for (int i = 0; i < size; ++i) {
    seq[i] = start + i * step;
  }
  return seq;
}

// TODO: rename dens* to something more generic
class BaseCorrelator {
 public:
  BaseCorrelator(const std::array<int, 3> shape, Float rmax, Float dr,
                 Float kmax, Float dk, int maxell, unsigned fftw_flags)
      : rmax_(rmax),
        kmax_(kmax),
        maxell_(maxell),
        work_(shape),
        fftw_flags_(fftw_flags),
        rhist_(maxell / 2 + 1, 0.0, rmax, dr),
        khist_(maxell / 2 + 1, 0.0, kmax, dk) {}

  virtual void autocorrelate(const ConfigSpaceGrid &dens) = 0;

  virtual void cross_correlate(const ConfigSpaceGrid &dens1) = 0;

  void cross_correlate(const ConfigSpaceGrid &dens1,
                       const ConfigSpaceGrid &dens2) {
    set_dens2(dens2);
    cross_correlate(dens1);
  }

  void set_dens2(const ConfigSpaceGrid &dens2) {
    setup_time_.start();
    array_ops::copy_into_padded_array(dens2.data(), work_.as_real_array());
    setup_fft();
    setup_time_.stop();

    fprintf(stdout, "# Computing the density 2 FFT...");
    fflush(NULL);
    work_.execute_fft();
    fprintf(stdout, "# Done!\n");
    fflush(NULL);

    // Save the FFT of dens2.
    setup_time_.start();
    if (!dens2_fft_.data()) {
      dens2_fft_.allocate(work_.as_complex_array().shape());
    }
    array_ops::copy(work_.as_complex_array(), dens2_fft_);
    setup_time_.stop();
  }

  // Output accessors.
  Float zerolag() const { return zerolag_; }
  const ArrayPtr1D<Float> &correlation_r() const { return rhist_.bins(); }
  const RowMajorArrayPtr<int, 2> &correlation_counts() const {
    return rhist_.counts();
  }
  const RowMajorArrayPtr<Float, 2> &correlation_histogram() const {
    return rhist_.hist_values();
  }
  const ArrayPtr1D<Float> &power_spectrum_k() const { return khist_.bins(); }
  const RowMajorArrayPtr<int, 2> &power_spectrum_counts() const {
    return khist_.counts();
  }
  const RowMajorArrayPtr<Float, 2> &power_spectrum_histogram() const {
    return khist_.hist_values();
  }

  // Timer accessors.
  Float setup_time() const {
    return setup_time_.elapsed_sec() + work_.setup_time();
  }
  Float fft_plan_time() const { return work_.plan_time(); }
  Float fft_time() const { return work_.fft_time(); }
  Float extract_time() const { return work_.extract_time(); }
  Float multiply_time() const { return mult_time_.elapsed_sec(); }
  Float ylm_time() const { return ylm_time_.elapsed_sec(); }
  Float histogram_time() const { return hist_time_.elapsed_sec(); }
  Float total_time() const { return total_time_.elapsed_sec(); }

 protected:
  void setup_fft() {
    if (!work_.fft_ready()) {
      work_.plan_fft(fftw_flags_);  // Planning may destroy data in work_.
    }
  }

  void setup_correlate(const ConfigSpaceGrid &dens) {
    setup_time_.start();
    setup_fft();
    if (!rgrid_.data()) {
      setup_rgrid(dens);
    }
    if (!kgrid_.data()) {
      setup_kgrid(dens);
    }
    rhist_.reset();
    khist_.reset();
    zerolag_ = -1.0;
    setup_time_.stop();
  }

  void setup_rgrid(const ConfigSpaceGrid &dens) {
    // Create the separation-space subgrid.
    Float cell_size = dens.cell_size();
    int rmax_cells = ceil(rmax_ / cell_size);  // rmax in grid cell units.
    fprintf(stderr, "rmax = %f, cell_size = %f, rmax_cells =%d\n", rmax_,
            cell_size, rmax_cells);
    // Number of cells in each dimension of the subgrid.
    int sizex = 2 * rmax_cells + 1;  // Include negative separation vectors.
    std::array<int, 3> rshape = {sizex, sizex, sizex};
    fprintf(stderr, "rgrid shape = [%d, %d, %d]\n", rshape[0], rshape[1],
            rshape[2]);
    rgrid_.allocate(rshape);

    // The axes of the cell centers in separation space in physical units.
    rx_ = sequence(-cell_size * rmax_cells, cell_size, rshape[0]);
    ry_ = sequence(-cell_size * rmax_cells, cell_size, rshape[1]);
    rz_ = sequence(-cell_size * rmax_cells, cell_size, rshape[2]);

    // Radius of each separation-space subgrid cell in physical units.
    rnorm_.allocate(rshape);
    for (int i = 0; i < rshape[0]; ++i) {
      for (int j = 0; j < rshape[1]; ++j) {
        for (int k = 0; k < rshape[2]; ++k) {
          // TODO: use get_row or something faster everywhere I call at() in a
          // loop.
          rnorm_.at(i, j, k) =
              sqrt(rx_[i] * rx_[i] + ry_[j] * ry_[j] + rz_[k] * rz_[k]);
        }
      }
    }
    // Index corresponding to zero separation (r=0).
    for (int &x : rzero_) x = rmax_cells;
    rylm_.allocate(rshape);
  }

  void setup_kgrid(const ConfigSpaceGrid &dens) {
    // Create the Fourier-space subgrid.
    // Our box has cubic-sized cells, so k_Nyquist is the same in all
    // directions. The spacing of modes is therefore 2*k_Nyq/ngrid.
    const std::array<int, 3> &ngrid = dens.shape();
    Float cell_size = dens.cell_size();
    Float k_Nyq = M_PI / cell_size;  // The Nyquist frequency for our grid.
    // Number of cells in the subgrid.
    std::array<int, 3> kshape;
    for (int i = 0; i < 3; ++i) {
      kshape[i] = 2 * ceil(kmax_ / (2.0 * k_Nyq / ngrid[i])) + 1;
    }
    fprintf(stdout, "# Storing wavenumbers up to %6.4f, with k_Nyq = %6.4f\n",
            kmax_, k_Nyq);
    for (int i = 0; i < 3; ++i) {
      if (kshape[i] > ngrid[i]) {
        // TODO: in the corner case of kmax ~= k_Nyq, this can be greater than
        // ngrid[i].
        kshape[i] = 2 * floor(ngrid[i] / 2) + 1;
        fprintf(stdout,
                "# WARNING: Requested wavenumber is too big. Truncating "
                "shape[%d] to %d\n",
                i, kshape[i]);
      }
    }
    kgrid_.allocate(kshape);
    fprintf(stdout,
            "# Done setting up the wavevector submatrix of size +-%d, %d, %d\n",
            kshape[0] / 2, kshape[1] / 2, kshape[2] / 2);
    fprintf(stderr, "kshape = [%d, %d, %d]\n", kshape[0], kshape[1], kshape[2]);

    // The axes in Fourier space.
    kx_ = sequence((-kshape[0] / 2) * 2.0 * k_Nyq / ngrid[0],
                   2.0 * k_Nyq / ngrid[0], kshape[0]);
    ky_ = sequence((-kshape[1] / 2) * 2.0 * k_Nyq / ngrid[1],
                   2.0 * k_Nyq / ngrid[1], kshape[1]);
    kz_ = sequence((-kshape[2] / 2) * 2.0 * k_Nyq / ngrid[2],
                   2.0 * k_Nyq / ngrid[2], kshape[2]);

    // Frequency of each freqency subgrid cell in physical units.
    knorm_.allocate(kshape);
    for (int i = 0; i < kshape[0]; ++i) {
      for (int j = 0; j < kshape[1]; ++j) {
        for (int k = 0; k < kshape[2]; ++k) {
          knorm_.at(i, j, k) =
              sqrt(kx_[i] * kx_[i] + ky_[j] * ky_[j] + kz_[k] * kz_[k]);
        }
      }
    }
    kylm_.allocate(kshape);

    // Set up window correction.
    inv_window_.allocate(kshape);
    Float window = 1.0;
    for (int i = 0; i < kshape[0]; ++i) {
      for (int j = 0; j < kshape[1]; ++j) {
        for (int k = 0; k < kshape[2]; ++k) {
          switch (dens.window_type()) {
            case kNearestCell:
              window = 1.0;
              break;
            case kCloudInCell: {
              // For TSC, the square window is 1-sin^2(kL/2)+2/15*sin^4(kL/2)
              Float sinkxL = sin(kx_[i] * cell_size / 2.0);
              Float sinkyL = sin(ky_[j] * cell_size / 2.0);
              Float sinkzL = sin(kz_[k] * cell_size / 2.0);
              sinkxL *= sinkxL;
              sinkyL *= sinkyL;
              sinkzL *= sinkzL;
              Float Wx, Wy, Wz;
              Wx = 1 - sinkxL + 2.0 / 15.0 * sinkxL * sinkxL;
              Wy = 1 - sinkyL + 2.0 / 15.0 * sinkyL * sinkyL;
              Wz = 1 - sinkzL + 2.0 / 15.0 * sinkzL * sinkzL;
              window = Wx * Wy * Wz;  // This is the square of the window
              break;
            }
            case kWavelet:
              window = 1.0;
              break;
          }
          inv_window_.at(i, j, k) = 1.0 / window;
          // We will divide the power spectrum by the square of the window
        }
      }
    }
  }

  // Inputs.
  Float rmax_;
  Float kmax_;
  int maxell_;

  // Work space for FFTs.
  FftGrid work_;
  unsigned fftw_flags_;

  // Separation-space arrays.
  RowMajorArray<Float, 3> rgrid_;  // TODO: rtotal_?
  std::array<int, 3> rzero_;
  Array1D<Float> rx_;
  Array1D<Float> ry_;
  Array1D<Float> rz_;
  RowMajorArray<Float, 3> rnorm_;
  RowMajorArray<Float, 3> rylm_;

  // Fourier-space arrays.
  RowMajorArray<Complex, 3> dens2_fft_;
  RowMajorArray<Float, 3> kgrid_;  // TODO: ktotal_?
  Array1D<Float> kx_;
  Array1D<Float> ky_;
  Array1D<Float> kz_;
  RowMajorArray<Float, 3> knorm_;
  RowMajorArray<Float, 3> kylm_;
  RowMajorArray<Float, 3> inv_window_;

  // Outputs.
  HistogramList rhist_;
  HistogramList khist_;
  Float zerolag_;

  // Timers.
  mutable Timer setup_time_;
  mutable Timer mult_time_;
  mutable Timer ylm_time_;
  mutable Timer hist_time_;
  mutable Timer total_time_;
};

class PeriodicCorrelator : public BaseCorrelator {
 public:
  using BaseCorrelator::BaseCorrelator;  // Inherit constructors.

  void autocorrelate(const ConfigSpaceGrid &dens) {
    correlate_internal(dens, NULL);
  }

  void cross_correlate(const ConfigSpaceGrid &dens1) {
    correlate_internal(dens1, &dens2_fft_);
  }

 protected:
  void correlate_internal(const ConfigSpaceGrid &dens1,
                          const RowMajorArrayPtr<Complex, 3> *dens2_fft) {
    total_time_.start();
    setup_correlate(dens1);

    array_ops::copy_into_padded_array(dens1.data(), work_.as_real_array());
    fprintf(stdout, "# Computing the density 1 FFT...");
    fflush(NULL);
    work_.execute_fft();
    fprintf(stdout, "# Done!\n");
    fflush(NULL);

    if (!dens2_fft) {
      dens2_fft = &work_.as_complex_array();  // Autocorrelation.
    }

    fprintf(stdout, "# Multiply...");
    fflush(NULL);
    // TODO: inplace abs^2?
    mult_time_.start();
    array_ops::multiply_with_conjugation(*dens2_fft, work_.as_complex_array());
    mult_time_.stop();

    // We must multiply the DFT result by (1/ncells^2): DFT differs from
    // Fourier series coefficients by a factor of ncells, and we've multiplied
    // two DFTs together.
    uint64 ncells = dens1.data().size();
    Float k_rescale = (1.0 / ncells / ncells);

    // Extract the power spectrum, expanded in Legendre polynomials.
    for (int ell = 0; ell <= maxell_; ell += 2) {
      setup_time_.start();
      array_ops::set_all(0.0, kgrid_);
      setup_time_.stop();
      // P_l = Y_l0 * sqrt(4.0 * M_PI / (2 * ell + 1)) and then we need to
      // multiply by (2 * ell + 1) to account for the normalization of P_l's.
      // Also include the DFT scaling.
      Float coeff = sqrt((4.0 * M_PI) * (2 * ell + 1)) * k_rescale;
      // TODO: include &inv_window if appropriate in this case.
      ylm_time_.start();
      make_ylm(ell, 0, kx_, ky_, kz_, coeff, 0, NULL, &kylm_);
      ylm_time_.stop();
      work_.extract_fft_submatrix_c2r(&kgrid_, &kylm_);
      hist_time_.start();
      khist_.accumulate(ell / 2, knorm_, kgrid_);
      hist_time_.stop();
    }

    // iFFT the result, in place
    fprintf(stdout, "IFFT...");
    fflush(NULL);
    work_.execute_ifft();
    fprintf(stdout, "# Done!\n");
    fflush(NULL);

    // We must multiply the IFT by two factors of (1/ncells): the first one
    // completes the inverse FFT (FFTW doesn't include this factor
    // automatically) and the second is the factor converting the circular
    // cross-correlation of grids to an estimator of the cross-correlation
    // function of random fields.
    Float r_rescale = (1.0 / ncells / ncells);

    // Extract the 2PCF, expanded in Legendre polynomials.
    for (int ell = 0; ell <= maxell_; ell += 2) {
      setup_time_.start();
      array_ops::set_all(0.0, rgrid_);
      setup_time_.stop();
      // P_l = Y_l0 * sqrt(4.0 * M_PI / (2 * ell + 1)) and then we need to
      // multiply by (2 * ell + 1) to account for the normalization of P_l's.
      // Also include the IFT scaling.
      Float coeff = sqrt((4.0 * M_PI) * (2 * ell + 1)) * r_rescale;
      ylm_time_.start();
      make_ylm(ell, 0, rx_, ry_, rz_, coeff, 0, NULL, &rylm_);
      ylm_time_.stop();
      work_.extract_submatrix(&rgrid_, &rylm_);
      hist_time_.start();
      rhist_.accumulate(ell / 2, rnorm_, rgrid_);
      hist_time_.stop();
      if (ell == 0) {
        zerolag_ = rgrid_.at(rzero_[0], rzero_[1], rzero_[2]);
      }
    }
    total_time_.stop();
  }
};

class Correlator : public BaseCorrelator {
 public:
  using BaseCorrelator::BaseCorrelator;  // Inherit constructors.

  void autocorrelate(const ConfigSpaceGrid &dens) {
    BaseCorrelator::cross_correlate(dens, dens);
  }

  void cross_correlate(const ConfigSpaceGrid &dens1) {
    correlate_internal(dens1);
  }

 protected:
  void setup_correlate(const ConfigSpaceGrid &dens) {
    BaseCorrelator::setup_correlate(dens);
    if (!xcell_.data()) {
      setup_cell_coords(dens);
    }
  }

  void setup_cell_coords(const ConfigSpaceGrid &dens) {
    // Location of the observer relative to posmin, in grid units.
    std::array<Float, 3> observer;
    // Put the observer at the origin of the survey coordinate system.
    for (int i = 0; i < 3; ++i) {
      observer[i] = -dens.posmin(i) / dens.cell_size();
    }
    // We cab simulate a periodic box by puting the observer centered in the
    // grid, but displaced far away in the -x direction. This is an
    // inefficient way to compute the periodic case, but it's a good sanity
    // check. for (int i = 0; i < 3; ++i) {
    //   observer[i] = dens1.shape(i) / 2.0;
    // }
    // observer[0] -= dens1.shape(0) * 1e6;  // Observer far away!

    // Coordinates of the cell centers in each dimension, relative to the
    // observer. We're using grid units (scale doesn't matter when computing
    // Ylms).
    xcell_ = sequence(0.5 - observer[0], 1.0, dens.shape(0));
    ycell_ = sequence(0.5 - observer[1], 1.0, dens.shape(1));
    zcell_ = sequence(0.5 - observer[2], 1.0, dens.shape(2));
  }

  void correlate_internal(const ConfigSpaceGrid &dens1) {
    constexpr int wide_angle_exponent = 0;  // TODO: do we still need this?
    total_time_.start();
    setup_correlate(dens1);

    array_ops::copy_into_padded_array(dens1.data(), work_.as_real_array());
    fprintf(stdout, "# Computing the density 1 FFT...");
    fflush(NULL);
    work_.execute_fft();
    fprintf(stdout, "# Done!\n");
    fflush(NULL);

    /* ------------ Loop over ell & m --------------- */
    // Loop over each ell to compute the anisotropic correlations
    for (int ell = 0; ell <= maxell_; ell += 2) {
      // Initialize the submatrix
      setup_time_.start();
      array_ops::set_all(0.0, rgrid_);
      array_ops::set_all(0.0, kgrid_);
      setup_time_.stop();
      // Loop over m
      for (int m = -ell; m <= ell; m++) {
        fprintf(stdout, "# Computing %d %2d...", ell, m);

        // Create the Ylm matrix times work_
        // TODO: here, is it advantageous if dens1 is padded as well, so its
        // boundaries match with those of work?
        ylm_time_.start();
        make_ylm(ell, m, xcell_, ycell_, zcell_, 1.0, -wide_angle_exponent,
                 &dens1.data(), &work_.as_real_array());
        ylm_time_.stop();
        fprintf(stdout, "Ylm...");

        // FFT in place
        work_.execute_fft();

        // Multiply by conj(dens2_fft), as complex numbers
        // TODO: we could just store the conjugate form of dens2_fft.
        mult_time_.start();
        array_ops::multiply_with_conjugation(dens2_fft_,
                                             work_.as_complex_array());
        mult_time_.stop();

        // Extract the anisotropic power spectrum
        // Load the Ylm's. Include the window correction and the SE15
        // normalization.
        ylm_time_.start();
        make_ylm(ell, m, kx_, ky_, kz_, 4.0 * M_PI, wide_angle_exponent,
                 &inv_window_, &kylm_);
        ylm_time_.stop();
        // Multiply these Ylm by the power result, and then add to total.
        // TODO: we only need to extract the real part for an autocorrelation,
        // since autocorrelations of real functions are even and therefore
        // have purely real fourier transforms (another way to see this is
        // that we multiply by the conjugate in fourier space). But cross
        // correlations are not in general even and so the complex component
        // is not generally zero. But we may be able to argue that it's
        // negligible by isotropy of the universe...but that might not be true
        // in redshift space? This may sink the idea of using this class to
        // compute cross-correlations.
        work_.extract_fft_submatrix_c2r(&kgrid_, &kylm_);

        // iFFT the result, in place
        work_.execute_ifft();
        fprintf(stdout, "FFT...");

        // Create Ylm for the submatrix that we'll extract for histogramming
        // Include the SE15 normalization and the factor of ncells to finish
        // the inverse FFT (FFTW doesn't include this factor automatically).
        uint64 ncells = dens1.data().size();
        ylm_time_.start();
        make_ylm(ell, m, rx_, ry_, rz_, 4.0 * M_PI / ncells,
                 wide_angle_exponent, NULL, &rylm_);
        ylm_time_.stop();

        // Multiply these Ylm by the correlation result, and then add to
        // total.
        work_.extract_submatrix(&rgrid_, &rylm_);

        fprintf(stdout, "Done!\n");
        fflush(NULL);
      }
      hist_time_.start();
      rhist_.accumulate(ell / 2, rnorm_, rgrid_);
      khist_.accumulate(ell / 2, knorm_, kgrid_);
      hist_time_.stop();
      if (ell == 0) {
        zerolag_ = rgrid_.at(rzero_[0], rzero_[1], rzero_[2]);
      }
    }
    total_time_.stop();
  }

  // Configuration-space arrays
  Array1D<Float> xcell_;
  Array1D<Float> ycell_;
  Array1D<Float> zcell_;
};

#endif  // CORRELATE_H