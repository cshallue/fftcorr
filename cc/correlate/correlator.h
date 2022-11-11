#ifndef CORRELATE_H
#define CORRELATE_H

#include <array>
#include <cmath>
#include <vector>

#include "../array/array_ops.h"
#include "../array/row_major_array.h"
#include "../grid/config_space_grid.h"
#include "../grid/fft_grid.h"
#include "../histogram/histogram_list.h"
#include "../profiling/timer.h"
#include "../types.h"
#include "spherical_harmonics.h"
#include "window_correct.h"

class BaseCorrelator {
 public:
  BaseCorrelator(const std::array<int, 3> shape, Float cell_size,
                 WindowCorrection window_correct, Float rmax, Float dr,
                 Float kmax, Float dk, int maxell, unsigned fftw_flags)
      : shape_(shape),
        cell_size_(cell_size),
        window_correct_(window_correct),
        rmax_(rmax),
        dr_(dr),
        kmax_(kmax),
        dk_(dk),
        maxell_(maxell),
        work_(shape),
        rhist_(maxell_ / 2 + 1, 0.0, rmax_, dr_),
        khist_(maxell_ / 2 + 1, 0.0, kmax_, dk_) {
    setup_time_.start();
    work_.plan_fft(fftw_flags);
    setup_r_subgrid();
    setup_k_subgrid();
    setup_time_.stop();
  }

  virtual ~BaseCorrelator() {}

  virtual void autocorrelate(const RowMajorArrayPtr<Float, 3> &grid) = 0;

  virtual void cross_correlate(const RowMajorArrayPtr<Float, 3> &grid1) = 0;

  void cross_correlate(const RowMajorArrayPtr<Float, 3> &grid1,
                       const RowMajorArrayPtr<Float, 3> &grid2) {
    set_grid2(grid2);
    cross_correlate(grid1);
  }

  void set_grid2(const RowMajorArrayPtr<Float, 3> &grid2) {
    setup_time_.start();
    array_ops::copy_into_padded_array(grid2, work_.as_real_array());
    setup_time_.stop();

    fprintf(stdout, "# Computing grid 2 FFT...");
    fflush(NULL);
    work_.execute_fft();
    fprintf(stdout, "# Done!\n");
    fflush(NULL);

    // Save the FFT of grid2.
    set_grid2_fft(work_.as_complex_array());
  }

  void set_grid2_fft(const RowMajorArrayPtr<Complex, 3> &grid2_fft) {
    setup_time_.start();
    if (!grid2_fft_.data()) {
      // Lazy allocate because periodic autocorrelation doesn't require a
      // separate copy.
      grid2_fft_.allocate(work_.as_complex_array().shape());
    }
    array_ops::copy(grid2_fft, grid2_fft_);
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
  void setup_r_subgrid() {
    // Create the separation-space subgrid.
    int rmax_cells = ceil(rmax_ / cell_size_);  // rmax in grid cell units.
    fprintf(stderr, "rmax = %f, cell_size = %f, rmax_cells =%d\n", rmax_,
            cell_size_, rmax_cells);
    // Number of cells in each dimension of the subgrid.
    int sizex = 2 * rmax_cells + 1;  // Include negative separation vectors.
    std::array<int, 3> rshape = {sizex, sizex, sizex};
    fprintf(stderr, "r_subgrid shape = [%d, %d, %d]\n", rshape[0], rshape[1],
            rshape[2]);
    rsubgrid_.allocate(rshape);

    // The axes of the cell centers in separation space in physical units.
    const Float minsep = -cell_size_ * rmax_cells;
    rx_ = array_ops::sequence(minsep, cell_size_, rshape[0]);
    ry_ = array_ops::sequence(minsep, cell_size_, rshape[1]);
    rz_ = array_ops::sequence(minsep, cell_size_, rshape[2]);

    // Radius of each separation-space subgrid cell in physical units.
    rnorm_.allocate(rshape);
    for (int i = 0; i < rshape[0]; ++i) {
      for (int j = 0; j < rshape[1]; ++j) {
        Float *row = rnorm_.get_row(i, j);
        for (int k = 0; k < rshape[2]; ++k) {
          row[k] = sqrt(rx_[i] * rx_[i] + ry_[j] * ry_[j] + rz_[k] * rz_[k]);
        }
      }
    }
    // Index corresponding to zero separation (r=0).
    for (int &x : rzero_) x = rmax_cells;
    rylm_.allocate(rshape);
  }

  void setup_k_subgrid() {
    // Create the Fourier-space subgrid.
    // Our box has cubic-sized cells, so k_Nyquist is the same in all
    // directions. The spacing of modes is therefore 2*k_Nyq/ngrid.
    Float k_Nyq = M_PI / cell_size_;  // The Nyquist frequency for our grid.
    // Number of cells in the subgrid.
    std::array<int, 3> kshape;
    for (int i = 0; i < 3; ++i) {
      kshape[i] = 2 * ceil(kmax_ / (2.0 * k_Nyq / shape_[i])) + 1;
    }
    fprintf(stdout, "# Storing wavenumbers up to %6.4f, with k_Nyq = %6.4f\n",
            kmax_, k_Nyq);
    for (int i = 0; i < 3; ++i) {
      if (kshape[i] > shape_[i]) {
        // TODO: in the corner case of kmax ~= k_Nyq, this can be greater than
        // ngrid[i].
        kshape[i] = 2 * floor(shape_[i] / 2) + 1;
        fprintf(stdout,
                "# WARNING: Requested wavenumber is too big. Truncating "
                "shape[%d] to %d\n",
                i, kshape[i]);
      }
    }
    ksubgrid_.allocate(kshape);
    fprintf(stdout,
            "# Done setting up the wavevector submatrix of size +-%d, %d, %d\n",
            kshape[0] / 2, kshape[1] / 2, kshape[2] / 2);
    fprintf(stderr, "kshape = [%d, %d, %d]\n", kshape[0], kshape[1], kshape[2]);

    // The axes in Fourier space.
    kx_ = array_ops::sequence((-kshape[0] / 2) * 2.0 * k_Nyq / shape_[0],
                              2.0 * k_Nyq / shape_[0], kshape[0]);
    ky_ = array_ops::sequence((-kshape[1] / 2) * 2.0 * k_Nyq / shape_[1],
                              2.0 * k_Nyq / shape_[1], kshape[1]);
    kz_ = array_ops::sequence((-kshape[2] / 2) * 2.0 * k_Nyq / shape_[2],
                              2.0 * k_Nyq / shape_[2], kshape[2]);

    // Frequency of each freqency subgrid cell in physical units.
    knorm_.allocate(kshape);
    for (int i = 0; i < kshape[0]; ++i) {
      for (int j = 0; j < kshape[1]; ++j) {
        Float *row = knorm_.get_row(i, j);
        for (int k = 0; k < kshape[2]; ++k) {
          row[k] = sqrt(kx_[i] * kx_[i] + ky_[j] * ky_[j] + kz_[k] * kz_[k]);
        }
      }
    }
    kylm_.allocate(kshape);

    // Set up window correction.
    inv_window_.allocate(kshape);
    std::unique_ptr<WindowSquaredNorm> sq_norm =
        make_window_function(window_correct_);
    for (int i = 0; i < kshape[0]; ++i) {
      for (int j = 0; j < kshape[1]; ++j) {
        Float *row = inv_window_.get_row(i, j);
        for (int k = 0; k < kshape[2]; ++k) {
          row[k] = 1.0 / (sq_norm->evaluate(kx_[i], cell_size_) *
                          sq_norm->evaluate(ky_[j], cell_size_) *
                          sq_norm->evaluate(kz_[k], cell_size_));
        }
      }
    }
  }

  // Physical dimensions of the input grid.
  std::array<int, 3> shape_;
  Float cell_size_;
  WindowCorrection window_correct_;

  // Histogram arguments.
  Float rmax_;
  Float dr_;
  Float kmax_;
  Float dk_;
  Float maxell_;

  // Work space for FFTs.
  FftGrid work_;

  // Separation-space arrays.
  RowMajorArray<Float, 3> rsubgrid_;
  std::array<int, 3> rzero_;
  Array1D<Float> rx_;
  Array1D<Float> ry_;
  Array1D<Float> rz_;
  RowMajorArray<Float, 3> rnorm_;
  RowMajorArray<Float, 3> rylm_;

  // Fourier-space arrays.
  RowMajorArray<Complex, 3> grid2_fft_;
  RowMajorArray<Float, 3> ksubgrid_;
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

  void autocorrelate(const RowMajorArrayPtr<Float, 3> &grid) {
    correlate_internal(grid, NULL);
  }

  void cross_correlate(const RowMajorArrayPtr<Float, 3> &grid1) {
    correlate_internal(grid1, &grid2_fft_);
  }

  // TODO: this appears to be necessary for cython?
  void cross_correlate(const RowMajorArrayPtr<Float, 3> &grid1,
                       const RowMajorArrayPtr<Float, 3> &grid2) {
    BaseCorrelator::cross_correlate(grid1, grid2);
  }

 protected:
  void correlate_internal(const RowMajorArrayPtr<Float, 3> &grid1,
                          const RowMajorArrayPtr<Complex, 3> *grid2_fft) {
    total_time_.start();
    rhist_.reset();
    khist_.reset();

    array_ops::copy_into_padded_array(grid1, work_.as_real_array());
    fprintf(stdout, "# Computing grid 1 FFT...");
    fflush(NULL);
    work_.execute_fft();
    fprintf(stdout, "# Done!\n");
    fflush(NULL);

    if (!grid2_fft) {
      grid2_fft = &work_.as_complex_array();  // Autocorrelation.
    }

    fprintf(stdout, "# Multiply...");
    fflush(NULL);
    mult_time_.start();
    array_ops::multiply_with_conjugation(*grid2_fft, work_.as_complex_array());
    mult_time_.stop();

    // The periodic cross spectrum is (1/V) FT[f] conj(FT[g]), expressed in a
    // Fourier convention where the Fourier series coefficients have units of
    // volume and (1/V) appears in the inverse transform. With this convention,
    // the output of the DFT is (N/V) times the corresponding Fourier series
    // coefficient, where N is the number of cells. Thus, we must multiply by
    // (1/V) / (N/V) / (N/V) = V / N^2 = cell_size^3 / ncells.
    uint64 ncells = grid1.size();
    Float k_rescale = cell_size_ * cell_size_ * cell_size_ / ncells;

    // Extract the power spectrum, expanded in Legendre polynomials.
    for (int ell = 0; ell <= maxell_; ell += 2) {
      setup_time_.start();
      array_ops::set_all(0.0, ksubgrid_);
      setup_time_.stop();
      // P_l = Y_l0 * sqrt(4.0 * M_PI / (2 * ell + 1)) and then we need to
      // multiply by (2 * ell + 1) to account for the normalization of P_l's.
      // Also include the DFT scaling.
      Float coeff = sqrt((4.0 * M_PI) * (2 * ell + 1)) * k_rescale;
      ylm_time_.start();
      make_ylm(ell, 0, kx_, ky_, kz_, coeff, 0, &inv_window_, &kylm_);
      ylm_time_.stop();
      work_.extract_fft_submatrix_c2r(&ksubgrid_, &kylm_);
      hist_time_.start();
      khist_.accumulate(ell / 2, knorm_, ksubgrid_);
      hist_time_.stop();
    }

    // iFFT the result, in place
    fprintf(stdout, "IFFT...");
    fflush(NULL);
    work_.execute_ifft();
    fprintf(stdout, "# Done!\n");
    fflush(NULL);

    // The 2PCF is (1 / ncells) (f * g), where * denotes periodic
    // cross-correlation. This is independent of Fourier conventions, but we
    // must rescale by another factor of (1 / ncells) to complete the inverse
    // DFT (FFTW doesn't include this factor automatically).
    Float r_rescale = (1.0 / ncells / ncells);

    // Extract the 2PCF, expanded in Legendre polynomials.
    for (int ell = 0; ell <= maxell_; ell += 2) {
      setup_time_.start();
      array_ops::set_all(0.0, rsubgrid_);
      setup_time_.stop();
      // P_l = Y_l0 * sqrt(4.0 * M_PI / (2 * ell + 1)) and then we need to
      // multiply by (2 * ell + 1) to account for the normalization of P_l's.
      // Also include the IFT scaling.
      Float coeff = sqrt((4.0 * M_PI) * (2 * ell + 1)) * r_rescale;
      ylm_time_.start();
      make_ylm(ell, 0, rx_, ry_, rz_, coeff, 0, NULL, &rylm_);
      ylm_time_.stop();
      work_.extract_submatrix(&rsubgrid_, &rylm_);
      hist_time_.start();
      rhist_.accumulate(ell / 2, rnorm_, rsubgrid_);
      hist_time_.stop();
      if (ell == 0) {
        zerolag_ = rsubgrid_.at(rzero_[0], rzero_[1], rzero_[2]);
      }
    }
    total_time_.stop();
  }
};

class Correlator : public BaseCorrelator {
 public:
  Correlator(const std::array<int, 3> shape, Float cell_size,
             const std::array<Float, 3> posmin, WindowCorrection window_correct,
             Float rmax, Float dr, Float kmax, Float dk, int maxell,
             unsigned fftw_flags)
      : BaseCorrelator(shape, cell_size, window_correct, rmax, dr, kmax, dk,
                       maxell, fftw_flags),
        posmin_(posmin) {
    setup_cell_coords();
  }

  void autocorrelate(const RowMajorArrayPtr<Float, 3> &grid) {
    BaseCorrelator::cross_correlate(grid, grid);
  }

  void cross_correlate(const RowMajorArrayPtr<Float, 3> &grid1) {
    correlate_internal(grid1);
  }

 protected:
  void setup_cell_coords() {
    // Location of the observer relative to posmin, in grid units.
    std::array<Float, 3> observer;
    // Put the observer at the origin of the survey coordinate system.
    for (int i = 0; i < 3; ++i) {
      observer[i] = -posmin_[i] / cell_size_;
    }
    // We cab simulate a periodic box by puting the observer centered in the
    // grid, but displaced far away in the -x direction. This is an
    // inefficient way to compute the periodic case, but it's a good sanity
    // check. for (int i = 0; i < 3; ++i) {
    //   observer[i] = grid1.shape(i) / 2.0;
    // }
    // observer[0] -= grid1.shape(0) * 1e6;  // Observer far away!

    // Coordinates of the cell centers in each dimension, relative to the
    // observer. We're using grid units (scale doesn't matter when computing
    // Ylms).
    xcell_ = array_ops::sequence(0.5 - observer[0], 1.0, shape_[0]);
    ycell_ = array_ops::sequence(0.5 - observer[1], 1.0, shape_[1]);
    zcell_ = array_ops::sequence(0.5 - observer[2], 1.0, shape_[2]);
  }

  // TODO: the normalization doesn't match the periodic case, but that doesn't
  // really matter because here we take the ratios of NN and RR.
  void correlate_internal(const RowMajorArrayPtr<Float, 3> &grid1) {
    constexpr int wide_angle_exponent = 0;  // TODO: do we still need this?
    total_time_.start();
    rhist_.reset();
    khist_.reset();

    array_ops::copy_into_padded_array(grid1, work_.as_real_array());
    fprintf(stdout, "# Computing grid 1 FFT...");
    fflush(NULL);
    work_.execute_fft();
    fprintf(stdout, "# Done!\n");
    fflush(NULL);

    /* ------------ Loop over ell & m --------------- */
    // Loop over each ell to compute the anisotropic correlations
    for (int ell = 0; ell <= maxell_; ell += 2) {
      // Initialize the submatrix
      setup_time_.start();
      array_ops::set_all(0.0, rsubgrid_);
      array_ops::set_all(0.0, ksubgrid_);
      setup_time_.stop();
      // Loop over m
      for (int m = -ell; m <= ell; m++) {
        fprintf(stdout, "# Computing %d %2d...", ell, m);

        // Create the Ylm matrix times work_
        // TODO: here, is it advantageous if grid1 is padded as well, so its
        // boundaries match with those of work?
        ylm_time_.start();
        make_ylm(ell, m, xcell_, ycell_, zcell_, 1.0, -wide_angle_exponent,
                 &grid1, &work_.as_real_array());
        ylm_time_.stop();
        fprintf(stdout, "Ylm...");

        // FFT in place
        work_.execute_fft();

        // Multiply by conj(grid2_fft), as complex numbers
        mult_time_.start();
        array_ops::multiply_with_conjugation(grid2_fft_,
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
        work_.extract_fft_submatrix_c2r(&ksubgrid_, &kylm_);

        // iFFT the result, in place
        work_.execute_ifft();
        fprintf(stdout, "FFT...");

        // Create Ylm for the submatrix that we'll extract for histogramming
        // Include the SE15 normalization and the factor of ncells to finish
        // the inverse FFT (FFTW doesn't include this factor automatically).
        uint64 ncells = grid1.size();
        ylm_time_.start();
        make_ylm(ell, m, rx_, ry_, rz_, 4.0 * M_PI / ncells,
                 wide_angle_exponent, NULL, &rylm_);
        ylm_time_.stop();

        // Multiply these Ylm by the correlation result, and then add to
        // total.
        work_.extract_submatrix(&rsubgrid_, &rylm_);

        fprintf(stdout, "Done!\n");
        fflush(NULL);
      }
      hist_time_.start();
      rhist_.accumulate(ell / 2, rnorm_, rsubgrid_);
      khist_.accumulate(ell / 2, knorm_, ksubgrid_);
      hist_time_.stop();
      if (ell == 0) {
        zerolag_ = rsubgrid_.at(rzero_[0], rzero_[1], rzero_[2]);
      }
    }
    total_time_.stop();
  }

  std::array<Float, 3> posmin_;

  // Configuration-space arrays
  Array1D<Float> xcell_;
  Array1D<Float> ycell_;
  Array1D<Float> zcell_;
};

#endif  // CORRELATE_H