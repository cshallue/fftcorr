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

class Correlator {
 public:
  Correlator(const ConfigSpaceGrid &dens, Float rmax, Float dr, Float kmax,
             Float dk, int maxell)
      : dens_(dens),
        rmax_(rmax),
        kmax_(kmax),
        maxell_(maxell),
        work_(dens_.ngrid()),
        rhist_(maxell / 2 + 1, 0.0, rmax, dr),
        khist_(maxell / 2 + 1, 0.0, kmax, dk) {}

  void correlate_periodic() {
    setup(true);

    // Setup.Stop();

    // Correlate .Start();  // Starting the main work
    fprintf(stdout, "# Computing the density FFT...");
    fflush(NULL);
    work_.execute_fft();
    fprintf(stdout, "# Done!\n");
    fflush(NULL);

    fprintf(stdout, "# Multiply...");
    fflush(NULL);
    // TODO: inplace abs^2
    array_ops::multiply_with_conjugation(work_.carr(), work_.carr());

    // We must multiply the DFT result by (1/ncells^2): DFT differs from Fourier
    // series coefficients by a factor of ncells, and we've squared the DFT.
    uint64 ncells = dens_.data().size();
    Float k_rescale = (1.0 / ncells / ncells);

    // Extract the power spectrum, expanded in Legendre polynomials.
    for (int ell = 0; ell <= maxell_; ell += 2) {
      array_ops::set_all(0.0, kgrid_);
      // P_l = Y_l0 * sqrt(4.0 * M_PI / (2 * ell + 1)) and then we need to
      // multiply by (2 * ell + 1) to account for the normalization of P_l's.
      // Also include the DFT scaling.
      Float coeff = sqrt((4.0 * M_PI) * (2 * ell + 1)) * k_rescale;
      // TODO: include &inv_window if appropriate in this case.
      make_ylm(ell, 0, kx_, ky_, kz_, coeff, 0, NULL, &kylm_);
      work_.extract_submatrix_C2R(&kgrid_, &kylm_);
      khist_.accumulate(ell / 2, knorm_, kgrid_);
    }

    // iFFT the result, in place
    fprintf(stdout, "IFFT...");
    fflush(NULL);
    work_.execute_ifft();
    fprintf(stdout, "# Done!\n");
    fflush(NULL);

    // We must multiply the IFT by two factors of (1/ncells): the  first one
    // completes the inverse FFT (FFTW doesn't include this factor
    // automatically) and the second is the factor converting the
    // autocorrelation to the 2PCF.
    Float r_rescale = (1.0 / ncells / ncells);

    // Extract the 2PCF, expanded in Legendre polynomials.
    for (int ell = 0; ell <= maxell_; ell += 2) {
      array_ops::set_all(0.0, rgrid_);
      // P_l = Y_l0 * sqrt(4.0 * M_PI / (2 * ell + 1)) and then we need to
      // multiply by (2 * ell + 1) to account for the normalization of P_l's.
      // Also include the IFT scaling.
      Float coeff = sqrt((4.0 * M_PI) * (2 * ell + 1)) * r_rescale;
      make_ylm(ell, 0, rx_, ry_, rz_, coeff, 0, NULL, &rylm_);
      work_.extract_submatrix(&rgrid_, &rylm_);
      rhist_.accumulate(ell / 2, rnorm_, rgrid_);
      if (ell == 0) {
        zerolag_ = rgrid_.at(rzero_[0], rzero_[1], rzero_[2]);
      }
    }

    // Hist.Stop();
    // Correlate.Stop();
  }

  void correlate_nonperiodic(int wide_angle_exponent) {
    setup(false);

    // Correlate .Start();  // Starting the main work
    fprintf(stdout, "# Computing the density FFT...");
    fflush(NULL);
    work_.execute_fft();
    fprintf(stdout, "# Done!\n");
    fflush(NULL);

    // Correlate.Stop();  // We're tracking initialization separately
    // TODO: we could copy with conjugation.
    array_ops::copy(work_.carr(), dens_fft_);
    // Correlate.Start();

    /* ------------ Loop over ell & m --------------- */
    // Loop over each ell to compute the anisotropic correlations
    for (int ell = 0; ell <= maxell_; ell += 2) {
      // Initialize the submatrix
      array_ops::set_all(0.0, rgrid_);
      array_ops::set_all(0.0, kgrid_);
      // Loop over m
      for (int m = -ell; m <= ell; m++) {
        fprintf(stdout, "# Computing %d %2d...", ell, m);

        // Create the Ylm matrix times work_
        // TODO: here, is it advantageous if dens_ is padded as well, so its
        // boundaries match with those of work?
        make_ylm(ell, m, xcell_, ycell_, zcell_, 1.0, -wide_angle_exponent,
                 &dens_.data(), &work_.arr());
        fprintf(stdout, "Ylm...");

        // FFT in place
        work_.execute_fft();

        // Multiply by conj(dens_fft), as complex numbers
        // AtimesB.Start();
        // TODO: we could just store the conjugate form of dens_fft.
        array_ops::multiply_with_conjugation(dens_fft_, work_.carr());
        // AtimesB.Stop();

        // Extract the anisotropic power spectrum
        // Load the Ylm's. Include the window correction and the SE15
        // normalization.
        make_ylm(ell, m, kx_, ky_, kz_, 4.0 * M_PI, wide_angle_exponent,
                 &inv_window_, &kylm_);
        // Multiply these Ylm by the power result, and then add to total.
        work_.extract_submatrix_C2R(&kgrid_, &kylm_);

        // iFFT the result, in place
        work_.execute_ifft();
        fprintf(stdout, "FFT...");

        // Create Ylm for the submatrix that we'll extract for histogramming
        // Include the SE15 normalization and the factor of ncells to finish the
        // inverse FFT (FFTW doesn't include this factor automatically).
        uint64 ncells = dens_.data().size();
        make_ylm(ell, m, rx_, ry_, rz_, 4.0 * M_PI / ncells,
                 wide_angle_exponent, NULL, &rylm_);

        // Multiply these Ylm by the correlation result, and then add to total.
        work_.extract_submatrix(&rgrid_, &rylm_);

        fprintf(stdout, "Done!\n");
        fflush(NULL);
      }

      // Extract.Start();
      // Extract.Stop();
      // HistogramList total by rnorm
      // Hist.Start();
      rhist_.accumulate(ell / 2, rnorm_, rgrid_);
      khist_.accumulate(ell / 2, knorm_, kgrid_);
      // Hist.Stop();
      if (ell == 0) {
        zerolag_ = rgrid_.at(rzero_[0], rzero_[1], rzero_[2]);
      }
    }
    // Correlate.Stop();
  }

  // Output accessors.
  Float zerolag() const { return zerolag_; }
  const Array1D<Float> &correlation_r() const { return rhist_.bins(); }
  const RowMajorArray<int, 2> &correlation_counts() const {
    return rhist_.counts();
  }
  const RowMajorArray<Float, 2> &correlation_histogram() const {
    return rhist_.hist_values();
  }
  const Array1D<Float> &power_spectrum_k() const { return khist_.bins(); }
  const RowMajorArray<int, 2> &power_spectrum_counts() const {
    return khist_.counts();
  }
  const RowMajorArray<Float, 2> &power_spectrum_histogram() const {
    return khist_.hist_values();
  }

 private:
  // Sets up a fresh call to correlate_{periodic,nonperiodic}.
  void setup(bool periodic) {
    // TODO: consistency between dens.data() and work.arr()
    array_ops::copy_into_padded_array(dens_.data(), work_.arr());
    if (!periodic && !xcell_.data()) {
      setup_cell_coords();
    }
    if (!rgrid_.data()) {
      setup_rgrid();
    }
    if (!periodic && !dens_fft_.data()) {
      dens_fft_.allocate(work_.carr().shape());
    }
    if (!kgrid_.data()) {
      setup_kgrid();
    }
    rhist_.reset();
    khist_.reset();
    zerolag_ = -1.0;
  }

  void setup_cell_coords() {
    // Location of the observer relative to posmin, in grid units.
    std::array<Float, 3> observer;
    // Put the observer at the origin of the survey coordinate system.
    for (int i = 0; i < 3; ++i) {
      observer[i] = -dens_.posmin(i) / dens_.cell_size();
    }
    // We cab simulate a periodic box by puting the observer centered in the
    // grid, but displaced far away in the -x direction. This is an inefficient
    // way to compute the periodic case, but it's a good sanity check.
    // for (int i = 0; i < 3; ++i) {
    //   observer[i] = dens_.ngrid(i) / 2.0;
    // }
    // observer[0] -= dens_.ngrid(0) * 1e6;  // Observer far away!

    // Coordinates of the cell centers in each dimension, relative to the
    // observer. We're using grid units (scale doesn't matter when computing
    // Ylms).
    xcell_ = sequence(0.5 - observer[0], 1.0, dens_.ngrid(0));
    ycell_ = sequence(0.5 - observer[1], 1.0, dens_.ngrid(1));
    zcell_ = sequence(0.5 - observer[2], 1.0, dens_.ngrid(2));
  }

  void setup_rgrid() {
    // Create the separation-space subgrid.
    Float cell_size = dens_.cell_size();
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

  void setup_kgrid() {
    // Create the Fourier-space subgrid.
    // Our box has cubic-sized cells, so k_Nyquist is the same in all
    // directions. The spacing of modes is therefore 2*k_Nyq/ngrid.
    const std::array<int, 3> &ngrid = dens_.ngrid();
    Float cell_size = dens_.cell_size();
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
    Float window;
    for (int i = 0; i < kshape[0]; ++i) {
      for (int j = 0; j < kshape[1]; ++j) {
        for (int k = 0; k < kshape[2]; ++k) {
          switch (dens_.window_type()) {
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
  const ConfigSpaceGrid &dens_;
  Float rmax_;
  Float kmax_;
  int maxell_;

  // Workspace.
  FftGrid work_;

  // Configuration-space arrays
  Array1D<Float> xcell_;
  Array1D<Float> ycell_;
  Array1D<Float> zcell_;

  // Separation-space arrays.
  RowMajorArray<Float, 3> rgrid_;  // TODO: rtotal_?
  std::array<int, 3> rzero_;
  Array1D<Float> rx_;
  Array1D<Float> ry_;
  Array1D<Float> rz_;
  RowMajorArray<Float, 3> rnorm_;
  RowMajorArray<Float, 3> rylm_;

  // Fourier-space arrays.
  RowMajorArray<Complex, 3> dens_fft_;
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
};

#endif  // CORRELATE_H