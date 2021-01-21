#ifndef CORRELATE_H
#define CORRELATE_H

#include <assert.h>

#include <array>
#include <cmath>

#include "array/array_ops.h"
#include "grid.h"
#include "grid/config_space_grid.h"
#include "grid/fft_grid.h"
#include "histogram.h"
#include "particle_mesh/window_functions.h"
#include "spherical_harmonics.h"
#include "types.h"

class Correlator {
 public:
  Correlator(const ConfigSpaceGrid &dens, Float rmax, Float kmax)
      : dens_(dens), rmax_(rmax), kmax_(kmax), work_(dens_.ngrid()) {
    setup_rgrid();
    setup_kgrid();

    // TODO: we could use the quick FFTW setup for the isotropic case, since we
    // only FFT and inverse FFT once each.
    work_.setup_fft();
    fprintf(stderr, "work size = [%d, %d, %d]\n", work_.dshape()[0],
            work_.dshape()[1], work_.dshape()[2]);
  }

  ~Correlator() {
    delete rgrid_;
    delete rx_;
    delete ry_;
    delete rz_;
    delete rnorm_;

    delete kgrid_;
    delete kx_;
    delete ky_;
    delete kz_;
    delete knorm_;
    delete inv_window_;
  }

  void correlate_iso(Histogram1D *h, Histogram1D *kh, Float *zerolag) {
    // Copy the density field into work_. We do this after setup_fft, because
    // that can estroy the input. TODO: possible optimization of initializing
    // work_ by copy and then hoping setup_fft doesn't destroy the input, but
    // we'd also need to make to ensure that we touch the whole padded work
    // array if we initialize by copy.
    array_ops::copy_into_padded_array(dens_.data().arr(),  // TODO: yuck
                                      work_.arr());

    // Setup.Stop();

    fprintf(stdout, "# Computing the density FFT...");
    work_.execute_fft();

    fprintf(stdout, "# Multiply...");
    fflush(NULL);
    // TODO: inplace abs^2
    array_ops::multiply_with_conjugation(work_.carr(), work_.carr());

    // Extract power spectrum.
    // TODO: should this include a CICwindow correction like the aniso case?
    work_.extract_submatrix_C2R(&kgrid_->arr());

    // We must divide by ncells^2: DFT differs from Fourier series coefficients
    // by a factor of ncells, and we've squared the DFT result.
    // TODO: there are too many variables called 'norm' and they have different
    // types.
    uint64 ncells = dens_.data().size();  // TODO: yuck
    Float pnorm = 1.0 / ncells / ncells;
    kgrid_->multiply_by(pnorm);
    kh->histcorr(*knorm_, *kgrid_);

    // iFFT the result, in place
    fprintf(stdout, "IFFT...");
    fflush(NULL);
    work_.execute_ifft();

    fprintf(stdout, "# Done!\n");
    fflush(NULL);

    work_.extract_submatrix(&rgrid_->arr());

    // We must divide by two factors of ncells: the first one completes the
    // inverse FFT (FFTW doesn't include this factor automatically) and the
    // second is the factor converting the autocorrelation to the 2PCF.
    Float norm = 1.0 / ncells / ncells;
    rgrid_->multiply_by(norm);
    h->histcorr(*rnorm_, *rgrid_);

    // Hist.Stop();
    // Correlate.Stop();
    // TODO: restore
    // *zerolag = corr.at(rzero[0], rzero[1], rzero[2]);
  }

  // zerolag is needed because that information is not in the output histograms
  // (small but nonzero separations are put in the same bin as the zero
  // separation)
  void correlate_aniso(int maxell, int wide_angle_exponent,
                       std::array<Float, 3> observer, Histogram2D *h,
                       Histogram2D *kh, Float *zerolag) {
    // Copy the density field into work_. We do this after setup_fft, because
    // that can estroy the input. TODO: possible optimization of initializing
    // work_ by copy and then hoping setup_fft doesn't destroy the input, but
    // we'd also need to make to ensure that we touch the whole padded work
    // array if we initialize by copy.
    array_ops::copy_into_padded_array(dens_.data().arr(),  // TODO: yuck
                                      work_.arr());

    // Compute xcell, ycell, zcell, which are the coordinates of the cell
    // centers in each dimension, relative to the origin. Now set up the cell
    // centers relative to the origin, in grid units.
    // {0.5, 0.5, 0.5} is the center of first cell in grid coords. Subtracting
    // the location of the observer gives the origin with respect to the
    // observer.
    Array1D xcell(dens_.ngrid(0));
    xcell.range(0.5 - observer[0], 1);
    Array1D ycell(dens_.ngrid(1));
    ycell.range(0.5 - observer[1], 1);
    Array1D zcell(dens_.ngrid(2));
    zcell.range(0.5 - observer[2], 1);

    // Multiply total by 4*pi, to match SE15 normalization
    // Include the FFTW normalization
    uint64 ncells = dens_.data().size();  // TODO: yuck
    Float norm = 4.0 * M_PI / ncells;
    Float pnorm = 4.0 * M_PI;

    // TODO: okay so we keep one of these as a class member and create the other
    // every time? We could do a lazy initialization. Ultimately it probably
    // doesn't matter because they're not expensive, but I just want to be
    // consistent.
    Array3D total(rgrid_->shape());
    Array3D ktotal(kgrid_->shape());

    // Correlate .Start();  // Starting the main work
    // Now compute the FFT of the density field and conjugate it
    // FFT(work) in place and conjugate it, storing in dens_fft
    fprintf(stdout, "# Computing the density FFT...");
    fflush(NULL);
    work_.execute_fft();
    fprintf(stdout, "# Done!\n");
    fflush(NULL);

    // Correlate.Stop();  // We're tracking initialization separately
    // TODO: we could copy with conjugation in one fell swoop.
    // TODO: are there cases where the dens_fft is not the entire Complex work
    // grid?
    // TODO: abstract all this away into a copy() op or something?
    Complex *fft_data = array_ops::allocate_array<Complex>(work_.carr().size());
    RowMajorArray<Complex> dens_fft(fft_data, work_.carr().shape());
    // TODO: is it okay to do the copy initialization with complex rather than
    // floats for the purpose of assigning to physical hardware?
    array_ops::copy(work_.carr(), dens_fft);
    // Correlate.Start();

    /* ------------ Loop over ell & m --------------- */
    // Loop over each ell to compute the anisotropic correlations
    for (int ell = 0; ell <= maxell; ell += 2) {
      // Initialize the submatrix
      total.set_all(0.0);   // TODO: needed?
      ktotal.set_all(0.0);  // TODO: needed?
      // Loop over m
      for (int m = -ell; m <= ell; m++) {
        fprintf(stdout, "# Computing %d %2d...", ell, m);
        // Create the Ylm matrix times work_
        // TODO: here, is it advantageous if dens_ is padded as well, so its
        // boundaries match with those of work?
        makeYlm(&work_.arr(), ell, m, dens_.ngrid(), xcell, ycell, zcell,
                &dens_.data().arr(), -wide_angle_exponent);
        fprintf(stdout, "Ylm...");

        // FFT in place
        work_.execute_fft();

        // Multiply by conj(dens_fft), as complex numbers
        // AtimesB.Start();
        // TODO: we could just store the conjugate form of dens_fft.
        array_ops::multiply_with_conjugation(dens_fft, work_.carr());
        // AtimesB.Stop();

        // Extract the anisotropic power spectrum
        // Load the Ylm's and include the CICwindow correction
        makeYlm(&kgrid_->arr(), ell, m, kgrid_->shape(), *kx_, *ky_, *kz_,
                &inv_window_->arr(), wide_angle_exponent);
        // Multiply these Ylm by the power result, and then add to total.
        work_.extract_submatrix_C2R(&ktotal.arr(), &kgrid_->arr());

        // iFFT the result, in place
        work_.execute_ifft();
        fprintf(stdout, "FFT...");

        // Create Ylm for the submatrix that we'll extract for histogramming
        // The extra multiplication by one here is of negligible cost, since
        // this array is so much smaller than the FFT grid.
        makeYlm(&rgrid_->arr(), ell, m, rgrid_->shape(), *rx_, *ry_, *rz_, NULL,
                wide_angle_exponent);

        // Multiply these Ylm by the correlation result, and then add to total.
        work_.extract_submatrix(&total.arr(), &rgrid_->arr());

        fprintf(stdout, "Done!\n");
        fflush(NULL);
      }

      // Extract.Start();
      total.multiply_by(norm);
      ktotal.multiply_by(pnorm);
      // Extract.Stop();
      // Histogram total by rnorm
      // Hist.Start();
      h->histcorr(ell, *rnorm_, total);
      kh->histcorr(ell, *knorm_, ktotal);
      // Hist.Stop();
      // TODO: restore
      // if (ell == 0) {
      //   *zerolag = total.at(rzero[0], rzero[1], rzero[2]);
      // }
    }
    // Correlate.Stop();
  }

 private:
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
    rgrid_ = new Array3D(rshape);

    // The axes of the cell centers in separation space in physical units.
    rx_ = new Array1D(rshape[0]);
    rx_->range(-cell_size * rmax_cells, cell_size);
    ry_ = new Array1D(rshape[1]);
    ry_->range(-cell_size * rmax_cells, cell_size);
    rz_ = new Array1D(rshape[2]);
    rz_->range(-cell_size * rmax_cells, cell_size);

    // Radius of each separation-space subgrid cell in physical units.
    rnorm_ = new Array3D(rshape);
    for (int i = 0; i < rshape[0]; ++i) {
      for (int j = 0; j < rshape[1]; ++j) {
        for (int k = 0; k < rshape[2]; ++k) {
          // TODO: use get_row or something faster everywhere I call at() in a
          // loop.
          rnorm_->at(i, j, k) =
              sqrt((*rx_)[i] * (*rx_)[i] + (*ry_)[j] * (*ry_)[j] +
                   (*rz_)[k] * (*rz_)[k]);
        }
      }
    }
    fprintf(stdout, "# Done setting up the separation submatrix of size +-%d\n",
            rmax_cells);
    // Index of r=0.
    // TODO: need this
    // std::array<int, 3> rzero = {rmax_cells, rmax_cells, rmax_cells};
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
    kgrid_ = new Array3D(kshape);
    fprintf(stdout,
            "# Done setting up the wavevector submatrix of size +-%d, %d, %d\n",
            kshape[0] / 2, kshape[1] / 2, kshape[2] / 2);
    fprintf(stderr, "kshape = [%d, %d, %d]\n", kshape[0], kshape[1], kshape[2]);

    // The axes in Fourier space.
    kx_ = new Array1D(kshape[0]);
    kx_->range((-kshape[0] / 2) * 2.0 * k_Nyq / ngrid[0],
               2.0 * k_Nyq / ngrid[0]);
    ky_ = new Array1D(kshape[1]);
    ky_->range((-kshape[1] / 2) * 2.0 * k_Nyq / ngrid[1],
               2.0 * k_Nyq / ngrid[1]);
    kz_ = new Array1D(kshape[2]);
    kz_->range((-kshape[2] / 2) * 2.0 * k_Nyq / ngrid[2],
               2.0 * k_Nyq / ngrid[2]);

    // Frequency of each freqency subgrid cell in physical units.
    knorm_ = new Array3D(kshape);
    for (int i = 0; i < kshape[0]; ++i) {
      for (int j = 0; j < kshape[1]; ++j) {
        for (int k = 0; k < kshape[2]; ++k) {
          knorm_->at(i, j, k) =
              sqrt((*kx_)[i] * (*kx_)[i] + (*ky_)[j] * (*ky_)[j] +
                   (*kz_)[k] * (*kz_)[k]);
        }
      }
    }

    inv_window_ = new Array3D(kshape);
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
              Float sinkxL = sin((*kx_)[i] * cell_size / 2.0);
              Float sinkyL = sin((*ky_)[j] * cell_size / 2.0);
              Float sinkzL = sin((*kz_)[k] * cell_size / 2.0);
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
          inv_window_->at(i, j, k) = 1.0 / window;
          // We will divide the power spectrum by the square of the window
        }
      }
    }
  }

  const ConfigSpaceGrid &dens_;
  Float rmax_;  // TODO: needed?
  Float kmax_;
  // TODO: use RowMajorArray and allocate on the stack.
  Array3D *rgrid_;
  Array1D *rx_;
  Array1D *ry_;
  Array1D *rz_;
  Array3D *rnorm_;

  Array3D *kgrid_;
  Array1D *kx_;
  Array1D *ky_;
  Array1D *kz_;
  Array3D *knorm_;
  Array3D *inv_window_;

  FftGrid work_;
};

#endif  // CORRELATE_H