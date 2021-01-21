#ifndef CORRELATE_H
#define CORRELATE_H

#include <assert.h>

#include <array>

#include "array/array_ops.h"
#include "grid.h"
#include "grid/config_space_grid.h"
#include "grid/fft_grid.h"
#include "histogram.h"
#include "particle_mesh/window_functions.h"
#include "spherical_harmonics.h"
#include "types.h"

// TODO: share common code between correlate_iso and correlate_aniso.
class Correlator {
 public:
  Correlator(const ConfigSpaceGrid &dens) : dens_(dens), work_(dens_.ngrid()) {
    // TODO: we could use the quick FFTW setup for the isotropic case, since we
    // only FFT and inverse FFT once each.
    work_.setup_fft();

    // Copy the density field into work_. We do this after setup_fft, because
    // that can estroy the input. TODO: possible optimization of initializing
    // work_ by copy and then hoping setup_fft doesn't destroy the input, but
    // we'd also need to make to ensure that we touch the whole padded work
    // array if we initialize by copy.
    array_ops::copy_into_padded_array(dens_.data().arr(),  // TODO: yuck
                                      work_.arr());
    fprintf(stderr, "work size = [%d, %d, %d]\n", work_.dshape()[0],
            work_.dshape()[1], work_.dshape()[2]);
  }

  void correlate_iso(Float rmax, Float kmax, Histogram1D *h, Histogram1D *kh,
                     Float *zerolag) {
    const std::array<int, 3> &ngrid = dens_.ngrid();
    Float cell_size = dens_.cell_size();

    // Storage for the r-space submatrices
    // TODO: consistency with using r or s for separation space
    int rmax_cell = ceil(rmax / cell_size);  // rmax in grid units
    fprintf(stderr, "rmax = %f, cell_size = %f, rmax_cell =%d\n", rmax,
            cell_size, rmax_cell);
    // How many cells we must extract as a submatrix to do the histogramming.
    int csizex = 2 * rmax_cell + 1;
    assert(csizex % 2 == 1);
    std::array<int, 3> csize = {csizex, csizex, csizex};
    fprintf(stderr, "csize = [%d, %d, %d]\n", csize[0], csize[1], csize[2]);

    Array3D rnorm(csize);  // Radius of each histogram bin in separation space.
    for (uint64 i = 0; i < csize[0]; i++)
      for (int j = 0; j < csize[1]; j++)
        for (int k = 0; k < csize[2]; k++) {
          rnorm.at(i, j, k) =
              cell_size * sqrt((i - rmax_cell) * (i - rmax_cell) +
                               (j - rmax_cell) * (j - rmax_cell) +
                               (k - rmax_cell) * (k - rmax_cell));
        }
    fprintf(stdout, "# Done setting up the separation submatrix of size +-%d\n",
            rmax_cell);
    // Index of r=0.
    std::array<int, 3> rzero = {rmax_cell, rmax_cell, rmax_cell};

    // Our box has cubic-sized cells, so k_Nyquist is the same in all
    // directions. The spacing of modes is therefore 2*k_Nyq/ngrid
    Float k_Nyq = M_PI / cell_size;  // The Nyquist frequency for our grid.
    fprintf(stdout, "# Storing wavenumbers up to %6.4f, with k_Nyq = %6.4f\n",
            kmax, k_Nyq);
    // How many cells we must extract as a submatrix to do the histogramming.
    std::array<int, 3> ksize;
    for (int i = 0; i < 3; i++)
      ksize[i] = 2 * ceil(kmax / (2.0 * k_Nyq / ngrid[i])) + 1;
    assert(ksize[0] % 2 == 1);
    assert(ksize[1] % 2 == 1);
    assert(ksize[2] % 2 == 1);
    for (int i = 0; i < 3; i++) {
      if (ksize[i] > ngrid[i]) {
        // TODO: in the corner case of kmax ~= k_Nyq, this can be greater than
        // ngrid[i].
        ksize[i] = 2 * floor(ngrid[i] / 2) + 1;
        fprintf(stdout,
                "# WARNING: Requested wavenumber is too big.  Truncating "
                "ksize_[%d] to %d\n",
                i, ksize[i]);
      }
    }

    // The wavenumber of each cell, in a flattened submatrix.
    Array3D knorm(ksize);

    // The cell centers, relative to zero lag.
    // Allocate kx_cell to [ksize_] and knorm_ to [ksize_**3]
    Array1D kx_cell = range((-ksize[0] / 2) * 2.0 * k_Nyq / ngrid[0],
                            2.0 * k_Nyq / ngrid[0], ksize[0]);
    Array1D ky_cell = range((-ksize[1] / 2) * 2.0 * k_Nyq / ngrid[1],
                            2.0 * k_Nyq / ngrid[0], ksize[0]);
    Array1D kz_cell = range((-ksize[1] / 2) * 2.0 * k_Nyq / ngrid[0],
                            2.0 * k_Nyq / ngrid[0], ksize[1]);

    for (uint64 i = 0; i < ksize[0]; i++) {
      for (int j = 0; j < ksize[1]; j++) {
        for (int k = 0; k < ksize[2]; k++) {
          knorm.at(i, j, k) =
              sqrt(kx_cell[i] * kx_cell[i] + ky_cell[j] * ky_cell[j] +
                   kz_cell[k] * kz_cell[k]);
        }
      }
    }

    fprintf(stdout,
            "# Done setting up the wavevector submatrix of size +-%d, %d, %d\n",
            ksize[0] / 2, ksize[1] / 2, ksize[2] / 2);
    fprintf(stderr, "ksize = [%d, %d, %d]\n", ksize[0], ksize[1], ksize[2]);

    // Allocate total[csize**3] and corr[csize**3]
    Array3D corr(csize);
    Array3D kcorr(ksize);

    // Setup.Stop();

    // Here's where most of the work occurs.

    fprintf(stdout, "# Computing the density FFT...");
    work_.execute_fft();

    fprintf(stdout, "# Multiply...");
    fflush(NULL);
    // TODO: inplace abs^2
    array_ops::multiply_with_conjugation(work_.carr(), work_.carr());

    // Extract power spectrum.
    // TODO: should this include a CICwindow correction like the aniso case?
    work_.extract_submatrix_C2R(&kcorr.arr());

    // iFFT the result, in place
    fprintf(stdout, "IFFT...");
    fflush(NULL);
    work_.execute_ifft();

    fprintf(stdout, "# Done!\n");
    fflush(NULL);

    work_.extract_submatrix(&corr.arr());

    // We must divide by two factors of ncells: the first one completes the
    // inverse FFT (FFTW doesn't include this factor automatically) and the
    // second is the factor converting the autocorrelation to the 2PCF.
    Float ncells = ngrid[0] * ngrid[1] * ngrid[2];
    Float norm = 1.0 / ncells / ncells;
    corr.multiply_by(norm);

    // We must divide by ncells^2: DFT differs from Fourier series coefficients
    // by a factor of ncells, and we've squared the DFT result.
    // TODO: there are too many variables called 'norm'.
    Float pnorm = 1.0 / ncells / ncells;
    kcorr.multiply_by(pnorm);

    // Correlate .Start();  // Starting the main work

    // Histogram total by rnorm
    // Hist.Start();
    h->histcorr(rnorm, corr);
    kh->histcorr(knorm, kcorr);
    // Hist.Stop();
    // Correlate.Stop();
    *zerolag = corr.at(rzero[0], rzero[1], rzero[2]);
  }

  // zerolag is needed because that information is not in the output histograms
  // (small but nonzero separations are put in the same bin as the zero
  // separation)
  void correlate_aniso(Float rmax, Float kmax, int maxell,
                       int wide_angle_exponent, std::array<Float, 3> observer,
                       Histogram2D *h, Histogram2D *kh, Float *zerolag) {
    const std::array<int, 3> &ngrid = dens_.ngrid();
    Float cell_size = dens_.cell_size();
    // Set up the sub-matrix information, assuming that we'll extract
    // -rmax..+rmax cells around zero-lag.
    // Setup.Start();

    // Compute xcell, ycell, zcell, which are the coordinates of the cell
    // centers in each dimension, relative to the origin. Now set up the cell
    // centers relative to the origin, in grid units.
    // {0.5, 0.5, 0.5} is the center of first cell in grid coords. Subtracting
    // the location of the observer gives the origin with respect to the
    // observer.
    Array1D xcell = range(0.5 - observer[0], 1, ngrid[0]);
    Array1D ycell = range(0.5 - observer[1], 1, ngrid[1]);
    Array1D zcell = range(0.5 - observer[2], 1, ngrid[2]);

    // Storage for the r-space submatrices
    int rmax_cell = ceil(rmax / cell_size);
    fprintf(stderr, "rmax = %f, cell_size = %f, rmax_cell =%d\n", rmax,
            cell_size, rmax_cell);
    // How many cells we must extract as a submatrix to do the histogramming.
    int csizex = 2 * rmax_cell + 1;
    assert(csizex % 2 == 1);
    std::array<int, 3> csize = {csizex, csizex, csizex};
    fprintf(stderr, "csize = [%d, %d, %d]\n", csize[0], csize[1], csize[2]);

    // Allocate corr_cell to [csize] and rnorm to [csize**3]
    // The cell centers, relative to zero lag.
    // Normalizing by cell_size just so that the Ylm code can do the wide-angle
    // corrections in the same units.
    Array1D cx_cell = range(-cell_size * rmax_cell, cell_size, csize[0]);
    Array1D cy_cell = range(-cell_size * rmax_cell, cell_size, csize[1]);
    Array1D cz_cell = range(-cell_size * rmax_cell, cell_size, csize[2]);

    Array3D rnorm(csize);  // The radius of each cell.
    for (uint64 i = 0; i < csize[0]; i++)
      for (int j = 0; j < csize[1]; j++)
        for (int k = 0; k < csize[2]; k++)
          rnorm.at(i, j, k) =
              cell_size * sqrt((i - rmax_cell) * (i - rmax_cell) +
                               (j - rmax_cell) * (j - rmax_cell) +
                               (k - rmax_cell) * (k - rmax_cell));
    fprintf(stdout, "# Done setting up the separation submatrix of size +-%d\n",
            rmax_cell);
    // Index of r=0.
    std::array<int, 3> rzero = {rmax_cell, rmax_cell, rmax_cell};

    // Our box has cubic-sized cells, so k_Nyquist is the same in all
    // directions. The spacing of modes is therefore 2*k_Nyq/ngrid
    Float k_Nyq = M_PI / cell_size;  // The Nyquist frequency for our grid.
    fprintf(stdout, "# Storing wavenumbers up to %6.4f, with k_Nyq = %6.4f\n",
            kmax, k_Nyq);
    // How many cells we must extract as a submatrix to do the histogramming.
    std::array<int, 3> ksize;
    for (int i = 0; i < 3; i++)
      ksize[i] = 2 * ceil(kmax / (2.0 * k_Nyq / ngrid[i])) + 1;
    assert(ksize[0] % 2 == 1);
    assert(ksize[1] % 2 == 1);
    assert(ksize[2] % 2 == 1);
    for (int i = 0; i < 3; i++) {
      if (ksize[i] > ngrid[i]) {
        ksize[i] = 2 * floor(ngrid[i] / 2) + 1;
        fprintf(stdout,
                "# WARNING: Requested wavenumber is too big.  Truncating "
                "ksize_[%d] to %d\n",
                i, ksize[i]);
      }
    }
    // The cell centers, relative to zero lag.
    // Allocate kx_cell to [ksize_] and knorm_ to [ksize_**3]
    Array1D kx_cell = range((-ksize[0] / 2) * 2.0 * k_Nyq / ngrid[0],
                            2.0 * k_Nyq / ngrid[0], ksize[0]);
    Array1D ky_cell = range((-ksize[1] / 2) * 2.0 * k_Nyq / ngrid[1],
                            2.0 * k_Nyq / ngrid[0], ksize[0]);
    Array1D kz_cell = range((-ksize[1] / 2) * 2.0 * k_Nyq / ngrid[0],
                            2.0 * k_Nyq / ngrid[0], ksize[1]);

    // The wavenumber of each cell, in a flattened submatrix.
    Array3D knorm(ksize);
    // The inverse of the window function for the CIC cell assignment.
    Array3D CICwindow(ksize);

    for (uint64 i = 0; i < ksize[0]; i++)
      for (int j = 0; j < ksize[1]; j++)
        for (int k = 0; k < ksize[2]; k++) {
          knorm.at(i, j, k) =
              sqrt(kx_cell[i] * kx_cell[i] + ky_cell[j] * ky_cell[j] +
                   kz_cell[k] * kz_cell[k]);
          Float window;
          switch (dens_.window_type()) {
            case kNearestCell:
              window = 1.0;
              break;
            case kCloudInCell: {
              // For TSC, the square window is 1-sin^2(kL/2)+2/15*sin^4(kL/2)
              Float sinkxL = sin(kx_cell[i] * cell_size / 2.0);
              Float sinkyL = sin(ky_cell[j] * cell_size / 2.0);
              Float sinkzL = sin(kz_cell[k] * cell_size / 2.0);
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
          CICwindow.at(i, j, k) = 1.0 / window;
          // We will divide the power spectrum by the square of the window
        }

    fprintf(stdout,
            "# Done setting up the wavevector submatrix of size +-%d, %d, %d\n",
            ksize[0] / 2, ksize[1] / 2, ksize[2] / 2);
    fprintf(stderr, "ksize = [%d, %d, %d]\n", ksize[0], ksize[1], ksize[2]);

    // Setup.Stop();

    // Here's where most of the work occurs.

    // Multiply total by 4*pi, to match SE15 normalization
    // Include the FFTW normalization
    Float norm = 4.0 * M_PI / ngrid[0] / ngrid[1] / ngrid[2];
    Float pnorm = 4.0 * M_PI;

    // Allocate total[csize**3] and corr[csize**3]
    Array3D total(csize);
    Array3D corr(csize);
    Array3D ktotal(ksize);
    Array3D kcorr(ksize);

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
    RowMajorArray<Complex> dens_fft(work_.carr().shape());
    // TODO: is it okay to do the copy initialization with complex rather than
    // floats for the purpose of assigning to physical hardware?
    array_ops::copy(work_.carr(), dens_fft);
    // Correlate.Start();

    /* ------------ Loop over ell & m --------------- */
    // Loop over each ell to compute the anisotropic correlations
    for (int ell = 0; ell <= maxell; ell += 2) {
      // Initialize the submatrix
      total.set_all(0.0);
      ktotal.set_all(0.0);
      // Loop over m
      for (int m = -ell; m <= ell; m++) {
        fprintf(stdout, "# Computing %d %2d...", ell, m);
        // Create the Ylm matrix times work_
        // TODO: here, is it advantageous if dens_ is padded as well, so its
        // boundaries match with those of work?
        makeYlm(&work_.arr(), ell, m, ngrid, xcell, ycell, zcell,
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
        makeYlm(&kcorr.arr(), ell, m, ksize, kx_cell, ky_cell, kz_cell,
                &CICwindow.arr(), wide_angle_exponent);
        // Multiply these Ylm by the power result, and then add to total.
        work_.extract_submatrix_C2R(&ktotal.arr(), &kcorr.arr());

        // iFFT the result, in place
        work_.execute_ifft();
        fprintf(stdout, "FFT...");

        // Create Ylm for the submatrix that we'll extract for histogramming
        // The extra multiplication by one here is of negligible cost, since
        // this array is so much smaller than the FFT grid.
        makeYlm(&corr.arr(), ell, m, csize, cx_cell, cy_cell, cz_cell, NULL,
                wide_angle_exponent);

        // Multiply these Ylm by the correlation result, and then add to total.
        work_.extract_submatrix(&total.arr(), &corr.arr());

        fprintf(stdout, "Done!\n");
        fflush(NULL);
      }

      // Extract.Start();
      total.multiply_by(norm);
      ktotal.multiply_by(pnorm);
      // Extract.Stop();
      // Histogram total by rnorm
      // Hist.Start();
      h->histcorr(ell, rnorm, total);
      kh->histcorr(ell, knorm, ktotal);
      // Hist.Stop();
      if (ell == 0) {
        *zerolag = total.at(rzero[0], rzero[1], rzero[2]);
      }
    }
    // Correlate.Stop();
  }

 private:
  const ConfigSpaceGrid &dens_;
  FftGrid work_;
};

#endif  // CORRELATE_H