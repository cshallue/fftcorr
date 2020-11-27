#ifndef CORRELATE_H
#define CORRELATE_H

#include <assert.h>

#include <array>

#include "discrete_field.h"
#include "grid.h"
#include "histogram.h"
#include "spherical_harmonics.h"
#include "types.h"
#include "window_functions.h"

// TODO: share common code between correlate_iso and correlate_aniso.
class Correlator {
 public:
  void correlate_iso(DiscreteField *dens, Float cell_size, Float sep,
                     Float kmax, WindowType window_type, Histogram1D *h,
                     Histogram1D *kh, Float *zerolag) {
    std::array<int, 3> ngrid = dens->rshape();  // TODO: rename?

    // Storage for the r-space submatrices
    int sep_cell = ceil(sep / cell_size);
    fprintf(stderr, "sep = %f, cell_size = %f, sep_cell =%d\n", sep, cell_size,
            sep_cell);
    // How many cells we must extract as a submatrix to do the histogramming.
    int csizex = 2 * sep_cell + 1;
    assert(csizex % 2 == 1);
    std::array<int, 3> csize = {csizex, csizex, csizex};
    fprintf(stderr, "csize = [%d, %d, %d]\n", csize[0], csize[1], csize[2]);

    Array3D rnorm;  // The radius of each cell.
    rnorm.initialize(csize);
    for (uint64 i = 0; i < csize[0]; i++)
      for (int j = 0; j < csize[1]; j++)
        for (int k = 0; k < csize[2]; k++)
          rnorm.at(i, j, k) = cell_size * sqrt((i - sep_cell) * (i - sep_cell) +
                                               (j - sep_cell) * (j - sep_cell) +
                                               (k - sep_cell) * (k - sep_cell));
    fprintf(stdout, "# Done setting up the separation submatrix of size +-%d\n",
            sep_cell);
    // Index of r=0.
    std::array<int, 3> zerosep = {sep_cell, sep_cell, sep_cell};

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

    // The wavenumber of each cell, in a flattened submatrix.
    Array3D knorm;
    knorm.initialize(ksize);

    fprintf(stdout,
            "# Done setting up the wavevector submatrix of size +-%d, %d, %d\n",
            ksize[0] / 2, ksize[1] / 2, ksize[2] / 2);
    fprintf(stderr, "ksize = [%d, %d, %d]\n", ksize[0], ksize[1], ksize[2]);

    // Allocate total[csize**3] and corr[csize**3]
    Array3D corr, kcorr;
    corr.initialize(csize);
    kcorr.initialize(ksize);

    // Setup.Stop();

    // Here's where most of the work occurs.

    // TODO: we should use the quick FFTW setup, since we only FFT and inverse
    // FFT once.
    dens->setup_fft();

    fprintf(stdout, "# Computing the density FFT...");
    dens->execute_fft();

    fprintf(stdout, "# Multiply...");
    fflush(NULL);
    dens->multiply_with_conjugation(*dens);

    // Extract power spectrum.
    // TODO: should this include a CICwindow correction like the aniso case?
    dens->extract_submatrix(&kcorr);

    // iFFT the result, in place
    fprintf(stdout, "IFFT...");
    fflush(NULL);
    dens->execute_ifft();

    fprintf(stdout, "# Done!\n");
    fflush(NULL);

    dens->extract_submatrix(&corr);

    // We must divide by two factors of ncells: the first one completes the
    // inverse FFT (FFTW doesn't include this factor automatically) and the
    // second is the factor converting the autocorrelation to the 2PCF.
    Float ncells = ngrid[0] * ngrid[1] * ngrid[2];
    Float norm = 1.0 / ncells / ncells;
    corr.multiply_by(norm);
    // Float Pnorm = 4.0 * M_PI;

    // Correlate .Start();  // Starting the main work

    // Histogram total by rnorm
    // Hist.Start();
    h->histcorr(rnorm, corr);
    kh->histcorr(knorm, kcorr);
    // Hist.Stop();
    // Correlate.Stop();
    *zerolag = corr.at(zerosep[0], zerosep[1], zerosep[2]);
  }

  // zerolag is needed because that information is not in the output histograms
  // (small but nonzero separations are put in the same bin as the zero
  // separation)
  void correlate_aniso(const DiscreteField &dens, std::array<Float, 3> origin,
                       Float cell_size, Float sep, Float kmax, int maxell,
                       int wide_angle_exponent, WindowType window_type,
                       Histogram2D *h, Histogram2D *kh, Float *zerolag) {
    // Set up the sub-matrix information, assuming that we'll extract
    // -sep..+sep cells around zero-lag.
    // Setup.Start();

    std::array<int, 3> ngrid = dens.rshape();  // TODO: rename?

    // Compute xcell, ycell, zcell, which are the coordinates of the cell
    // centers in each dimension, relative to the origin. Now set up the cell
    // centers relative to the origin, in grid units
    Array1D xcell = range(0.5 - origin[0], 1, ngrid[0]);
    Array1D ycell = range(0.5 - origin[1], 1, ngrid[1]);
    Array1D zcell = range(0.5 - origin[2], 1, ngrid[2]);

    // Storage for the r-space submatrices
    int sep_cell = ceil(sep / cell_size);
    fprintf(stderr, "sep = %f, cell_size = %f, sep_cell =%d\n", sep, cell_size,
            sep_cell);
    // How many cells we must extract as a submatrix to do the histogramming.
    int csizex = 2 * sep_cell + 1;
    assert(csizex % 2 == 1);
    std::array<int, 3> csize = {csizex, csizex, csizex};
    fprintf(stderr, "csize = [%d, %d, %d]\n", csize[0], csize[1], csize[2]);

    // Allocate corr_cell to [csize] and rnorm to [csize**3]
    // The cell centers, relative to zero lag.
    // Normalizing by cell_size just so that the Ylm code can do the wide-angle
    // corrections in the same units.
    Array1D cx_cell = range(-cell_size * sep_cell, cell_size, csize[0]);
    Array1D cy_cell = range(-cell_size * sep_cell, cell_size, csize[1]);
    Array1D cz_cell = range(-cell_size * sep_cell, cell_size, csize[2]);

    Array3D rnorm;  // The radius of each cell.
    rnorm.initialize(csize);
    for (uint64 i = 0; i < csize[0]; i++)
      for (int j = 0; j < csize[1]; j++)
        for (int k = 0; k < csize[2]; k++)
          rnorm.at(i, j, k) = cell_size * sqrt((i - sep_cell) * (i - sep_cell) +
                                               (j - sep_cell) * (j - sep_cell) +
                                               (k - sep_cell) * (k - sep_cell));
    fprintf(stdout, "# Done setting up the separation submatrix of size +-%d\n",
            sep_cell);
    // Index of r=0.
    std::array<int, 3> zerosep = {sep_cell, sep_cell, sep_cell};

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
    Array3D knorm;
    knorm.initialize(ksize);
    // The inverse of the window function for the CIC cell assignment.
    Array3D CICwindow;
    CICwindow.initialize(ksize);

    for (uint64 i = 0; i < ksize[0]; i++)
      for (int j = 0; j < ksize[1]; j++)
        for (int k = 0; k < ksize[2]; k++) {
          knorm.at(i, j, k) =
              sqrt(kx_cell[i] * kx_cell[i] + ky_cell[j] * ky_cell[j] +
                   kz_cell[k] * kz_cell[k]);
          Float window;
          switch (window_type) {
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
    Float Pnorm = 4.0 * M_PI;

    // Allocate the work matrix and load it with the density
    // Ensure that the array is touched before FFT planning
    DiscreteField work(ngrid);
    work.copy_from(dens);
    work.setup_fft();
    // FFTW might have destroyed the contents of work; need to restore
    // work[]==dens_[] So far, I haven't seen this happen.
    work.restore_from(dens);

    // Allocate total[csize**3] and corr[csize**3]
    Array3D total, corr, ktotal, kcorr;
    total.initialize(csize);
    corr.initialize(csize);
    ktotal.initialize(ksize);
    kcorr.initialize(ksize);

    // Correlate .Start();  // Starting the main work
    // Now compute the FFT of the density field and conjugate it
    // FFT(work) in place and conjugate it, storing in densFFT
    fprintf(stdout, "# Computing the density FFT...");
    fflush(NULL);
    work.execute_fft();
    fprintf(stdout, "# Done!\n");
    fflush(NULL);

    // Correlate.Stop();  // We're tracking initialization separately
    DiscreteField densFFT(ngrid);
    densFFT.copy_from(work);
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
        // Create the Ylm matrix times dens_
        makeYlm(&work.arr(), ell, m, ngrid, xcell, ycell, zcell, &dens.arr(),
                -wide_angle_exponent);
        fprintf(stdout, "Ylm...");

        // FFT in place
        work.execute_fft();

        // Multiply by conj(densFFT), as complex numbers
        // AtimesB.Start();
        work.multiply_with_conjugation(densFFT);
        // AtimesB.Stop();

        // Extract the anisotropic power spectrum
        // Load the Ylm's and include the CICwindow correction
        makeYlm(&kcorr, ell, m, ksize, kx_cell, ky_cell, kz_cell, &CICwindow,
                wide_angle_exponent);
        // Multiply these Ylm by the power result, and then add to total.
        work.extract_submatrix_C2R(&ktotal, &kcorr);

        // iFFT the result, in place
        work.execute_ifft();
        fprintf(stdout, "FFT...");

        // Create Ylm for the submatrix that we'll extract for histogramming
        // The extra multiplication by one here is of negligible cost, since
        // this array is so much smaller than the FFT grid.
        makeYlm(&corr, ell, m, csize, cx_cell, cy_cell, cz_cell, NULL,
                wide_angle_exponent);

        // Multiply these Ylm by the correlation result, and then add to total.
        work.extract_submatrix(&total, &corr);

        fprintf(stdout, "Done!\n");
        fflush(NULL);
      }

      // Extract.Start();
      total.multiply_by(norm);
      ktotal.multiply_by(Pnorm);
      // Extract.Stop();
      // Histogram total by rnorm
      // Hist.Start();
      h->histcorr(ell, rnorm, total);
      kh->histcorr(ell, knorm, ktotal);
      // Hist.Stop();
      if (ell == 0) {
        *zerolag = total.at(zerosep[0], zerosep[1], zerosep[2]);
      }
    }
    // Correlate.Stop();
  }
};

#endif  // CORRELATE_H