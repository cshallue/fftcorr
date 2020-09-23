#ifndef CORRELATE_H
#define CORRELATE_H

#include <assert.h>

#include "fft_utils.h"
#include "grid.h"
#include "types.h"

void correlate(Grid &g, Float sep, Float kmax, int maxell, Histogram &h,
               Histogram &kh, int wide_angle_exponent) {
  // Set up the sub-matrix information, assuming that we'll extract
  // -sep..+sep cells around zero-lag.
  // sep<0 causes a default to the value in the file.
  // Setup.Start();
  fprintf(stdout, "# Chosen separation %f vs max %f\n", sep, g.max_sep());
  assert(sep <= g.max_sep());

  // Storage for the r-space submatrices
  Float cell_size = g.cell_size();
  int sep_cell = ceil(sep / cell_size);
  // How many cells we must extract as a submatrix to do the histogramming.
  int csize[3];
  csize[0] = 2 * sep_cell + 1;
  csize[1] = csize[2] = csize[0];
  assert(csize[0] % 2 == 1);
  assert(csize[1] % 2 == 1);
  assert(csize[2] % 2 == 1);
  // The number of submatrix cells
  int csize3 = csize[0] * csize[1] * csize[2];
  // Allocate corr_cell to [csize] and rnorm to [csize**3]
  // The cell centers, relative to zero lag.
  Float *cx_cell, *cy_cell, *cz_cell;
  Float *rnorm = NULL;  // The radius of each cell, in a flattened submatrix.
  int err;
  err =
      posix_memalign((void **)&cx_cell, PAGE, sizeof(Float) * csize[0] + PAGE);
  assert(err == 0);
  err =
      posix_memalign((void **)&cy_cell, PAGE, sizeof(Float) * csize[1] + PAGE);
  assert(err == 0);
  err =
      posix_memalign((void **)&cz_cell, PAGE, sizeof(Float) * csize[2] + PAGE);
  assert(err == 0);
  initialize_matrix(rnorm, csize3, csize[0]);

  // Normalizing by cell_size just so that the Ylm code can do the wide-angle
  // corrections in the same units.
  for (int i = 0; i < csize[0]; i++) cx_cell[i] = cell_size * (i - sep_cell);
  for (int i = 0; i < csize[1]; i++) cy_cell[i] = cell_size * (i - sep_cell);
  for (int i = 0; i < csize[2]; i++) cz_cell[i] = cell_size * (i - sep_cell);

  for (uint64 i = 0; i < csize[0]; i++)
    for (int j = 0; j < csize[1]; j++)
      for (int k = 0; k < csize[2]; k++)
        rnorm[k + csize[2] * (j + i * csize[1])] =
            cell_size * sqrt((i - sep_cell) * (i - sep_cell) +
                             (j - sep_cell) * (j - sep_cell) +
                             (k - sep_cell) * (k - sep_cell));
  fprintf(stdout, "# Done setting up the separation submatrix of size +-%d\n",
          sep_cell);

  // Our box has cubic-sized cells, so k_Nyquist is the same in all
  // directions. The spacing of modes is therefore 2*k_Nyq/ngrid
  Float k_Nyq = M_PI / cell_size;  // The Nyquist frequency for our grid.
  fprintf(stdout, "# Storing wavenumbers up to %6.4f, with k_Nyq = %6.4f\n",
          kmax, k_Nyq);
  // How many cells we must extract as a submatrix to do the histogramming.
  int *ngrid = g.ngrid();
  int ngrid2 = g.ngrid2();
  uint64 ngrid3 = g.ngrid3();
  int ksize[3];
  for (int i = 0; i < 3; i++)
    ksize[i] = 2 * ceil(kmax / (2.0 * k_Nyq / ngrid[i])) + 1;
  assert(ksize[0] % 2 == 1);
  assert(ksize[1] % 2 == 1);
  assert(ksize[2] % 2 == 1);
  for (int i = 0; i < 3; i++)
    if (ksize[i] > ngrid[i]) {
      ksize[i] = 2 * floor(ngrid[i] / 2) + 1;
      fprintf(stdout,
              "# WARNING: Requested wavenumber is too big.  Truncating "
              "ksize_[%d] to %d\n",
              i, ksize[i]);
    }
  // The number of submatrix cells.
  int ksize3 = ksize[0] * ksize[1] * ksize[2];
  // The cell centers, relative to zero lag.
  // Allocate kx_cell to [ksize_] and knorm_ to [ksize_**3]
  Float *kx_cell, *ky_cell, *kz_cell;
  err =
      posix_memalign((void **)&kx_cell, PAGE, sizeof(Float) * ksize[0] + PAGE);
  assert(err == 0);
  err =
      posix_memalign((void **)&ky_cell, PAGE, sizeof(Float) * ksize[1] + PAGE);
  assert(err == 0);
  err =
      posix_memalign((void **)&kz_cell, PAGE, sizeof(Float) * ksize[2] + PAGE);
  assert(err == 0);
  // The wavenumber of each cell, in a flattened submatrix.
  Float *knorm = NULL;
  initialize_matrix(knorm, ksize3, ksize[0]);
  // The inverse of the window function for the CIC cell assignment.
  Float *CICwindow = NULL;
  initialize_matrix(CICwindow, ksize3, ksize[0]);

  for (int i = 0; i < ksize[0]; i++)
    kx_cell[i] = (i - ksize[0] / 2) * 2.0 * k_Nyq / ngrid[0];
  for (int i = 0; i < ksize[1]; i++)
    ky_cell[i] = (i - ksize[1] / 2) * 2.0 * k_Nyq / ngrid[1];
  for (int i = 0; i < ksize[2]; i++)
    kz_cell[i] = (i - ksize[2] / 2) * 2.0 * k_Nyq / ngrid[2];

  for (uint64 i = 0; i < ksize[0]; i++)
    for (int j = 0; j < ksize[1]; j++)
      for (int k = 0; k < ksize[2]; k++) {
        knorm[k + ksize[2] * (j + i * ksize[1])] =
            sqrt(kx_cell[i] * kx_cell[i] + ky_cell[j] * ky_cell[j] +
                 kz_cell[k] * kz_cell[k]);
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
        Float window = Wx * Wy * Wz;  // This is the square of the window
#ifdef NEAREST_CELL
        // For this case, the window is unity
        window = 1.0;
#endif
#ifdef WAVELET
        // For this case, the window is unity
        window = 1.0;
#endif
        CICwindow[k + ksize[2] * (j + i * ksize[1])] = 1.0 / window;
        // We will divide the power spectrum by the square of the window
      }

  fprintf(stdout,
          "# Done setting up the wavevector submatrix of size +-%d, %d, %d\n",
          ksize[0] / 2, ksize[1] / 2, ksize[2] / 2);

  // Setup.Stop();

  // Here's where most of the work occurs.

  // Multiply total by 4*pi, to match SE15 normalization
  // Include the FFTW normalization
  Float norm = 4.0 * M_PI / ngrid[0] / ngrid[1] / ngrid[2];
  Float Pnorm = 4.0 * M_PI;

  // Allocate the work matrix and load it with the density
  // We do this here so that the array is touched before FFT planning
  Float *dens = g.dens();
  Float *work = NULL;  // work space for each (ell,m), in a flattened grid.
  initialize_matrix_by_copy(work, ngrid3, ngrid[0], dens);

  // Allocate total[csize**3] and corr[csize**3]
  Float *total = NULL;
  initialize_matrix(total, csize3, csize[0]);
  Float *corr = NULL;
  initialize_matrix(corr, csize3, csize[0]);
  Float *ktotal = NULL;
  initialize_matrix(ktotal, ksize3, ksize[0]);
  Float *kcorr = NULL;
  initialize_matrix(kcorr, ksize3, ksize[0]);

  /* Setup FFTW */
  fftw_plan fft, fftYZ, fftX, ifft, ifftYZ, ifftX;
  setup_FFTW(fft, fftYZ, fftX, ifft, ifftYZ, ifftX, ngrid, ngrid2, work);

  // FFTW might have destroyed the contents of work; need to restore
  // work[]==dens_[] So far, I haven't seen this happen.
  if (dens[1] != work[1] || dens[1 + ngrid[2]] != work[1 + ngrid[2]] ||
      dens[ngrid3 - 1] != work[ngrid3 - 1]) {
    fprintf(stdout, "Restoring work matrix\n");
    // Init.Start();
    copy_matrix(work, dens, ngrid3, ngrid[0]);
    // Init.Stop();
  }

  // Correlate .Start();  // Starting the main work
  // Now compute the FFT of the density field and conjugate it
  // FFT(work) in place and conjugate it, storing in densFFT
  fprintf(stdout, "# Computing the density FFT...");
  fflush(NULL);
  FFT_Execute(fft, fftYZ, fftX, ngrid, ngrid2, work);

  // Correlate.Stop();  // We're tracking initialization separately
  Float *densFFT = NULL;  // The FFT of the density field, in a flattened grid.
  initialize_matrix_by_copy(densFFT, ngrid3, ngrid[0], work);
  fprintf(stdout, "Done!\n");
  fflush(NULL);
  // Correlate.Start();

  // Let's try a check as well -- convert with the 3D code and compare
  /* copy_matrix(work, dens_, ngrid3_, ngrid[0]);
fftw_execute(fft);
for (uint64 j=0; j<ngrid3_; j++)
if (densFFT[j]!=work[j]) {
  int z = j%ngrid2_;
  int y = j/ngrid2_; y=y%ngrid2_;
  int x = j/ngrid_[1]/ngrid2_;
  printf("%d %d %d  %f  %f\n", x, y, z, densFFT[j], work[j]);
}
*/

  /* ------------ Loop over ell & m --------------- */
  // Loop over each ell to compute the anisotropic correlations
  for (int ell = 0; ell <= maxell; ell += 2) {
    // Initialize the submatrix
    set_matrix(total, 0.0, csize3, csize[0]);
    set_matrix(ktotal, 0.0, ksize3, ksize[0]);
    // Loop over m
    for (int m = -ell; m <= ell; m++) {
      fprintf(stdout, "# Computing %d %2d...", ell, m);
      // Create the Ylm matrix times dens_
      makeYlm(work, ell, m, ngrid, ngrid2, g.xcell_, g.ycell_, g.zcell_, dens,
              -wide_angle_exponent);
      fprintf(stdout, "Ylm...");

      // FFT in place
      FFT_Execute(fft, fftYZ, fftX, ngrid, ngrid2, work);

      // Multiply by conj(densFFT), as complex numbers
      // AtimesB.Start();
      multiply_matrix_with_conjugation((Complex *)work, (Complex *)densFFT,
                                       ngrid3 / 2, ngrid[0]);
      // AtimesB.Stop();

      // Extract the anisotropic power spectrum
      // Load the Ylm's and include the CICwindow correction
      makeYlm(kcorr, ell, m, ksize, ksize[2], kx_cell, ky_cell, kz_cell,
              CICwindow, wide_angle_exponent);
      // Multiply these Ylm by the power result, and then add to total.
      extract_submatrix_C2R(ktotal, kcorr, ksize, (Complex *)work, ngrid,
                            ngrid2);

      // iFFT the result, in place
      IFFT_Execute(ifft, ifftYZ, ifftX, ngrid, ngrid2, work);
      fprintf(stdout, "FFT...");

      // Create Ylm for the submatrix that we'll extract for histogramming
      // The extra multiplication by one here is of negligible cost, since
      // this array is so much smaller than the FFT grid.
      makeYlm(corr, ell, m, csize, csize[2], cx_cell, cy_cell, cz_cell, NULL,
              wide_angle_exponent);

      // Multiply these Ylm by the correlation result, and then add to total.
      extract_submatrix(total, corr, csize, work, ngrid, ngrid2);

      fprintf(stdout, "Done!\n");
    }

    // Extract.Start();
    scale_matrix(total, norm, csize3, csize[0]);
    scale_matrix(ktotal, Pnorm, ksize3, ksize[0]);
    // Extract.Stop();
    // Histogram total by rnorm
    // Hist.Start();
    h.histcorr(ell, csize3, rnorm, total);
    kh.histcorr(ell, ksize3, knorm, ktotal);
    // Hist.Stop();
  }

  /* ------------------- Clean up -------------------*/
  free(rnorm);
  free(cx_cell);
  free(cy_cell);
  free(cz_cell);
  free(knorm);
  free(kx_cell);
  free(ky_cell);
  free(kz_cell);
  free(CICwindow);
  free(work);
  free(densFFT);
  free(corr);
  free(total);
  free(kcorr);
  free(ktotal);
  free_FFTW(fft, fftYZ, fftX, ifft, ifftYZ, ifftX);

  // Correlate.Stop();
}

#endif  // CORRELATE_H