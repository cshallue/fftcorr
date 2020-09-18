#ifndef CORRELATE_H
#define CORRELATE_H

#include <assert.h>

#include "fft_utils.h"
#include "grid.h"
#include "types.h"

void correlate(Grid &g, int maxell, Histogram &h, Histogram &kh,
               int wide_angle_exponent) {
  // Here's where most of the work occurs.

  // Multiply total by 4*pi, to match SE15 normalization
  // Include the FFTW normalization
  Float norm = 4.0 * M_PI / g.ngrid_[0] / g.ngrid_[1] / g.ngrid_[2];
  Float Pnorm = 4.0 * M_PI;
  assert(g.sep_ > 0);  // This is a check that the submatrix got set up.

  // Allocate the work matrix and load it with the density
  // We do this here so that the array is touched before FFT planning
  Float *work = NULL;  // work space for each (ell,m), in a flattened grid.
  initialize_matrix_by_copy(work, g.ngrid3_, g.ngrid_[0], g.dens_);

  // Allocate total[csize_**3] and corr[csize_**3]
  Float *total = NULL;
  initialize_matrix(total, g.csize3_, g.csize_[0]);
  Float *corr = NULL;
  initialize_matrix(corr, g.csize3_, g.csize_[0]);
  Float *ktotal = NULL;
  initialize_matrix(ktotal, g.ksize3_, g.ksize_[0]);
  Float *kcorr = NULL;
  initialize_matrix(kcorr, g.ksize3_, g.ksize_[0]);

  /* Setup FFTW */
  fftw_plan fft, fftYZ, fftX, ifft, ifftYZ, ifftX;
  setup_FFTW(fft, fftYZ, fftX, ifft, ifftYZ, ifftX, g.ngrid_, g.ngrid2_, work);

  // FFTW might have destroyed the contents of work; need to restore
  // work[]==dens_[] So far, I haven't seen this happen.
  if (g.dens_[1] != work[1] ||
      g.dens_[1 + g.ngrid_[2]] != work[1 + g.ngrid_[2]] ||
      g.dens_[g.ngrid3_ - 1] != work[g.ngrid3_ - 1]) {
    fprintf(stdout, "Restoring work matrix\n");
    // Init.Start();
    copy_matrix(work, g.dens_, g.ngrid3_, g.ngrid_[0]);
    // Init.Stop();
  }

  // Correlate .Start();  // Starting the main work
  // Now compute the FFT of the density field and conjugate it
  // FFT(work) in place and conjugate it, storing in densFFT_
  fprintf(stdout, "# Computing the density FFT...");
  fflush(NULL);
  FFT_Execute(fft, fftYZ, fftX, g.ngrid_, g.ngrid2_, work);

  // Correlate.Stop();  // We're tracking initialization separately
  initialize_matrix_by_copy(g.densFFT_, g.ngrid3_, g.ngrid_[0], work);
  fprintf(stdout, "Done!\n");
  fflush(NULL);
  // Correlate.Start();

  // Let's try a check as well -- convert with the 3D code and compare
  /* copy_matrix(work, dens_, ngrid3_, ngrid_[0]);
fftw_execute(fft);
for (uint64 j=0; j<ngrid3_; j++)
if (densFFT_[j]!=work[j]) {
  int z = j%ngrid2_;
  int y = j/ngrid2_; y=y%ngrid2_;
  int x = j/ngrid_[1]/ngrid2_;
  printf("%d %d %d  %f  %f\n", x, y, z, densFFT_[j], work[j]);
}
*/

  /* ------------ Loop over ell & m --------------- */
  // Loop over each ell to compute the anisotropic correlations
  for (int ell = 0; ell <= maxell; ell += 2) {
    // Initialize the submatrix
    set_matrix(total, 0.0, g.csize3_, g.csize_[0]);
    set_matrix(ktotal, 0.0, g.ksize3_, g.ksize_[0]);
    // Loop over m
    for (int m = -ell; m <= ell; m++) {
      fprintf(stdout, "# Computing %d %2d...", ell, m);
      // Create the Ylm matrix times dens_
      makeYlm(work, ell, m, g.ngrid_, g.ngrid2_, g.xcell_, g.ycell_, g.zcell_,
              g.dens_, -wide_angle_exponent);
      fprintf(stdout, "Ylm...");

      // FFT in place
      FFT_Execute(fft, fftYZ, fftX, g.ngrid_, g.ngrid2_, work);

      // Multiply by conj(densFFT_), as complex numbers
      // AtimesB.Start();
      multiply_matrix_with_conjugation((Complex *)work, (Complex *)g.densFFT_,
                                       g.ngrid3_ / 2, g.ngrid_[0]);
      // AtimesB.Stop();

      // Extract the anisotropic power spectrum
      // Load the Ylm's and include the CICwindow_ correction
      makeYlm(kcorr, ell, m, g.ksize_, g.ksize_[2], g.kx_cell_, g.ky_cell_,
              g.kz_cell_, g.CICwindow_, wide_angle_exponent);
      // Multiply these Ylm by the power result, and then add to total.
      extract_submatrix_C2R(ktotal, kcorr, g.ksize_, (Complex *)work, g.ngrid_,
                            g.ngrid2_);

      // iFFT the result, in place
      IFFT_Execute(ifft, ifftYZ, ifftX, g.ngrid_, g.ngrid2_, work);
      fprintf(stdout, "FFT...");

      // Create Ylm for the submatrix that we'll extract for histogramming
      // The extra multiplication by one here is of negligible cost, since
      // this array is so much smaller than the FFT grid.
      makeYlm(corr, ell, m, g.csize_, g.csize_[2], g.cx_cell_, g.cy_cell_,
              g.cz_cell_, NULL, wide_angle_exponent);

      // Multiply these Ylm by the correlation result, and then add to total.
      extract_submatrix(total, corr, g.csize_, work, g.ngrid_, g.ngrid2_);

      fprintf(stdout, "Done!\n");
    }

    // Extract.Start();
    scale_matrix(total, norm, g.csize3_, g.csize_[0]);
    scale_matrix(ktotal, Pnorm, g.ksize3_, g.ksize_[0]);
    // Extract.Stop();
    // Histogram total by rnorm_
    // Hist.Start();
    h.histcorr(ell, g.csize3_, g.rnorm_, total);
    kh.histcorr(ell, g.ksize3_, g.knorm_, ktotal);
    // Hist.Stop();
  }

  /* ------------------- Clean up -------------------*/
  free(work);
  // Free densFFT_ and Ylm
  free(corr);
  free(total);
  free(kcorr);
  free(ktotal);
  free_FFTW(fft, fftYZ, fftX, ifft, ifftYZ, ifftX);

  // Correlate.Stop();
}

#endif  // CORRELATE_H