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
  fprintf(stdout, "# Chosen separation %f vs max %f\n", sep, g.max_sep_);
  assert(sep <= g.max_sep_);

  int sep_cell = ceil(sep / g.cell_size_);
  g.csize_[0] = 2 * sep_cell + 1;
  g.csize_[1] = g.csize_[2] = g.csize_[0];
  assert(g.csize_[0] % 2 == 1);
  assert(g.csize_[1] % 2 == 1);
  assert(g.csize_[2] % 2 == 1);
  g.csize3_ = g.csize_[0] * g.csize_[1] * g.csize_[2];
  // Allocate corr_cell to [g.csize_] and g.rnorm_ to [g.csize_**3]
  int err;
  err = posix_memalign((void **)&g.cx_cell_, PAGE,
                       sizeof(Float) * g.csize_[0] + PAGE);
  assert(err == 0);
  err = posix_memalign((void **)&g.cy_cell_, PAGE,
                       sizeof(Float) * g.csize_[1] + PAGE);
  assert(err == 0);
  err = posix_memalign((void **)&g.cz_cell_, PAGE,
                       sizeof(Float) * g.csize_[2] + PAGE);
  assert(err == 0);
  initialize_matrix(g.rnorm_, g.csize3_, g.csize_[0]);

  // Normalizing by g.cell_size_ just so that the Ylm code can do the wide-angle
  // corrections in the same units.
  for (int i = 0; i < g.csize_[0]; i++)
    g.cx_cell_[i] = g.cell_size_ * (i - sep_cell);
  for (int i = 0; i < g.csize_[1]; i++)
    g.cy_cell_[i] = g.cell_size_ * (i - sep_cell);
  for (int i = 0; i < g.csize_[2]; i++)
    g.cz_cell_[i] = g.cell_size_ * (i - sep_cell);

  for (uint64 i = 0; i < g.csize_[0]; i++)
    for (int j = 0; j < g.csize_[1]; j++)
      for (int k = 0; k < g.csize_[2]; k++)
        g.rnorm_[k + g.csize_[2] * (j + i * g.csize_[1])] =
            g.cell_size_ * sqrt((i - sep_cell) * (i - sep_cell) +
                                (j - sep_cell) * (j - sep_cell) +
                                (k - sep_cell) * (k - sep_cell));
  fprintf(stdout, "# Done setting up the separation submatrix of size +-%d\n",
          sep_cell);

  // Our box has cubic-sized cells, so k_Nyquist is the same in all
  // directions. The spacing of modes is therefore 2*g.k_Nyq_/ngrid
  g.k_Nyq_ = M_PI / g.cell_size_;
  g.kmax_ = kmax;
  fprintf(stdout, "# Storing wavenumbers up to %6.4f, with g.k_Nyq_ = %6.4f\n",
          g.kmax_, g.k_Nyq_);
  for (int i = 0; i < 3; i++)
    g.ksize_[i] = 2 * ceil(g.kmax_ / (2.0 * g.k_Nyq_ / g.ngrid_[i])) + 1;
  assert(g.ksize_[0] % 2 == 1);
  assert(g.ksize_[1] % 2 == 1);
  assert(g.ksize_[2] % 2 == 1);
  for (int i = 0; i < 3; i++)
    if (g.ksize_[i] > g.ngrid_[i]) {
      g.ksize_[i] = 2 * floor(g.ngrid_[i] / 2) + 1;
      fprintf(stdout,
              "# WARNING: Requested wavenumber is too big.  Truncating "
              "ksize_[%d] to %d\n",
              i, g.ksize_[i]);
    }

  g.ksize3_ = g.ksize_[0] * g.ksize_[1] * g.ksize_[2];
  // Allocate g.kx_cell_ to [ksize_] and knorm_ to [ksize_**3]
  err = posix_memalign((void **)&g.kx_cell_, PAGE,
                       sizeof(Float) * g.ksize_[0] + PAGE);
  assert(err == 0);
  err = posix_memalign((void **)&g.ky_cell_, PAGE,
                       sizeof(Float) * g.ksize_[1] + PAGE);
  assert(err == 0);
  err = posix_memalign((void **)&g.kz_cell_, PAGE,
                       sizeof(Float) * g.ksize_[2] + PAGE);
  assert(err == 0);
  initialize_matrix(g.knorm_, g.ksize3_, g.ksize_[0]);
  initialize_matrix(g.CICwindow_, g.ksize3_, g.ksize_[0]);

  for (int i = 0; i < g.ksize_[0]; i++)
    g.kx_cell_[i] = (i - g.ksize_[0] / 2) * 2.0 * g.k_Nyq_ / g.ngrid_[0];
  for (int i = 0; i < g.ksize_[1]; i++)
    g.ky_cell_[i] = (i - g.ksize_[1] / 2) * 2.0 * g.k_Nyq_ / g.ngrid_[1];
  for (int i = 0; i < g.ksize_[2]; i++)
    g.kz_cell_[i] = (i - g.ksize_[2] / 2) * 2.0 * g.k_Nyq_ / g.ngrid_[2];

  for (uint64 i = 0; i < g.ksize_[0]; i++)
    for (int j = 0; j < g.ksize_[1]; j++)
      for (int k = 0; k < g.ksize_[2]; k++) {
        g.knorm_[k + g.ksize_[2] * (j + i * g.ksize_[1])] =
            sqrt(g.kx_cell_[i] * g.kx_cell_[i] + g.ky_cell_[j] * g.ky_cell_[j] +
                 g.kz_cell_[k] * g.kz_cell_[k]);
        // For TSC, the square window is 1-sin^2(kL/2)+2/15*sin^4(kL/2)
        Float sinkxL = sin(g.kx_cell_[i] * g.cell_size_ / 2.0);
        Float sinkyL = sin(g.ky_cell_[j] * g.cell_size_ / 2.0);
        Float sinkzL = sin(g.kz_cell_[k] * g.cell_size_ / 2.0);
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
        g.CICwindow_[k + g.ksize_[2] * (j + i * g.ksize_[1])] = 1.0 / window;
        // We will divide the power spectrum by the square of the window
      }

  fprintf(stdout,
          "# Done setting up the wavevector submatrix of size +-%d, %d, %d\n",
          g.ksize_[0] / 2, g.ksize_[1] / 2, g.ksize_[2] / 2);

  // Setup.Stop();

  // Here's where most of the work occurs.

  // Multiply total by 4*pi, to match SE15 normalization
  // Include the FFTW normalization
  Float norm = 4.0 * M_PI / g.ngrid_[0] / g.ngrid_[1] / g.ngrid_[2];
  Float Pnorm = 4.0 * M_PI;

  // Allocate the work matrix and load it with the density
  // We do this here so that the array is touched before FFT planning
  Float *work = NULL;  // work space for each (ell,m), in a flattened grid.
  initialize_matrix_by_copy(work, g.ngrid3_, g.ngrid_[0], g.dens_);

  // Allocate total[g.csize_**3] and corr[g.csize_**3]
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
  // FFT(work) in place and conjugate it, storing in densFFT
  fprintf(stdout, "# Computing the density FFT...");
  fflush(NULL);
  FFT_Execute(fft, fftYZ, fftX, g.ngrid_, g.ngrid2_, work);

  // Correlate.Stop();  // We're tracking initialization separately
  Float *densFFT = NULL;  // The FFT of the density field, in a flattened grid.
  initialize_matrix_by_copy(densFFT, g.ngrid3_, g.ngrid_[0], work);
  fprintf(stdout, "Done!\n");
  fflush(NULL);
  // Correlate.Start();

  // Let's try a check as well -- convert with the 3D code and compare
  /* copy_matrix(work, dens_, ngrid3_, g.ngrid_[0]);
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

      // Multiply by conj(densFFT), as complex numbers
      // AtimesB.Start();
      multiply_matrix_with_conjugation((Complex *)work, (Complex *)densFFT,
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
    // Histogram total by g.rnorm_
    // Hist.Start();
    h.histcorr(ell, g.csize3_, g.rnorm_, total);
    kh.histcorr(ell, g.ksize3_, g.knorm_, ktotal);
    // Hist.Stop();
  }

  /* ------------------- Clean up -------------------*/
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