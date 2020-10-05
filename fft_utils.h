#ifndef FFT_UTIL_H
#define FFT_UTIL_H

#include <assert.h>

#include "types.h"

/* ===============================  FFTW wrapper routines =====================
 */

void setup_FFTW(fftw_plan &fft, fftw_plan &fftYZ, fftw_plan &fftX,
                fftw_plan &ifft, fftw_plan &ifftYZ, fftw_plan &ifftX,
                int ngrid[3], const int ngrid2, Float *work) {
  // Setup the FFTW plans, possibly from disk, and save the wisdom
  fprintf(stdout, "# Planning the FFTs...");
  fflush(NULL);
  // FFTW.Start();
  FILE *fp = NULL;
#ifdef OPENMP
#ifndef FFTSLAB
  {
    int errval = fftw_init_threads();
    assert(errval);
  }
  fftw_plan_with_nthreads(omp_get_max_threads());
#endif
#define WISDOMFILE "wisdom_fftw_omp"
#else
#define WISDOMFILE "wisdom_fftw"
#endif
#ifdef FFTSLAB
#undef WISDOMFILE
#define WISDOMFILE "wisdom_fftw"
#endif
  fp = fopen(WISDOMFILE, "r");
  if (fp != NULL) {
    fprintf(stdout, "Reading %s...", WISDOMFILE);
    fflush(NULL);
    fftw_import_wisdom_from_file(fp);
    fclose(fp);
  }

#ifndef FFTSLAB
  // The following interface should work even if ngrid2 was 'non-minimal',
  // as might be desired by padding.
  int nfft[3], nfftc[3];
  nfft[0] = nfftc[0] = ngrid[0];
  nfft[1] = nfftc[1] = ngrid[1];
  // Since ngrid2 is always even, this will trick
  // FFTW to assume ngrid2/2 Complex numbers in the result, while
  // fulfilling that nfft[2]>=ngrid[2].
  nfft[2] = ngrid2;
  nfftc[2] = nfft[2] / 2;
  fftw_complex *cwork = (fftw_complex *)work;  // Interpret work as complex.
  int howmany = 1;  // Only one forward and inverse FFT.
  int dist = 0;     // Unused because howmany = 1.
  int stride = 1;   // Array is continuous in memory.
  fft = fftw_plan_many_dft_r2c(3, ngrid, howmany, work, nfft, stride, dist,
                               cwork, nfftc, stride, dist, FFTW_MEASURE);
  ifft = fftw_plan_many_dft_c2r(3, ngrid, howmany, cwork, nfftc, stride, dist,
                                work, nfft, stride, dist, FFTW_MEASURE);

  /*	// The original interface, which only works if ngrid2 is tightly packed.
  fft = fftw_plan_dft_r2c_3d(ngrid[0], ngrid[1], ngrid[2],
                  work, (fftw_complex *)work, FFTW_MEASURE);
  ifft = fftw_plan_dft_c2r_3d(ngrid[0], ngrid[1], ngrid[2],
                  (fftw_complex *)work, work, FFTW_MEASURE);
*/

#else
  // If we wanted to split into 2D and 1D by hand (and therefore handle the OMP
  // aspects ourselves), then we need to have two plans each.
  int nfft2[2], nfft2c[2];
  nfft2[0] = nfft2c[0] = ngrid[1];
  nfft2[1] = ngrid2;  // Since ngrid2 is always even, this will trick
  nfft2c[1] = nfft2[1] / 2;
  int ngridYZ[2];
  ngridYZ[0] = ngrid[1];
  ngridYZ[1] = ngrid[2];
  fftYZ =
      fftw_plan_many_dft_r2c(2, ngridYZ, 1, work, nfft2, 1, 0,
                             (fftw_complex *)work, nfft2c, 1, 0, FFTW_MEASURE);
  ifftYZ = fftw_plan_many_dft_c2r(2, ngridYZ, 1, (fftw_complex *)work, nfft2c,
                                  1, 0, work, nfft2, 1, 0, FFTW_MEASURE);

  // After we've done the 2D r2c FFT, we have to do the 1D c2c transform.
  // We'll plan to parallelize over Y, so that we're doing (ngrid[2]/2+1)
  // 1D FFTs at a time.
  // Elements in the X direction are separated by ngrid[1]*ngrid2/2 complex
  // numbers.
  int ngridX = ngrid[0];
  fftX =
      fftw_plan_many_dft(1, &ngridX, (ngrid[2] / 2 + 1), (fftw_complex *)work,
                         NULL, ngrid[1] * ngrid2 / 2, 1, (fftw_complex *)work,
                         NULL, ngrid[1] * ngrid2 / 2, 1, -1, FFTW_MEASURE);
  ifftX =
      fftw_plan_many_dft(1, &ngridX, (ngrid[2] / 2 + 1), (fftw_complex *)work,
                         NULL, ngrid[1] * ngrid2 / 2, 1, (fftw_complex *)work,
                         NULL, ngrid[1] * ngrid2 / 2, 1, +1, FFTW_MEASURE);
#endif

  fp = fopen(WISDOMFILE, "w");
  assert(fp != NULL);
  fftw_export_wisdom_to_file(fp);
  fclose(fp);
  fprintf(stdout, "Done!\n");
  fflush(NULL);
  // FFTW.Stop();
  return;
}

void FFT_Execute(fftw_plan fft, fftw_plan fftYZ, fftw_plan fftX, int ngrid[3],
                 const int ngrid2, Float *work) {
  // Note that if FFTSLAB is not set, then the *work input is ignored!
  // Routine will use the array that was called for setup!
  // TODO: Might fix this behavior, but note alignment issues!
  // FFTonly.Start();
#ifndef FFTSLAB
  fftw_execute(fft);
#else
  // FFTyz.Start();
// Then need to call this for every slab.  Can OMP these lines
#pragma omp parallel for MY_SCHEDULE
  for (uint64 x = 0; x < ngrid[0]; x++)
    fftw_execute_dft_r2c(fftYZ, work + x * ngrid[1] * ngrid2,
                         (fftw_complex *)work + x * ngrid[1] * ngrid2 / 2);
    // FFTyz.Stop();
    // FFTx.Start();
#pragma omp parallel for schedule(dynamic, 1)
  for (uint64 y = 0; y < ngrid[1]; y++)
    fftw_execute_dft(fftX, (fftw_complex *)work + y * ngrid2 / 2,
                     (fftw_complex *)work + y * ngrid2 / 2);
    // FFTx.Stop();
#endif
  // FFTonly.Stop();
}

void IFFT_Execute(fftw_plan ifft, fftw_plan ifftYZ, fftw_plan ifftX,
                  int ngrid[3], const int ngrid2, Float *work) {
  // Note that if FFTSLAB is not set, then the *work input is ignored!
  // Routine will use the array that was called for setup!
  // TODO: Might fix this behavior, but note alignment issues!
  // FFTonly.Start();
#ifndef FFTSLAB
  fftw_execute(ifft);
#else
  // FFTx.Start();
// Then need to call this for every slab.  Can OMP these lines
#pragma omp parallel for schedule(dynamic, 1)
  for (uint64 y = 0; y < ngrid[1]; y++)
    fftw_execute_dft(ifftX, (fftw_complex *)work + y * ngrid2 / 2,
                     (fftw_complex *)work + y * ngrid2 / 2);
    // FFTx.Stop();
    // FFTyz.Start();
#pragma omp parallel for MY_SCHEDULE
  for (uint64 x = 0; x < ngrid[0]; x++)
    fftw_execute_dft_c2r(ifftYZ,
                         (fftw_complex *)work + x * ngrid[1] * ngrid2 / 2,
                         work + x * ngrid[1] * ngrid2);
    // FFTyz.Stop();
#endif
  // FFTonly.Stop();
}

#endif  // FFT_UTIL_H