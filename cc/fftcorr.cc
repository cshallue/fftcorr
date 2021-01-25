/* fftcorr.cpp -- Daniel Eisenstein, July 2016
This computes anisotropic correlation functions using the FFT techniques of
Slepian & Eisenstein 2015.

Input particles files have a specific binary format:

64 bytes header:  double posmin[3], posmax[3], max_sep, blank8;
Then 4 doubles per particle: x,y,z,weight, repeating for each particle.

Posmin and posmax are the bounding box for the particle data.
Input posmin and posmax must be oversized so that no periodic replicas are
within max_sep of each other.  Don't forget to include the cell-size
and CIC effects in this estimate; we recommend padding by another
50 Mpc or so, just to be safe.  This code does not periodic wrap, so
it is required that posmin and posmax be padded.

There is a python code that can check the cubic box case without
CIC.  The last time I checked, it matched to 9 digits, so one can
do very well.  Remember that this is only true for NEAREST_CELL.

I have not compared in detail to a pair-counting code, but I did
verify that the monopole of RR matches to O(10%).  That is, the
basic normalization makes sense.  Also, results are stable as the
grid changes; no grid normalization issues.  Padding the box out
further (with constant cell size and registration) doesn't change
the results at all, as one would expect.


Memory usage: Dominated by 3 double-precision arrays of size Nx*Ny*Nz.
Note that the 3 box dimensions need not be equal, although one should pick
FFT-friendly values.  There is value in rotating the survey to fit in a
minimum rectangular box.

Use the -cell (-c) option to force a cell size, which may then
oversize the box.  This might be useful in combining disjoint survey
regions, for example.

Use the -periodic (-p) option to configure for a cubic box.  In
this case, the posmin/posmax inputs should not have any padding,
but should reflect the periodic wrapping length.  The observer
will be placed far away in the -x direction from the center of
the box.

Using the -zeromean (-z) option will invoke -periodic and then also
set the mean density of the box to zero.  This may allow one to
avoid using a random catalog for zero mean cases, as it is the
equivalent to entering N=D-R with an infinite set of R, but this
requires further testing.

Run time: On a single-core MacBook Pro, 512^3, ell=2 is taking about
40 seconds.  The time is 90+% dominated by the FFTs.  Computing the
Ylm's is 5%.  Both of these scale with Nx*Ny*Nz.

The primary correlation routine is multi-threaded.  On Odyssey
(64-core Athlons), it takes about 45 seconds to do 1024*1024*768
to ell=4.

On the 4-socket Opteron machine, I found noticeable performance gains if
the 3 main arrays were assigned YZ slabs to individual (and consistent)
threads.  This is because the memory then ends up on the memory bank
specific to each thread.  Only the X FFT needs to cross memory banks.
Use -DSLAB -DFFTSLAB to turn this on.  I peak around 50-60 GFLOPS
for this case.

Loading and gridding the particles is subdominant for BOSS-sized
problems.  But this would be much larger if we had a particle density
typical of N-body simulations rather than SDSS BOSS galaxy density!
This part is multi-threaded and does sorting to try to help the
cloud-in-cell to keep up.

*/

/* ======================= Preamble ================= */
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <algorithm>
#include <array>
#include <vector>

#include "STimer.cc"
#include "array/array_ops.h"
#include "array/row_major_array.h"
#include "correlate.h"
#include "grid.h"
#include "grid/config_space_grid.h"
#include "histogram.h"
#include "multithreading.h"
#include "particle_mesh/mass_assignor.h"
#include "particle_mesh/window_functions.h"
#include "read_catalog.h"
#include "survey_box.h"
#include "types.h"

STimer IO, Setup, FFTW, Correlate, YlmTime, Total, CIC, Misc, FFTonly, Hist,
    Extract, AtimesB, Init, FFTyz, FFTx;

void ReportTimes(FILE *fp, uint64 nfft, uint64 ngrid3, int cnt) {
  fflush(NULL);
  fprintf(fp, "#\n# Timing Report: \n");
  fprintf(fp, "# Setup time:       %8.4f s\n", Setup.Elapsed());
  fprintf(
      fp,
      "# I/O time:         %8.4f s, %6.3f Mparticles/sec, %6.2f MB/sec Read\n",
      IO.Elapsed(), cnt / IO.Elapsed() / 1e6, cnt / IO.Elapsed() * 32.0 / 1e6);
  fprintf(fp,
          "# CIC Grid time:    %8.4f s, %6.3f Mparticles/sec, %6.2f GB/sec\n",
          CIC.Elapsed(), cnt / CIC.Elapsed() / 1e6,
          1e-9 * cnt / CIC.Elapsed() * 27.0 * 2.0 * sizeof(Float));
  fprintf(fp, "#        Sorting:   %8.4f s\n", Sorting.Elapsed());
  fprintf(fp, "#        Merging:   %8.4f s\n", Merging.Elapsed());
  fprintf(fp, "#            CIC:   %8.4f s\n",
          CIC.Elapsed() - Merging.Elapsed() - Sorting.Elapsed());
  fprintf(fp, "# FFTW Prep time:   %8.4f s\n", FFTW.Elapsed());
  fprintf(fp, "# Array Init time:  %8.4f s, %6.3f GB/s\n", Init.Elapsed(),
          1e-9 * ngrid3 * sizeof(Float) * 5 / Init.Elapsed());
  fprintf(fp, "# Correlate time:   %8.4f s\n", Correlate.Elapsed());
  // Expecting 6 Floats of load/store
  fprintf(fp,
          "#       FFT time:   %8.4f s, %6.3f Mcells/s, %6.3f GB/s, %6.3f "
          "GFLOPS/s\n",
          FFTonly.Elapsed(), nfft / 1e6 / FFTonly.Elapsed(),
          nfft / 1e9 / FFTonly.Elapsed() * 6.0 * sizeof(Float),
          nfft / 1e6 / FFTonly.Elapsed() * 2.5 * log(ngrid3) / log(2) / 1e3);
#ifdef FFTSLAB
  fprintf(fp,
          "#     FFTyz time:   %8.4f s, %6.3f Mcells/s, %6.3f GB/s, %6.3f "
          "GFLOPS/s\n",
          FFTyz.Elapsed(), nfft / 1e6 / FFTyz.Elapsed() * 2.0 / 3.0,
          nfft / 1e9 / FFTyz.Elapsed() * 6.0 * sizeof(Float) * 2.0 / 3.0,
          nfft / 1e6 / FFTyz.Elapsed() * 2.5 * log(ngrid3) / log(2) / 1e3 *
              2.0 / 3.0);
  fprintf(fp,
          "#      FFTx time:   %8.4f s, %6.3f Mcells/s, %6.3f GB/s, %6.3f "
          "GFLOPS/s\n",
          FFTx.Elapsed(), nfft / 1e6 / FFTx.Elapsed() / 3.0,
          nfft / 1e9 / FFTx.Elapsed() * 6.0 * sizeof(Float) / 3.0,
          nfft / 1e6 / FFTx.Elapsed() * 2.5 * log(ngrid3) / log(2) / 1e3 / 3.0);
#endif
  // Approximating number of Ylm cells as FFT/2.
  // Each stores one float, but nearly all load one float too.
  fprintf(
      fp, "#       Ylm time:   %8.4f s, %6.3f GB/s\n", YlmTime.Elapsed(),
      (nfft - ngrid3) / 2.0 / 1e9 / YlmTime.Elapsed() * sizeof(Float) * 2.0);
  fprintf(fp, "#      Hist time:   %8.4f s\n", Hist.Elapsed());
  fprintf(fp, "#   Extract time:   %8.4f s\n", Extract.Elapsed());
  // We're doing two FFTs per loop and then one extra, so like 2*N+1
  // Hence N examples of A*Bt, each of which is 3 Floats of load/store
  fprintf(fp, "#      A*Bt time:   %8.4f s, %6.3f M/s of A=A*Bt, %6.3f GB/s\n",
          AtimesB.Elapsed(),
          (nfft / 2.0 / ngrid3 - 0.5) * ngrid3 / 1e6 / AtimesB.Elapsed(),
          (nfft / 2.0 / ngrid3 - 0.5) * ngrid3 / 1e9 / AtimesB.Elapsed() * 3.0 *
              sizeof(Float));
  fprintf(fp, "# Total time:       %8.4f s\n", Total.Elapsed());
  if (Misc.Elapsed() > 0.0) {
    fprintf(fp, "#\n# Misc time:          %8.4f s\n", Misc.Elapsed());
  }
  return;
}

class ThreadCount {
  int *cnt;
  int max;

 public:
  ThreadCount(int max_threads) {
    max = max_threads;
    int err = posix_memalign((void **)&cnt, PAGE, sizeof(int) * 8 * max);
    assert(err == 0);
    for (int j = 0; j < max * 8; j++) cnt[j] = 0;
    return;
  }
  ~ThreadCount() {
    free(cnt);
    return;
  }
  void add() { cnt[omp_get_thread_num() * 8]++; }
  void print(FILE *fp) {
    for (int j = 0; j < max; j++)
      if (cnt[j * 8] > 0) fprintf(fp, "# Thread %2d = %d\n", j, cnt[j * 8]);
  }
};

#define MAX_THREADS 128
ThreadCount Ylm_count(MAX_THREADS);

void usage() {
  fprintf(stderr, "FFTCORR: Error in command-line \n");
  fprintf(stderr,
          "   -n <int> (or -ngrid): FFT linear grid size for a cubic box\n");
  fprintf(stderr,
          "   -n3 <int> <int> <int> (or -ngrid3): FFT linear grid sizes for "
          "rectangle\n");
  fprintf(stderr, "             -n3 will outrank -n\n");
  fprintf(stderr, "   -ell <int> (or -maxell): Multipole to compute.\n");
  fprintf(stderr,
          "   -b <float> (or -box): Bounding box size.  Must exceed value in "
          "input file.\n");

  fprintf(stderr, "             <0 will default to value in input file.\n");
  fprintf(stderr,
          "   -r <float> (or -sep): Max separation.  Cannot exceed value in "
          "input file.\n");
  fprintf(stderr, "             <0 will default to value in input file.\n");
  fprintf(stderr, "   -dr <float> (or -dsep): Binning of separation.\n");
  fprintf(stderr, "   -kmax <float>: Maximum wavenumber for power spectrum.\n");
  fprintf(stderr, "   -dk <float>: Binning of wavenumber.\n");
  fprintf(stderr, "   -periodic (or -p): Configure for cubic periodic box.\n");
  fprintf(stderr,
          "   -zeromean (or -z): Configure for cubic periodic box and set mean "
          "density to zero.\n");
  fprintf(stderr,
          "   -normalize: Configure for cubic periodic box, set mean "
          "density to zero and divide by the mean.\n");
  fprintf(stderr, "   -iso: Isotropic correlations.\n");
  fprintf(stderr,
          "   -w <int> (or -window): Window function for mass assignment. See "
          "enum WindowType.\n");
  fprintf(stderr, "   -in <filename>:  Input file name\n");
  fprintf(stderr, "   -in2 <filename>: Second input file name\n");
  fprintf(stderr, "   -out <filename>: Output file name, default to stdout\n");
  fprintf(stderr,
          "   -exp <int> (or -e): Use the wide-angle exponent of Slepian & "
          "Eisenstein 2016 (niche users only)\n");
  fprintf(stderr, "\n");
  exit(1);
}

void print_hist(const Histogram &h, FILE *fp, int prefix, bool norm) {
  // Print out the results
  // If norm==1, divide by counts
  const Array1D<Float> &bins = h.bins();
  const Array1D<int> &count = h.count();
  const RowMajorArray<Float, 2> &accum = h.accum();
  for (uint64 j = 0; j < bins.size(); ++j) {
    fprintf(fp, "%1d ", prefix);
    fprintf(fp, "%7.4f %8.0f", bins[j], (Float)count[j]);
    Float denom = norm && count[j] != 0 ? count[j] : 1.0;
    for (int i = 0; i < accum.shape(0); ++i)
      fprintf(fp, " %16.9e", accum.at(i, j) / denom);
    fprintf(fp, "\n");
  }
}

/* ===========================================================================
 */

int main(int argc, char *argv[]) {
  // Need to get this information.
  Total.Start();
  // Here are some defaults
  Float sep = -123.0;
  Float dsep = 10.0;
  Float kmax = 0.03;
  Float dk = 0.01;
  int maxell = 0;
  int wide_angle_exponent = 0;
  int ngridCube = 256;
  int qperiodic = 0;
  WindowType window_type = kWavelet;
  bool isotropic = false;
  std::array<int, 3> ngrid = {-1, -1, -1};
  Float cell_size = -123.0;  // Default to what's implied by the file
  const char default_fname[] = "/tmp/corrRR.dat";
  char *infile = NULL;
  char *infile2 = NULL;
  char *outfile = NULL;

  int i = 1;
  while (i < argc) {
    if (!strcmp(argv[i], "-ngrid") || !strcmp(argv[i], "-n"))
      ngridCube = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-ngrid3") || !strcmp(argv[i], "-n3")) {
      ngrid[0] = atoi(argv[++i]);
      ngrid[1] = atoi(argv[++i]);
      ngrid[2] = atoi(argv[++i]);
    } else if (!strcmp(argv[i], "-maxell") || !strcmp(argv[i], "-ell"))
      maxell = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-exp") || !strcmp(argv[i], "-e"))
      wide_angle_exponent = atoi(argv[++i]);
    else if (!strcmp(argv[i], "-sep") || !strcmp(argv[i], "-r"))
      sep = atof(argv[++i]);
    else if (!strcmp(argv[i], "-dsep") || !strcmp(argv[i], "-dr"))
      dsep = atof(argv[++i]);
    else if (!strcmp(argv[i], "-kmax") || !strcmp(argv[i], "-k"))
      kmax = atof(argv[++i]);
    else if (!strcmp(argv[i], "-dk") || !strcmp(argv[i], "-dk"))
      dk = atof(argv[++i]);
    else if (!strcmp(argv[i], "-cell") || !strcmp(argv[i], "-c"))
      cell_size = atof(argv[++i]);
    else if (!strcmp(argv[i], "-in") || !strcmp(argv[i], "-i"))
      infile = argv[++i];
    else if (!strcmp(argv[i], "-in2") || !strcmp(argv[i], "-i2"))
      infile2 = argv[++i];
    else if (!strcmp(argv[i], "-out") || !strcmp(argv[i], "-o"))
      outfile = argv[++i];
    else if (!strcmp(argv[i], "-periodic") || !strcmp(argv[i], "-p"))
      qperiodic = 1;
    else if (!strcmp(argv[i], "-zeromean") || !strcmp(argv[i], "-z"))
      qperiodic = 2;
    else if (!strcmp(argv[i], "-normalize"))
      qperiodic = 3;
    else if (!strcmp(argv[i], "-iso"))
      isotropic = true;
    else if (!strcmp(argv[i], "-window") || !strcmp(argv[i], "-w"))
      window_type = WindowType(atoi(argv[++i]));
    else
      usage();
    i++;
  }

  assert(ngridCube > 0);
  assert(maxell >= 0 && maxell % 2 == 0);
  assert(wide_angle_exponent % 2 == 0);  // Must be an even number
  assert(sep > 0.0);
  assert(dsep > 0.0);
  assert(kmax != 0.0);
  assert(dk > 0.0);
  // If qperiodic is set, user cannot supply a cell_size
  assert(qperiodic == 0 || cell_size < 0);
  if (infile == NULL) infile = (char *)default_fname;
  if (outfile != NULL) {
    // printf("%s\n", outfile);
    FILE *discard = freopen(outfile, "w", stdout);
    assert(discard != NULL && stdout != NULL);
  }

  if (ngrid[0] <= 0) ngrid[0] = ngrid[1] = ngrid[2] = ngridCube;
  assert(ngrid[0] > 0);
  assert(ngrid[1] > 0);
  assert(ngrid[2] > 0);

#ifdef OPENMP
  fprintf(stdout, "# Running with %d threads\n", omp_get_max_threads());
#else
  fprintf(stdout, "# Running single threaded.\n");
#endif

  setup_wavelet();

  // Read box dimensions from catalog header.
  SurveyBox box;
  box.read_header(infile);
  if (!qperiodic) {
    box.pad_to_sep(sep);
  }

  // TODO: get rid of Grid class?
  Grid g(ngrid);
  cell_size = g.cover_box(box, qperiodic, cell_size);
  ConfigSpaceGrid grid(ngrid, g.posmin(), cell_size,
                       window_type);  // TODO: grid name clash

  int galaxy_batch_size = 1000000;
  MassAssignor mass_assignor(&grid, galaxy_batch_size);
  SurveyReader reader(&mass_assignor);
  reader.read_galaxies(infile);
  if (infile2 != NULL) {
    reader.read_galaxies(infile2);
  }
  // TODO: it should be the same if this goes at the end of read_galaxies(), but
  // right now it causes different behavior! Figure this out: does the same
  // thing happen with NEAREST_CELL?
  mass_assignor.flush();

  RowMajorArray<Float, 3> &dens = grid.data();
  fprintf(stdout, "# Found %d particles. Total weight %10.4e.\n",
          mass_assignor.count(), mass_assignor.totw());
  Float totw2 = array_ops::sum(dens);
  fprintf(stdout, "# Sum of grid is %10.4e (delta = %10.4e)\n", totw2,
          totw2 - mass_assignor.totw());
  fprintf(stdout, "# Sum of squares of grid is %10.4e \n",
          array_ops::sumsq(dens));
  if (qperiodic >= 2) {
    // We're asked to set the mean to zero
    Float mean = mass_assignor.totw() / dens.size();
    array_ops::add_scalar(-mean, dens);
    fprintf(stdout, "# Subtracting mean cell density %10.4e\n", mean);
    if (qperiodic == 3) {
      // Also divide by the mean.
      array_ops::multiply_by(1.0 / mean, dens);
      fprintf(stdout, "# Also dividing by mean cell density %10.4e\n", mean);
    }
  }

  Float totwsq = mass_assignor.totwsq();
  Float sumsq_dens = array_ops::sumsq(dens);  // TODO: repeated from above
  fprintf(stdout, "# Sum of squares of density = %14.7e\n", sumsq_dens);
  fprintf(stdout,
          "# Sum of squares of weights (divide by I for Pshot) = %14.7e\n",
          totwsq);
  // When run with N=D-R, this divided by I would be the shot noise.

  // Meanwhile, an estimate of I when running with only R is
  // (sum of R^2)/Vcell - (11/20)**3*(sum_R w^2)/Vcell
  // The latter is correcting the estimate for shot noise
  // The 11/20 factor is for triangular cloud in cell.
  switch (window_type) {
    case kNearestCell:
      fprintf(stdout, "# Using nearest cell method\n");
      break;
    case kCloudInCell:
      totwsq *= 0.55 * 0.55 * 0.55;
      fprintf(stdout, "# Using triangular cloud-in-cell\n");
      break;
    case kWavelet:
      fprintf(stdout, "# Using D12 wavelet\n");
      break;
  }

  Float Vcell = cell_size * cell_size * cell_size;
  fprintf(stdout, "# Estimate of I (denominator) = %14.7e - %14.7e = %14.7e\n",
          sumsq_dens / Vcell, totwsq / Vcell, (sumsq_dens - totwsq) / Vcell);

  // In the limit of infinite homogeneous particles in a periodic box:
  // If W=sum(w), then each particle has w = W/N.  totwsq = N*(W/N)^2 =
  // W^2/N. Meanwhile, each cell has density (W/N)*(N/Ncell) = W/Ncell.
  // sumsq_dens/Vcell = W^2/(Ncell*Vcell) = W^2/V.
  // Hence the real shot noise is V/N = 1/n.

  /* Done setup Grid ======================================================= */

  // Compute the correlations.
  Histogram h(maxell / 2 + 1, 0.0, sep, dsep);
  Histogram kh(maxell / 2 + 1, 0.0, kmax, dk);
  Float zerolag = -12345.0;
  Correlator corr(grid, sep, kmax);
  if (isotropic) {
    corr.correlate_iso(h, kh, zerolag);
  } else {
    fprintf(stdout, "# Using wide-angle exponent %d\n", wide_angle_exponent);
    corr.correlate_aniso(maxell, wide_angle_exponent, g.observer(), h, kh,
                         zerolag);
  }
  Ylm_count.print(stdout);
  fprintf(stdout, "# Anisotropic power spectrum:\n");
  print_hist(kh, stdout, 1, true);
  fprintf(stdout, "# Anisotropic correlations:\n");
  print_hist(h, stdout, 0, isotropic);
  // We want to use the correlation at zero lag as the I normalization
  // factor in the FKP power spectrum.
  fprintf(stdout, "#\n# Zero-lag correlations are %14.7e\n", zerolag);
  // Integral of power spectrum needs a d^3k/(2 pi)^3, which is (1/L)^3 =
  // (1/(cell_size*ngrid))^3
  Float sum_ell0 = 0.0;
  for (int j = 0; j < kh.nbins(); ++j) sum_ell0 += kh.accum().at(0, j);
  fprintf(stdout, "#\n# Integral of power spectrum is %14.7e\n",
          sum_ell0 / (g.cell_size() * g.cell_size() * g.cell_size() *
                       ngrid[0] * ngrid[1] * ngrid[2]));

  Total.Stop();
  uint64 nfft = 1;
  uint64 ngrid3 = dens.size();  // TODO: this should be the padded size
  for (int j = 0; j <= maxell; j += 2) nfft += 2 * (2 * j + 1);
  nfft *= ngrid3;
  fprintf(stdout, "#\n");
  ReportTimes(stdout, nfft, ngrid3, mass_assignor.count());
  return 0;
}
