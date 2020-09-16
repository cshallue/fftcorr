#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include "types.h"

class Histogram {
  // This should set up the binning and the space to hold the answers
 public:
  int maxell;
  Float sep;
  int nbins;

  Float *cnt;
  Float *hist;
  Float binsize;
  Float zerolag;  // The value at zero lag

  Histogram(int _maxell, Float _sep, Float _dsep) {
    int err;
    maxell = _maxell;
    sep = _sep;
    binsize = _dsep;
    zerolag = -12345.0;
    nbins = floor(sep / binsize);
    fprintf(stderr, "nbins = %d\n", nbins);
    // Too big probably means data entry error.
    assert(nbins > 0 && nbins < 1e6);

    // Allocate cnt[nbins], hist[maxell/2+1, nbins]
    err = posix_memalign((void **)&cnt, PAGE, sizeof(Float) * nbins);
    assert(err == 0);
    err = posix_memalign((void **)&hist, PAGE,
                         sizeof(Float) * nbins * (maxell / 2 + 1));
    assert(err == 0);
    assert(cnt != NULL);
    assert(hist != NULL);
  }
  ~Histogram() {
    // For some reason, these cause a crash!  Weird!
    // free(hist);
    // free(cnt);
  }

  // TODO: Might consider creating more flexible ways to select a binning.
  inline int r2bin(Float r) { return floor(r / binsize); }

  void histcorr(int ell, int n, Float *rnorm, Float *total) {
    // Histogram into bins by rnorm[n], adding up weighting by total[n].
    // Add to multipole ell.
    if (ell == 0) {
      for (int j = 0; j < nbins; j++) cnt[j] = 0.0;
      for (int j = 0; j < nbins; j++) hist[j] = 0.0;
      for (int j = 0; j < n; j++) {
        int b = r2bin(rnorm[j]);
        if (rnorm[j] < binsize * 1e-6) {
          zerolag = total[j];
        }
        if (b >= nbins || b < 0) continue;
        cnt[b]++;
        hist[b] += total[j];
      }
    } else {
      // ell>0
      Float *h = hist + ell / 2 * nbins;
      for (int j = 0; j < nbins; j++) h[j] = 0.0;
      for (int j = 0; j < n; j++) {
        int b = r2bin(rnorm[j]);
        if (b >= nbins || b < 0) continue;
        h[b] += total[j];
      }
    }
  }

  Float sum() {
    // Add up the histogram values for ell=0
    Float total = 0.0;
    for (int j = 0; j < nbins; j++) total += hist[j];
    return total;
  }

  void print(FILE *fp, int norm) {
    // Print out the results
    // If norm==1, divide by counts
    Float denom;
    for (int j = 0; j < nbins; j++) {
      fprintf(fp, "%1d ", norm);
      if (sep > 2)
        fprintf(fp, "%6.2f %8.0f", (j + 0.5) * binsize, cnt[j]);
      else
        fprintf(fp, "%7.4f %8.0f", (j + 0.5) * binsize, cnt[j]);
      if (cnt[j] != 0 && norm)
        denom = cnt[j];
      else
        denom = 1.0;
      for (int k = 0; k <= maxell / 2; k++)
        fprintf(fp, " %16.9e", hist[k * nbins + j] / denom);
      fprintf(fp, "\n");
    }
  }
};  // end Histogram

#endif  // HISTOGRAM_H