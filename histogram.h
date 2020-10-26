#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include "types.h"

class Histogram {
  // This should set up the binning and the space to hold the answers
 public:
  Histogram(int maxell, Float sep, Float dsep) {
    int err;
    maxell_ = maxell;
    sep_ = sep;
    binsize_ = dsep;
    zerolag_ = -12345.0;
    nbins_ = floor(sep_ / binsize_);
    fprintf(stderr, "nbins_ = %d\n", nbins_);
    // Too big probably means data entry error.
    assert(nbins_ > 0 && nbins_ < 1e6);

    // Allocate cnt_[nbins_], hist_[maxell/2+1, nbins_]
    err = posix_memalign((void **)&cnt_, PAGE, sizeof(Float) * nbins_);
    assert(err == 0);
    err = posix_memalign((void **)&hist_, PAGE,
                         sizeof(Float) * nbins_ * (maxell_ / 2 + 1));
    assert(err == 0);
    assert(cnt_ != NULL);
    assert(hist_ != NULL);
  }
  ~Histogram() {
    // For some reason, these cause a crash!  Weird!
    // free(hist_);
    // free(cnt_);
  }

  // TODO: Might consider creating more flexible ways to select a binning.
  inline int r2bin(Float r) { return floor(r / binsize_); }

  void histcorr(int ell, const Array3D &rnorm, Array3D *total) {
    // Histogram into bins by rnorm[n], adding up weighting by total[n].
    // Add to multipole ell.
    if (ell == 0) {
      for (int j = 0; j < nbins_; j++) cnt_[j] = 0.0;
      for (int j = 0; j < nbins_; j++) hist_[j] = 0.0;
      for (int j = 0; j < rnorm.size(); j++) {
        int b = r2bin(rnorm[j]);
        if (rnorm[j] < binsize_ * 1e-6) {
          zerolag_ = (*total)[j];
        }
        if (b >= nbins_ || b < 0) continue;
        cnt_[b]++;
        hist_[b] += (*total)[j];
      }
    } else {
      // ell>0
      Float *h = hist_ + ell / 2 * nbins_;
      for (int j = 0; j < nbins_; j++) h[j] = 0.0;
      for (int j = 0; j < rnorm.size(); j++) {
        int b = r2bin(rnorm[j]);
        if (b >= nbins_ || b < 0) continue;
        h[b] += (*total)[j];
      }
    }
  }

  Float sum() {
    // Add up the histogram values for ell=0
    Float total = 0.0;
    for (int j = 0; j < nbins_; j++) total += hist_[j];
    return total;
  }

  void print(FILE *fp, int norm) {
    // Print out the results
    // If norm==1, divide by counts
    Float denom;
    for (int j = 0; j < nbins_; j++) {
      fprintf(fp, "%1d ", norm);
      if (sep_ > 2)
        fprintf(fp, "%6.2f %8.0f", (j + 0.5) * binsize_, cnt_[j]);
      else
        fprintf(fp, "%7.4f %8.0f", (j + 0.5) * binsize_, cnt_[j]);
      if (cnt_[j] != 0 && norm)
        denom = cnt_[j];
      else
        denom = 1.0;
      for (int k = 0; k <= maxell_ / 2; k++)
        fprintf(fp, " %16.9e", hist_[k * nbins_ + j] / denom);
      fprintf(fp, "\n");
    }
  }

  Float zerolag() { return zerolag_; }

 private:
  int maxell_;
  Float sep_;
  int nbins_;

  Float *cnt_;
  Float *hist_;
  Float binsize_;
  Float zerolag_;  // The value at zero lag
};                 // end Histogram

#endif  // HISTOGRAM_H