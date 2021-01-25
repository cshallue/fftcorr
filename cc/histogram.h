#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include "array/array_ops.h"
#include "array/row_major_array.h"
#include "types.h"

class Histogram {
 public:
  Histogram(int maxell, Float sep, Float dsep)
      : maxell_(maxell),
        sep_(sep),
        binsize_(dsep),
        nbins_(floor(sep_ / binsize_)),
        cnt_({nbins_}),
        hist_({maxell_ / 2 + 1, nbins_}) {
    for (int &x : cnt_) x = 0;
    for (Float &x : hist_) x = 0.0;
  }

  // TODO: Might consider creating more flexible ways to select a binning.
  inline int r2bin(Float r) { return floor(r / binsize_); }

  void histcorr(const RowMajorArrayPtr<Float, 3> &rnorm,
                const RowMajorArrayPtr<Float, 3> &total, int ih) {
    Float *h = hist_.get_row(ih);
    for (uint64 i = 0; i < rnorm.size(); ++i) {
      int b = r2bin(rnorm[i]);
      if (b >= nbins_ || b < 0) continue;
      if (ih == 0) cnt_[b]++;
      h[b] += total[i];
    }
  }

  Float sum(int ih) {
    Float *h = hist_.get_row(ih);
    Float total = 0.0;
    for (int b = 0; b < nbins_; ++b) total += h[b];
    return total;
  }

  void print(FILE *fp, int prefix, bool norm) {
    // Print out the results
    // If norm==1, divide by counts
    Float denom;
    for (int j = 0; j < nbins_; j++) {
      fprintf(fp, "%1d ", prefix);
      if (sep_ > 2)
        fprintf(fp, "%6.2f %8.0f", (j + 0.5) * binsize_, (Float)cnt_[j]);
      else
        fprintf(fp, "%7.4f %8.0f", (j + 0.5) * binsize_, (Float)cnt_[j]);
      if (cnt_[j] != 0 && norm)
        denom = cnt_[j];
      else
        denom = 1.0;
      for (int k = 0; k <= maxell_ / 2; k++)
        fprintf(fp, " %16.9e", hist_.at(k, j) / denom);
      fprintf(fp, "\n");
    }
  }

 private:
  int maxell_;
  Float sep_;
  Float binsize_;

  int nbins_;
  Array1D<int> cnt_;              // (rbin)
  RowMajorArray<Float, 2> hist_;  // (ell, rbin)
};

#endif  // HISTOGRAM_H