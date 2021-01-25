#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include "array/array_ops.h"
#include "array/row_major_array.h"
#include "types.h"

class Histogram2D {
 public:
  Histogram2D(int maxell, Float sep, Float dsep)
      : maxell_(maxell),
        sep_(sep),
        binsize_(dsep),
        nbins_(floor(sep_ / binsize_)),
        cnt_({nbins_}),
        hist_({maxell_ / 2 + 1, nbins_}) {
    for (Float &x : cnt_) x = 0.0;
    for (Float &x : hist_) x = 0.0;
  }

  // TODO: Might consider creating more flexible ways to select a binning.
  inline int r2bin(Float r) { return floor(r / binsize_); }

  void histcorr(int ell, const RowMajorArrayPtr<Float, 3> &rnorm,
                const RowMajorArrayPtr<Float, 3> &total) {
    // Histogram into bins by rnorm[n], adding up weighting by total[n].
    // Add to multipole ell.
    int ih = ell / 2;
    // TODO: really we only care about rnorm and total as flattened arrays.
    // There are a few possibilities to simplify this: have a wrapper class
    // that treats the array as flat (e.g. pass rnorm.flatten() into this
    // function); make Array3D iterable; make Array3D indexable by a row-major
    // index (but this is confusing because it's also indexable by a 3-tuple
    // index).
    for (uint64 i = 0; i < rnorm.size(); ++i) {
      int b = r2bin(rnorm[i]);
      if (b >= nbins_ || b < 0) continue;
      if (ell == 0) cnt_[b]++;
      hist_.at(ih, b) += total[i];
    }
  }

  Float sum(int ell) {
    // Add up the histogram values for ell
    int i = ell / 2;
    Float total = 0.0;
    for (int j = 0; j < nbins_; j++) total += hist_.at(i, j);
    return total;
  }

  void print(FILE *fp, int prefix, bool norm) {
    // Print out the results
    // If norm==1, divide by counts
    Float denom;
    for (int j = 0; j < nbins_; j++) {
      fprintf(fp, "%1d ", prefix);
      if (sep_ > 2)
        fprintf(fp, "%6.2f %8.0f", (j + 0.5) * binsize_, cnt_[j]);
      else
        fprintf(fp, "%7.4f %8.0f", (j + 0.5) * binsize_, cnt_[j]);
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
  Array1D<Float> cnt_;            // (rbin)  TODO: uint64?
  RowMajorArray<Float, 2> hist_;  // (ell, rbin)
};

class Histogram1D : public Histogram2D {
 public:
  Histogram1D(Float sep, Float dsep) : Histogram2D(0, sep, dsep) {}

  void histcorr(const RowMajorArrayPtr<Float, 3> &rnorm,
                const RowMajorArrayPtr<Float, 3> &total) {
    Histogram2D::histcorr(0, rnorm, total);
  }
};

#endif  // HISTOGRAM_H