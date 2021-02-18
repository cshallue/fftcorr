#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include "../array/array.h"
#include "../array/row_major_array.h"
#include "../types.h"

class Histogram {
 public:
  Histogram(int n, Float minval, Float maxval, Float binsize)
      : minval_(minval),
        binsize_(binsize),
        nbins_(floor((maxval - minval) / binsize)),
        bins_({nbins_}),
        count_({nbins_}),
        accum_({n, nbins_}) {
    for (int i = 0; i < nbins_; ++i) bins_[i] = (i + 0.5) * binsize_;
    for (int &x : count_) x = 0;
    for (Float &x : accum_) x = 0.0;
  }

  int nbins() const { return nbins_; }
  const Array1D<Float> &bins() const { return bins_; }
  const Array1D<int> &count() const { return count_; }
  const RowMajorArray<Float, 2> &accum() const { return accum_; }

  int to_bin_index(Float val) { return floor((val - minval_) / binsize_); }

  void accumulate(const Array<Float> &rnorm, const Array<Float> &total,
                  int ih) {
    Float *h = accum_.get_row(ih);
    for (uint64 i = 0; i < rnorm.size(); ++i) {
      int b = to_bin_index(rnorm[i]);
      if (b >= nbins_ || b < 0) continue;
      if (ih == 0) ++count_[b];
      h[b] += total[i];
    }
  }

 private:
  Float minval_;
  Float binsize_;

  int nbins_;
  Array1D<Float> bins_;            // Bin midpoints
  Array1D<int> count_;             // (rbin)
  RowMajorArray<Float, 2> accum_;  // (ell, rbin)
};

#endif  // HISTOGRAM_H