#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <assert.h>

#include "../array/array.h"
#include "../array/row_major_array.h"
#include "../types.h"

class HistogramList {
 public:
  HistogramList(int n, Float minval, Float maxval, Float binsize)
      : minval_(minval),
        binsize_(binsize),
        nbins_(floor((maxval - minval) / binsize)),
        bins_(nbins_),
        counts_({n, nbins_}),
        hist_values_({n, nbins_}) {
    assert(n > 0);
    assert(maxval > minval);
    assert(binsize > 0);
    for (int i = 0; i < nbins_; ++i) bins_[i] = (i + 0.5) * binsize_;
    for (int &c : counts_) c = 0;
    for (Float &x : hist_values_) x = 0.0;
  }

  int nbins() const { return nbins_; }
  const Array1D<Float> &bins() const { return bins_; }
  const RowMajorArray<int, 2> &counts() const { return counts_; }
  const RowMajorArray<Float, 2> &hist_values() const { return hist_values_; }

  int to_bin_index(Float val) { return floor((val - minval_) / binsize_); }

  void accumulate(int ih, const Array<Float> &x, const Array<Float> &y) {
    assert(ih < counts_.shape(0));
    int *count = counts_.get_row(ih);
    Float *hist = hist_values_.get_row(ih);
    for (uint64 j = 0; j < x.size(); ++j) {
      int bin = to_bin_index(x[j]);
      if (bin >= nbins_ || bin < 0) continue;
      ++count[bin];
      hist[bin] += y[j];
    }
  }

 private:
  Float minval_;
  Float binsize_;

  int nbins_;
  Array1D<Float> bins_;  // Bin midpoints
  RowMajorArray<int, 2> counts_;
  RowMajorArray<Float, 2> hist_values_;
};

#endif  // HISTOGRAM_H