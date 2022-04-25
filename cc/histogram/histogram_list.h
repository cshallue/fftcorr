#ifndef HISTOGRAM_H
#define HISTOGRAM_H

#include <stdexcept>

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
    if (n <= 0) throw std::invalid_argument("n must be positive.");
    if (minval >= maxval) {
      throw std::invalid_argument("maxval must be greater than minval.");
    }
    if (binsize <= 0) throw std::invalid_argument("binsize must be positive.");
    for (int i = 0; i < nbins_; ++i) bins_[i] = minval_ + (i + 0.5) * binsize_;
    reset();
  }

  int nbins() const { return nbins_; }
  const ArrayPtr1D<Float> &bins() const { return bins_; }
  const RowMajorArrayPtr<int, 2> &counts() const { return counts_; }
  const RowMajorArrayPtr<Float, 2> &hist_values() const { return hist_values_; }

  void accumulate(int ih, const Array<Float> &x, const Array<Float> &y) {
    if (ih >= counts_.shape(0)) throw std::out_of_range("ih out of range.");
    int *count = counts_.get_row(ih);
    Float *hist = hist_values_.get_row(ih);
    const Float *xdata = x.data();
    const Float *ydata = y.data();
    for (uint64 j = 0; j < x.size(); ++j) {
      int bin = to_bin_index(xdata[j]);
      if (bin >= nbins_ || bin < 0) continue;
      ++count[bin];
      hist[bin] += ydata[j];
    }
  }

  void reset() {
    int *count = counts_.data();
    Float *hist = hist_values_.data();
    for (uint64 i = 0; i < counts_.size(); ++i) {
      count[i] = 0;
      hist[i] = 0.0;
    }
  }

 private:
  int to_bin_index(Float val) { return floor((val - minval_) / binsize_); }

  Float minval_;
  Float binsize_;

  // TODO: should rename bins_ to bin_midpoints_ or something and also expose or
  // store the endpoints.
  int nbins_;
  Array1D<Float> bins_;  // Bin midpoints
  RowMajorArray<int, 2> counts_;
  RowMajorArray<Float, 2> hist_values_;
};

#endif  // HISTOGRAM_H