#ifndef ROW_MAJOR_ARRAY_H
#define ROW_MAJOR_ARRAY_H

#include <array>

#include "../types.h"

// TODO: make the dimension a template parameter too? need to make sure for
// loops are unrolled?
template <typename dtype>
class RowMajorArray {
 public:
  RowMajorArray(dtype *data, std::array<int, 3> shape)
      : data_(data),
        shape_(shape),
        size_((uint64)shape_[0] * shape_[1] * shape_[2]) {}

  // TODO: needed?
  const std::array<int, 3> &shape() const { return shape_; }
  int shape(int i) const { return shape_[i]; }
  uint64 size() const { return size_; }

  // Indexing.
  inline uint64 get_index(int ix, int iy, int iz) const {
    return (uint64)iz + shape_[2] * (iy + ix * shape_[1]);
  }
  inline dtype &at(int ix, int iy, int iz) {
    return data_[get_index(ix, iy, iz)];
  }
  inline const dtype &at(int ix, int iy, int iz) const {
    return data_[get_index(ix, iy, iz)];
  }
  // TODO: just use at() syntax instead of a new function? Fewer chars,
  // but then again, it returns a pointer and this is a RowMajorArray.
  dtype *get_row(int ix, int iy) { return &at(ix, iy, 0); }
  const dtype *get_row(int ix, int iy) const { return &at(ix, iy, 0); }

 private:
  dtype *data_;
  std::array<int, 3> shape_;
  uint64 size_;
};

#endif  // ROW_MAJOR_ARRAY_H
