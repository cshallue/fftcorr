#ifndef ROW_MAJOR_ARRAY_H
#define ROW_MAJOR_ARRAY_H

#include <array>
#include <utility>

#include "../types.h"

// TODO: make the dimension a template parameter too? need to make sure for
// loops are unrolled?
// TODO: the RowMajorArray/RowMajorArrayPtr split is nice, but it makes the
// naming weird across the rest of the code. Can we make RowMajorArray the one
// used most commonly?
// TODO: a possibly unnecessary optimization would be to wrap the shape as well
template <typename dtype>
class RowMajorArrayPtr {
 public:
  RowMajorArrayPtr(dtype *data, std::array<int, 3> shape)
      : data_(data),
        shape_(shape),
        size_((uint64)shape[0] * shape[1] * shape[2]) {
    // TODO: check data_ not null? Depends on whether subclass accepts data or
    // allocates it itself.
  }

  // TODO: needed?
  const std::array<int, 3> &shape() const { return shape_; }
  int shape(int i) const { return shape_[i]; }
  uint64 size() const { return size_; }
  dtype *data() { return data_; }
  const dtype *data() const { return data_; }

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

 protected:
  dtype *data_;
  std::array<int, 3> shape_;
  uint64 size_;
};

template <typename dtype>
class RowMajorArray : public RowMajorArrayPtr<dtype> {
 public:
  RowMajorArray(std::array<int, 3> shape)
      : RowMajorArrayPtr<dtype>(NULL, shape) {
    // TODO: I think this class should probably accept an already allocated
    // pointer, since it needn't insist on a particular memory alignment that's
    // specific to FFTs. Confirm this, and then do the posix_memalign just in
    // FftGrid.
    int err = posix_memalign((void **)&this->data_, PAGE,
                             sizeof(dtype) * this->size_ + PAGE);
    assert(err == 0);
    assert(this->data_ != NULL);
  }

  ~RowMajorArray() { free(this->data_); }

  // Disable default copy constructors because this causes lifetime issues.
  // This also disables move operations. We could implement copy and move
  // operations if needed.
  RowMajorArray(const RowMajorArray<dtype> &) = delete;
  RowMajorArray &operator=(const RowMajorArray &) = delete;
};

#endif  // ROW_MAJOR_ARRAY_H