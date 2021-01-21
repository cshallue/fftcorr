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
// Can be freely copied and moved.
template <typename dtype>
class RowMajorArrayPtr {
 public:
  RowMajorArrayPtr(dtype *data, std::array<int, 3> shape)
      : data_(data),
        shape_(shape),
        size_((uint64)shape[0] * shape[1] * shape[2]) {}

  // Default constructor. This object is effectively a null pointer until
  // initialize() is called. Like a null pointer, operations should not be
  // called before initialization or bad things will happen. We need this
  // constructor so that classes can allocate a RowMajorArrayPtr on the stack
  // even if they construct the array in the body of their constructor.
  RowMajorArrayPtr() : data_(NULL) {}

  void initialize(dtype *data, const std::array<int, 3> &shape) {
    assert(data_ == NULL);  // Only allowed to initialize once.
    data_ = data;
    assert(data_ != NULL);  // Can't initialize to a nullptr.
    shape_ = shape;
    size_ = (uint64)shape[0] * shape[1] * shape[2];
  }

  // TODO: needed?
  const std::array<int, 3> &shape() const { return shape_; }
  int shape(int i) const { return shape_[i]; }
  uint64 size() const { return size_; }
  dtype *data() { return data_; }
  const dtype *data() const { return data_; }

  // Indexing.
  uint64 get_index(int ix, int iy, int iz) const {
    return (uint64)iz + shape_[2] * (iy + ix * shape_[1]);
  }
  dtype &at(int ix, int iy, int iz) { return data_[get_index(ix, iy, iz)]; }
  const dtype &at(int ix, int iy, int iz) const {
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
  RowMajorArray(dtype *data, std::array<int, 3> shape)
      : RowMajorArrayPtr<dtype>(data, shape) {}

  RowMajorArray() : RowMajorArrayPtr<dtype>() {}

  ~RowMajorArray() {
    if (this->data_ != NULL) free(this->data_);
  }

  // Disable copy and move operations. We could implement these if needed, but
  // we'd need to copy the data / transfer ownership.
  RowMajorArray(const RowMajorArray<dtype> &) = delete;
  RowMajorArray &operator=(const RowMajorArray &) = delete;
};

#endif  // ROW_MAJOR_ARRAY_H