#ifndef ROW_MAJOR_ARRAY_H
#define ROW_MAJOR_ARRAY_H

#include <array>
#include <utility>

#include "../types.h"

// TODO: the RowMajorArray/RowMajorArrayPtr split is nice, but it makes the
// naming weird across the rest of the code. Can we make RowMajorArray the one
// used most commonly?
// TODO: a possibly unnecessary optimization would be to wrap the shape as well
// Can be freely copied and moved.
// TODO: virtual destructor?
template <typename dtype, std::size_t N>
class RowMajorArrayPtrBase {
 public:
  RowMajorArrayPtrBase(dtype *data, std::array<int, N> shape)
      : data_(data), shape_(shape), size_(1) {
    for (int nx : shape_) size_ *= nx;
  }

  // Default constructor.
  // A default-constructed RowMajorArrayPtr is effectively a null pointer until
  // initialize() is called. Operations should not be called before
  // initialization. We need this constructor so that classes can allocate a
  // RowMajorArrayPtr on the stack even if they construct the array in the body
  // of their constructor.
  RowMajorArrayPtrBase() : data_(NULL), size_(0) {}

  virtual ~RowMajorArrayPtrBase() = default;

  void initialize(dtype *data, const std::array<int, N> &shape) {
    assert(data_ == NULL);  // Only allowed to initialize once.
    data_ = data;
    assert(data_ != NULL);  // Can't initialize to a nullptr.
    shape_ = shape;
    size_ = 1;
    for (int nx : shape_) size_ *= nx;
  }

  // TODO: needed?
  const std::array<int, N> &shape() const { return shape_; }
  int shape(int i) const { return shape_[i]; }
  uint64 size() const { return size_; }
  dtype *data() { return data_; }
  const dtype *data() const { return data_; }

 protected:
  dtype *data_;
  std::array<int, N> shape_;
  uint64 size_;
};

template <typename dtype, std::size_t N>
class RowMajorArrayPtr : public RowMajorArrayPtrBase<dtype, N> {
 public:
  RowMajorArrayPtr() : RowMajorArrayPtrBase<dtype, N>() {}
  RowMajorArrayPtr(dtype *data, std::array<int, N> shape)
      : RowMajorArrayPtrBase<dtype, N>(data, shape) {}
};

template <typename dtype>
class RowMajorArrayPtr<dtype, 3> : public RowMajorArrayPtrBase<dtype, 3> {
 public:
  RowMajorArrayPtr() : RowMajorArrayPtrBase<dtype, 3>() {}
  RowMajorArrayPtr(dtype *data, std::array<int, 3> shape)
      : RowMajorArrayPtrBase<dtype, 3>(data, shape) {}

  // Indexing.
  uint64 get_index(int ix, int iy, int iz) const {
    return (uint64)iz + this->shape_[2] * (iy + ix * this->shape_[1]);
  }
  dtype &at(int ix, int iy, int iz) {
    return this->data_[get_index(ix, iy, iz)];
  }
  const dtype &at(int ix, int iy, int iz) const {
    return this->data_[get_index(ix, iy, iz)];
  }
  // TODO: just use at() syntax instead of a new function? Fewer chars,
  // but then again, it returns a pointer and this is a RowMajorArray.
  // Could rename to row()
  dtype *get_row(int ix, int iy) { return &this->at(ix, iy, 0); }
  const dtype *get_row(int ix, int iy) const { return &this->at(ix, iy, 0); }
};

template <typename dtype, std::size_t N>
class RowMajorArray : public RowMajorArrayPtr<dtype, N> {
 public:
  RowMajorArray(dtype *data, std::array<int, N> shape)
      : RowMajorArrayPtr<dtype, N>(data, shape) {}

  RowMajorArray() : RowMajorArrayPtr<dtype, N>() {}

  ~RowMajorArray() {
    if (this->data_ != NULL) free(this->data_);
  }

  // Disable copy and move operations. We could implement these if needed, but
  // we'd need to copy the data / transfer ownership.
  RowMajorArray(const RowMajorArray<dtype, N> &) = delete;
  RowMajorArray &operator=(const RowMajorArray<dtype, N> &) = delete;
};

#endif  // ROW_MAJOR_ARRAY_H