#ifndef ROW_MAJOR_ARRAY_H
#define ROW_MAJOR_ARRAY_H

#include <array>
#include <utility>

#include "../types.h"

namespace {

template <std::size_t N>
uint64 compute_size(const std::array<int, N> &shape) {
  uint64 size = 1;
  for (int nx : shape) size *= nx;
  return size;
}

}  // namespace

// TODO: the RowMajorArray/RowMajorArrayPtr split is nice, but it makes the
// naming weird across the rest of the code. Can we make RowMajorArray the one
// used most commonly?
// TODO: a possibly unnecessary optimization would be to wrap the shape as well
// Can be freely copied and moved.
// TODO: virtual destructor?
template <typename dtype, std::size_t N>
class RowMajorArrayPtrBase {
 public:
  // Default constructor.
  // A default-constructed RowMajorArrayPtr is effectively a null pointer until
  // initialize() is called. Operations should not be called before
  // initialization. We need this constructor so that classes can allocate a
  // RowMajorArrayPtr on the stack even if they construct the array in the body
  // of their constructor.
  RowMajorArrayPtrBase() : size_(0), data_(NULL) {}

  RowMajorArrayPtrBase(std::array<int, N> shape, dtype *data)
      : shape_(shape), size_(compute_size(shape_)), data_(data) {}

  virtual ~RowMajorArrayPtrBase() = default;

  void initialize(const std::array<int, N> &shape, dtype *data) {
    assert(data_ == NULL);  // Only allowed to initialize once.
    shape_ = shape;
    size_ = compute_size(shape_);
    data_ = data;
  }

  // TODO: needed?
  const std::array<int, N> &shape() const { return shape_; }
  int shape(int i) const { return shape_[i]; }
  uint64 size() const { return size_; }
  dtype *data() { return data_; }
  const dtype *data() const { return data_; }

  inline dtype &operator[](int idx) { return data_[idx]; }
  inline const dtype &operator[](int idx) const { return data_[idx]; }

  // Iterability.
  dtype *begin() { return data_; }
  dtype *end() { return &data_[size()]; }

 protected:
  std::array<int, N> shape_;
  uint64 size_;
  dtype *data_;
};

template <typename dtype, std::size_t N>
class RowMajorArrayPtr : public RowMajorArrayPtrBase<dtype, N> {
 public:
  RowMajorArrayPtr() : RowMajorArrayPtrBase<dtype, N>() {}
  RowMajorArrayPtr(std::array<int, N> shape, dtype *data)
      : RowMajorArrayPtrBase<dtype, N>(shape, data) {}
};

template <typename dtype>
class RowMajorArrayPtr<dtype, 2> : public RowMajorArrayPtrBase<dtype, 2> {
 public:
  RowMajorArrayPtr() : RowMajorArrayPtrBase<dtype, 2>() {}
  RowMajorArrayPtr(std::array<int, 2> shape, dtype *data)
      : RowMajorArrayPtrBase<dtype, 2>(shape, data) {}

  // Indexing.
  uint64 get_index(int ix, int iy) const {
    return (uint64)iy + ix * this->shape_[1];
  }
  dtype &at(int ix, int iy) { return this->data_[get_index(ix, iy)]; }
  const dtype &at(int ix, int iy) const {
    return this->data_[get_index(ix, iy)];
  }
  // TODO: just use at() syntax instead of a new function? Fewer chars,
  // but then again, it returns a pointer and this is a RowMajorArray.
  // Could rename to row()
  dtype *get_row(int ix) { return &this->at(ix, 0); }
  const dtype *get_row(int ix) const { return &this->at(ix, 0); }
};

template <typename dtype>
class RowMajorArrayPtr<dtype, 3> : public RowMajorArrayPtrBase<dtype, 3> {
 public:
  RowMajorArrayPtr() : RowMajorArrayPtrBase<dtype, 3>() {}
  RowMajorArrayPtr(std::array<int, 3> shape, dtype *data)
      : RowMajorArrayPtrBase<dtype, 3>(shape, data) {}

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
  // Default constructor: empty array.
  RowMajorArray() : RowMajorArrayPtr<dtype, N>() {}

  // Take ownership of allocated memory.
  RowMajorArray(std::array<int, N> shape, dtype *data)
      : RowMajorArrayPtr<dtype, N>(shape, data) {}

  // Allocate memory.
  RowMajorArray(std::array<int, N> shape) : RowMajorArray(shape, NULL) {
    allocate_data();
  }

  void initialize(const std::array<int, N> &shape) {
    assert(this->data_ == NULL);  // Make sure unitialized.
    RowMajorArrayPtr<dtype, N>::initialize(shape, NULL);  // Set shape and size.
    allocate_data();
  }

  ~RowMajorArray() {
    if (this->data_ != NULL) free(this->data_);
  }

  // Disable copy construction and copy assignment. We could implement these if
  // needed, but we'd need to copy the data, and we have specific initialization
  // requirements for different arrays.
  RowMajorArray(const RowMajorArray<dtype, N> &) = delete;
  RowMajorArray &operator=(const RowMajorArray<dtype, N> &) = delete;

  // Move constructor and assignment.
  RowMajorArray(RowMajorArray &&other) : RowMajorArray() {
    std::swap(this->shape_, other.shape_);
    std::swap(this->size_, other.size_);
    std::swap(this->data_, other.data_);
  }
  RowMajorArray &operator=(RowMajorArray &&other) {
    std::swap(this->shape_, other.shape_);
    std::swap(this->size_, other.size_);
    std::swap(this->data_, other.data_);
    return *this;
  }

 private:
  void allocate_data() {
    // Page alignment is only important for the big 3D grids, but we don't have
    // too many smaller arrays, so we just do it for all arrays.
    assert(this->size_ > 0);
    int err = posix_memalign((void **)&this->data_, PAGE,
                             sizeof(dtype) * this->size_ + PAGE);
    assert(err == 0);
    assert(this->data_ != NULL);
  }
};

// TODO: I could think about making the 1D case have a constructor that's just
// an integer (not a list). But that'll involve RowMajorArrayBase. Or could I
// somehow make Array<type> a base class of the higher dimensional arrays, and
// avoid this?
// TODO: Perhaps just 'Array'.
template <typename dtype>
using ArrayPtr1D = RowMajorArrayPtr<dtype, 1>;

template <typename dtype>
using Array1D = RowMajorArray<dtype, 1>;

#endif  // ROW_MAJOR_ARRAY_H