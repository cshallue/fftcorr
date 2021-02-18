#ifndef ROW_MAJOR_ARRAY_H
#define ROW_MAJOR_ARRAY_H

#include <array>
#include <utility>

#include "../types.h"
#include "array.h"

// TODO: move this into a class with ArrayNd interface.
// Base class for an n-dimensional array pointer. Does not implement a data
// layout.
template <typename dtype, std::size_t N>
class ArrayNdPtrBase : public Array<dtype> {
 public:
  // A default-constructed array pointer is effectively a null pointer until
  // set_data() is called. Operations should not be called before
  // initialization.
  ArrayNdPtrBase() : Array<dtype>(), size_(0), data_(NULL) {}

  ArrayNdPtrBase(const std::array<int, N> &shape, dtype *data)
      : ArrayNdPtrBase() {
    set_data(shape, data);
  }

  virtual ~ArrayNdPtrBase() = default;

  void set_data(const std::array<int, N> &shape, dtype *data) {
    assert(data_ == NULL);  // Make sure unitialized.
    assert(data != NULL);   // Can't initialize with NULL data.
    set_shape(shape);
    data_ = data;
  }

  const std::array<int, N> &shape() const { return shape_; }
  int shape(int i) const { return shape_[i]; }
  uint64 size() const { return size_; }
  dtype *data() { return data_; }
  const dtype *data() const { return data_; }

  // TODO: delete? random access raises questions about indexing for nd arrays
  // and the user can already get data* or iterate directly - are these enough?
  dtype &operator[](uint64 idx) { return data_[idx]; }
  const dtype &operator[](uint64 idx) const { return data_[idx]; }

  // Iterator through the flattened array.
  dtype *begin() { return data_; }
  dtype *end() { return &data_[size()]; }

 protected:
  void set_shape(const std::array<int, N> &shape) {
    shape_ = shape;
    size_ = 1;
    for (int nx : shape_) {
      assert(nx > 0);
      size_ *= nx;
    }
  }

  std::array<int, N> shape_;
  uint64 size_;
  dtype *data_;
};

template <typename dtype, std::size_t N>
class RowMajorArrayPtr : public ArrayNdPtrBase<dtype, N> {
 public:
  RowMajorArrayPtr() : ArrayNdPtrBase<dtype, N>() {}
  RowMajorArrayPtr(std::array<int, N> shape, dtype *data)
      : ArrayNdPtrBase<dtype, N>(shape, data) {}
};

template <typename dtype>
class RowMajorArrayPtr<dtype, 1> : public ArrayNdPtrBase<dtype, 1> {
 public:
  RowMajorArrayPtr() : ArrayNdPtrBase<dtype, 1>() {}
  RowMajorArrayPtr(std::array<int, 1> shape, dtype *data)
      : ArrayNdPtrBase<dtype, 1>(shape, data) {}
  // In 1D, it's more natural to use integers than single-element tuples.
  RowMajorArrayPtr(int size, dtype *data)
      : ArrayNdPtrBase<dtype, 1>({size}, data) {}
  void set_data(int size, dtype *data) { this->set_data({size}, data); }
};

template <typename dtype>
class RowMajorArrayPtr<dtype, 2> : public ArrayNdPtrBase<dtype, 2> {
 public:
  RowMajorArrayPtr() : ArrayNdPtrBase<dtype, 2>() {}
  RowMajorArrayPtr(std::array<int, 2> shape, dtype *data)
      : ArrayNdPtrBase<dtype, 2>(shape, data) {}

  // Indexing.
  uint64 get_index(int ix, int iy) const {
    return (uint64)iy + ix * this->shape_[1];
  }
  dtype &at(int ix, int iy) { return this->data_[get_index(ix, iy)]; }
  const dtype &at(int ix, int iy) const {
    return this->data_[get_index(ix, iy)];
  }
  dtype *get_row(int ix) { return &this->at(ix, 0); }
  const dtype *get_row(int ix) const { return &this->at(ix, 0); }
};

template <typename dtype>
class RowMajorArrayPtr<dtype, 3> : public ArrayNdPtrBase<dtype, 3> {
 public:
  RowMajorArrayPtr() : ArrayNdPtrBase<dtype, 3>() {}
  RowMajorArrayPtr(std::array<int, 3> shape, dtype *data)
      : ArrayNdPtrBase<dtype, 3>(shape, data) {}

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
  dtype *get_row(int ix, int iy) { return &this->at(ix, iy, 0); }
  const dtype *get_row(int ix, int iy) const { return &this->at(ix, iy, 0); }
};

template <typename dtype, std::size_t N>
class RowMajorArrayBase : public RowMajorArrayPtr<dtype, N> {
 public:
  // Default constructor: empty array.
  RowMajorArrayBase() : RowMajorArrayPtr<dtype, N>() {}

  // Allocates memory.
  RowMajorArrayBase(const std::array<int, N> &shape) : RowMajorArrayBase() {
    allocate(shape);
  }

  ~RowMajorArrayBase() {
    if (this->data_ != NULL) free(this->data_);
  }

  // Allocates memory.
  void allocate(const std::array<int, N> &shape) {
    assert(this->data_ == NULL);  // Make sure unitialized.
    this->set_shape(shape);
    // Page alignment is only important for our big 3D grids, but we just do it
    // for all arrays.
    int err = posix_memalign((void **)&this->data_, PAGE,
                             sizeof(dtype) * this->size_ + PAGE);
    assert(err == 0);
    assert(this->data_ != NULL);
  }

  // Disable copy construction and copy assignment: users must copy explicitly.
  RowMajorArrayBase(const RowMajorArrayBase<dtype, N> &) = delete;
  RowMajorArrayBase &operator=(const RowMajorArrayBase<dtype, N> &) = delete;

  // Move constructor and assignment.
  RowMajorArrayBase &operator=(RowMajorArrayBase<dtype, N> &&other) {
    std::swap(this->shape_, other.shape_);
    std::swap(this->size_, other.size_);
    std::swap(this->data_, other.data_);
    return *this;
  }
  RowMajorArrayBase(RowMajorArrayBase<dtype, N> &&other) : RowMajorArrayBase() {
    *this = std::move(other);
  }
};

template <typename dtype, std::size_t N>
class RowMajorArray : public RowMajorArrayBase<dtype, N> {
 public:
  RowMajorArray() : RowMajorArrayBase<dtype, N>() {}
  RowMajorArray(const std::array<int, N> &shape)
      : RowMajorArrayBase<dtype, N>(shape) {}
};

template <typename dtype>
class RowMajorArray<dtype, 1> : public RowMajorArrayBase<dtype, 1> {
 public:
  RowMajorArray() : RowMajorArrayBase<dtype, 1>() {}
  RowMajorArray(const std::array<int, 1> &shape)
      : RowMajorArrayBase<dtype, 1>(shape) {}
  // In 1D, it's more natural to use integers than single-element tuples.
  RowMajorArray(int size) : RowMajorArrayBase<dtype, 1>({size}) {}
  void allocate(int size) { this->allocate({size}); }
};

template <typename dtype>
using ArrayPtr1D = RowMajorArrayPtr<dtype, 1>;

template <typename dtype>
using Array1D = RowMajorArray<dtype, 1>;

#endif  // ROW_MAJOR_ARRAY_H