#ifndef ROW_MAJOR_ARRAY_H
#define ROW_MAJOR_ARRAY_H

#include <array>
#include <utility>

#include "../types.h"
#include "array_nd.h"

// N-dimensional array pointer with row-major (C-contiguous) data layout.
// Partial specializations of this class for small N are defined below.
template <typename dtype, std::size_t N>
class RowMajorArrayPtr : public ArrayNdPtrBase<dtype, N> {
 public:
  RowMajorArrayPtr() : ArrayNdPtrBase<dtype, N>() {}
  RowMajorArrayPtr(std::array<int, N> shape, dtype *data)
      : ArrayNdPtrBase<dtype, N>(shape, data) {}
};

// 1D row-major array pointer.
template <typename dtype>
class RowMajorArrayPtr<dtype, 1> : public ArrayNdPtrBase<dtype, 1> {
 public:
  RowMajorArrayPtr() : ArrayNdPtrBase<dtype, 1>() {}
  RowMajorArrayPtr(std::array<int, 1> shape, dtype *data)
      : ArrayNdPtrBase<dtype, 1>(shape, data) {}
  RowMajorArrayPtr(int size, dtype *data)
      : ArrayNdPtrBase<dtype, 1>({size}, data) {}
  dtype &operator[](uint64 idx) { return this->data_[idx]; }
  const dtype &operator[](uint64 idx) const { return this->data_[idx]; }
  void set_data(int size, dtype *data) { this->set_data({size}, data); }
};

// 2D row-major array pointer.
template <typename dtype>
class RowMajorArrayPtr<dtype, 2> : public ArrayNdPtrBase<dtype, 2> {
 public:
  RowMajorArrayPtr() : ArrayNdPtrBase<dtype, 2>() {}
  RowMajorArrayPtr(std::array<int, 2> shape, dtype *data)
      : ArrayNdPtrBase<dtype, 2>(shape, data) {}
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

// Base class for an N-dimensional array with row-major data layout that
// allocates and owns its own data.
template <typename dtype, std::size_t N>
class RowMajorArrayBase : public RowMajorArrayPtr<dtype, N> {
 public:
  // Default constructor: empty array, no memory allocated. User must call
  // allocate() explicitly.
  RowMajorArrayBase() : RowMajorArrayPtr<dtype, N>() {}
  // This constructor allocates memory.
  RowMajorArrayBase(const std::array<int, N> &shape) : RowMajorArrayBase() {
    allocate(shape);
  }
  ~RowMajorArrayBase() {
    if (this->data_ != NULL) free(this->data_);
  }

  // Disable copy construction and copy assignment: users must copy manually.
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

  // Allocates memory to store array data.
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
};

// N-dimensional array with row-major data layout.
template <typename dtype, std::size_t N>
class RowMajorArray : public RowMajorArrayBase<dtype, N> {
 public:
  RowMajorArray() : RowMajorArrayBase<dtype, N>() {}
  RowMajorArray(const std::array<int, N> &shape)
      : RowMajorArrayBase<dtype, N>(shape) {}
};

// 1D row-major array special case.
template <typename dtype>
class RowMajorArray<dtype, 1> : public RowMajorArrayBase<dtype, 1> {
 public:
  RowMajorArray() : RowMajorArrayBase<dtype, 1>() {}
  RowMajorArray(const std::array<int, 1> &shape)
      : RowMajorArrayBase<dtype, 1>(shape) {}
  RowMajorArray(int size) : RowMajorArrayBase<dtype, 1>({size}) {}
  void allocate(int size) { this->allocate({size}); }
};

template <typename dtype>
using ArrayPtr1D = RowMajorArrayPtr<dtype, 1>;

template <typename dtype>
using Array1D = RowMajorArray<dtype, 1>;

#endif  // ROW_MAJOR_ARRAY_H