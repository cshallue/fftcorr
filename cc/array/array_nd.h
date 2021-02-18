#ifndef ARRAY_ND_H
#define ARRAY_ND_H

#include <array>
#include <utility>

#include "../types.h"
#include "array.h"

// Base interface for an n-dimensional array of numbers.
template <typename dtype, std::size_t N>
class ArrayNdBase : public Array<dtype> {
 public:
  virtual ~ArrayNdBase() = default;
  virtual const std::array<int, N> &shape() const = 0;
  virtual int shape(int i) const = 0;
};

// Interface for an n-dimensional array of numbers.
template <typename dtype, std::size_t N>
class ArrayNd : public ArrayNdBase<dtype, N> {
 public:
  virtual ~ArrayNd() = default;
};

// Interface for 2d array.
template <typename dtype>
class ArrayNd<dtype, 2> : public ArrayNdBase<dtype, 2> {
 public:
  virtual ~ArrayNd() = default;
  // Indexing.
  virtual uint64 get_index(int ix, int iy) const = 0;
  virtual dtype &at(int ix, int iy) = 0;
  virtual const dtype &at(int ix, int iy) const = 0;
};

// Interface for 3d array.
template <typename dtype>
class ArrayNd<dtype, 3> : public ArrayNdBase<dtype, 3> {
 public:
  virtual ~ArrayNd() = default;
  // Indexing.
  virtual uint64 get_index(int ix, int iy, int iz) const = 0;
  virtual dtype &at(int ix, int iy, int iz) = 0;
  virtual const dtype &at(int ix, int iy, int iz) const = 0;
};

// Base class for an n-dimensional array pointer. Does not implement a data
// layout.
template <typename dtype, std::size_t N>
class ArrayNdPtrBase : public ArrayNd<dtype, N> {
 public:
  // A default-constructed array pointer is effectively a null pointer until
  // set_data() is called. Operations should not be called before
  // initialization.
  ArrayNdPtrBase() : size_(0), data_(NULL) {}

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

#endif  // ARRAY_ND_H