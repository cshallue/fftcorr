#ifndef ARRAY_ND_H
#define ARRAY_ND_H

#include <array>
#include <stdexcept>
#include <utility>

#include "../types.h"
#include "array.h"

// Base interface for an N-dimensional array of numbers.
template <typename dtype, std::size_t N>
class ArrayNdBase : public Array<dtype> {
 public:
  virtual ~ArrayNdBase() = default;
  virtual const std::array<int, N> &shape() const = 0;
  virtual int shape(int i) const = 0;
};

// Interface for an N-dimensional array of numbers.
// Partial specializations of this class for small N are defined below.
template <typename dtype, std::size_t N>
class ArrayNd : public ArrayNdBase<dtype, N> {
 public:
  virtual ~ArrayNd() = default;
};

// Interface for 1D array.
template <typename dtype>
class ArrayNd<dtype, 1> : public ArrayNdBase<dtype, 1> {
 public:
  virtual ~ArrayNd() = default;
  virtual dtype &operator[](uint64 idx) = 0;
  virtual const dtype &operator[](uint64 idx) const = 0;
};

// Interface for 2D array.
template <typename dtype>
class ArrayNd<dtype, 2> : public ArrayNdBase<dtype, 2> {
 public:
  virtual ~ArrayNd() = default;
  // Indexing.
  virtual uint64 get_index(int i, int j) const = 0;
  virtual dtype &at(int i, int j) = 0;
  virtual const dtype &at(int i, int j) const = 0;
};

// Interface for 3D array.
template <typename dtype>
class ArrayNd<dtype, 3> : public ArrayNdBase<dtype, 3> {
 public:
  virtual ~ArrayNd() = default;
  // Indexing.
  virtual uint64 get_index(int i, int j, int k) const = 0;
  virtual dtype &at(int i, int j, int k) = 0;
  virtual const dtype &at(int i, int j, int k) const = 0;
};

// Interface for 4D array.
template <typename dtype>
class ArrayNd<dtype, 4> : public ArrayNdBase<dtype, 4> {
 public:
  virtual ~ArrayNd() = default;
  // Indexing.
  virtual uint64 get_index(int i, int j, int k, int l) const = 0;
  virtual dtype &at(int i, int j, int k, int l) = 0;
  virtual const dtype &at(int i, int j, int k, int l) const = 0;
};

// Base class for an N-dimensional array pointer. Does not implement a data
// layout.
template <typename dtype, std::size_t N>
class ArrayNdPtrBase : public ArrayNd<dtype, N> {
 public:
  // The default constructor yields what is effectively a null pointer until
  // set_data() is called. Operations should not be called before set_data().
  ArrayNdPtrBase() : size_(0), data_(NULL) {}

  ArrayNdPtrBase(const std::array<int, N> &shape, dtype *data)
      : ArrayNdPtrBase() {
    set_data(shape, data);
  }

  virtual ~ArrayNdPtrBase() = default;

  void set_data(const std::array<int, N> &shape, dtype *data) {
    set_shape(shape);
    data_ = data;
  }

  const std::array<int, N> &shape() const { return shape_; }
  int shape(int i) const { return shape_[i]; }
  uint64 size() const { return size_; }
  dtype *data() { return data_; }
  const dtype *data() const { return data_; }

 protected:
  void set_shape(const std::array<int, N> &shape) {
    shape_ = shape;
    size_ = 1;
    for (int nx : shape_) {
      if (nx <= 0) throw std::invalid_argument("Shape must be nonnegative.");
      size_ *= nx;
    }
  }

  std::array<int, N> shape_;
  uint64 size_;
  dtype *data_;
};

template <typename dtype, std::size_t N>
inline bool operator==(const ArrayNdPtrBase<dtype, N> &lhs,
                       const ArrayNdPtrBase<dtype, N> &rhs) {
  return lhs.shape() == rhs.shape() && lhs.data() == rhs.data();
}

#endif  // ARRAY_ND_H
