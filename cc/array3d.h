#ifndef ARRAY3D_H
#define ARRAY3D_H

#include <assert.h>

#include <array>

#include "array/row_major_array.h"
#include "types.h"

// TODO: rename this file or relocate the contents

// TODO: make this directly instantiable as Array or ArrayNd?
template <std::size_t N>
class ArrayBase {
 public:
  // TODO: These can be out of line if subclasses don't explicitly call it.
  ArrayBase(const std::array<int, N> shape)
      : shape_(shape), size_(1), data_(NULL) {
    for (int nx : shape_) size_ *= nx;
    // Allocate data_ array.
    // TODO: we probably don't need aligned memory for non-FFT stuff.
    int err =
        posix_memalign((void **)&data_, PAGE, sizeof(Float) * size_ + PAGE);
    assert(err == 0);
    assert(data_ != NULL);
    set_all(0.0);  // TODO: needed?
  }
  ~ArrayBase() {
    if (data_ != NULL) free(data_);
  }

  const std::array<int, N> &shape() const { return shape_; }
  Float size() { return size_; }
  const Float *data() const { return data_; }  // TODO: remove?

  void set_all(Float value) {
    for (uint64 i = 0; i < size_; ++i) data_[i] = value;
  }

 protected:
  // TODO: private?
  const std::array<int, N> shape_;
  uint64 size_;
  Float *data_;
};

// TODO: not needed? Just use ArrayBase directly and rename it ArrayNd?
class Array1D : public ArrayBase<1> {
 public:
  Array1D(int size) : ArrayBase({size}) {}

  // TODO: this should be called something else, or it should take (start, stop,
  // step) instead.
  // TODO: there's surely a standard libary function for this
  void range(Float start, Float step) {
    for (int i = 0; i < size_; ++i) {
      // TODO: could be slightly more efficient
      data_[i] = start + i * step;
    }
  }

  // TODO: at(), for consistency?
  inline Float &operator[](int idx) { return data_[idx]; }
  inline const Float &operator[](int idx) const { return data_[idx]; }
};

// TODO: not needed? Just use ArrayBase directly and rename it ArrayNd?
class Array2D : public ArrayBase<2> {
 public:
  Array2D(int nx, int ny) : ArrayBase({nx, ny}) {}

  // Indexing.
  inline Float &at(int ix, int iy) { return data_[iy + ix * shape_[1]]; }
};

#endif  // ARRAY3D_H
