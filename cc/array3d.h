#ifndef ARRAY3D_H
#define ARRAY3D_H

#include <assert.h>

#include <array>

#include "types.h"

// TODO: rename this file arrays.h

// TODO: make this multidimensional?
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
  // TODO: just use at() syntax instead of a new function? Fewer chars...
  dtype *get_row(int ix, int iy) { return &at(ix, iy, 0); }
  const dtype *get_row(int ix, int iy) const { return &at(ix, iy, 0); }

 private:
  dtype *data_;
  std::array<int, 3> shape_;
  uint64 size_;
};

// TODO: make this directly instantiable as Array or ArrayNd?
template <std::size_t N>
class ArrayBase {
 public:
  // TODO: These can be out of line if subclasses don't explicitly call it.
  ArrayBase(const std::array<int, N> shape)
      : shape_(shape), size_(1), data_(NULL) {
    for (int nx : shape_) size_ *= nx;
    // Allocate data_ array.
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

  // TODO: at(), for consistency?
  inline Float &operator[](int idx) { return data_[idx]; }
  inline const Float &operator[](int idx) const { return data_[idx]; }
};

// TODO: make this a static function of Array1D?
// Then we can remove the operator[] from Array1D
// TODO: this should be called something else, or it should take (start, stop,
// step) instead.
Array1D range(Float start, Float step, int size);

// TODO: not needed? Just use ArrayBase directly and rename it ArrayNd?
class Array2D : public ArrayBase<2> {
 public:
  Array2D(int nx, int ny) : ArrayBase({nx, ny}) {}

  // Indexing.
  inline Float &at(int ix, int iy) { return data_[iy + ix * shape_[1]]; }
};

// This class allocates and owns a 3D grid of Floats, stored in a flattened
// array in row-major order.
//
// It also provides a set of elementwise primitive operations, which interpret
// the grid as either as an array of real numbers, or an array of complex
// numbers with the last dimension having half the size. It is the
// responsibility of the caller to keep keep track of what kind of data the
// array represents at which time.
//
// It would be nice to separate out the real and complex operations into
// separate classes (which could be backed by the same underlying Array3D).
// Doing this properly would be a little involved because our real-space
// arrays sometimes - but not always - have a different logical shape to the
// data grid due to padding for in-place FFTs. So we would ideally change the
// operations (like sum()) to be operate only the logical sub-grid, and change
// other parts of the code that assume the padded structure. Such a change
// might be worth it because it would abstract away the FFT padding from the
// rest of the code.

// TODO: inherit from base Array class
class Array3D : public ArrayBase<3> {
 public:
  // TODO: add Array3D(nx, ny, nz)?
  Array3D(std::array<int, 3> shape);
  ~Array3D();

  void set_all(Float value);

  // Indexing.
  inline Float &at(int ix, int iy, int iz) {
    return data_[get_index(ix, iy, iz)];
  }
  inline const Float &at(int ix, int iy, int iz) const {
    return data_[get_index(ix, iy, iz)];
  }

  // Operations.
  void add_scalar(Float s);
  void multiply_by(Float s);
  Float sum() const;
  Float sumsq() const;

  const std::array<int, 3> &shape() const { return shape_; }
  int shape(int i) const { return shape_[i]; }
  uint64 size() const { return size_; }

  // TODO: remove.
  RowMajorArray<Float> &arr() { return *arr_; }
  const RowMajorArray<Float> &arr() const { return *arr_; }

  // TODO: private
  inline uint64 get_index(int ix, int iy, int iz) const {
    return arr_->get_index(ix, iy, iz);
  }

 private:
  Float *data() { return data_; }
  const Float *data() const { return data_; }

  // TODO: allocate on stack not heap? Need an initialize() method then.
  RowMajorArray<Float> *arr_;
};

#endif  // ARRAY3D_H
