#ifndef ARRAY3D_H
#define ARRAY3D_H

#include <array>

#include "types.h"

// TODO: rename this file arrays.h

// TODO: make this directly instantiable as Array or ArrayNd?
template <std::size_t N>
class ArrayBase {
 public:
  ArrayBase(const std::array<int, N> shape);
  ~ArrayBase() {
    if (data_ != NULL) free(data_);
  }

  const std::array<int, N> &shape() const { return shape_; }
  Float size() { return size_; }
  const Float *data() const { return data_; }  // TODO: remove?

  // TODO: not needed/desired for N>1?
  inline Float &operator[](int idx) { return data_[idx]; }
  inline const Float &operator[](int idx) const { return data_[idx]; }

 protected:
  // TODO: private?
  const std::array<int, N> shape_;
  uint64 size_;
  Float *data_;
};

class Array1D : public ArrayBase<1> {
 public:
  Array1D(int size) : ArrayBase({size}) {}
};

// TODO: make this a static function of Array1D?
Array1D range(Float start, Float step, int size);

// TODO: not needed? Just use ArrayBase directly and rename it ArrayNd?
class Array2D : public ArrayBase<2> {};

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
class Array3D {
 public:
  Array3D();
  ~Array3D();

  // TODO: really it makes more sense to do this in the constructor, which
  // would save us calling this explicitly and mean we wouldn't have to check
  // for initialization in all internal operations, but DiscreteField needs to
  // figure out the size in the body of its constructor.
  void initialize(std::array<int, 3> shape);
  // TODO: this might just become a copy constructor. Then initialize could be
  // in the normal constructor.
  void copy_from(const Array3D &other);
  void copy_with_scalar_multiply(const Array3D &other, Float s);

  void set_all(Float value);

  // Indexing.
  inline Float &at(int ix, int iy, int iz) {
    return data_[get_index(ix, iy, iz)];
  }
  inline const Float &at(int ix, int iy, int iz) const {
    return data_[get_index(ix, iy, iz)];
  }

  // TODO: these are only needed to iterate over the values in an Array3D
  // (actually, a pair of Array3D with corresponding elements) in the
  // histogram. Come up with a better way.
  inline Float &operator[](uint64 idx) { return data_[idx]; }
  inline const Float &operator[](uint64 idx) const { return data_[idx]; }

  // Real-space operations.
  void add_scalar(Float s);
  void multiply_by(Float s);
  Float sum() const;
  Float sumsq() const;

  const std::array<int, 3> &shape() const { return shape_; }
  uint64 size() const { return size_; }

 private:
  inline uint64 get_index(int ix, int iy, int iz) const {
    return (uint64)iz + shape_[2] * (iy + ix * shape_[1]);
  }

  Float *data() { return data_; }
  const Float *data() const { return data_; }

  // Complex-space operations.
  void multiply_with_conjugation(const Array3D &other);
  const Complex *cdata() const { return (Complex *)data_; }

  std::array<int, 3> shape_;
  uint64 size_;
  Float *data_;

  std::array<int, 3> cshape_;
  uint64 csize_;
  Complex *cdata_;

  friend class DiscreteField;
};

#endif  // ARRAY3D_H