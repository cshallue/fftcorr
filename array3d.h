#ifndef ARRAY3D_H
#define ARRAY3D_H

#include <array>

#include "types.h"

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
// Doing this properly would be a little involved because our real-space arrays
// sometimes - but not always - have a different logical shape to the data grid
// due to padding for in-place FFTs. So we would ideally change the operations
// (like sum()) to be operate only the logical sub-grid, and change other parts
// of the code that assume the padded structure. Such a change might be worth it
// because it would abstract away the FFT padding from the rest of the code.
class Array3D {
 public:
  Array3D();
  ~Array3D();

  void initialize(std::array<int, 3> shape);
  // TODO: this might just become a copy constructor. Then initialize could be
  // in the normal constructor.
  void copy_from(const Array3D &other);

  // Indexing.
  // TODO: might want to only index by flat or 3d index, right now both are
  // supported. Could change arr.at(i,j,k) to arr(i,j,k).
  inline uint64 get_index(int ix, int iy, int iz) const {
    return (uint64)iz + shape_[2] * (iy + ix * shape_[1]);
  }
  inline Float &at(int ix, int iy, int iz) {
    return data_[get_index(ix, iy, iz)];
  }
  inline Float &operator[](uint64 idx) { return data_[idx]; }
  inline const Float &operator[](uint64 idx) const { return data_[idx]; }

  // Real-space operations.
  void add_scalar(Float s);
  Float sum() const;
  Float sumsq() const;

  // Complex-space operations.
  void multiply_with_conjugation(const Array3D &other);

  const std::array<int, 3> &shape() const { return shape_; }
  uint64 size() const { return size_; }
  // TODO: I think I can remove this as long as DiscreteField and SurveyReader
  // are friend classes.
  Float *data() { return data_; }
  const Float *data() const { return data_; }

  const std::array<int, 3> &cshape() const { return cshape_; }
  uint64 csize() const { return csize_; }
  Complex *cdata() { return (Complex *)data_; }  // TODO: remove?

 private:
  void set_all(Float value);

  std::array<int, 3> shape_;
  uint64 size_;
  Float *data_;

  std::array<int, 3> cshape_;
  uint64 csize_;
  Complex *cdata_;
};

#endif  // ARRAY3D_H