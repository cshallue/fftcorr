#ifndef DISCRETE_FIELD_H
#define DISCRETE_FIELD_H

#include <fftw3.h>

#include <array>

#include "array3d.h"
#include "types.h"

class DiscreteField {
 public:
  DiscreteField(std::array<int, 3> shape);
  ~DiscreteField();

  // TODO: allow the user to pass fft flags? I.e. FFT_MEASURE, etc.
  void setup_fft();
  void execute_fft();
  void execute_ifft();

  // TODO: this should be a copy constructor. Then initialize could be
  // in the normal constructor.
  void copy_from(const DiscreteField& other);
  // TODO: this could be private if DiscreteField initializes the FFTs in its
  // copy constructor, and it is a friend class. Then it would (a) copy, (b)
  // setup_fft, (c) restore_from
  void restore_from(const DiscreteField& other);

  const std::array<int, 3>& rshape() const { return rshape_; }
  const std::array<int, 3>& cshape() const { return cshape_; }
  const std::array<int, 3>& dshape() const { return arr_.shape(); }

  uint64 rsize() const { return rsize_; }
  uint64 csize() const { return csize_; }
  uint64 dsize() const { return arr_.size(); }

  inline uint64 get_index(int ix, int iy, int iz) const {
    return arr_.get_index(ix, iy, iz);
  }

  // TODO: make private and accessible to friend classes only?
  Array3D& arr() { return arr_; }
  const Array3D& arr() const { return arr_; }
  Float* data() { return arr_.data(); }
  const Float* data() const { return arr_.data(); }
  Complex* cdata() { return arr_.cdata(); }

  // Real-space operations.
  // TODO: sum and sumsq are over padded elements too!
  void add_scalar(Float s);
  Float sum() const;
  Float sumsq() const;

  // Complex-space operations.
  void multiply_with_conjugation(const DiscreteField& other);

  // TODO: rename arguments something more general.
  void extract_submatrix(const Array3D& corr, Array3D* total) const;
  void extract_submatrix_C2R(const Array3D& corr, Array3D* total) const;

 private:
  std::array<int, 3> rshape_;  // Shape as a real array.
  std::array<int, 3> cshape_;  // Shape as a complex array.
  uint64 rsize_;
  uint64 csize_;

  // TODO: it would be nice to keep track of this, but currently we just
  // overwrite data from one space with data from another, so we'd need to take
  // that into account.
  bool is_fourier_space_;  // Whether data is in Fourier space

  Array3D arr_;

#ifndef FFTSLAB
  fftw_plan fft_;
  fftw_plan ifft_;
#else
  fftw_plan fftX_;
  fftw_plan fftYZ_;
  fftw_plan ifftYZ_;
  fftw_plan ifftX_;
#endif
};

#endif  // DISCRETE_FIELD_H