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

  // TODO: add shape(int i)?
  const std::array<int, 3>& rshape() const { return rshape_; }
  const std::array<int, 3>& dshape() const { return arr_.shape(); }

  uint64 rsize() const { return rsize_; }
  uint64 dsize() const { return arr_.size(); }

  Array3D& arr() { return arr_; }
  const Array3D& arr() const { return arr_; }

  // Real-space operations.
  // TODO: sum and sumsq are over padded elements too!
  void add_scalar(Float s);
  void multiply_by(Float s);
  Float sum() const;
  Float sumsq() const;

  // Complex-space operations.
  void multiply_with_conjugation(const DiscreteField& other);

  void extract_submatrix(Array3D* out) const;
  void extract_submatrix(Array3D* out, const Array3D* mult) const;
  void extract_submatrix_C2R(Array3D* out) const;
  void extract_submatrix_C2R(Array3D* out, const Array3D* mult) const;

 private:
  std::array<int, 3> rshape_;  // Shape as a real array.
  std::array<int, 3> cshape_;  // Shape as a complex array.
  uint64 rsize_;
  uint64 csize_;

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

  // TODO: we could get rid of this, but we'd need to make SurveyReader unaware
  // of padding. Careful use of arr.at(i,j,k) might be sufficient.
  friend class SurveyReader;
  inline uint64 get_index(int ix, int iy, int iz) const {
    return arr_.get_index(ix, iy, iz);
  }
  Float* data() { return arr_.data_; }
};

#endif  // DISCRETE_FIELD_H