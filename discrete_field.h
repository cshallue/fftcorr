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

  // TODO: there's actually usually only one object we need to be a
  // DiscreteField; the other objects are just plain data.
  void copy_from(const DiscreteField& other);
  // TODO: this could be private if DiscreteField initializes the FFTs in its
  // copy constructor, and it is a friend class. Then it would (a) copy, (b)
  // setup_fft, (c) restore_from
  void restore_from(const DiscreteField& other);

  // TODO: add shape(int i)?
  const std::array<int, 3>& rshape() const { return rshape_; }
  const std::array<int, 3>& dshape() const { return arr_->shape(); }

  uint64 rsize() const { return rsize_; }
  uint64 dsize() const { return arr_->size(); }

  RowMajorArray<Float>& arr() { return *arr_; }
  const RowMajorArray<Float>& arr() const { return *arr_; }

  // Real-space operations.
  // TODO: sum and sumsq are over padded elements too! This actually matters if
  // we normalize the density field, but currently that just affects some
  // printed info, not the correlations or power spectrum.
  void add_scalar(Float s);
  void multiply_by(Float s);
  Float sum() const;
  Float sumsq() const;

  // Complex-space operations.
  void multiply_with_conjugation(const DiscreteField& other);

  void extract_submatrix(RowMajorArray<Float>* out) const;
  void extract_submatrix(RowMajorArray<Float>* out,
                         const RowMajorArray<Float>* mult) const;
  void extract_submatrix_C2R(RowMajorArray<Float>* out) const;
  void extract_submatrix_C2R(RowMajorArray<Float>* out,
                             const RowMajorArray<Float>* mult) const;

 private:
  std::array<int, 3> rshape_;  // Shape as a real array.
  std::array<int, 3> cshape_;  // Shape as a complex array.
  uint64 rsize_;
  uint64 csize_;

  Float* data_;
  Complex* cdata_;  // TODO: needed?
  // TODO: allocate on stack not heap? Need an initialize() method then.
  RowMajorArray<Float>* arr_;
  RowMajorArray<Complex>* carr_;

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