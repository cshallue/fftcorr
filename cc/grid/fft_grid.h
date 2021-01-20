#ifndef FFT_GRID_H
#define FFT_GRID_H

#include <fftw3.h>

#include <array>

#include "../array3d.h"
#include "../types.h"

class FftGrid {
 public:
  FftGrid(std::array<int, 3> shape);
  ~FftGrid();

  // TODO: allow the user to pass fft flags? I.e. FFT_MEASURE, etc.
  void setup_fft();
  void execute_fft();
  void execute_ifft();

  // TODO: consider which copy ops we need to support.
  // void copy_from(const RowMajorArrayPtr<Float>& other);
  // void copy_from(const FftGrid& other);
  // TODO: this could be private if FftGrid initializes the FFTs in its
  // copy constructor, and it is a friend class. Then it would (a) copy, (b)
  // setup_fft, (c) restore_from
  // void restore_from(const RowMajorArrayPtr<Float>& other);

  // TODO: add shape(int i)?
  const std::array<int, 3>& rshape() const { return rshape_; }
  const std::array<int, 3>& dshape() const { return arr_->shape(); }

  uint64 rsize() const { return rsize_; }
  uint64 dsize() const { return arr_->size(); }

  RowMajorArray<Float>& arr() { return *arr_; }
  const RowMajorArray<Float>& arr() const { return *arr_; }
  RowMajorArrayPtr<Complex>& carr() { return *carr_; }
  const RowMajorArrayPtr<Complex>& carr() const { return *carr_; }

  // TODO: these can be out-of-class operations on two RowMajorArrayPtr<>s?
  // They are quite natural here, because they assume FFT layout.
  void extract_submatrix(RowMajorArrayPtr<Float>* out) const;
  void extract_submatrix(RowMajorArrayPtr<Float>* out,
                         const RowMajorArrayPtr<Float>* mult) const;
  void extract_submatrix_C2R(RowMajorArrayPtr<Float>* out) const;
  void extract_submatrix_C2R(RowMajorArrayPtr<Float>* out,
                             const RowMajorArrayPtr<Float>* mult) const;

 private:
  std::array<int, 3> rshape_;  // Shape as a real array.
  std::array<int, 3> cshape_;  // Shape as a complex array.
  uint64 rsize_;
  uint64 csize_;

  Float* data_;     // TODO: needed? owned by arr_, so this is just an alias
  Complex* cdata_;  // TODO: needed? owned by carr_, so this is just an alias
  // TODO: allocate on stack not heap? Need an initialize() method then.
  RowMajorArray<Float>* arr_;
  RowMajorArrayPtr<Complex>* carr_;  // TODO: needed?

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

#endif  // FFT_GRID_H