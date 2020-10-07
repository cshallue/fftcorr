#ifndef ARRAY3D_H
#define ARRAY3D_H

#include <fftw3.h>

#include "types.h"

// TODO: consider renaming to Field or DiscreteField to reflect the fact that
// it can be in complex / Fourier space.
class Array3D {
 public:
  Array3D(int ngrid[3]);
  ~Array3D();

  // TODO: allow the user to pass fft flags? I.e. FFT_MEASURE, etc.
  void setup_fft();
  void execute_fft();
  void execute_ifft();

  // TODO: maybe call this automatically if copy initialization is always done
  // by a copy constructor. Or else check that it's been initialized.
  void set_value(Float value);
  // TODO: this might just become a copy constructor.
  void copy_from(const Array3D &other);
  void restore_from(const Array3D &other);

  // TODO: make this class indexable? But would need to know whether in
  // Fourier space or not.
  inline uint64 to_grid_index(uint64 ix, uint64 iy, uint64 iz) {
    return iz + ngrid2_ * (iy + ix * ngrid_[1]);
  }

  void add_scalar(Float s);
  void multiply_with_conjugation(const Array3D &other);
  Float sum() const;
  Float sumsq() const;

  // TODO: rename these to something sensible. shape, sizexy, sizez?
  const int *ngrid() const { return ngrid_; }
  int ngrid2() const { return ngrid2_; }
  Float ngrid3() const { return ngrid3_; }
  // TODO: access to these should be internal-only.
  // Or, at the very least, we should have two functions, one for Fourier
  // space that casts to complex.
  const Float *data() const { return data_; };
  Float *raw_data() { return data_; }

 private:
  int ngrid_[3];
  int ngrid2_;             // ngrid_[2] padded out for the FFT work
  uint64 ngrid3_;          // The total number of FFT grid cells
  Float *data_;            // The flattened grid
  bool is_fourier_space_;  // Whether data is in Fourier space

#ifndef FFTSLAB
  fftw_plan fft_;
  fftw_plan ifft_;
#else
  fftw_plan fftX_;
  fftw_plan fftYZ_;
  fftw_plan ifftYZ_;
  fftw_plan ifftX_;
#endif

  friend class SurveyReader;  // TODO: remove
};

#endif  // ARRAY3D_H