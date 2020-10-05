#ifndef ARRAY3D_H
#define ARRAY3D_H

#include <fftw3.h>

#include "types.h"

class Array3D {
 public:
  Array3D(int ngrid[3]);
  ~Array3D();

  void setup_fft();
  void execute_fft();
  void execute_ifft();

  // TODO: maybe call this automatically if copy initialization is always done
  // by a copy constructor. Or else check that it's been initialized.
  void set_value(Float value);
  // TODO: accept another Array3D.
  // TODO: this might just become a copy constructor.
  void copy_from(const Float *other);

  inline uint64 to_grid_index(uint64 ix, uint64 iy, uint64 iz) {
    return iz + ngrid2_ * (iy + ix * ngrid_[1]);
  }

  void add_scalar(Float s);
  Float sum() const;
  Float sumsq() const;

  const int *ngrid() const { return ngrid_; }
  int ngrid2() const { return ngrid2_; }
  Float ngrid3() const { return ngrid3_; }
  const Float *data() const { return data_; };
  Float *raw_data() { return data_; }  // TODO: come up with a better solution

 private:
  int ngrid_[3];
  int ngrid2_;     // ngrid_[2] padded out for the FFT work
  uint64 ngrid3_;  // The total number of FFT grid cells
  Float *data_;    // The flattened grid

  // TODO: define these conditionally on compiler flags.
  fftw_plan fft_;
  fftw_plan fftYZ_;
  fftw_plan fftX_;
  fftw_plan ifft_;
  fftw_plan ifftYZ_;
  fftw_plan ifftX_;

  friend class SurveyReader;  // TODO: remove
};

#endif  // ARRAY3D_H