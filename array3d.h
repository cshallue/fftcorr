#ifndef ARRAY3D_H
#define ARRAY3D_H

#include "types.h"

class Array3D {
 public:
  Array3D(int ngrid[3]);
  ~Array3D();

  inline uint64 to_grid_index(uint64 ix, uint64 iy, uint64 iz) {
    return iz + ngrid2_ * (iy + ix * ngrid_[1]);
  }

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

  // TODO: make public, don't call automatically in constructor, and
  // remove from matrix_utils.
  void initialize(Float *&m, const uint64 size, const int nx);

  friend class SurveyReader;  // TODO: remove
};

#endif  // ARRAY3D_H