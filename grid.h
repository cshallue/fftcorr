#ifndef GRID_H
#define GRID_H

#include "array3d.h"
#include "d12.cpp"
#include "galaxy.h"
#include "histogram.h"
#include "matrix_utils.h"
#include "merge_sort_omp.cpp"
#include "spherical_harmonics.h"
#include "types.h"

class Grid {
 public:
  // Positions need to arrive in a coordinate system that has the observer at
  // the origin

  Grid(const Float posmin[3], int ngrid[3], Float cell_size) : dens_(ngrid) {
    for (int j = 0; j < 3; j++) {
      posmin_[j] = posmin[j];
    }
    cell_size_ = cell_size;
  }

  /* ------------------------------------------------------------------- */

  inline uint64 change_to_grid_coords(Float tmp[4]) {
    // Given tmp[4] = x,y,z,w,
    // Modify to put them in box coordinates.
    // We'll have no use for the original coordinates!
    // tmp[3] (w) is unchanged
    tmp[0] = (tmp[0] - posmin_[0]) / cell_size_;
    tmp[1] = (tmp[1] - posmin_[1]) / cell_size_;
    tmp[2] = (tmp[2] - posmin_[2]) / cell_size_;
    uint64 ix = floor(tmp[0]);
    uint64 iy = floor(tmp[1]);
    uint64 iz = floor(tmp[2]);
    return dens_.to_grid_index(ix, iy, iz);
  }

  /* ------------------------------------------------------------------- */

  Float cell_size() { return cell_size_; }
  const Float *posmin() { return posmin_; }
  Array3D &dens() { return dens_; };  // TODO: make const

 private:
  // Inputs
  Float posmin_[3];  // Including the border; we don't support periodic wrapping
                     // in CIC
  Float cell_size_;  // The size of the cubic cells

  Array3D dens_;

  friend class SurveyReader;  // TODO: remove this
};

#endif  // GRID_H