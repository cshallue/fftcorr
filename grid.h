#ifndef GRID_H
#define GRID_H

#include <array>

#include "types.h"

class Grid {
 public:
  // Positions need to arrive in a coordinate system that has the observer at
  // the origin

  Grid(std::array<Float, 3> posmin, Float cell_size)
      : posmin_(posmin), cell_size_(cell_size) {}

  /* ------------------------------------------------------------------- */

  inline void change_to_grid_coords(Float tmp[4]) const {
    // Given tmp[4] = x,y,z,w,
    // Modify to put them in box coordinates.
    // We'll have no use for the original coordinates!
    // tmp[3] (w) is unchanged
    tmp[0] = (tmp[0] - posmin_[0]) / cell_size_;
    tmp[1] = (tmp[1] - posmin_[1]) / cell_size_;
    tmp[2] = (tmp[2] - posmin_[2]) / cell_size_;
  }

  /* ------------------------------------------------------------------- */

  Float cell_size() const { return cell_size_; }
  const std::array<Float, 3>& posmin() const { return posmin_; }

 private:
  // Inputs
  std::array<Float, 3> posmin_;  // Including the border; we don't support
                                 // periodic wrapping in CIC
  Float cell_size_;              // The size of the cubic cells
};

#endif  // GRID_H