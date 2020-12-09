#ifndef GRID_H
#define GRID_H

#include <array>

#include "survey_box.h"
#include "types.h"

// Represents a rectangular grid in physical space. Converts between the
// following coordinate systems:
//  1. survey coordinates s[i] in physical units, relative to the survey origin;
//  2. grid coordinates, g[i] := (s[i] - posmin_[i]) / cell_size;
//  3. observer coordinates. o[i] := g[i] - observer_[i].
class Grid {
 public:
  Grid(std::array<int, 3> ngrid) : ngrid_(ngrid), cell_size_(0.0) {}

  // Sets posmin_, cell_size, observer_ to cover a survey box.
  // If cell_size > 0 is passed in, we just check that it covers the box.
  // Otherwise, we choose the minimum cell size that covers the box.
  Float cover_box(const SurveyBox& box, bool periodic, Float cell_size) {
    posmin_ = box.posmin();

    // Choose the cell size.
    std::array<Float, 3> posrange;
    for (int i = 0; i < 3; ++i) {
      posrange[i] = box.posmax()[i] - box.posmin()[i];
    }
    // Compute the box size required in each direction.
    if (cell_size <= 0) {
      // We've been given 3 ngrid and we have the bounding box.
      // Need to pick the most conservative choice
      cell_size =
          std::max(posrange[0] / ngrid_[0],
                   std::max(posrange[1] / ngrid_[1], posrange[2] / ngrid_[2]));
    }
    assert(cell_size * ngrid_[0] >= posrange[0]);
    assert(cell_size * ngrid_[1] >= posrange[1]);
    assert(cell_size * ngrid_[2] >= posrange[2]);
    fprintf(stdout, "# Adopting cell_size=%f for ngrid=%d, %d, %d\n", cell_size,
            ngrid_[0], ngrid_[1], ngrid_[2]);
    fprintf(stdout, "# Adopted boxsize: %6.1f %6.1f %6.1f\n",
            cell_size * ngrid_[0], cell_size * ngrid_[1],
            cell_size * ngrid_[2]);
    fprintf(stdout, "# Input pos range: %6.1f %6.1f %6.1f\n", posrange[0],
            posrange[1], posrange[2]);
    fprintf(stdout, "# Minimum ngrid=%d, %d, %d\n",
            int(ceil(posrange[0] / cell_size)),
            int(ceil(posrange[1] / cell_size)),
            int(ceil(posrange[2] / cell_size)));

    cell_size_ = cell_size;

    // Set the observer coordinates.
    if (periodic) {
      // Place the observer centered in the grid, but displaced far away in the
      // -x direction
      for (int j = 0; j < 3; j++) {
        observer_[j] = ngrid_[j] / 2.0;
      }
      observer_[0] -= ngrid_[0] * 1e6;  // Observer far away!
    } else {
      for (int j = 0; j < 3; j++) {
        // The origin of the survey coordinates.
        observer_[j] = -posmin_[j] / cell_size;
      }
    }

    return cell_size_;
  }

  /* ------------------------------------------------------------------- */

  inline void change_grid_to_observer_coords(Float* pos) const {
    pos[0] = pos[0] - observer_[0];
    pos[1] = pos[1] - observer_[1];
    pos[2] = pos[2] - observer_[2];
  }

  inline void change_survey_to_grid_coords(Float* pos) const {
    pos[0] = (pos[0] - posmin_[0]) / cell_size_;
    pos[1] = (pos[1] - posmin_[1]) / cell_size_;
    pos[2] = (pos[2] - posmin_[2]) / cell_size_;
  }

  /* ------------------------------------------------------------------- */

  const std::array<int, 3>& ngrid() const { return ngrid_; }
  const std::array<Float, 3>& posmin() const { return posmin_; }
  Float cell_size() const { return cell_size_; }

 private:
  // Dimensions of the grid.
  std::array<int, 3> ngrid_;
  // The origin of the grid coordinate system, expressed in survey coordinates.
  std::array<Float, 3> posmin_;
  // Origin of the observer coordinate system, expressed in grid coordinates.
  std::array<Float, 3> observer_;
  // Scale factor converting grid distances to survey distances.
  Float cell_size_;
};

#endif  // GRID_H