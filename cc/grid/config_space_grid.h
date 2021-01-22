#ifndef CONFIG_SPACE_GRID_H
#define CONFIG_SPACE_GRID_H

#include <array>

#include "../array/array_ops.h"
#include "../array/row_major_array.h"
#include "../particle_mesh/window_functions.h"
#include "../types.h"

class ConfigSpaceGrid {
 public:
  ConfigSpaceGrid(std::array<int, 3> ngrid, std::array<Float, 3> posmin,
                  Float cell_size, WindowType window_type)
      : ngrid_(ngrid),
        posmin_(posmin),
        cell_size_(cell_size),
        window_type_(window_type),
        grid_(array_ops::allocate_array<Float>(ngrid_), ngrid_) {}

  // TODO: rename ngrid to shape.
  const std::array<int, 3>& ngrid() const { return ngrid_; }
  int ngrid(int i) const { return ngrid_[i]; }
  const std::array<Float, 3>& posmin() const { return posmin_; }
  Float cell_size() const { return cell_size_; }
  WindowType window_type() const { return window_type_; }
  // TODO: needed for MassAssignor and for fftcorr.cpp normalization.
  RowMajorArray<Float>& data() { return grid_; }
  const RowMajorArray<Float>& data() const { return grid_; }

  // TODO: just for testing wrapping; delete.
  void add_scalar(Float s) { array_ops::add_scalar(s, grid_); }
  void multiply_by(Float s) { array_ops::multiply_by(s, grid_); }
  Float sum() const { return array_ops::sum(grid_); }
  Float sumsq() const { return array_ops::sumsq(grid_); }

  // TODO: for wrapping.
  Float* raw_data() { return grid_.data(); }

  inline void change_survey_to_grid_coords(Float& x, Float& y, Float& z) const {
    x = (x - posmin_[0]) / cell_size_;
    y = (y - posmin_[1]) / cell_size_;
    z = (z - posmin_[2]) / cell_size_;
  }

 private:
  // Number of cells in each dimension.
  const std::array<int, 3> ngrid_;
  // The origin of the grid coordinate system, expressed in survey coordinates.
  const std::array<Float, 3> posmin_;
  // Size of each grid cell, in survey coordinates.
  const Float cell_size_;
  // Type of window used in mass assignment.
  WindowType window_type_;

  RowMajorArray<Float> grid_;
};

#endif  // CONFIG_SPACE_GRID_H