#ifndef CONFIG_SPACE_GRID_H
#define CONFIG_SPACE_GRID_H

#include <array>

#include "../array3d.h"
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
        data_(ngrid_) {}

  // TODO: rename ngrid to shape.
  const std::array<int, 3>& ngrid() const { return ngrid_; }
  int ngrid(int i) const { return ngrid_[i]; }
  const std::array<Float, 3>& posmin() const { return posmin_; }
  Float cell_size() const { return cell_size_; }
  WindowType window_type() const { return window_type_; }
  const Array3D& data() const { return data_; }

  // TODO: just for testing wrapping; delete.
  void add_scalar(Float s) { data_.add_scalar(s); }
  void multiply_by(Float s) { data_.multiply_by(s); }
  Float sum() const { return data_.sum(); }
  Float sumsq() const { return data_.sumsq(); }

  // TODO: needed for MassAssignor and for fftcorr.cpp normalization.
  Array3D& data() { return data_; }

  // TODO: for wrapping.
  Float* raw_data() { return data_.arr().get_row(0, 0); }

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

  // TODO: figure out the minimal array type this class needs. It needs to own
  // and allocate the data, it needs indexing operations, and we need
  // real-space operations add_scalar, multiply_by, sum, sumsq, although these
  // could be out-of-class operations on a RowMajorArray.
  Array3D data_;
};

#endif  // CONFIG_SPACE_GRID_H