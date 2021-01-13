#ifndef CONFIG_SPACE_GRID_H
#define CONFIG_SPACE_GRID_H

#include <array>

#include "../array3d.h"
#include "../types.h"

class ConfigSpaceGrid {
 public:
  ConfigSpaceGrid(std::array<int, 3> ngrid, std::array<Float, 3> posmin,
                  Float cell_size)
      : ngrid_(ngrid), posmin_(posmin), cell_size_(cell_size), data_(ngrid_) {}

  const std::array<int, 3>& ngrid() const { return ngrid_; }
  const std::array<Float, 3>& posmin() const { return posmin_; }
  Float cell_size() const { return cell_size_; }
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

  inline void change_survey_to_grid_coords(Float* pos) const {
    pos[0] = (pos[0] - posmin_[0]) / cell_size_;
    pos[1] = (pos[1] - posmin_[1]) / cell_size_;
    pos[2] = (pos[2] - posmin_[2]) / cell_size_;
  }

 private:
  // Number of cells in each dimension.
  std::array<int, 3> ngrid_;
  // The origin of the grid coordinate system, expressed in survey coordinates.
  std::array<Float, 3> posmin_;
  // Size of each grid cell, in survey coordinates.
  Float cell_size_;

  // TODO: figure out the minimal array type this class needs. It needs to own
  // and allocate the data, it needs indexing operations, and we need
  // real-space operations add_scalar, multiply_by, sum, sumsq, although these
  // could be out-of-class operations on a RowMajorArray.
  Array3D data_;
};

#endif  // CONFIG_SPACE_GRID_H