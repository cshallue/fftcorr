#ifndef CONFIG_SPACE_GRID_H
#define CONFIG_SPACE_GRID_H

#include <array>

#include "../array/array_ops.h"
#include "../array/row_major_array.h"
#include "../particle_mesh/window_functions.h"
#include "../types.h"

class ConfigSpaceGrid {
 public:
  ConfigSpaceGrid(std::array<int, 3> shape, std::array<Float, 3> posmin,
                  Float cell_size, WindowType window_type)
      : shape_(shape),
        posmin_(posmin),
        cell_size_(cell_size),
        window_type_(window_type),
        grid_(shape_) {
    clear();
  }

  const std::array<int, 3>& shape() const { return shape_; }
  int shape(int i) const { return shape_[i]; }
  uint64 size() const { return grid_.size(); }
  Float posmin(int i) const { return posmin_[i]; }
  Float cell_size() const { return cell_size_; }
  WindowType window_type() const { return window_type_; }
  RowMajorArray<Float, 3>& data() { return grid_; }
  const RowMajorArray<Float, 3>& data() const { return grid_; }

  void clear() { array_ops::set_all(0.0, grid_); }

 private:
  // Number of cells in each dimension.
  const std::array<int, 3> shape_;
  // The origin of the grid coordinate system, expressed in survey coordinates.
  const std::array<Float, 3> posmin_;
  // Size of each grid cell, in survey coordinates.
  const Float cell_size_;
  // Type of window used in mass assignment.
  const WindowType window_type_;

  RowMajorArray<Float, 3> grid_;
};

#endif  // CONFIG_SPACE_GRID_H