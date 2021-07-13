#ifndef WINDOW_FUNCTIONS_H
#define WINDOW_FUNCTIONS_H

#include <array>
#include <memory>

#include "../array/row_major_array.h"
#include "../types.h"
#include "d12.h"  // TODO: possibly incorporate into this file

enum WindowType {
  kNearestCell = 0,
  kCloudInCell = 1,
  kWavelet = 2,
};

class WindowFunction {
 public:
  virtual ~WindowFunction() {}
  virtual int width() = 0;
  virtual void add_particle_to_grid(const Float* pos, Float weight,
                                    RowMajorArrayPtr<Float, 3>& dens) = 0;
};

class NearestCellWindow : public WindowFunction {
  int width() override { return 1; }

  void add_particle_to_grid(const Float* pos, Float weight,
                            RowMajorArrayPtr<Float, 3>& dens) override {
    dens.at(floor(pos[0]), floor(pos[1]), floor(pos[2])) += weight;
  }
};

class CloudInCellWindow : public WindowFunction {
  int width() override { return 3; }

  void add_particle_to_grid(const Float* pos, Float weight,
                            RowMajorArrayPtr<Float, 3>& dens) override {
    // This implementation can correctly handle a padded data layout, i.e. the
    // memory layout of dens is a row-major (C-contiguous) array with dimensions
    // [nx1, ny1, nz1], whereas the logical shape of the density field grid is
    // [nx2, ny2, nz2], where nx1 == nx2, ny1 == ny2 and nz1 >= nz2. Currently,
    // we're assuming that nz1 == nz2, but the more general case can be handled
    // simply by making ngrid2 an additional parameter to this function.
    const std::array<int, 3>& ngrid = dens.shape();  // [nx1, ny1, nz1]
    const int ngrid2 = ngrid[2];                     // nz2

    // TODO: when I have tests covering this window function, try to use
    // indexing functions.
    Float* d = dens.get_row(0, 0);
    // 27-point triangular cloud-in-cell.
    uint64 index;
    int ix = floor(pos[0]);
    int iy = floor(pos[1]);
    int iz = floor(pos[2]);

    Float rx = pos[0] - ix;
    Float ry = pos[1] - iy;
    Float rz = pos[2] - iz;
    //
    Float xm = 0.5 * (1 - rx) * (1 - rx) * weight;
    Float xp = 0.5 * rx * rx * weight;
    Float x0 = (0.5 + rx - rx * rx) * weight;
    Float ym = 0.5 * (1 - ry) * (1 - ry);
    Float yp = 0.5 * ry * ry;
    Float y0 = 0.5 + ry - ry * ry;
    Float zm = 0.5 * (1 - rz) * (1 - rz);
    Float zp = 0.5 * rz * rz;
    Float z0 = 0.5 + rz - rz * rz;
    //
    if (ix == 0 || ix == ngrid[0] - 1 || iy == 0 || iy == ngrid[1] - 1 ||
        iz == 0 || iz == ngrid[2] - 1) {
      // This code does periodic wrapping
      const int ng0 = ngrid[0];
      const int ng1 = ngrid[1];
      const int ng2 = ngrid[2];
      ix += ngrid[0];  // Just to put away any fears of negative mods
      iy += ngrid[1];
      iz += ngrid[2];
      const int izm = (iz - 1) % ng2;
      const int iz0 = (iz) % ng2;
      const int izp = (iz + 1) % ng2;
      //
      index = ngrid2 * (((iy - 1) % ng1) + ((ix - 1) % ng0) * ng1);
      d[index + izm] += xm * ym * zm;
      d[index + iz0] += xm * ym * z0;
      d[index + izp] += xm * ym * zp;
      index = ngrid2 * (((iy) % ng1) + ((ix - 1) % ng0) * ng1);
      d[index + izm] += xm * y0 * zm;
      d[index + iz0] += xm * y0 * z0;
      d[index + izp] += xm * y0 * zp;
      index = ngrid2 * (((iy + 1) % ng1) + ((ix - 1) % ng0) * ng1);
      d[index + izm] += xm * yp * zm;
      d[index + iz0] += xm * yp * z0;
      d[index + izp] += xm * yp * zp;
      //
      index = ngrid2 * (((iy - 1) % ng1) + ((ix) % ng0) * ng1);
      d[index + izm] += x0 * ym * zm;
      d[index + iz0] += x0 * ym * z0;
      d[index + izp] += x0 * ym * zp;
      index = ngrid2 * (((iy) % ng1) + ((ix) % ng0) * ng1);
      d[index + izm] += x0 * y0 * zm;
      d[index + iz0] += x0 * y0 * z0;
      d[index + izp] += x0 * y0 * zp;
      index = ngrid2 * (((iy + 1) % ng1) + ((ix) % ng0) * ng1);
      d[index + izm] += x0 * yp * zm;
      d[index + iz0] += x0 * yp * z0;
      d[index + izp] += x0 * yp * zp;
      //
      index = ngrid2 * (((iy - 1) % ng1) + ((ix + 1) % ng0) * ng1);
      d[index + izm] += xp * ym * zm;
      d[index + iz0] += xp * ym * z0;
      d[index + izp] += xp * ym * zp;
      index = ngrid2 * (((iy) % ng1) + ((ix + 1) % ng0) * ng1);
      d[index + izm] += xp * y0 * zm;
      d[index + iz0] += xp * y0 * z0;
      d[index + izp] += xp * y0 * zp;
      index = ngrid2 * (((iy + 1) % ng1) + ((ix + 1) % ng0) * ng1);
      d[index + izm] += xp * yp * zm;
      d[index + iz0] += xp * yp * z0;
      d[index + izp] += xp * yp * zp;
    } else {
      // This code is faster, but doesn't do periodic wrapping
      index = (iz - 1) + ngrid2 * ((iy - 1) + (ix - 1) * ngrid[1]);
      d[index++] += xm * ym * zm;
      d[index++] += xm * ym * z0;
      d[index] += xm * ym * zp;
      index += ngrid2 - 2;  // Step to the next row in y
      d[index++] += xm * y0 * zm;
      d[index++] += xm * y0 * z0;
      d[index] += xm * y0 * zp;
      index += ngrid2 - 2;  // Step to the next row in y
      d[index++] += xm * yp * zm;
      d[index++] += xm * yp * z0;
      d[index] += xm * yp * zp;
      index = (iz - 1) + ngrid2 * ((iy - 1) + ix * ngrid[1]);
      d[index++] += x0 * ym * zm;
      d[index++] += x0 * ym * z0;
      d[index] += x0 * ym * zp;
      index += ngrid2 - 2;  // Step to the next row in y
      d[index++] += x0 * y0 * zm;
      d[index++] += x0 * y0 * z0;
      d[index] += x0 * y0 * zp;
      index += ngrid2 - 2;  // Step to the next row in y
      d[index++] += x0 * yp * zm;
      d[index++] += x0 * yp * z0;
      d[index] += x0 * yp * zp;
      index = (iz - 1) + ngrid2 * ((iy - 1) + (ix + 1) * ngrid[1]);
      d[index++] += xp * ym * zm;
      d[index++] += xp * ym * z0;
      d[index] += xp * ym * zp;
      index += ngrid2 - 2;  // Step to the next row in y
      d[index++] += xp * y0 * zm;
      d[index++] += xp * y0 * z0;
      d[index] += xp * y0 * zp;
      index += ngrid2 - 2;  // Step to the next row in y
      d[index++] += xp * yp * zm;
      d[index++] += xp * yp * z0;
      d[index] += xp * yp * zp;
    }
  }
};

class WaveletWindow : public WindowFunction {
  int width() override { return WCELLS; }

  void add_particle_to_grid(const Float* pos, Float weight,
                            RowMajorArrayPtr<Float, 3>& dens) override {
    // This implementation can correctly handle a padded data layout, i.e. the
    // memory layout of dens is a row-major (C-contiguous) array with dimensions
    // [nx1, ny1, nz1], whereas the logical shape of the density field grid is
    // [nx2, ny2, nz2], where nx1 == nx2, ny1 == ny2 and nz1 >= nz2. Currently,
    // we're assuming that nz1 == nz2, but the more general case can be handled
    // simply by making ngrid2 an additional parameter to this function.
    const std::array<int, 3>& ngrid = dens.shape();  // [nx1, ny1, nz1]
    const int ngrid2 = ngrid[2];                     // nz2

    // We truncate to 1/WAVESAMPLE resolution in each
    // cell and use a lookup table.  Table is set up so that each sub-cell
    // resolution has the values for the various integral cell offsets
    // contiguous in memory.
    int ix = floor(pos[0]);
    int iy = floor(pos[1]);
    int iz = floor(pos[2]);
    int sx = floor((pos[0] - ix) * WAVESAMPLE);
    int sy = floor((pos[1] - iy) * WAVESAMPLE);
    int sz = floor((pos[2] - iz) * WAVESAMPLE);
    const Float* xwave = wave + sx * WCELLS;
    const Float* ywave = wave + sy * WCELLS;
    const Float* zwave = wave + sz * WCELLS;
    // This code does periodic wrapping
    const int ng0 = ngrid[0];
    const int ng1 = ngrid[1];
    const int ng2 = ngrid[2];
    // Offset to the lower-most cell, taking care to handle unsigned int
    ix = (ix + ng0 + WMIN) % ng0;
    iy = (iy + ng1 + WMIN) % ng1;
    iz = (iz + ng2 + WMIN) % ng2;
    Float* px = dens.get_row(ix, 0);
    for (int ox = 0; ox < WCELLS; ox++, px += ngrid2 * ng1) {
      if (ix + ox == ng0) px -= ng0 * ng1 * ngrid2;  // Periodic wrap in X
      Float Dx = xwave[ox] * weight;
      Float* py = px + iy * ngrid2;
      for (int oy = 0; oy < WCELLS; oy++, py += ngrid2) {
        if (iy + oy == ng1) py -= ng1 * ngrid2;  // Periodic wrap in Y
        Float* pz = py + iz;
        Float Dxy = Dx * ywave[oy];
        if (iz + WCELLS > ng2) {  // Z Wrap is needed
          for (int oz = 0; oz < WCELLS; oz++) {
            if (iz + oz == ng2) pz -= ng2;  // Periodic wrap in Z
            pz[oz] += zwave[oz] * Dxy;
          }
        } else {
          for (int oz = 0; oz < WCELLS; oz++) pz[oz] += zwave[oz] * Dxy;
        }
      }
    }
  }
};

// TODO: this can go at the top of the file if the implementations are in a cc
// file.
std::unique_ptr<WindowFunction> make_window_function(WindowType type) {
  WindowFunction* func = NULL;
  switch (type) {
    case kNearestCell:
      func = new NearestCellWindow();
      break;
    case kCloudInCell:
      func = new CloudInCellWindow();
      break;
    case kWavelet:
      func = new WaveletWindow();
      break;
  }
  return std::unique_ptr<WindowFunction>(func);
}

#endif  // WINDOW_FUNCTIONS_H