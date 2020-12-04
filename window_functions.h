#ifndef WINDOW_FUNCTIONS_H
#define WINDOW_FUNCTIONS_H

#include <array>
#include <memory>

#include "d12.cpp"  // TODO: possibly incorporate into this file
#include "galaxy.h"
#include "types.h"

enum WindowType {
  kNearestCell = 0,
  kCloudInCell = 1,
  kWavelet = 2,
};

class WindowFunction {
 public:
  virtual ~WindowFunction() {}
  virtual int width() = 0;
  virtual void add_galaxy_to_density_field(const Galaxy& g, Float* dens,
                                           const std::array<int, 3>& ngrid,
                                           int ngrid2) = 0;
};

class NearestCellWindow : public WindowFunction {
  int width() override { return 1; }

  void add_galaxy_to_density_field(const Galaxy& g, Float* dens,
                                   const std::array<int, 3>& ngrid,
                                   int ngrid2) override {
    uint64 ix = floor(g.x);
    uint64 iy = floor(g.y);
    uint64 iz = floor(g.z);
    uint64 index = iz + ngrid2 * (iy + ix * ngrid[1]);
    dens[index] += g.w;
  }
};

class CloudInCellWindow : public WindowFunction {
  int width() override { return 3; }

  void add_galaxy_to_density_field(const Galaxy& g, Float* dens,
                                   const std::array<int, 3>& ngrid,
                                   int ngrid2) override {
    // 27-point triangular cloud-in-cell.
    uint64 index;
    uint64 ix = floor(g.x);
    uint64 iy = floor(g.y);
    uint64 iz = floor(g.z);

    Float rx = g.x - ix;
    Float ry = g.y - iy;
    Float rz = g.z - iz;
    //
    Float xm = 0.5 * (1 - rx) * (1 - rx) * g.w;
    Float xp = 0.5 * rx * rx * g.w;
    Float x0 = (0.5 + rx - rx * rx) * g.w;
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
      const uint64 ng0 = ngrid[0];
      const uint64 ng1 = ngrid[1];
      const uint64 ng2 = ngrid[2];
      ix += ngrid[0];  // Just to put away any fears of negative mods
      iy += ngrid[1];
      iz += ngrid[2];
      const uint64 izm = (iz - 1) % ng2;
      const uint64 iz0 = (iz) % ng2;
      const uint64 izp = (iz + 1) % ng2;
      //
      index = ngrid2 * (((iy - 1) % ng1) + ((ix - 1) % ng0) * ng1);
      dens[index + izm] += xm * ym * zm;
      dens[index + iz0] += xm * ym * z0;
      dens[index + izp] += xm * ym * zp;
      index = ngrid2 * (((iy) % ng1) + ((ix - 1) % ng0) * ng1);
      dens[index + izm] += xm * y0 * zm;
      dens[index + iz0] += xm * y0 * z0;
      dens[index + izp] += xm * y0 * zp;
      index = ngrid2 * (((iy + 1) % ng1) + ((ix - 1) % ng0) * ng1);
      dens[index + izm] += xm * yp * zm;
      dens[index + iz0] += xm * yp * z0;
      dens[index + izp] += xm * yp * zp;
      //
      index = ngrid2 * (((iy - 1) % ng1) + ((ix) % ng0) * ng1);
      dens[index + izm] += x0 * ym * zm;
      dens[index + iz0] += x0 * ym * z0;
      dens[index + izp] += x0 * ym * zp;
      index = ngrid2 * (((iy) % ng1) + ((ix) % ng0) * ng1);
      dens[index + izm] += x0 * y0 * zm;
      dens[index + iz0] += x0 * y0 * z0;
      dens[index + izp] += x0 * y0 * zp;
      index = ngrid2 * (((iy + 1) % ng1) + ((ix) % ng0) * ng1);
      dens[index + izm] += x0 * yp * zm;
      dens[index + iz0] += x0 * yp * z0;
      dens[index + izp] += x0 * yp * zp;
      //
      index = ngrid2 * (((iy - 1) % ng1) + ((ix + 1) % ng0) * ng1);
      dens[index + izm] += xp * ym * zm;
      dens[index + iz0] += xp * ym * z0;
      dens[index + izp] += xp * ym * zp;
      index = ngrid2 * (((iy) % ng1) + ((ix + 1) % ng0) * ng1);
      dens[index + izm] += xp * y0 * zm;
      dens[index + iz0] += xp * y0 * z0;
      dens[index + izp] += xp * y0 * zp;
      index = ngrid2 * (((iy + 1) % ng1) + ((ix + 1) % ng0) * ng1);
      dens[index + izm] += xp * yp * zm;
      dens[index + iz0] += xp * yp * z0;
      dens[index + izp] += xp * yp * zp;
    } else {
      // This code is faster, but doesn't do periodic wrapping
      index = (iz - 1) + ngrid2 * ((iy - 1) + (ix - 1) * ngrid[1]);
      dens[index++] += xm * ym * zm;
      dens[index++] += xm * ym * z0;
      dens[index] += xm * ym * zp;
      index += ngrid2 - 2;  // Step to the next row in y
      dens[index++] += xm * y0 * zm;
      dens[index++] += xm * y0 * z0;
      dens[index] += xm * y0 * zp;
      index += ngrid2 - 2;  // Step to the next row in y
      dens[index++] += xm * yp * zm;
      dens[index++] += xm * yp * z0;
      dens[index] += xm * yp * zp;
      index = (iz - 1) + ngrid2 * ((iy - 1) + ix * ngrid[1]);
      dens[index++] += x0 * ym * zm;
      dens[index++] += x0 * ym * z0;
      dens[index] += x0 * ym * zp;
      index += ngrid2 - 2;  // Step to the next row in y
      dens[index++] += x0 * y0 * zm;
      dens[index++] += x0 * y0 * z0;
      dens[index] += x0 * y0 * zp;
      index += ngrid2 - 2;  // Step to the next row in y
      dens[index++] += x0 * yp * zm;
      dens[index++] += x0 * yp * z0;
      dens[index] += x0 * yp * zp;
      index = (iz - 1) + ngrid2 * ((iy - 1) + (ix + 1) * ngrid[1]);
      dens[index++] += xp * ym * zm;
      dens[index++] += xp * ym * z0;
      dens[index] += xp * ym * zp;
      index += ngrid2 - 2;  // Step to the next row in y
      dens[index++] += xp * y0 * zm;
      dens[index++] += xp * y0 * z0;
      dens[index] += xp * y0 * zp;
      index += ngrid2 - 2;  // Step to the next row in y
      dens[index++] += xp * yp * zm;
      dens[index++] += xp * yp * z0;
      dens[index] += xp * yp * zp;
    }
  }
};

class WaveletWindow : public WindowFunction {
  int width() override { return WCELLS; }

  void add_galaxy_to_density_field(const Galaxy& g, Float* dens,
                                   const std::array<int, 3>& ngrid,
                                   int ngrid2) override {
    // We truncate to 1/WAVESAMPLE resolution in each
    // cell and use a lookup table.  Table is set up so that each sub-cell
    // resolution has the values for the various integral cell offsets
    // contiguous in memory.
    uint64 ix = floor(g.x);
    uint64 iy = floor(g.y);
    uint64 iz = floor(g.z);
    uint64 sx = floor((g.x - ix) * WAVESAMPLE);
    uint64 sy = floor((g.y - iy) * WAVESAMPLE);
    uint64 sz = floor((g.z - iz) * WAVESAMPLE);
    const Float* xwave = wave + sx * WCELLS;
    const Float* ywave = wave + sy * WCELLS;
    const Float* zwave = wave + sz * WCELLS;
    // This code does periodic wrapping
    const uint64 ng0 = ngrid[0];
    const uint64 ng1 = ngrid[1];
    const uint64 ng2 = ngrid[2];
    // Offset to the lower-most cell, taking care to handle unsigned int
    ix = (ix + ng0 + WMIN) % ng0;
    iy = (iy + ng1 + WMIN) % ng1;
    iz = (iz + ng2 + WMIN) % ng2;
    Float* px = dens + ngrid2 * ng1 * ix;
    for (int ox = 0; ox < WCELLS; ox++, px += ngrid2 * ng1) {
      if (ix + ox == ng0) px -= ng0 * ng1 * ngrid2;  // Periodic wrap in X
      Float Dx = xwave[ox] * g.w;
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