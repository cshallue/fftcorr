#ifndef MASS_ASSIGNMENT_H
#define MASS_ASSIGNMENT_H

#include <vector>

#include "d12.cpp"
#include "discrete_field.h"
#include "galaxy.h"
#include "merge_sort_omp.cpp"
#include "types.h"

#define GALAXY_BATCH_SIZE 1000000

class MassAssignor {
 public:
  MassAssignor(const Grid &grid, DiscreteField *field)
      : grid_(grid), field_(field) {
    gal_.reserve(GALAXY_BATCH_SIZE);
  }

  void add_galaxy(Float posw[4]) {
    grid_.change_to_grid_coords(posw);
    uint64 index =
        field_->get_index(floor(posw[0]), floor(posw[1]), floor(posw[2]));
    gal_.push_back(Galaxy(posw, index));
    if (gal_.size() >= GALAXY_BATCH_SIZE) {
      // IO.Stop();
      flush_to_density_field();
      // IO.Start();
    }
  }

  void flush_to_density_field() {
    const std::array<int, 3> &ngrid = field_->rshape();

    // Given a set of Galaxies, add them to the grid and then reset the list
    // CIC.Start();
    const int galsize = gal_.size();

#ifdef DEPRICATED
    // This works, but appears to be slower
    for (int j = 0; j < galsize; j++) add_galaxy_to_density_field(grid, gal[j]);
#else
    // If we're parallelizing this, then we need to keep the threads from
    // stepping on each other.  Do this in slabs, but with only every third
    // slab active at any time.

    // Let's sort the particles by x.
    // Need to supply an equal amount of temporary space to merge sort.
    // Do this by another vector.
    std::vector<Galaxy> tmp;
    tmp.reserve(galsize);
    mergesort_parallel_omp(gal_.data(), galsize, tmp.data(),
                           omp_get_max_threads());
    // This just falls back to std::sort if omp_get_max_threads==1

    // Now we need to find the starting point of each slab
    // Galaxies between N and N+1 should be in indices [first[N], first[N+1]).
    // That means that first[N] should be the index of the first galaxy to
    // exceed N.
    int first[ngrid[0] + 1], ptr = 0;
    for (int j = 0; j < galsize; j++)
      while (gal_[j].x > ptr) first[ptr++] = j;
    for (; ptr <= ngrid[0]; ptr++) first[ptr] = galsize;

    // Now, we'll loop, with each thread in charge of slab x.
    // Not bothering with NUMA issues.  a) Most of the time is spent waiting
    // for memory to respond, not actually piping between processors.  b)
    // Adjacent slabs may not be on the same memory bank anyways.  Keep it
    // simple.
    int slabset = 3;
#ifdef WAVELET
    slabset = WCELLS;
#endif
    for (int mod = 0; mod < slabset; mod++) {
#pragma omp parallel for schedule(dynamic, 1)
      for (int x = mod; x < ngrid[0]; x += slabset) {
        // For each slab, insert these particles
        for (int j = first[x]; j < first[x + 1]; j++)
          add_galaxy_to_density_field(gal_[j]);
      }
    }
#endif
    gal_.clear();
    // CIC.Stop();
  }

 private:
  void add_galaxy_to_density_field(Galaxy g) {
    Float *dens = field_->data();
    const std::array<int, 3> &ngrid = field_->rshape();
    int ngrid2 = field_->dshape()[2];

    // Add one particle to the density grid.
    // This does a 27-point triangular cloud-in-cell, unless one invokes
    // NEAREST_CELL.
    uint64 index;  // Trying not to assume that ngrid**3 won't spill 32-bits.
    uint64 ix = floor(g.x);
    uint64 iy = floor(g.y);
    uint64 iz = floor(g.z);

// If we're just doing nearest cell.
#ifdef NEAREST_CELL
    index = (iz) + ngrid2 * ((iy) + (ix)*ngrid[1]);
    dens[index] += g.w;
    return;
#endif

#ifdef WAVELET
    // In the wavelet version, we truncate to 1/WAVESAMPLE resolution in each
    // cell and use a lookup table.  Table is set up so that each sub-cell
    // resolution has the values for the various integral cell offsets
    // contiguous in memory.
    uint64 sx = floor((g.x - ix) * WAVESAMPLE);
    uint64 sy = floor((g.y - iy) * WAVESAMPLE);
    uint64 sz = floor((g.z - iz) * WAVESAMPLE);
    const Float *xwave = wave + sx * WCELLS;
    const Float *ywave = wave + sy * WCELLS;
    const Float *zwave = wave + sz * WCELLS;
    // This code does periodic wrapping
    const uint64 ng0 = ngrid[0];
    const uint64 ng1 = ngrid[1];
    const uint64 ng2 = ngrid[2];
    // Offset to the lower-most cell, taking care to handle unsigned int
    ix = (ix + ng0 + WMIN) % ng0;
    iy = (iy + ng1 + WMIN) % ng1;
    iz = (iz + ng2 + WMIN) % ng2;
    Float *px = dens + ngrid2 * ng1 * ix;
    for (int ox = 0; ox < WCELLS; ox++, px += ngrid2 * ng1) {
      if (ix + ox == ng0) px -= ng0 * ng1 * ngrid2;  // Periodic wrap in X
      Float Dx = xwave[ox] * g.w;
      Float *py = px + iy * ngrid2;
      for (int oy = 0; oy < WCELLS; oy++, py += ngrid2) {
        if (iy + oy == ng1) py -= ng1 * ngrid2;  // Periodic wrap in Y
        Float *pz = py + iz;
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
    return;
#endif

    // Now to Cloud-in-Cell
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

  const Grid &grid_;
  DiscreteField *field_;

  std::vector<Galaxy> gal_;
};

#endif  // MASS_ASSIGNMENT_H