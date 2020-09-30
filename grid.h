#ifndef GRID_H
#define GRID_H

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

  ~Grid() {
    if (dens_ != NULL) free(dens_);
    free(zcell_);
    free(ycell_);
    free(xcell_);
  }

  Grid(Float posmin[3], int ngrid[3], Float cell_size, int qperiodic) {
    for (int j = 0; j < 3; j++) {
      posmin_[j] = posmin[j];
      ngrid_[j] = ngrid[j];
      assert(ngrid_[j] > 0 && ngrid_[j] < 1e4);
    }
    cell_size_ = cell_size;
    qperiodic_ = qperiodic;

    // Have to set these to null so that the initialization will work.
    dens_ = NULL;

    // ngrid2_ pads out the array for the in-place FFT.
    // The default 3d FFTW format must have the following:
    ngrid2_ = (ngrid_[2] / 2 + 1) * 2;  // For the in-place FFT
#ifdef FFTSLAB
// That said, the rest of the code should work even if extra space is used.
// Some operations will blindly apply to the pad cells, but that's ok.
// In particular, we might consider having ngrid2_ be evenly divisible by
// the critical alignment stride (32 bytes for AVX, but might be more for cache
// lines) or even by a full PAGE for NUMA memory.  Doing this *will* force a
// more complicated FFT, but at least for the NUMA case this is desired: we want
// to force the 2D FFT to run on its socket, and only have the last 1D FFT
// crossing sockets.  Re-using FFTW plans requires the consistent memory
// alignment.
#define FFT_ALIGN 16
    // This is in units of Floats.  16 doubles is 1024 bits.
    ngrid2_ = FFT_ALIGN * (ngrid2_ / FFT_ALIGN + 1);
#endif
    assert(ngrid2_ % 2 == 0);
    fprintf(stdout, "# Using ngrid2_=%d for FFT r2c padding\n", ngrid2_);
    ngrid3_ = (uint64)ngrid_[0] * ngrid_[1] * ngrid2_;

    // Convert origin to grid units
    if (qperiodic_) {
      // In this case, we'll place the observer centered in the grid, but
      // then displaced far away in the -x direction
      for (int j = 0; j < 3; j++) origin_[j] = ngrid_[j] / 2.0;
      origin_[0] -= ngrid_[0] * 1e6;  // Observer far away!
    } else {
      for (int j = 0; j < 3; j++) origin_[j] = (0.0 - posmin_[j]) / cell_size_;
    }

    // Allocate xcell_, ycell_, zcell_ to [ngrid]
    int err;
    err = posix_memalign((void **)&xcell_, PAGE,
                         sizeof(Float) * ngrid_[0] + PAGE);
    assert(err == 0);
    err = posix_memalign((void **)&ycell_, PAGE,
                         sizeof(Float) * ngrid_[1] + PAGE);
    assert(err == 0);
    err = posix_memalign((void **)&zcell_, PAGE,
                         sizeof(Float) * ngrid_[2] + PAGE);
    assert(err == 0);
    assert(xcell_ != NULL);
    assert(ycell_ != NULL);
    assert(zcell_ != NULL);
    // Now set up the cell centers relative to the origin, in grid units
    for (int j = 0; j < ngrid_[0]; j++) xcell_[j] = 0.5 + j - origin_[0];
    for (int j = 0; j < ngrid_[1]; j++) ycell_[j] = 0.5 + j - origin_[1];
    for (int j = 0; j < ngrid_[2]; j++) zcell_[j] = 0.5 + j - origin_[2];
    // Setup.Stop();

    // Allocate dens_ to [ngrid**2*ngrid2_] and set it to zero
    initialize_matrix(dens_, ngrid3_, ngrid_[0]);
    return;
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
    return (iz) + ngrid2_ * ((iy) + (ix)*ngrid_[1]);
  }

  /* ------------------------------------------------------------------- */

  Float cell_size() { return cell_size_; }
  const int *ngrid() { return ngrid_; }
  int ngrid2() { return ngrid2_; }
  Float ngrid3() { return ngrid3_; }
  const Float *dens() { return dens_; };
  const Float *xcell() { return xcell_; }
  const Float *ycell() { return ycell_; }
  const Float *zcell() { return zcell_; }

 private:
  // Inputs
  int ngrid_[3];     // We might prefer a non-cubic box.  The cells are always
                     // cubic!
  Float posmin_[3];  // Including the border; we don't support periodic wrapping
                     // in CIC
  Float cell_size_;  // The size of the cubic cells

  // TODO: does this class need to store this?
  int qperiodic_;

  Float origin_[3];                 // The location of the origin in grid units.
  Float *xcell_, *ycell_, *zcell_;  // The cell centers, relative to the origin

  // The big grids
  int ngrid2_;     // ngrid_[2] padded out for the FFT work
  uint64 ngrid3_;  // The total number of FFT grid cells
  Float *dens_;    // The density field, in a flattened grid

  friend class CatalogReader;  // TODO: remove this
};

#endif  // GRID_H