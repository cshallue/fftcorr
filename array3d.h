#ifndef ARRAY3D_H
#define ARRAY3D_H

#include "matrix_utils.h"
#include "types.h"

class Array3D {
 public:
  // Positions need to arrive in a coordinate system that has the observer at
  // the origin

  ~Array3D() {
    if (data_ != NULL) free(data_);
  }

  Array3D(int ngrid[3]) {
    for (int j = 0; j < 3; j++) {
      ngrid_[j] = ngrid[j];
      assert(ngrid_[j] > 0 && ngrid_[j] < 1e4);
    }

    // Setup ngrid2_.
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
#else
    // ngrid2_ pads out the array for the in-place FFT.
    // The default 3d FFTW format must have the following:
    ngrid2_ = (ngrid_[2] / 2 + 1) * 2;  // For the in-place FFT
#endif
    assert(ngrid2_ % 2 == 0);
    fprintf(stdout, "# Using ngrid2_=%d for FFT r2c padding\n", ngrid2_);

    ngrid3_ = (uint64)ngrid_[0] * ngrid_[1] * ngrid2_;

    // Allocate dens_ to [ngrid3_] and set it to zero
    // TODO: do we always want to initialize on construction?
    data_ = NULL;
    initialize_matrix(data_, ngrid3_, ngrid_[0]);
  }

  /* ------------------------------------------------------------------- */

  inline uint64 to_grid_index(uint64 ix, uint64 iy, uint64 iz) {
    return iz + ngrid2_ * (iy + ix * ngrid_[1]);
  }

  /* ------------------------------------------------------------------- */

  const int *ngrid() const { return ngrid_; }
  int ngrid2() const { return ngrid2_; }
  Float ngrid3() const { return ngrid3_; }
  const Float *data() const { return data_; };
  Float *raw_data() { return data_; }  // TODO: come up with a better solution

 private:
  int ngrid_[3];
  int ngrid2_;     // ngrid_[2] padded out for the FFT work
  uint64 ngrid3_;  // The total number of FFT grid cells
  Float *data_;    // The flattened grid

  friend class SurveyReader;  // TODO: remove
};

#endif  // ARRAY3D_H