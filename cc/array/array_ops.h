#ifndef ARRAY_OPS_H
#define ARRAY_OPS_H

#include <stdexcept>

#include "../multithreading.h"
#include "../types.h"
#include "row_major_array.h"

namespace array_ops {

// Ops for 1D Float arrays.
Array1D<Float> sequence(Float start, Float step, int size);

// Ops for 3D Float arrays.
void set_all(Float value, RowMajorArrayPtr<Float, 3> &arr);
void add_scalar(Float s, RowMajorArray<Float, 3> &arr);
void multiply_by(Float s, RowMajorArray<Float, 3> &arr);
Float sum(const RowMajorArray<Float, 3> &arr);
Float sumsq(const RowMajorArray<Float, 3> &arr);
void copy_into_padded_array(const RowMajorArrayPtr<Float, 3> &in,
                            RowMajorArrayPtr<Float, 3> &out);

// Ops for 3D complex arrays.
void multiply_with_conjugation(const RowMajorArrayPtr<Complex, 3> &in,
                               RowMajorArrayPtr<Complex, 3> &out);

// Ops for 3D arrays of any type.
template <typename dtype>
void copy(const RowMajorArrayPtr<dtype, 3> &in,
          RowMajorArrayPtr<dtype, 3> &out) {
  if (in.shape() != out.shape()) {
    throw std::invalid_argument("Incompatible shapes for copy.");
  }
  const dtype *in_data = in.data();
  dtype *out_data = out.data();
#ifdef SLAB
  int nx = in.shape(0);
  uint64 nyz = in.size() / nx;
#pragma omp parallel for MY_SCHEDULE
  for (int x = 0; x < nx; ++x) {
    const dtype *in_slab = in_data + x * nyz;
    dtype *out_slab = out_data + x * nyz;
    for (uint64 i = 0; i < nyz; ++i) {
      out_slab[i] = in_slab[i];
    }
  }
#else
#pragma omp parallel for MY_SCHEDULE
  for (uint64 i = 0; i < in.size(); i++) {
    out_data[i] = in_data[i];
  }
#endif
}

}  // namespace array_ops

#endif  // ARRAY_OPS_H
