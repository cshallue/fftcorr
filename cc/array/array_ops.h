#ifndef ARRAY_OPS_H
#define ARRAY_OPS_H

#include "row_major_array.h"
#include "../types.h"

// TODO: most/all of these ops are in too general a namespace. they're really
// implemented specifically for 3D large grids.
// We could template many of these functions, but really we only need the Float
// type mostly.
namespace array_ops {

template <typename dtype>
void set_all(dtype value, RowMajorArrayPtr<dtype, 3> &arr);

void add_scalar(Float s, RowMajorArray<Float, 3> &arr);
void multiply_by(Float s, RowMajorArray<Float, 3> &arr);
Float sum(const RowMajorArray<Float, 3> &arr);
Float sumsq(const RowMajorArray<Float, 3> &arr);

template <typename dtype>
void copy(const RowMajorArrayPtr<dtype, 3> &in,
          RowMajorArrayPtr<dtype, 3> &out);

void copy_into_padded_array(const RowMajorArrayPtr<Float, 3> &in,
                            RowMajorArrayPtr<Float, 3> &out);

void multiply_with_conjugation(const RowMajorArrayPtr<Complex, 3> &in,
                               RowMajorArrayPtr<Complex, 3> &out);

}  // namespace array_ops

#endif  // ARRAY_OPS_H
