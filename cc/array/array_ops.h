#ifndef ARRAY_OPS_H
#define ARRAY_OPS_H

#include "row_major_array.h"

// TODO: most/all of these ops are in too general a namespace. they're really
// implemented specifically for 3D large grids.
// We could template many of these functions, but really we only need the Float
// type mostly.
namespace array_ops {

template <typename dtype>
void set_all(dtype value, RowMajorArrayPtr<dtype> &arr);

void add_scalar(Float s, RowMajorArray<Float> &arr);
void multiply_by(Float s, RowMajorArray<Float> &arr);
Float sum(const RowMajorArray<Float> &arr);
Float sumsq(const RowMajorArray<Float> &arr);

// RowMajorArray<Float> create(const std::array<int, 3> &shape) {
//   RowMajorArray<Float> arr = create_uninitialized(shape);
//   set_all(0.0, arr);
//   return std::move(arr);
// }

template <typename dtype>
inline dtype *allocate_array(const std::array<int, 3> &shape) {
  uint64 size = (uint64)shape[0] * shape[1] * shape[2];
  dtype *data;
  int err = posix_memalign((void **)&data, PAGE, sizeof(dtype) * size + PAGE);
  assert(err == 0);
  assert(data != NULL);
  return data;
}

template <typename dtype>
void copy(const RowMajorArrayPtr<dtype> &in, RowMajorArrayPtr<dtype> &out);

void copy_into_padded_array(const RowMajorArrayPtr<Float> &in,
                            RowMajorArrayPtr<Float> &out);

void multiply_with_conjugation(const RowMajorArrayPtr<Complex> &in,
                               RowMajorArrayPtr<Complex> &out);

}  // namespace array_ops

#endif  // ARRAY_OPS_H
