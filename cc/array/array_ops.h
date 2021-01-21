#ifndef ARRAY_OPS_H
#define ARRAY_OPS_H

#include "row_major_array.h"

// TODO: most/all of these ops are in too general a namespace. they're really
// implemented specifically for 3D large grids.
namespace array_ops {

template <typename dtype>
void set_all(dtype value, RowMajorArrayPtr<dtype> &arr);

// RowMajorArray<Float> create(const std::array<int, 3> &shape) {
//   RowMajorArray<Float> arr = create_uninitialized(shape);
//   set_all(0.0, arr);
//   return std::move(arr);
// }

template <typename dtype>
inline dtype *allocate_array(uint64 size) {
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
