#ifndef ARRAY_OPS_H
#define ARRAY_OPS_H

#include "row_major_array.h"

namespace array_ops {

template <typename dtype>
void set_all(dtype value, RowMajorArrayPtr<dtype> &arr);

// RowMajorArray<Float> create(const std::array<int, 3> &shape) {
//   RowMajorArray<Float> arr = create_uninitialized(shape);
//   set_all(0.0, arr);
//   return std::move(arr);
// }

template <typename dtype>
void copy(const RowMajorArrayPtr<dtype> &in, RowMajorArrayPtr<dtype> &out);

void copy_into_padded_array(const RowMajorArrayPtr<Float> &in,
                            RowMajorArrayPtr<Float> &out);

void multiply_with_conjugation(const RowMajorArrayPtr<Complex> &in,
                               RowMajorArrayPtr<Complex> &out);

}  // namespace array_ops

#endif  // ARRAY_OPS_H
