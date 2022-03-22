#include "array_ops.h"

#include "row_major_array.h"

namespace array_ops {

void set_all(Float value, RowMajorArrayPtr<Float, 3> &arr) {
#ifdef SLAB
  int nx = arr.shape(0);
  uint64 nyz = arr.size() / nx;
#pragma omp parallel for MY_SCHEDULE
  for (int ix = 0; ix < nx; ++ix) {
    Float *slab = arr.get_row(ix);
    for (uint64 iyz = 0; iyz < nyz; ++iyz) {
      slab[iyz] = value;
    }
  }
#else
  Float *data = arr.data();
#pragma omp parallel for MY_SCHEDULE
  for (uint64 i = 0; i < arr.size(); i++) {
    data[i] = value;
  }
#endif
}

void copy_into_padded_array(const RowMajorArrayPtr<Float, 3> &in,
                            RowMajorArrayPtr<Float, 3> &out) {
  bool valid_shape =
      (in.shape(0) == out.shape(0) && in.shape(1) == out.shape(1) &&
       in.shape(2) <= out.shape(2));
  if (!valid_shape) {
    throw std::invalid_argument("Incompatible shapes for copy.");
  }
#pragma omp parallel for MY_SCHEDULE
  for (int i = 0; i < in.shape(0); ++i) {
    for (int j = 0; j < in.shape(1); ++j) {
      const Float *in_row = in.get_row(i, j);
      Float *out_row = out.get_row(i, j);
      for (int k = 0; k < in.shape(2); ++k) {
        out_row[k] = in_row[k];
      }
    }
  }
}

void add_scalar(Float s, RowMajorArray<Float, 3> &arr) {
#ifdef SLAB
  int nx = arr.shape(0);
  uint64 nyz = arr.size() / nx;
#pragma omp parallel for MY_SCHEDULE
  for (int ix = 0; ix < nx; ++ix) {
    Float *slab = arr.get_row(ix);
    for (uint64 iyz = 0; iyz < nyz; ++iyz) {
      slab[i] += s;
    }
  }
#else
  Float *data = arr.data();
#pragma omp parallel for MY_SCHEDULE
  for (uint64 i = 0; i < arr.size(); ++i) {
    data[i] += s;
  }
#endif
}

void multiply_by(Float s, RowMajorArray<Float, 3> &arr) {
#ifdef SLAB
  int nx = arr.shape(0);
  uint64 nyz = arr.size() / nx;
#pragma omp parallel for MY_SCHEDULE
  for (int ix = 0; ix < nx; ++ix) {
    Float *slab = arr.get_row(ix);
    for (uint64 iyz = 0; iyz < nyz; ++iyz) {
      slab[i] *= s;
    }
  }
#else
  Float *data = arr.data();
#pragma omp parallel for MY_SCHEDULE
  for (uint64 i = 0; i < arr.size(); ++i) {
    data[i] *= s;
  }
#endif
}

Float sum(const RowMajorArray<Float, 3> &arr) {
  Float tot = 0.0;
#ifdef SLAB
  int nx = arr.shape(0);
  uint64 nyz = arr.size() / nx;
#pragma omp parallel for MY_SCHEDULE
  for (int ix = 0; ix < nx; ++ix) {
    Float *slab = arr.get_row(ix);
    for (uint64 iyz = 0; iyz < nyz; ++iyz) {
      tot += slab[i];
    }
  }
#else
  const Float *data = arr.data();
#pragma omp parallel for MY_SCHEDULE reduction(+ : tot)
  for (uint64 i = 0; i < arr.size(); ++i) {
    tot += data[i];
  }
#endif
  return tot;
}

Float sumsq(const RowMajorArray<Float, 3> &arr) {
  Float tot = 0.0;
#ifdef SLAB
  int nx = arr.shape(0);
  uint64 nyz = arr.size() / nx;
#pragma omp parallel for MY_SCHEDULE
  for (int ix = 0; ix < nx; ++ix) {
    Float *slab = arr.get_row(ix);
    for (uint64 iyz = 0; iyz < nyz; ++iyz) {
      tot += slab[i];
    }
  }
#else
  const Float *data = arr.data();
#pragma omp parallel for MY_SCHEDULE reduction(+ : tot)
  for (uint64 i = 0; i < arr.size(); ++i) {
    tot += data[i] * data[i];
  }
#endif
  return tot;
}

void multiply_with_conjugation(const RowMajorArrayPtr<Complex, 3> &in,
                               RowMajorArrayPtr<Complex, 3> &out) {
  if (in.shape() != out.shape()) {
    throw std::invalid_argument("Incompatible shapes for multiply.");
  }
#ifdef SLAB
  int nx = in.shape(0);
  uint64 nyz = in.size() / nx;
#pragma omp parallel for MY_SCHEDULE
  for (int ix = 0; ix < nx; ++ix) {
    const Complex *in_slab = in.get_row(ix);
    Complex *out_slab = out.get_row(ix);
    for (uint64 iyz = 0; iyz < nyz; ++iyz) {
      out_slab[iyz] *= std::conj(in_slab[iyz]);
    }
  }
#else
  const Complex *in_data = in.data();
  Complex *out_data = out.data();
#pragma omp parallel for MY_SCHEDULE
  for (uint64 i = 0; i < in.size(); i++) {
    out_data[i] *= std::conj(in_data[i]);
  }
#endif
}

}  // namespace array_ops
