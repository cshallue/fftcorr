#include "array_ops.h"

#include <assert.h>

#include "../multithreading.h"
#include "row_major_array.h"

namespace array_ops {

template <typename dtype>
void set_all(dtype value, RowMajorArrayPtr<dtype, 3> &arr) {
  dtype *data = arr.data();
#ifdef SLAB
  int nx = arr.shape(0);
  uint64 nyz = arr.size() / nx;
#pragma omp parallel for MY_SCHEDULE
  for (int x = 0; x < nx; ++x) {
    dtype *slab = data + x * nyz;
    for (uint64 i = 0; i < nyz; ++i) {
      slab[i] = value;
    }
  }
#else
#pragma omp parallel for MY_SCHEDULE
  for (uint64 i = 0; i < arr.size(); i++) {
    data[i] = value;
  }
#endif
}
// Since we're using this template function in separate compilation units, we
// can't put its definition in the header or it'll be defined in both
// compilation units, causing a linker error. The other two options are to (1)
// mark it inline in the header, or (2) let the compiler know which definitions
// to create. Or the third option is to just use one compilation unit for the
// entire project.
// https://stackoverflow.com/questions/115703/storing-c-template-function-definitions-in-a-cpp-file
template void set_all(Float, RowMajorArrayPtr<Float, 3> &);
template void set_all(Complex, RowMajorArrayPtr<Complex, 3> &);

template <typename dtype>
void copy(const RowMajorArrayPtr<dtype, 3> &in,
          RowMajorArrayPtr<dtype, 3> &out) {
  assert(in.shape(0) == out.shape(0));
  assert(in.shape(1) == out.shape(1));
  assert(in.shape(2) == out.shape(2));
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
template void copy(const RowMajorArrayPtr<Float, 3> &,
                   RowMajorArrayPtr<Float, 3> &);
template void copy(const RowMajorArrayPtr<Complex, 3> &,
                   RowMajorArrayPtr<Complex, 3> &);

void copy_into_padded_array(const RowMajorArrayPtr<Float, 3> &in,
                            RowMajorArrayPtr<Float, 3> &out) {
  assert(in.shape(0) == out.shape(0));
  assert(in.shape(1) == out.shape(1));
  assert(in.shape(2) <= out.shape(2));
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
  Float *data = arr.data();
#ifdef SLAB
  int nx = arr.shape(0);
  const uint64 nyz = arr.size() / nx;
#pragma omp parallel for MY_SCHEDULE
  for (int x = 0; x < nx; ++x) {
    // TODO: I could use arr_->get_row() here.
    Float *slab = data + x * nyz;
    for (uint64 i = 0; i < nyz; ++i) {
      slab[i] += s;
    }
  }
#else
#pragma omp parallel for MY_SCHEDULE
  for (uint64 i = 0; i < arr.size(); ++i) {
    data[i] += s;
  }
#endif
}

void multiply_by(Float s, RowMajorArray<Float, 3> &arr) {
  Float *data = arr.data();
#ifdef SLAB
  int nx = arr.shape(0);
  const uint64 nyz = arr.size() / nx;
#pragma omp parallel for MY_SCHEDULE
  for (int x = 0; x < nx; ++x) {
    Float *slab = data + x * nyz;
    for (uint64 i = 0; i < nyz; ++i) {
      slab[i] *= s;
    }
  }
#else
#pragma omp parallel for MY_SCHEDULE
  for (uint64 i = 0; i < arr.size(); ++i) {
    data[i] *= s;
  }
#endif
}

Float sum(const RowMajorArray<Float, 3> &arr) {
  const Float *data = arr.data();
  Float tot = 0.0;
#ifdef SLAB
  int nx = arr.shape(0);
  const uint64 nyz = arr.size() / nx;
#pragma omp parallel for MY_SCHEDULE reduction(+ : tot)
  for (int x = 0; x < nx; ++x) {
    const Float *slab = data + x * nyz;
    for (uint64 i = 0; i < nyz; ++i) {
      tot += slab[i];
    }
  }
#else
#pragma omp parallel for MY_SCHEDULE reduction(+ : tot)
  for (uint64 i = 0; i < arr.size(); ++i) {
    tot += data[i];
  }
#endif
  return tot;
}

// TODO: come up with a way to template these parallelizable ops
Float sumsq(const RowMajorArray<Float, 3> &arr) {
  const Float *data = arr.data();
  Float tot = 0.0;
#ifdef SLAB
  int nx = arr.shape(0);
  const uint64 nyz = arr.size() / nx;
#pragma omp parallel for MY_SCHEDULE reduction(+ : tot)
  for (int x = 0; x < nx; ++x) {
    const Float *slab = data + x * nyz;
    for (uint64 i = 0; i < nyz; ++i) {
      tot += slab[i];
    }
  }
#else
#pragma omp parallel for MY_SCHEDULE reduction(+ : tot)
  for (uint64 i = 0; i < arr.size(); ++i) {
    tot += data[i] * data[i];
  }
#endif
  return tot;
}

void multiply_with_conjugation(const RowMajorArrayPtr<Complex, 3> &in,
                               RowMajorArrayPtr<Complex, 3> &out) {
  assert(in.shape(0) == out.shape(0));
  assert(in.shape(1) == out.shape(1));
  assert(in.shape(2) == out.shape(2));
  const Complex *in_data = in.data();
  Complex *out_data = out.data();
#ifdef SLAB
  int nx = in.shape(0);
  uint64 nyz = in.size() / nx;
#pragma omp parallel for MY_SCHEDULE
  for (int x = 0; x < nx; ++x) {
    const Complex *in_slab = in_data + x * nyz;
    Complex *out_slab = out_data + x * nyz;
    for (uint64 i = 0; i < nyz; ++i) {
      out_slab[i] *= std::conj(in_slab[i]);
    }
  }
#else
#pragma omp parallel for MY_SCHEDULE
  for (uint64 i = 0; i < in.size(); i++) {
    out_data[i] *= std::conj(in_data[i]);
  }
#endif
}

}  // namespace array_ops
