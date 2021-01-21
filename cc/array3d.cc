#include "array3d.h"

// template <std::size_t N>
// ArrayBase<N>::ArrayBase(const std::array<int, N> shape)
//     : shape_(shape), size_(1), data_(NULL) {
//   for (int nx : shape_) size_ *= nx;
//   // Allocate data_ array.
//   int err = posix_memalign((void **)&data_, PAGE, sizeof(Float) * size_ +
//   PAGE); assert(err == 0); assert(data_ != NULL);
// }

Array3D::Array3D(std::array<int, 3> shape) : ArrayBase(shape) {
  arr_ = new RowMajorArrayPtr<Float>(data_, shape_);
}

Array3D::~Array3D() { delete arr_; }

void Array3D::set_all(Float value) {
  // Initialize data_ by setting each element.
  // We want to touch the whole matrix, because in NUMA this defines the
  // association of logical memory into the physical banks.
  // Init.Start();
  assert(data_ != NULL);
#ifdef SLAB
  int nx = shape_[0];
  const uint64 nyz = size_ / nx;
#pragma omp parallel for MY_SCHEDULE
  for (int x = 0; x < nx; ++x) {
    Float *slab = data_ + x * nyz;
    for (uint64 i = 0; i < nyz; ++i) {
      slab[i] = value;
    }
  }
#else
#pragma omp parallel for MY_SCHEDULE
  for (uint64 i = 0; i < size_; i++) {
    data_[i] = value;
  }
#endif
  // Init.Stop();
}

void Array3D::add_scalar(Float s) {
  assert(data_ != NULL);
#ifdef SLAB
  int nx = shape_[0];
  const uint64 nyz = size_ / nx;
#pragma omp parallel for MY_SCHEDULE
  for (int x = 0; x < nx; ++x) {
    // TODO: I could use arr_->get_row() here.
    Float *slab = data_ + x * nyz;
    for (uint64 i = 0; i < nyz; ++i) {
      slab[i] += s;
    }
  }
#else
#pragma omp parallel for MY_SCHEDULE
  for (uint64 i = 0; i < size_; ++i) {
    data_[i] += s;
  }
#endif
}

void Array3D::multiply_by(Float s) {
  assert(data_ != NULL);
#ifdef SLAB
  int nx = shape_[0];
  const uint64 nyz = size_ / nx;
#pragma omp parallel for MY_SCHEDULE
  for (int x = 0; x < nx; ++x) {
    Float *slab = data_ + x * nyz;
    for (uint64 i = 0; i < nyz; ++i) {
      slab[i] *= s;
    }
  }
#else
#pragma omp parallel for MY_SCHEDULE
  for (uint64 i = 0; i < size_; ++i) {
    data_[i] *= s;
  }
#endif
}

Float Array3D::sum() const {
  assert(data_ != NULL);
  Float tot = 0.0;
#ifdef SLAB
  int nx = shape_[0];
  const uint64 nyz = size_ / nx;
#pragma omp parallel for MY_SCHEDULE reduction(+ : tot)
  for (int x = 0; x < nx; ++x) {
    Float *slab = data_ + x * nyz;
    for (uint64 i = 0; i < nyz; ++i) {
      tot += slab[i];
    }
  }
#else
#pragma omp parallel for MY_SCHEDULE reduction(+ : tot)
  for (uint64 i = 0; i < size_; ++i) {
    tot += data_[i];
  }
#endif
  return tot;
}

// TODO: come up with a way to template these parallelizable ops
Float Array3D::sumsq() const {
  assert(data_ != NULL);
  Float tot = 0.0;
#ifdef SLAB
  int nx = shape_[0];
  const uint64 nyz = size_ / nx;
#pragma omp parallel for MY_SCHEDULE reduction(+ : tot)
  for (int x = 0; x < nx; ++x) {
    Float *slab = data_ + x * nyz;
    for (uint64 i = 0; i < nyz; ++i) {
      tot += slab[i];
    }
  }
#else
#pragma omp parallel for MY_SCHEDULE reduction(+ : tot)
  for (uint64 i = 0; i < size_; ++i) {
    tot += data_[i] * data_[i];
  }
#endif
  return tot;
}