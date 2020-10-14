#include "array3d.h"

Array1D::Array1D(int size) : size_(size), data_(NULL) {
  // Allocate data_ array.
  int err = posix_memalign((void **)&data_, PAGE, sizeof(Float) * size_ + PAGE);
  assert(err == 0);
  assert(data_ != NULL);
}

Array1D::~Array1D() {
  if (data_ != NULL) free(data_);
}

Array1D range(Float start, Float step, int size) {
  Array1D arr(size);
  for (int i = 0; i < size; ++i) {
    arr[i] = start + i * step;
  }
  return arr;
}

Array3D::Array3D() : data_(NULL), cdata_(NULL) {}

Array3D::~Array3D() {
  if (data_ != NULL) free(data_);
}

void Array3D::initialize(std::array<int, 3> shape) {
  shape_ = shape;
  size_ = (uint64)shape_[0] * shape_[1] * shape_[2];
  // Allocate data_ array.
  int err = posix_memalign((void **)&data_, PAGE, sizeof(Float) * size_ + PAGE);
  assert(err == 0);
  assert(data_ != NULL);
  set_all(0.0);

  cshape_ = std::array<int, 3>({shape_[0], shape_[1], shape_[2] / 2});
  csize_ = (uint64)cshape_[0] * cshape_[1] * cshape_[2];
  cdata_ = (Complex *)data_;
}

void Array3D::copy_from(const Array3D &other) {
  assert(data_ != NULL);
  // TODO: check same dimensions.
  // Init.Start();
  const Float *other_data = other.data_;
#ifdef SLAB
  int nx = shape_[0];
  const uint64 nyz = size_ / nx;
#pragma omp parallel for MY_SCHEDULE
  for (int x = 0; x < nx; ++x) {
    Float *slab = data_ + x * nyz;
    for (uint64 i = 0; i < nyz; ++i) {
      slab[i] = other_data[i];
    }
  }
#else
#pragma omp parallel for MY_SCHEDULE
  for (uint64 i = 0; i < size_; i++) {
    data_[i] = other_data[i];
  }
#endif
  // Init.Stop();
}

void Array3D::set_all(Float value) {
  // Initialize data_ by setting each element.
  // We want to touch the whole matrix, because in NUMA this defines the
  // association of logical memory into the physical banks.
  // Init.Start();
  assert(data_ != NULL);
#ifdef SLAB
  int nx = rshape_[0];
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

void Array3D::multiply_with_conjugation(const Array3D &other) {
  assert(data_ != NULL);
  // Element-wise multiply by conjugate of other
  // TODO: check same dimensions.
#ifdef SLAB
  int nx = shape_[0];
  const uint64 nyz = csize_ / nx;
#pragma omp parallel for MY_SCHEDULE
  for (int x = 0; x < nx; ++x) {
    Complex *slab = cdata_ + x * nyz;
    Complex *other_slab = other.cdata_ + x * nyz;
    for (uint64 i = 0; i < nyz; ++i) {
      slab[i] *= std::conj(other_slab[i]);
    }
  }
#else
#pragma omp parallel for MY_SCHEDULE
  for (uint64 i = 0; i < csize_; ++i) {
    cdata_[i] *= std::conj(other.cdata_[i]);
  }
#endif
}
