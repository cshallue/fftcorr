#include "array3d.h"

// template <std::size_t N>
// ArrayBase<N>::ArrayBase(const std::array<int, N> shape)
//     : shape_(shape), size_(1), data_(NULL) {
//   for (int nx : shape_) size_ *= nx;
//   // Allocate data_ array.
//   int err = posix_memalign((void **)&data_, PAGE, sizeof(Float) * size_ +
//   PAGE); assert(err == 0); assert(data_ != NULL);
// }

Array1D range(Float start, Float step, int size) {
  Array1D arr(size);
  for (int i = 0; i < size; ++i) {
    arr[i] = start + i * step;
  }
  return arr;
}

Array3D::Array3D() : data_(NULL), cdata_(NULL), arr_(NULL), carr_(NULL) {}

Array3D::~Array3D() {
  if (data_ != NULL) free(data_);
  if (arr_ != NULL) delete arr_;
  if (carr_ != NULL) delete carr_;
}

void Array3D::initialize(std::array<int, 3> shape) {
  shape_ = shape;
  size_ = (uint64)shape[0] * shape[1] * shape[2];
  // Allocate data_ array.
  int err = posix_memalign((void **)&data_, PAGE, sizeof(Float) * size_ + PAGE);
  assert(err == 0);
  assert(data_ != NULL);

  arr_ = new RowMajorArray<Float>(data_, shape);

  cdata_ = (Complex *)data_;
  carr_ =
      new RowMajorArray<Complex>(cdata_, {shape[0], shape[1], shape[2] / 2});

  set_all(0.0);

  // cshape_ = std::array<int, 3>({shape_[0], shape_[1], shape_[2] / 2});
  // csize_ = (uint64)cshape_[0] * cshape_[1] * cshape_[2];
  // cdata_ = (Complex *)data_;
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

void Array3D::copy_with_scalar_multiply(const Array3D &other, Float s) {
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
      slab[i] = other_data[i] * s;
    }
  }
#else
#pragma omp parallel for MY_SCHEDULE
  for (uint64 i = 0; i < size_; i++) {
    data_[i] = other_data[i] * s;
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
