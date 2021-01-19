#ifndef ROW_MAJOR_ARRAY_H
#define ROW_MAJOR_ARRAY_H

#include <array>

#include "../types.h"

// TODO: make the dimension a template parameter too? need to make sure for
// loops are unrolled?
// TODO: it's easy to split this into two separate classes, one that owns the
// data and one that doesn't. Or one that doesn't own the data and one that
// might or might not. Either way, we'd do RowMajorArray -> RowMajorArrayPtr and
// then RowMajorArray : RowMajorArrayPtr. Not sure if that would be useful in
// any way.
template <typename dtype>
class RowMajorArray {
 public:
  explicit RowMajorArray(std::array<int, 3> shape)
      : RowMajorArray<dtype>(nullptr, shape) {}

  RowMajorArray(dtype *data, std::array<int, 3> shape)
      : data_(data),
        own_data_(false),
        shape_(shape),
        size_((uint64)shape_[0] * shape_[1] * shape_[2]) {
    if (data_ != NULL) return;
    // Allocate data_ array.
    int err =
        posix_memalign((void **)&data_, PAGE, sizeof(dtype) * size_ + PAGE);
    assert(err == 0);
    assert(data_ != NULL);
    own_data_ = true;

    // Initialize data_ by setting each element.
    // TODO: do we want to do this in every case? Or do we want to initialize by
    // copy sometimes? We want to touch the whole matrix, because in NUMA this
    // defines the association of logical memory into the physical banks.
    // Init.Start();
#ifdef SLAB
    int nx = shape[0];
    const uint64 nyz = shape[1] * shape[2];
#pragma omp parallel for MY_SCHEDULE
    for (int x = 0; x < nx; ++x) {
      Float *slab = data_ + x * nyz;
      for (uint64 i = 0; i < nyz; ++i) {
        slab[i] = 0.0;
      }
    }
#else
#pragma omp parallel for MY_SCHEDULE
    for (uint64 i = 0; i < size_; ++i) {
      data_[i] = 0.0;
    }
#endif
    // Init.Stop();
  }

  ~RowMajorArray() {
    if (own_data_ && data_ != NULL) free(data_);
  }

  // TODO: needed?
  const std::array<int, 3> &shape() const { return shape_; }
  int shape(int i) const { return shape_[i]; }
  uint64 size() const { return size_; }
  dtype *data() { return data_; }

  // Indexing.
  inline uint64 get_index(int ix, int iy, int iz) const {
    return (uint64)iz + shape_[2] * (iy + ix * shape_[1]);
  }
  inline dtype &at(int ix, int iy, int iz) {
    return data_[get_index(ix, iy, iz)];
  }
  inline const dtype &at(int ix, int iy, int iz) const {
    return data_[get_index(ix, iy, iz)];
  }
  // TODO: just use at() syntax instead of a new function? Fewer chars,
  // but then again, it returns a pointer and this is a RowMajorArray.
  dtype *get_row(int ix, int iy) { return &at(ix, iy, 0); }
  const dtype *get_row(int ix, int iy) const { return &at(ix, iy, 0); }

 private:
  dtype *data_;
  bool own_data_;
  std::array<int, 3> shape_;
  uint64 size_;
};

#endif  // ROW_MAJOR_ARRAY_H
