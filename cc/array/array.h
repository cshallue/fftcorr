#ifndef ARRAY_H
#define ARRAY_H

#include <array>
#include <utility>

#include "../types.h"

template <typename dtype>
class ArrayPtr {
 public:
  // Default constructor.
  // A default-constructed ArrayPtr is effectively a null pointer until
  // initialize() is called. Operations should not be called before
  // initialization. We need this constructor so that classes can allocate an
  // ArrayPtr on the stack even if they construct the array in the body
  // of their constructor.
  ArrayPtr() : size_(0), data_(NULL) {}

  ArrayPtr(uint64 size, dtype *data) : size_(size), data_(data) {}

  virtual ~ArrayPtr() = default;

  void initialize(uint64 size, dtype *data) {
    assert(data_ == NULL);  // Only allowed to initialize once.
    size_ = size;
    data_ = data;
    // TODO: add this assertion
    // assert(data_ != NULL);  // Can't initialize with NULL data.
  }

  uint64 size() const { return size_; }
  dtype *data() { return data_; }
  const dtype *data() const { return data_; }

  inline dtype &operator[](int idx) { return data_[idx]; }
  inline const dtype &operator[](int idx) const { return data_[idx]; }

  // Iterability.
  dtype *begin() { return data_; }
  dtype *end() { return &data_[size()]; }

 protected:
  uint64 size_;
  dtype *data_;
};

template <typename dtype>
class Array : public ArrayPtr<dtype> {
 public:
  // Default constructor: empty array.
  Array() : ArrayPtr<dtype>() {}

  // Takes ownership of allocated memory.
  Array(uint64 size, dtype *data) : ArrayPtr<dtype>(size, data) {}

  // Allocates memory.
  Array(uint64 size) { initialize(size); }

  virtual ~Array() {
    if (this->data_ != NULL) free(this->data_);
  }

  // Allocates memory.
  void initialize(uint64 size) {
    assert(this->data_ == NULL);  // Make sure unitialized.
    // Page alignment is only important for the big 3D grids, but we don't have
    // too many smaller arrays, so we just do it for all arrays.
    assert(size > 0);
    int err = posix_memalign((void **)&this->data_, PAGE,
                             sizeof(dtype) * size + PAGE);
    assert(err == 0);
    assert(this->data_ != NULL);
    this->size_ = size;
  }

  // Disable copy construction and copy assignment. We could implement these if
  // needed, but we'd need to copy the data.
  Array(const Array<dtype> &) = delete;
  Array &operator=(const Array<dtype> &) = delete;

  // Move assignment and constructor. TODO: is this just the default operation?
  Array &operator=(Array &&other) {
    std::swap(this->size_, other.size_);
    std::swap(this->data_, other.data_);
    return *this;
  }
  Array(Array &&other) : Array() { *this = std::move(other); }
};

#endif  // ARRAY_H