#ifndef ARRAY_H
#define ARRAY_H

#include <array>
#include <utility>

#include "../types.h"

// Interface for a general array of numbers.
template <typename dtype>
class Array {
 public:
  // TODO: do I need to make the constructor protected? Or is the fact that
  // there are = 0's enough to make it non constructible?
  virtual ~Array() = default;
  virtual uint64 size() const = 0;
  virtual dtype *data() = 0;
  virtual const dtype *data() const = 0;

  // TODO: delete? random access raises questions about indexing for nd arrays
  // and the user can already get data* or iterate directly - are these enough?
  virtual dtype &operator[](uint64 idx) = 0;
  virtual const dtype &operator[](uint64 idx) const = 0;

  // Iterator through the flattened array. TODO: needed?
  virtual dtype *begin() = 0;
  virtual dtype *end() = 0;
};

#endif  // ARRAY_H