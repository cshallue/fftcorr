#ifndef ARRAY_H
#define ARRAY_H

#include <array>
#include <utility>

#include "../types.h"

// Interface for a general array of numbers backed by a contiguous memory block.
template <typename dtype>
class Array {
 public:
  virtual ~Array() = default;
  virtual uint64 size() const = 0;
  virtual dtype *data() = 0;
  virtual const dtype *data() const = 0;
};

#endif  // ARRAY_H