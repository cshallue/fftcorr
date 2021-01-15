#ifndef GALAXY_H
#define GALAXY_H

#include "types.h"

// A very simple class to contain the input objects
// TODO: rename to Particle. Make it a struct.
class Galaxy {
 public:
  Float x, y, z, w;
  uint64 index;
  Galaxy(Float _x, Float _y, Float _z, Float _w, uint64 _index)
      : x(_x), y(_y), z(_z), w(_w), index(_index) {}
  ~Galaxy() {}
  // We'll want to be able to sort in 'x' order
  // bool operator < (const Galaxy& str) const { return (x < str.x); }
  // Sort in cell order
  bool operator<(const Galaxy &str) const { return (index < str.index); }
};

#endif  // GALAXY_H