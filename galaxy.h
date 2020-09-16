#ifndef GALAXY_H
#define GALAXY_H

#include "types.h"

// A very simple class to contain the input objects
class Galaxy {
 public:
  Float x, y, z, w;
  uint64 index;
  Galaxy(Float a[4], uint64 i) {
    x = a[0];
    y = a[1];
    z = a[2];
    w = a[3];
    index = i;
    // fprintf(stderr, "Galaxy %llu: (%f, %f, %f) weight %f\n", i, x, y, z, w);
    return;
  }
  ~Galaxy() {}
  // We'll want to be able to sort in 'x' order
  // bool operator < (const Galaxy& str) const { return (x < str.x); }
  // Sort in cell order
  bool operator<(const Galaxy &str) const { return (index < str.index); }
};

#endif  // GALAXY_H