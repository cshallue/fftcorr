#ifndef PARTICLE_H
#define PARTICLE_H

#include "../types.h"

struct Particle {
  Particle(Float _x, Float _y, Float _z, Float _w, uint64 _index)
      : x(_x), y(_y), z(_z), w(_w), index(_index) {}

  // So we can sort in index order.
  bool operator<(const Particle &other) const { return index < other.index; }

  Float x;
  Float y;
  Float z;
  Float w;
  uint64 index;
};

#endif  // PARTICLE_H