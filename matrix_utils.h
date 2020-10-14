#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

#include <assert.h>

#include <array>

#include "types.h"

// Here are some matrix handling routines, which may need OMP attention

#ifndef OPENMP
#undef SLAB  // We probably don't want to do this if single threaded
#endif

#ifdef SLAB
// Try treating the matrices explicitly by x slab;
// this might allow NUMA memory to be closer to the socket running the thread.
#define MY_SCHEDULE schedule(static, 1)
#define YLM_SCHEDULE schedule(static, 1)
#else
// Just treat the matrices as one big object
#define MY_SCHEDULE schedule(dynamic, 512)
#define YLM_SCHEDULE schedule(dynamic, 1)
#endif

#endif  // MATRIX_UTILS_H