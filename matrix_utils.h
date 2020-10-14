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

Float *allocate_array(uint64 size) {
  Float *arr;
  int err = posix_memalign((void **)&arr, PAGE, sizeof(Float) * size + PAGE);
  assert(err == 0);
  assert(arr != NULL);
  return arr;
}

void set_matrix(Float *a, const Float b, const uint64 size, const int nx) {
  fprintf(stderr, "set_matrix: b = %f, size = %lld, nx = %d\n", b, size, nx);
  // Set a equal to a scalar b
  // nx will be our slab decomposition; it must divide into size evenly
  assert(size % nx == 0);
#ifdef SLAB
  const uint64 nyz = size / nx;
#pragma omp parallel for MY_SCHEDULE
  for (int x = 0; x < nx; x++) {
    Float *aslab = a + x * nyz;
    for (uint64 j = 0; j < nyz; j++) aslab[j] = b;
  }
#else
#pragma omp parallel for MY_SCHEDULE
  for (uint64 j = 0; j < size; j++) a[j] = b;
  fprintf(stderr, "done\n");
#endif
}

void copy_matrix(Float *a, const Float *b, const uint64 size, const int nx) {
  // Set a equal to a vector b
  // nx will be our slab decomposition; it must divide into size evenly
  assert(size % nx == 0);
#ifdef SLAB
  const uint64 nyz = size / nx;
#pragma omp parallel for MY_SCHEDULE
  for (int x = 0; x < nx; x++) {
    Float *aslab = a + x * nyz;
    Float *bslab = b + x * nyz;
    for (uint64 j = 0; j < nyz; j++) aslab[j] = bslab[j];
  }
#else
#pragma omp parallel for MY_SCHEDULE
  for (uint64 j = 0; j < size; j++) a[j] = b[j];
#endif
}

void copy_matrix(Float *a, const Float *b, const Float c, const uint64 size,
                 const int nx) {
  // Set a equal to a vector b times a scalar c
  // nx will be our slab decomposition; it must divide into size evenly
  assert(size % nx == 0);
#ifdef SLAB
  const uint64 nyz = size / nx;
#pragma omp parallel for MY_SCHEDULE
  for (int x = 0; x < nx; x++) {
    Float *aslab = a + x * nyz;
    Float *bslab = b + x * nyz;
    for (uint64 j = 0; j < nyz; j++) aslab[j] = bslab[j] * c;
  }
#else
#pragma omp parallel for MY_SCHEDULE
  for (uint64 j = 0; j < size; j++) a[j] = b[j] * c;
#endif
}

#endif  // MATRIX_UTILS_H