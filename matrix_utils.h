#ifndef MATRIX_UTILS_H
#define MATRIX_UTILS_H

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

void print_submatrix(Float *m, int n, int p, FILE *fp, Float norm) {
  // Print the inner part of a matrix(n,n,n) for debugging
  int mid = n / 2;
  assert(p <= mid);
  for (int i = -p; i <= p; i++)
    for (int j = -p; j <= p; j++) {
      fprintf(fp, "%2d %2d", i, j);
      for (int k = -p; k <= p; k++) {
        // We want to print mid+i, mid+j, mid+k
        fprintf(fp, " %12.8g",
                m[((mid + i) * n + (mid + j)) * n + mid + k] * norm);
      }
      fprintf(fp, "\n");
    }
  return;
}

void initialize_matrix(Float *&m, const uint64 size, const int nx) {
  // Initialize a matrix m and set it to zero.
  // We want to touch the whole matrix, because in NUMA this defines the
  // association of logical memory into the physical banks. nx will be our slab
  // decomposition; it must divide into size evenly Warning: This will only
  // allocate new space if m==NULL.  This allows one to reuse space.  But(!!)
  // there is no check that you've not changed the size of the matrix -- you
  // could overflow the previously allocated space.
  fprintf(stderr, "size: %llu, nx: %d\n", size, nx);
  assert(size % nx == 0);
  // Init.Start();
  if (m == NULL) {
    int err = posix_memalign((void **)&m, PAGE, sizeof(Float) * size + PAGE);
    assert(err == 0);
  }
  assert(m != NULL);
#ifdef SLAB
  const uint64 nyz = size / nx;
#pragma omp parallel for MY_SCHEDULE
  for (int x = 0; x < nx; x++) {
    Float *mslab = m + x * nyz;
    for (uint64 j = 0; j < nyz; j++) mslab[j] = 0.0;
  }
#else
#pragma omp parallel for MY_SCHEDULE
  for (uint64 j = 0; j < size; j++) m[j] = 0.0;
#endif
  // Init.Stop();
  return;
}

void initialize_matrix_by_copy(Float *&m, const uint64 size, const int nx,
                               const Float *copy) {
  // Initialize a matrix m and set it to copy[size].
  // nx will be our slab decomposition; it must divide into size evenly
  // Warning: This will only allocate new space if m==NULL.  This allows
  // one to reuse space.  But(!!) there is no check that you've not changed
  // the size of the matrix -- you could overflow the previously allocated
  // space.
  assert(size % nx == 0);
  // Init.Start();
  if (m == NULL) {
    int err = posix_memalign((void **)&m, PAGE, sizeof(Float) * size + PAGE);
    assert(err == 0);
  }
  assert(m != NULL);
#ifdef SLAB
  const uint64 nyz = size / nx;
#pragma omp parallel for MY_SCHEDULE
  for (int x = 0; x < nx; x++) {
    Float *mslab = m + x * nyz;
    Float *cslab = copy + x * nyz;
    for (uint64 j = 0; j < nyz; j++) mslab[j] = cslab[j];
    // memcpy(mslab, cslab, sizeof(Float)*nyz);    // Slower for some reason!
  }
#else
#pragma omp parallel for MY_SCHEDULE
  for (uint64 j = 0; j < size; j++) m[j] = copy[j];
#endif
  // Init.Stop();
  return;
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

void scale_matrix(Float *a, const Float b, const uint64 size, const int nx) {
  // Multiply a by a scalar b
  // nx will be our slab decomposition; it must divide into size evenly
  assert(size % nx == 0);
#ifdef SLAB
  const uint64 nyz = size / nx;
#pragma omp parallel for MY_SCHEDULE
  for (int x = 0; x < nx; x++) {
    Float *aslab = a + x * nyz;
    for (uint64 j = 0; j < nyz; j++) aslab[j] *= b;
  }
#else
#pragma omp parallel for MY_SCHEDULE
  for (uint64 j = 0; j < size; j++) a[j] *= b;
#endif
}

void addscalarto_matrix(Float *a, const Float b, const uint64 size,
                        const int nx) {
  // Add scalar b to matrix a
  // nx will be our slab decomposition; it must divide into size evenly
  assert(size % nx == 0);
#ifdef SLAB
  const uint64 nyz = size / nx;
#pragma omp parallel for MY_SCHEDULE
  for (int x = 0; x < nx; x++) {
    Float *aslab = a + x * nyz;
    for (uint64 j = 0; j < nyz; j++) aslab[j] += b;
  }
#else
#pragma omp parallel for MY_SCHEDULE
  for (uint64 j = 0; j < size; j++) a[j] += b;
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

Float sum_matrix(const Float *a, const uint64 size, const int nx) {
  // Sum the elements of the matrix
  // nx will be our slab decomposition; it must divide into size evenly
  assert(size % nx == 0);
  Float tot = 0.0;
#ifdef SLAB
  const uint64 nyz = size / nx;
#pragma omp parallel for MY_SCHEDULE reduction(+ : tot)
  for (int x = 0; x < nx; x++) {
    Float *aslab = a + x * nyz;
    for (uint64 j = 0; j < nyz; j++) tot += aslab[j];
  }
#else
#pragma omp parallel for MY_SCHEDULE reduction(+ : tot)
  for (uint64 j = 0; j < size; j++) tot += a[j];
#endif
  return tot;
}

Float sumsq_matrix(const Float *a, const uint64 size, const int nx) {
  // Sum the square of elements of the matrix
  // nx will be our slab decomposition; it must divide into size evenly
  assert(size % nx == 0);
  Float tot = 0.0;
#ifdef SLAB
  const uint64 nyz = size / nx;
#pragma omp parallel for MY_SCHEDULE reduction(+ : tot)
  for (int x = 0; x < nx; x++) {
    Float *aslab = a + x * nyz;
    for (uint64 j = 0; j < nyz; j++) tot += aslab[j] * aslab[j];
  }
#else
#pragma omp parallel for MY_SCHEDULE reduction(+ : tot)
  for (uint64 j = 0; j < size; j++) tot += a[j] * a[j];
#endif
  return tot;
}

void multiply_matrix_with_conjugation(Complex *a, Complex *b, const uint64 size,
                                      const int nx) {
  // Element-wise multiply a[] by conjugate of b[]
  // Note that size refers to the Complex size; the calling routine
  // is responsible for dividing the Float size by 2.
  // nx will be our slab decomposition; it must divide into size evenly
  assert(size % nx == 0);
#ifdef SLAB
  const uint64 nyz = size / nx;
#pragma omp parallel for MY_SCHEDULE
  for (int x = 0; x < nx; x++) {
    Complex *aslab = a + x * nyz;
    Complex *bslab = b + x * nyz;
    for (uint64 j = 0; j < nyz; j++) aslab[j] *= std::conj(bslab[j]);
  }
#else
#pragma omp parallel for MY_SCHEDULE
  for (uint64 j = 0; j < size; j++) a[j] *= std::conj(b[j]);
#endif
}

/* ==========================  Submatrix extraction =================== */

void extract_submatrix(Float *total, Float *corr, int csize[3], Float *work,
                       int ngrid[3], const int ngrid2) {
  // Given a large matrix work[ngrid^3],
  // extract out a submatrix of size csize^3, centered on work[0,0,0].
  // Multiply the result by corr[csize^3] and add it onto total[csize^3]
  // Again, zero lag is mapping to corr(csize/2, csize/2, csize/2),
  // but it is at (0,0,0) in the FFT grid.
  // Extract.Start();
  int cx = csize[0] / 2;  // This is the middle of the submatrix
  int cy = csize[1] / 2;  // This is the middle of the submatrix
  int cz = csize[2] / 2;  // This is the middle of the submatrix
#pragma omp parallel for schedule(dynamic, 1)
  for (uint64 i = 0; i < csize[0]; i++) {
    uint64 ii = (ngrid[0] - cx + i) % ngrid[0];
    for (int j = 0; j < csize[1]; j++) {
      uint64 jj = (ngrid[1] - cy + j) % ngrid[1];
      Float *t = total + (i * csize[1] + j) * csize[2];  // This is (i,j,0)
      Float *cc = corr + (i * csize[1] + j) * csize[2];  // This is (i,j,0)
      Float *Y = work + (ii * ngrid[1] + jj) * ngrid2 + ngrid[2] - cz;
      // This is (ii,jj,ngrid[2]-c)
      for (int k = 0; k < cz; k++) t[k] += cc[k] * Y[k];
      Y = work + (ii * ngrid[1] + jj) * ngrid2 - cz;
      // This is (ii,jj,-c)
      for (int k = cz; k < csize[2]; k++) t[k] += cc[k] * Y[k];
    }
  }
  // Extract.Stop();
}

void extract_submatrix_C2R(Float *total, Float *corr, int csize[3],
                           Complex *work, int ngrid[3], const int ngrid2) {
  // Given a large matrix work[ngrid^3/2],
  // extract out a submatrix of size csize^3, centered on work[0,0,0].
  // The input matrix is Complex * with the half-domain Fourier convention.
  // We are only summing the real part; the imaginary part always sums to zero.
  // Need to reflect the -z part around the origin, which also means reflecting
  // x & y. ngrid[2] and ngrid2 are given as their Float values, not yet divided
  // by two. Multiply the result by corr[csize^3] and add it onto total[csize^3]
  // Again, zero lag is mapping to corr(csize/2, csize/2, csize/2),
  // but it is at (0,0,0) in the FFT grid.
  // Extract.Start();
  int cx = csize[0] / 2;  // This is the middle of the submatrix
  int cy = csize[1] / 2;  // This is the middle of the submatrix
  int cz = csize[2] / 2;  // This is the middle of the submatrix
#pragma omp parallel for schedule(dynamic, 1)
  for (uint64 i = 0; i < csize[0]; i++) {
    uint64 ii = (ngrid[0] - cx + i) % ngrid[0];
    uint64 iin = (ngrid[0] - ii) % ngrid[0];  // The reflected coord
    for (int j = 0; j < csize[1]; j++) {
      uint64 jj = (ngrid[1] - cy + j) % ngrid[1];
      uint64 jjn = (ngrid[1] - jj) % ngrid[1];           // The reflected coord
      Float *t = total + (i * csize[1] + j) * csize[2];  // This is (i,j,0)
      Float *cc = corr + (i * csize[1] + j) * csize[2];  // This is (i,j,0)
      // The positive half-plane (inclusize)
      Complex *Y = work + (ii * ngrid[1] + jj) * ngrid2 / 2 - cz;
      // This is (ii,jj,-cz)
      for (int k = cz; k < csize[2]; k++) t[k] += cc[k] * std::real(Y[k]);
      // The negative half-plane (inclusize), reflected.
      // k=cz-1 should be +1, k=0 should be +cz
      Y = work + (iin * ngrid[1] + jjn) * ngrid2 / 2 + cz;
      // This is (iin,jjn,+cz)
      for (int k = 0; k < cz; k++) t[k] += cc[k] * std::real(Y[-k]);
    }
  }
  // Extract.Stop();
}

#endif  // MATRIX_UTILS_H