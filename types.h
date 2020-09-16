#ifndef TYPES_H
#define TYPES_H

#include <complex>

// In principle, this gives flexibility, but one also would need to adjust the
// FFTW calls.
typedef double Float;
typedef std::complex<double> Complex;

// We want to allow that the FFT grid could exceed 2^31 cells
typedef unsigned long long int uint64;

#define PAGE 4096  // To force some memory alignment; unit is bytes.

#endif  // TYPES_H