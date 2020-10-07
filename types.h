#ifndef TYPES_H
#define TYPES_H

#include <complex>

// Our in-place FFTs require sizeof(Complex) = 2*sizeof(Float).
typedef double Float;
typedef std::complex<Float> Complex;

// We want to allow that the FFT grid could exceed 2^31 cells
typedef unsigned long long int uint64;

#define PAGE 4096  // To force some memory alignment; unit is bytes.

#endif  // TYPES_H