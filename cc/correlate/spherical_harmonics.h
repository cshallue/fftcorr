#ifndef SPHERICAL_HARMONICS_H
#define SPHERICAL_HARMONICS_H

#include <assert.h>

#include <array>

#include "../array/row_major_array.h"
#include "../multithreading.h"
#include "../types.h"

// Evaluates the spherical harmonic function Y_lm on a Cartesian grid.
//   m = 0 will return Y_l0,
//   m > 0 will return sqrt(2) * Re(Y_lm),
//   m < 0 will return sqrt(2) * Im(Y_l|m|),
// up to an arbitrary choice of sign for each l, m.
//
// This convention is so we can use the following identity when summing over m:
//
//     Y_lm(r1) * conj(Y_lm(r2)) + Y_l(-m)(r1) * conj(Y_l(-m)(r2)) =
//       2 * Re[Y_lm(r1)] * Re[Y_lm(r2)] + 2 * Im[Y_lm(r1)] * Im[Y_lm(r2)]
//
// xcell, ycell, zcell are the x, y, z axes corresponding to the output grid
// Ylm, with the origin at Ylm[0, 0, 0]. These axes may be smaller than the
// output grid, in which case only the subgrid spanned by these axes will be
// filled.
//
// The polar angle and azimuthal angle are defined in the usual way with respect
// to the x, y, and z axes.
//
// In addition, the results are multiplied by coeff * r^exponent * mult[r],
// where mult (if not null) is a grid with the same dimensions as ylm.
void make_ylm(int ell, int m, const Array1D<Float> &xcell,
              const Array1D<Float> &ycell, const Array1D<Float> &zcell,
              Float coeff, int exponent, const RowMajorArrayPtr<Float, 3> *mult,
              RowMajorArrayPtr<Float, 3> *ylm) {
  // Check dimensions.
  const int n0 = xcell.size();
  const int n1 = ycell.size();
  const int n2 = zcell.size();
  assert(ylm->shape(0) >= n0);
  assert(ylm->shape(1) >= n1);
  assert(ylm->shape(2) >= n2);
  assert(exponent % 2 == 0);

  if (ell == 0 && m == 0 && exponent == 0) {
    // This case is so easy that we'll do it directly and skip the setup.
    Float value = coeff / sqrt(4.0 * M_PI);
#pragma omp parallel for YLM_SCHEDULE
    for (int i = 0; i < n0; ++i) {
      if (mult) {
        Float *Y;
        const Float *D;
        for (int j = 0; j < n1; ++j) {
          Y = ylm->get_row(i, j);
          D = mult->get_row(i, j);
          for (int k = 0; k < n2; ++k) Y[k] = D[k] * value;
        }
      } else {
        Float *Y;
        for (int j = 0; j < n1; j++) {
          Y = ylm->get_row(i, j);
          for (int k = 0; k < n2; ++k) Y[k] = value;
        }
      }
    }
    return;
  }

  (*ylm)[0] = -123456.0;  // A sentinal value.

  // Precompute some terms.
  Array1D<Float> z2(n2);
  Array1D<Float> z3(n2);
  Array1D<Float> z4(n2);
  Array1D<Float> ones(n2);
  const Float *z = zcell.data();
  for (int k = 0; k < n2; ++k) {
    z2[k] = z[k] * z[k];
    z3[k] = z2[k] * z[k];
    z4[k] = z3[k] * z[k];
    ones[k] = 1.0;
  }
  Float isqpi = sqrt(1.0 / M_PI);
  isqpi *= coeff;
  if (m != 0) isqpi *= sqrt(2.0);
  Float tiny = 1e-20;
#pragma omp parallel for YLM_SCHEDULE
  for (int i = 0; i < n0; ++i) {
    Array1D<Float> ir2(n2);
    Array1D<Float> rpow(n2);
    Float x = xcell[i];
    Float x2 = x * x;
    for (int j = 0; j < n1; ++j) {
      Float *Y = ylm->get_row(i, j);
      const Float *D = mult == NULL ? ones.data() : mult->get_row(i, j);
      Float y = ycell[j];
      Float y2 = y * y;
      Float y3 = y2 * y;
      Float y4 = y3 * y;
      for (int k = 0; k < n2; ++k) ir2[k] = 1.0 / (x2 + y2 + z2[k] + tiny);
      // Fill R with r^exponent
      Float *R;
      if (exponent == 0) {
        R = ones.data();
      } else {
        R = rpow.data();
        for (int k = 0; k < n2; ++k) {
          R[k] = 1.0;
          // r^(2*sign(exponent))
          Float rmult = exponent > 0 ? (x2 + y2 + z2[k]) : ir2[k];
          for (int e = 1; e <= abs(exponent) / 2; ++e) R[k] *= rmult;
        }
      }
      // Now ready to compute
      if (ell == 0) {
        for (int k = 0; k < n2; ++k) Y[k] = D[k] * R[k] * isqpi / 2.0;
      } else if (ell == 2) {
        if (m == 2)
          for (int k = 0; k < n2; ++k)
            Y[k] = D[k] * R[k] * isqpi * sqrt(15.0 / 32.0) * (x2 - y2) * ir2[k];
        else if (m == 1)
          for (int k = 0; k < n2; ++k)
            Y[k] = D[k] * R[k] * isqpi * sqrt(15.0 / 8.0) * x * z[k] * ir2[k];
        else if (m == 0)
          for (int k = 0; k < n2; ++k)
            Y[k] = D[k] * R[k] * isqpi * sqrt(5.0 / 16.0) *
                   (2.0 * z2[k] - x2 - y2) * ir2[k];
        else if (m == -1)
          for (int k = 0; k < n2; ++k)
            Y[k] = D[k] * R[k] * isqpi * sqrt(15.0 / 8.0) * y * z[k] * ir2[k];
        else if (m == -2)
          for (int k = 0; k < n2; ++k)
            Y[k] = D[k] * R[k] * isqpi * sqrt(15.0 / 8.0) * x * y * ir2[k];
      } else if (ell == 4) {
        if (m == 4)
          for (int k = 0; k < n2; ++k)
            Y[k] = D[k] * R[k] * isqpi * 3.0 / 16.0 * sqrt(35.0 / 2.0) *
                   (x2 * x2 - 6.0 * x2 * y2 + y4) * ir2[k] * ir2[k];
        else if (m == 3)
          for (int k = 0; k < n2; ++k)
            Y[k] = D[k] * R[k] * isqpi * 3.0 / 8.0 * sqrt(35.0) *
                   (x2 - 3.0 * y2) * z[k] * x * ir2[k] * ir2[k];
        else if (m == 2)
          for (int k = 0; k < n2; ++k)
            Y[k] = D[k] * R[k] * isqpi * 3.0 / 8.0 * sqrt(5.0 / 2.0) *
                   (6.0 * z2[k] * (x2 - y2) - x2 * x2 + y4) * ir2[k] * ir2[k];
        else if (m == 1)
          for (int k = 0; k < n2; ++k)
            Y[k] = D[k] * R[k] * isqpi * 3.0 / 8.0 * sqrt(5.0) * 3.0 *
                   (4.0 / 3.0 * z2[k] - x2 - y2) * x * z[k] * ir2[k] * ir2[k];
        else if (m == 0)
          for (int k = 0; k < n2; ++k)
            Y[k] = D[k] * R[k] * isqpi * 3.0 / 16.0 * 8.0 *
                   (z4[k] - 3.0 * z2[k] * (x2 + y2) +
                    3.0 / 8.0 * (x2 * x2 + 2.0 * x2 * y2 + y4)) *
                   ir2[k] * ir2[k];
        else if (m == -1)
          for (int k = 0; k < n2; ++k)
            Y[k] = D[k] * R[k] * isqpi * 3.0 / 8.0 * sqrt(5.0) * 3.0 *
                   (4.0 / 3.0 * z2[k] - x2 - y2) * y * z[k] * ir2[k] * ir2[k];
        else if (m == -2)
          for (int k = 0; k < n2; ++k)
            Y[k] = D[k] * R[k] * isqpi * 3.0 / 8.0 * sqrt(5.0 / 2.0) *
                   (2.0 * x * y * (6.0 * z2[k] - x2 - y2)) * ir2[k] * ir2[k];
        else if (m == -3)
          for (int k = 0; k < n2; ++k)
            Y[k] = D[k] * R[k] * isqpi * 3.0 / 8.0 * sqrt(35.0) *
                   (3.0 * x2 * y - y3) * z[k] * ir2[k] * ir2[k];
        else if (m == -4)
          for (int k = 0; k < n2; ++k)
            Y[k] = D[k] * R[k] * isqpi * 3.0 / 16.0 * sqrt(35.0 / 2.0) *
                   (4.0 * x * (x2 * y - y3)) * ir2[k] * ir2[k];
      }
    }
  }
  // Traps whether the user entered an illegal (ell,m)
  assert((*ylm)[0] != 123456.0);
}

#endif  // SPHERICAL_HARMONICS_H
