#ifndef SPHERICAL_HARMONICS_H
#define SPHERICAL_HARMONICS_H

#include <assert.h>

#include <array>

#include "types.h"

/* ============== Spherical Harmonic routine ============== */

void makeYlm(RowMajorArrayPtr<Float> *Ylm, int ell, int m,
             const std::array<int, 3> &n, const Array1D &xcell,
             const Array1D &ycell, const Array1D &zcell,
             const RowMajorArrayPtr<Float> *mult, int exponent) {
  // TODO: check dimensions

  // We're not actually returning Ylm here.
  // m>0 will return Re(Y_lm)*sqrt(2)
  // m<0 will return Im(Y_l|m|)*sqrt(2)
  // m=0 will return Y_l0
  // These are not including common minus signs, since we're only
  // using them in matched squares.
  //
  // Input x[n[0]], y[n[1]], z[n[2]] are the x,y,z centers of this row of bins
  //
  // If mult!=NULL, then it should point to a [n[0]][n[1]][n2] vector that will
  // be multiplied element-wise onto the results.  This can save a store/load to
  // main memory.
  //
  // If exponent!=0, then we will attach a dependence of r^exponent to the Ylm's
  // exponent must be an even number
  assert(exponent % 2 == 0);
  // YlmTime.Start();
  Float isqpi = sqrt(1.0 / M_PI);
  if (m != 0) isqpi *= sqrt(2.0);  // Do this up-front, so we don't forget
  Float tiny = 1e-20;

  const int cn2 = n[2];  // To help with loop vectorization

  if (ell == 0 && m == 0 && exponent == 0) {
    // This case is so easy that we'll do it directly and skip the setup.
    Float value = 1.0 / sqrt(4.0 * M_PI);
#pragma omp parallel for YLM_SCHEDULE
    for (uint64 i = 0; i < n[0]; i++) {
      if (mult) {
        Float *Y;
        const Float *D;
        for (int j = 0; j < n[1]; j++) {
          Y = Ylm->get_row(i, j);
          D = mult->get_row(i, j);
          for (int k = 0; k < cn2; k++) Y[k] = D[k] * value;
        }
      } else {
        // TODO: if I end up with set_all in RowMajorArray, just call that.
        Float *Y;
        for (int j = 0; j < n[1]; j++) {
          Y = Ylm->get_row(i, j);
          for (int k = 0; k < cn2; k++) Y[k] = value;
        }
      }
    }
    // YlmTime.Stop();
    return;
  }

  const Float *z = zcell.data();
  Float *z2, *z3, *z4, *ones;
  int err = posix_memalign((void **)&z2, PAGE, sizeof(Float) * n[2] + PAGE);
  assert(err == 0);
  err = posix_memalign((void **)&z3, PAGE, sizeof(Float) * n[2] + PAGE);
  assert(err == 0);
  err = posix_memalign((void **)&z4, PAGE, sizeof(Float) * n[2] + PAGE);
  assert(err == 0);
  err = posix_memalign((void **)&ones, PAGE, sizeof(Float) * n[2] + PAGE);
  assert(err == 0);
  for (int k = 0; k < cn2; k++) {
    z2[k] = z[k] * z[k];
    z3[k] = z2[k] * z[k];
    z4[k] = z3[k] * z[k];
    ones[k] = 1.0;
  }

  Ylm->at(0, 0, 0) = -123456.0;  // A sentinal value

#pragma omp parallel for YLM_SCHEDULE
  for (uint64 i = 0; i < n[0]; i++) {
    // Ylm_count.add();
    Float *ir2;  // Need some internal workspace
    err = posix_memalign((void **)&ir2, PAGE, sizeof(Float) * n[2] + PAGE);
    assert(err == 0);
    Float *rpow;
    err = posix_memalign((void **)&rpow, PAGE, sizeof(Float) * n[2] + PAGE);
    assert(err == 0);
    Float x = xcell[i], x2 = x * x;
    Float *R;
    Float *Y;
    const Float *D;
    for (int j = 0; j < n[1]; j++) {
      // Important to use .at() for when Ylm and mult are internally padded.
      Y = Ylm->get_row(i, j);                           // (i, j, 0)
      D = (mult == NULL) ? ones : mult->get_row(i, j);  // (i, j, 0)
      Float y = ycell[j], y2 = y * y, y3 = y2 * y, y4 = y3 * y;
      for (int k = 0; k < cn2; k++) ir2[k] = 1.0 / (x2 + y2 + z2[k] + tiny);
      // Now figure out the exponent r^n
      if (exponent == 0)
        R = ones;
      else if (exponent > 0) {
        // Fill R with r^exponent
        R = rpow;
        for (int k = 0; k < cn2; k++) R[k] = (x2 + y2 + z2[k]);
        for (int e = exponent; e > 2; e -= 2)
          for (int k = 0; k < cn2; k++) R[k] *= (x2 + y2 + z2[k]);
      } else {
        // Fill R with r^exponent
        R = rpow;
        for (int k = 0; k < cn2; k++) R[k] = ir2[k];
        for (int e = exponent; e < -2; e += 2)
          for (int k = 0; k < cn2; k++) R[k] *= ir2[k];
      }
      // Now ready to compute
      if (ell == 2) {
        if (m == 2)
          for (int k = 0; k < cn2; k++)
            Y[k] = D[k] * R[k] * isqpi * sqrt(15. / 32.) * (x2 - y2) * ir2[k];
        else if (m == 1)
          for (int k = 0; k < cn2; k++)
            Y[k] = D[k] * R[k] * isqpi * sqrt(15. / 8.) * x * z[k] * ir2[k];
        else if (m == 0)
          for (int k = 0; k < cn2; k++)
            Y[k] = D[k] * R[k] * isqpi * sqrt(5. / 16.) *
                   (2.0 * z2[k] - x2 - y2) * ir2[k];
        else if (m == -1)
          for (int k = 0; k < cn2; k++)
            Y[k] = D[k] * R[k] * isqpi * sqrt(15. / 8.) * y * z[k] * ir2[k];
        else if (m == -2)
          for (int k = 0; k < cn2; k++)
            Y[k] = D[k] * R[k] * isqpi * sqrt(15. / 32.) * 2.0 * x * y * ir2[k];
      } else if (ell == 4) {
        if (m == 4)
          for (int k = 0; k < cn2; k++)
            Y[k] = D[k] * R[k] * isqpi * 3.0 / 16.0 * sqrt(35. / 2.) *
                   (x2 * x2 - 6.0 * x2 * y2 + y4) * ir2[k] * ir2[k];
        else if (m == 3)
          for (int k = 0; k < cn2; k++)
            Y[k] = D[k] * R[k] * isqpi * 3.0 / 8.0 * sqrt(35.) *
                   (x2 - 3.0 * y2) * z[k] * x * ir2[k] * ir2[k];
        else if (m == 2)
          for (int k = 0; k < cn2; k++)
            Y[k] = D[k] * R[k] * isqpi * 3.0 / 8.0 * sqrt(5. / 2.) *
                   (6.0 * z2[k] * (x2 - y2) - x2 * x2 + y4) * ir2[k] * ir2[k];
        else if (m == 1)
          for (int k = 0; k < cn2; k++)
            Y[k] = D[k] * R[k] * isqpi * 3.0 / 8.0 * sqrt(5.) * 3.0 *
                   (4.0 / 3.0 * z2[k] - x2 - y2) * x * z[k] * ir2[k] * ir2[k];
        else if (m == 0)
          for (int k = 0; k < cn2; k++)
            Y[k] = D[k] * R[k] * isqpi * 3.0 / 16.0 * 8.0 *
                   (z4[k] - 3.0 * z2[k] * (x2 + y2) +
                    3.0 / 8.0 * (x2 * x2 + 2.0 * x2 * y2 + y4)) *
                   ir2[k] * ir2[k];
        else if (m == -1)
          for (int k = 0; k < cn2; k++)
            Y[k] = D[k] * R[k] * isqpi * 3.0 / 8.0 * sqrt(5.) * 3.0 *
                   (4.0 / 3.0 * z2[k] - x2 - y2) * y * z[k] * ir2[k] * ir2[k];
        else if (m == -2)
          for (int k = 0; k < cn2; k++)
            Y[k] = D[k] * R[k] * isqpi * 3.0 / 8.0 * sqrt(5. / 2.) *
                   (2.0 * x * y * (6.0 * z2[k] - x2 - y2)) * ir2[k] * ir2[k];
        else if (m == -3)
          for (int k = 0; k < cn2; k++)
            Y[k] = D[k] * R[k] * isqpi * 3.0 / 8.0 * sqrt(35.) *
                   (3.0 * x2 * y - y3) * z[k] * ir2[k] * ir2[k];
        else if (m == -4)
          for (int k = 0; k < cn2; k++)
            Y[k] = D[k] * R[k] * isqpi * 3.0 / 16.0 * sqrt(35. / 2.) *
                   (4.0 * x * (x2 * y - y3)) * ir2[k] * ir2[k];
      } else if (ell == 0) {
        // We only get here if exponent!=0
        for (int k = 0; k < cn2; k++) Y[k] = D[k] * R[k] / sqrt(4.0 * M_PI);
      }
    }
    free(ir2);
  }
  // This traps whether the user entered an illegal (ell,m)
  assert(Ylm->at(0, 0, 0) != 123456.0);
  free(z2);
  free(z3);
  free(z4);
  // YlmTime.Stop();
  return;
}

#endif  // SPHERICAL_HARMONICS_H
