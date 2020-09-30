#ifndef READ_GALAXIES_H
#define READ_GALAXIES_H

#include <assert.h>

#include "grid.h"
#include "types.h"

struct CatalogHeader {
  Float max_sep;
  Float posmin[3];
  Float posmax[3];
};

CatalogHeader read_header(const char filename[]) {
  CatalogHeader h;
  // Read posmin_[3], posmax_[3], max_sep_, blank8;
  FILE *fp = fopen(filename, "rb");
  assert(fp != NULL);
  double buf[8];
  int nread = fread(buf, sizeof(double), 8, fp);
  assert(nread == 8);
  fclose(fp);

  Float TOOBIG = 1e10;
  for (int i = 0; i < 7; i++) {
    assert(fabs(buf[i]) < TOOBIG);
  }

  h.posmin[0] = buf[0];
  h.posmin[1] = buf[1];
  h.posmin[2] = buf[2];
  h.posmax[0] = buf[3];
  h.posmax[1] = buf[4];
  h.posmax[2] = buf[5];
  h.max_sep = buf[6];
  // buf[7] not used, just for alignment.

  assert(h.max_sep >= 0);

  fprintf(stderr, "posmin = [%.4f, %.4f, %.4f]\n", h.posmin[0], h.posmin[1],
          h.posmin[2]);
  fprintf(stderr, "posmax = [%.4f, %.4f, %.4f]\n", h.posmax[0], h.posmax[1],
          h.posmax[2]);
  fprintf(stderr, "max_sep = %f\n", h.max_sep);

  return h;
}

#endif  // READ_GALAXIES_H