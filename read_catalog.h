#ifndef READ_GALAXIES_H
#define READ_GALAXIES_H

#include <assert.h>

#include "grid.h"
#include "types.h"

class CatalogHeader {
 public:
  Float max_sep_;
  Float posmin_[3];
  Float posmax_[3];

  void read(const char filename[]) {
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

    posmin_[0] = buf[0];
    posmin_[1] = buf[1];
    posmin_[2] = buf[2];
    posmax_[0] = buf[3];
    posmax_[1] = buf[4];
    posmax_[2] = buf[5];
    max_sep_ = buf[6];
    // buf[7] not used, just for alignment.

    assert(max_sep_ >= 0);

    fprintf(stderr, "posmin = [%.4f, %.4f, %.4f]", posmin_[0], posmin_[1],
            posmin_[2]);
    fprintf(stderr, "posmax = [%.4f, %.4f, %.4f]", posmax_[0], posmax_[1],
            posmax_[2]);
    fprintf(stderr, "max_sep_ = %f\n", max_sep_);
  }
};

#endif  // READ_GALAXIES_H