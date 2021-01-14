#ifndef SURVEY_BOX_H
#define SURVEY_BOX_H

#include <assert.h>

#include <array>

#include "types.h"

class SurveyBox {
 public:
  SurveyBox() : posmin_({0, 0, 0}), posmax_({0, 0, 0}) {}

  void read_header(const char filename[]) {
    // Read posmin_[3], posmax_[3], max_sep_, blank8;
    FILE* fp = fopen(filename, "rb");
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
    // buf[6] and buf[7] not used, just for alignment.

    fprintf(stderr, "Reading survey box from header of %s\n", filename);
    fprintf(stderr, "posmin = [%.4f, %.4f, %.4f]\n", posmin_[0], posmin_[1],
            posmin_[2]);
    fprintf(stderr, "posmax = [%.4f, %.4f, %.4f]\n", posmax_[0], posmax_[1],
            posmax_[2]);
  }

  void pad_to_sep(Float sep) {
    // Expand the survey box to ensure a minimum separation.
    Float pad = sep / 1.5;
    for (int i = 0; i < 3; i++) {
      posmin_[i] -= pad;
      posmax_[i] += pad;
    }
  }

  const std::array<Float, 3>& posmin() const { return posmin_; }
  const std::array<Float, 3>& posmax() const { return posmax_; }

 private:
  std::array<Float, 3> posmin_;
  std::array<Float, 3> posmax_;
};

#endif  // SURVEY_BOX_H