#ifndef SURVEY_BOX_H
#define SURVEY_BOX_H

#include <assert.h>

#include <array>

#include "types.h"

class SurveyBox {
 public:
  SurveyBox() : max_sep_(0), posmin_({0, 0, 0}), posmax_({0, 0, 0}) {}

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
    max_sep_ = buf[6];
    // buf[7] not used, just for alignment.

    assert(max_sep_ >= 0);

    fprintf(stderr, "Reading survey box from header of %s\n", filename);
    fprintf(stderr, "posmin = [%.4f, %.4f, %.4f]\n", posmin_[0], posmin_[1],
            posmin_[2]);
    fprintf(stderr, "posmax = [%.4f, %.4f, %.4f]\n", posmax_[0], posmax_[1],
            posmax_[2]);
    fprintf(stderr, "max_sep = %f\n", max_sep_);
  }

  // If the user wants periodic BC, then we can ignore separation issues.
  void set_periodic_boundary() { max_sep_ = (posmax_[0] - posmin_[0]) * 100; }

  void ensure_sep(Float sep) {
    // Expand the survey box to ensure a minimum separation.
    if (sep <= max_sep_) return;
    Float extra_pad = sep - max_sep_;
    for (int i = 0; i < 3; i++) {
      posmax_[i] += extra_pad;
    }
    max_sep_ = sep;
  }

  Float max_sep() const { return max_sep_; }
  const std::array<Float, 3>& posmin() const { return posmin_; }
  const std::array<Float, 3>& posmax() const { return posmax_; }

 private:
  Float max_sep_;
  std::array<Float, 3> posmin_;
  std::array<Float, 3> posmax_;
};

#endif  // SURVEY_BOX_H