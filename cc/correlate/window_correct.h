#ifndef WINDOW_CORRECT_H
#define WINDOW_CORRECT_H

#include <cmath>

#include "../types.h"

enum WindowCorrection {
  kNoCorrection = 0,
  kTscCorrection = 1,
  kTscAliasedCorrection = 2,
};

Float sinc(Float x) {
  if (x == 0) return 1;
  return sin(x) / x;
}

// This is the squared norm of the Fourier transform of the window function.
class WindowSquaredNorm {
 public:
  virtual ~WindowSquaredNorm() {}
  virtual Float evaluate(Float k, Float cell_size) = 0;
};

class UnitWindow : public WindowSquaredNorm {
  Float evaluate(Float k, Float cell_size) { return 1.0; }
};

class TscWindow : public WindowSquaredNorm {
  Float evaluate(Float k, Float cell_size) {
    Float w = sinc(k * cell_size / 2.0);
    return pow(w, 6);
  }
};

class TscAliasedWindow : public WindowSquaredNorm {
  Float evaluate(Float k, Float cell_size) {
    // The square window is 1-sin^2(kL/2)+2/15*sin^4(kL/2)
    Float sinsq = sin(k * cell_size / 2.0);
    sinsq *= sinsq;
    return 1 - sinsq + 2.0 / 15.0 * sinsq * sinsq;
  }
};

std::unique_ptr<WindowSquaredNorm> make_window_function(WindowCorrection type) {
  WindowSquaredNorm* func = NULL;
  switch (type) {
    case kNoCorrection:
      func = new UnitWindow();
      break;
    case kTscCorrection:
      func = new TscWindow();
      break;
    case kTscAliasedCorrection:
      func = new TscAliasedWindow();
      break;
  }
  return std::unique_ptr<WindowSquaredNorm>(func);
}

#endif  // WINDOW_CORRECT_H