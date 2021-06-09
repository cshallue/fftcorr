#ifndef FFT_GRID_H
#define FFT_GRID_H

#include <fftw3.h>

#include <array>

#include "../array/row_major_array.h"
#include "../profiling/timer.h"
#include "../types.h"

// Manager of in-place fast Fourier transforms of real data.
class FftGrid {
 public:
  FftGrid(std::array<int, 3> shape);
  ~FftGrid();

  void plan_fft(unsigned flags);  // Must be called before exectute_[i]fft()
  bool fft_ready();
  void execute_fft();
  void execute_ifft();

  void extract_submatrix(RowMajorArrayPtr<Float, 3>* out,
                         const RowMajorArrayPtr<Float, 3>* mult = NULL) const;
  void extract_fft_submatrix(RowMajorArrayPtr<Float, 3>* out,
                             const RowMajorArrayPtr<Float, 3>* mult = NULL) const;

  RowMajorArrayPtr<Float, 3>& as_real_array() { return grid_; }
  const RowMajorArrayPtr<Float, 3>& as_real_array() const { return grid_; }
  RowMajorArrayPtr<Complex, 3>& as_complex_array() { return cgrid_; }
  const RowMajorArrayPtr<Complex, 3>& as_complex_array() const {
    return cgrid_;
  }

  Float setup_time() const { return setup_time_.elapsed_sec(); }
  Float plan_time() const { return plan_time_.elapsed_sec(); }
  Float fft_time() const { return fft_time_.elapsed_sec(); }
  Float extract_time() const { return extract_time_.elapsed_sec(); }
  Float convolve_time() const { return convolve_time_.elapsed_sec(); }

 private:
  std::array<int, 3> rshape_;  // Dimensions of the real-space input grid.
  std::array<int, 3> cshape_;  // Dimensions of the complex-space FFT output.

  RowMajorArray<Float, 3> grid_;  // Underlying data grid, which is padded.
  RowMajorArrayPtr<Complex, 3> cgrid_;  // Complex view of underlying data grid.

  Timer setup_time_;
  Timer plan_time_;
  Timer fft_time_;
  mutable Timer extract_time_;
  Timer convolve_time_;

#ifndef FFTSLAB
  fftw_plan fft_;
  fftw_plan ifft_;
#else
  fftw_plan fftx_;
  fftw_plan fftyz_;
  fftw_plan ifftyz_;
  fftw_plan ifftx_;

  Timer fftx_time_;
  Timer fftyz_time_;
#endif
};

#endif  // FFT_GRID_H