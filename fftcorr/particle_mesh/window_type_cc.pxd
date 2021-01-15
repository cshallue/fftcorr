cdef extern from "window_functions.h":
  # TODO: cython is supposed to support C++ scoped enums as of
  # https://github.com/cython/cython/pull/3640/files, but my current cython
  # version (0.29.21) gives compile errors. Revisit in a later
  # cython version and make WindowType an enum class.
  # TODO: 
  ctypedef enum WindowType:
    NEAREST_CELL "WindowType::kNearestCell"
    CLOUD_IN_CELL "WindowType::kCloudInCell"
    WAVELET "WindowType::kWavelet"
