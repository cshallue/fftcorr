from fftcorr.array.row_major_array cimport RowMajorArrayPtr

cimport numpy as cnp

cdef extern from "numpy_adaptor.h":
  cnp.ndarray as_numpy[dtype, N](RowMajorArrayPtr[dtype, N]& c_arr)
  cnp.ndarray as_numpy[dtype, N](const RowMajorArrayPtr[dtype, N]& c_arr)
