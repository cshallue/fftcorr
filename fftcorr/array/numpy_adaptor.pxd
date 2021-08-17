from fftcorr.array.row_major_array cimport RowMajorArrayPtr

cimport numpy as cnp

cdef extern from "numpy_adaptor.h":
  cnp.ndarray copy_to_numpy[dtype, N](const RowMajorArrayPtr[dtype, N]& arr)
  cnp.ndarray as_numpy[dtype, N](RowMajorArrayPtr[dtype, N]& arr)
  cnp.ndarray as_numpy[dtype, N](const RowMajorArrayPtr[dtype, N]& arr)
  RowMajorArrayPtr[dtype, N] as_RowMajorArrayPtr[dtype, N](cnp.ndarray arr) except *