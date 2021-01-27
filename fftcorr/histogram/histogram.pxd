from fftcorr.array cimport RowMajorArrayPtr
from fftcorr.types cimport Float, array, One, Two
cimport numpy as cnp

cdef extern from "histogram.h":
  cdef cppclass Histogram_cc "Histogram":
    Histogram_cc(int n, Float minval, Float maxval, Float binsize) except +
    int nbins()
    RowMajorArrayPtr[Float, One]& bins()
    RowMajorArrayPtr[int, One]& count()
    RowMajorArrayPtr[Float, Two]& accum()

cdef class Histogram:
    cdef Histogram_cc* _cc_hist
    cdef cnp.ndarray _bins
    cdef cnp.ndarray _count
    cdef cnp.ndarray _accum
