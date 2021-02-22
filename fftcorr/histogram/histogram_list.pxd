from fftcorr.array cimport RowMajorArrayPtr
from fftcorr.types cimport Float, array, One, Two
cimport numpy as cnp

cdef extern from "histogram_list.h":
  cdef cppclass HistogramList_cc "HistogramList":
    HistogramList_cc(int n, Float minval, Float maxval, Float binsize) except +
    int nbins()
    RowMajorArrayPtr[Float, One]& bins()
    RowMajorArrayPtr[int, Two]& counts()
    RowMajorArrayPtr[Float, Two]& hist_values()

cdef class HistogramList:
    cdef HistogramList_cc* _cc_hist_list
    cdef cnp.ndarray _bins
    cdef cnp.ndarray _counts
    cdef cnp.ndarray _hist_values

    cdef HistogramList_cc* cc_hist_list(self)
