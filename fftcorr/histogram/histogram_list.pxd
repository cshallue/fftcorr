from fftcorr.array cimport Array, RowMajorArrayPtr
from fftcorr.types cimport Float, array, One, Two
cimport numpy as cnp

cdef extern from "histogram_list.h":
  cdef cppclass HistogramList_cc "HistogramList":
    HistogramList_cc(int n, Float minval, Float maxval, Float binsize) except +
    int nbins()
    const RowMajorArrayPtr[Float, One]& bins()
    const RowMajorArrayPtr[int, Two]& counts()
    const RowMajorArrayPtr[Float, Two]& hist_values()
    void accumulate(int ih, const Array[Float]& x, const Array[Float]& y)
    void reset()

cdef class HistogramList:
    cdef HistogramList_cc* _cc_hist_list
    cdef cnp.ndarray _bins
    cdef cnp.ndarray _counts
    cdef cnp.ndarray _hist_values

    cdef HistogramList_cc* cc_hist_list(self)
