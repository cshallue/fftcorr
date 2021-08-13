from fftcorr.types cimport Float, array, Three

from libcpp cimport bool

cimport numpy as cnp

from fftcorr.grid cimport ConfigSpaceGrid_cc
from fftcorr.array cimport RowMajorArrayPtr
from fftcorr.types cimport Float, array, One, Two

cdef extern from "correlator.h":
  cdef cppclass PeriodicCorrelator_cc "PeriodicCorrelator":
    PeriodicCorrelator_cc(const array[int, Three]& shape,
                          Float rmax,
                          Float dr,
                          Float kmax,
                          Float dk,
                          int maxell,
                          unsigned fftw_flags) except +
    void set_dens2(const ConfigSpaceGrid_cc& dens2)
    void autocorrelate(const ConfigSpaceGrid_cc& dens1)
    void cross_correlate(const ConfigSpaceGrid_cc& dens1,
                         const ConfigSpaceGrid_cc& dens2)
    Float zerolag()
    const RowMajorArrayPtr[Float, One]& correlation_r()
    const RowMajorArrayPtr[int, Two]& correlation_counts()
    const RowMajorArrayPtr[Float, Two]& correlation_histogram()
    const RowMajorArrayPtr[Float, One]& power_spectrum_k()
    const RowMajorArrayPtr[int, Two]& power_spectrum_counts()
    const RowMajorArrayPtr[Float, Two]& power_spectrum_histogram()

cdef class PeriodicCorrelator:
    cdef PeriodicCorrelator_cc* _periodic_correlator_cc
    cdef cnp.ndarray _correlation_r
    cdef cnp.ndarray _correlation_counts
    cdef cnp.ndarray _correlation_histogram
    cdef cnp.ndarray _power_spectrum_k
    cdef cnp.ndarray _power_spectrum_counts
    cdef cnp.ndarray _power_spectrum_histogram
