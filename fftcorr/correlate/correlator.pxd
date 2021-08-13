from libcpp cimport bool

cimport numpy as cnp

from fftcorr.grid cimport ConfigSpaceGrid_cc
from fftcorr.array cimport RowMajorArrayPtr
from fftcorr.types cimport Float, array, One, Two

cdef extern from "correlator.h":
  cdef cppclass PeriodicCorrelator_cc "PeriodicCorrelator":
    PeriodicCorrelator_cc(const ConfigSpaceGrid_cc& dens1,
                          const ConfigSpaceGrid_cc& dens2,
                          Float rmax,
                          Float dr,
                          Float kmax,
                          Float dk,
                          int maxell,
                          unsigned fftw_flags) except +
    void correlate()
    void correlate_nonperiodic(int wide_angle_exponent)
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
