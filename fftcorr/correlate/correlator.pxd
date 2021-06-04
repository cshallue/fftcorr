from libcpp cimport bool

cimport numpy as cnp

from fftcorr.grid cimport ConfigSpaceGrid_cc
from fftcorr.array cimport RowMajorArrayPtr
from fftcorr.types cimport Float, array, One, Two

cdef extern from "correlator.h":
  cdef cppclass Correlator_cc "Correlator":
    Correlator_cc(const ConfigSpaceGrid_cc& dens,
                  Float rmax,
                  Float dr,
                  Float kmax,
                  Float dk,
                  int maxell,
                  unsigned fftw_flags) except +
    void correlate_periodic()
    void correlate_nonperiodic(int wide_angle_exponent)
    Float zerolag()
    const RowMajorArrayPtr[Float, One]& correlation_r()
    const RowMajorArrayPtr[int, Two]& correlation_counts()
    const RowMajorArrayPtr[Float, Two]& correlation_histogram()
    const RowMajorArrayPtr[Float, One]& power_spectrum_k()
    const RowMajorArrayPtr[int, Two]& power_spectrum_counts()
    const RowMajorArrayPtr[Float, Two]& power_spectrum_histogram()

cdef class Correlator:
    cdef Correlator_cc* _correlator_cc
    cdef cnp.ndarray _correlation_r
    cdef cnp.ndarray _correlation_counts
    cdef cnp.ndarray _correlation_histogram
    cdef cnp.ndarray _power_spectrum_k
    cdef cnp.ndarray _power_spectrum_counts
    cdef cnp.ndarray _power_spectrum_histogram
