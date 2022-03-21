from fftcorr.types cimport Float, array, Three

from libcpp cimport bool

cimport numpy as cnp

from fftcorr.array cimport RowMajorArrayPtr
from fftcorr.types cimport Float, Complex, array, One, Two, Three

cdef extern from "correlator.h":
  cdef cppclass BaseCorrelator_cc "BaseCorrelator":
    BaseCorrelator_cc(const array[int, Three]& shape,
                      Float cell_size,
                      int window_type,  # TODO: enum type?
                      Float rmax,
                      Float dr,
                      Float kmax,
                      Float dk,
                      int maxell,
                      unsigned fftw_flags) except +
    void set_dens2(const RowMajorArrayPtr[Float, Three]& dens2)
    void set_dens2_fft(const RowMajorArrayPtr[Complex, Three]& dens2_fft)
    void autocorrelate(const RowMajorArrayPtr[Float, Three]& dens1)
    void cross_correlate(const RowMajorArrayPtr[Float, Three]& dens1)
    void cross_correlate(const RowMajorArrayPtr[Float, Three]& dens1,
                         const RowMajorArrayPtr[Float, Three]& dens2)
    Float zerolag()
    const RowMajorArrayPtr[Float, One]& correlation_r()
    const RowMajorArrayPtr[int, Two]& correlation_counts()
    const RowMajorArrayPtr[Float, Two]& correlation_histogram()
    const RowMajorArrayPtr[Float, One]& power_spectrum_k()
    const RowMajorArrayPtr[int, Two]& power_spectrum_counts()
    const RowMajorArrayPtr[Float, Two]& power_spectrum_histogram()

  cdef cppclass PeriodicCorrelator_cc "PeriodicCorrelator":
    PeriodicCorrelator_cc(const array[int, Three]& shape,
                          Float cell_size,
                          int window_type,  # TODO: enum type?
                          Float rmax,
                          Float dr,
                          Float kmax,
                          Float dk,
                          int maxell,
                          unsigned fftw_flags) except +

  cdef cppclass Correlator_cc "Correlator":
    Correlator_cc(const array[int, Three]& shape,
                  Float cell_size,
                  const array[Float, Three]& posmin,
                  int window_type,  # TODO: enum type?
                  Float rmax,
                  Float dr,
                  Float kmax,
                  Float dk,
                  int maxell,
                  unsigned fftw_flags) except +

cdef class BaseCorrelator:
    cdef BaseCorrelator_cc *_correlator_cc
    cdef cnp.ndarray _correlation_r(self)
    cdef cnp.ndarray _correlation_counts(self)
    cdef cnp.ndarray _correlation_histogram(self)
    cdef cnp.ndarray _power_spectrum_k(self)
    cdef cnp.ndarray _power_spectrum_counts(self)
    cdef cnp.ndarray _power_spectrum_histogram(self)

cdef class PeriodicCorrelator(BaseCorrelator):
  pass

cdef class Correlator(BaseCorrelator):
  pass