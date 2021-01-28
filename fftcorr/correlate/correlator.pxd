from fftcorr.grid cimport ConfigSpaceGrid_cc
from fftcorr.histogram cimport Histogram_cc
from fftcorr.types cimport Float, array, Three

cdef extern from "correlator.h":
  cdef cppclass Correlator_cc "Correlator":
    Correlator_cc(const ConfigSpaceGrid_cc& dens, Float rmax, Float kmax) except +
    void correlate_iso(Histogram_cc &h, Histogram_cc &kh, Float &zerolag)
    void correlate_aniso(
      int maxell,
      int wide_angle_exponent,
      const array[Float, Three] &observer,
      Histogram_cc &h,
      Histogram_cc &kh,
      Float &zerolag)

cdef class Correlator:
    cdef Correlator_cc* _correlator_cc
