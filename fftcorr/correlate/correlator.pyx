from fftcorr.grid cimport ConfigSpaceGrid
from fftcorr.histogram cimport Histogram

cdef class Correlator:
    def __cinit__(self, ConfigSpaceGrid dens, Float rmax, Float kmax):
        self._correlator_cc = new Correlator_cc(dens.cc_grid()[0], rmax, kmax)

    def correlate_iso(self, Histogram h, Histogram kh):
        cdef Float zerolag
        self._correlator_cc.correlate_iso(
            h.cc_hist()[0], kh.cc_hist()[0], zerolag)
        return zerolag