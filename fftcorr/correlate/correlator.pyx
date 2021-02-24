from fftcorr.grid cimport ConfigSpaceGrid
from fftcorr.histogram cimport HistogramList

cdef class Correlator:
    def __cinit__(self, ConfigSpaceGrid dens, Float rmax, Float kmax):
        self._correlator_cc = new Correlator_cc(dens.cc_grid()[0], rmax, kmax)

    def correlate_iso(self, int maxell, HistogramList h, HistogramList kh):
        cdef Float zerolag = -1234.00
        self._correlator_cc.correlate_iso(
            maxell, h.cc_hist_list()[0], kh.cc_hist_list()[0], zerolag)
        return zerolag

    def correlate_aniso(self,
                        int maxell,
                        HistogramList h,
                        HistogramList kh,
                        int wide_angle_exponent = 0,
                        bool periodic = 0):
        cdef Float zerolag = -1234.00
        self._correlator_cc.correlate_aniso(
            maxell, wide_angle_exponent, periodic, h.cc_hist_list()[0],
            kh.cc_hist_list()[0], zerolag)
        return zerolag