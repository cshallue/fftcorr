from fftcorr.array.numpy_adaptor cimport as_numpy
from fftcorr.grid cimport ConfigSpaceGrid
from fftcorr.histogram cimport HistogramList

# TODO: can numpy_adaptor take care of this? address boundary errors without it
cimport numpy as cnp
cnp.import_array()

cdef class Correlator:
    def __cinit__(self,
                  ConfigSpaceGrid dens,
                  Float rmax,
                  Float dr,
                  Float kmax,
                  Float dk,
                  int maxell,
                  ConfigSpaceGrid dens2 = None,
                  # TODO: use enum?
                  unsigned fftw_flags = 0):
        if dens2 is None:
            dens2 = dens
        # TODO: else check dimensions, etc, match

        self._correlator_cc = new Correlator_cc(
            dens.cc_grid()[0], dens2.cc_grid()[0], rmax, dr, kmax, dk, maxell,
            fftw_flags)
        self._correlation_r = as_numpy(self._correlator_cc.correlation_r())
        self._correlation_counts = as_numpy(self._correlator_cc.correlation_counts())
        self._correlation_histogram = as_numpy(self._correlator_cc.correlation_histogram())
        self._power_spectrum_k = as_numpy(self._correlator_cc.power_spectrum_k())
        self._power_spectrum_counts = as_numpy(self._correlator_cc.power_spectrum_counts())
        self._power_spectrum_histogram = as_numpy(self._correlator_cc.power_spectrum_histogram())

    def correlate_periodic(self):
        self._correlator_cc.correlate_periodic()

    def correlate_nonperiodic(self, int wide_angle_exponent = 0):
        self._correlator_cc.correlate_nonperiodic(wide_angle_exponent)

    @property
    def correlation_r(self):
        return self._correlation_r

    @property
    def correlation_counts(self):
        return self._correlation_counts

    @property
    def correlation_histogram(self):
        return self._correlation_histogram

    @property
    def power_spectrum_k(self):
        return self._power_spectrum_k

    @property
    def power_spectrum_counts(self):
        return self._power_spectrum_counts

    @property
    def power_spectrum_histogram(self):
        return self._power_spectrum_histogram