from fftcorr.array cimport as_const_numpy
from fftcorr.grid cimport ConfigSpaceGrid
from fftcorr.histogram cimport HistogramList

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
        self._correlation_r = as_const_numpy(
            1,
            self._correlator_cc.correlation_r().shape().data(),
            cnp.NPY_DOUBLE,
            self._correlator_cc.correlation_r().data(),
            self)
        self._correlation_counts = as_const_numpy(
            2,
            self._correlator_cc.correlation_counts().shape().data(),
            cnp.NPY_INT,
            self._correlator_cc.correlation_counts().data(),
            self)
        self._correlation_histogram = as_const_numpy(
            2,
            self._correlator_cc.correlation_histogram().shape().data(),
            cnp.NPY_DOUBLE,
            self._correlator_cc.correlation_histogram().data(),
            self)
        self._power_spectrum_k = as_const_numpy(
            1,
            self._correlator_cc.power_spectrum_k().shape().data(),
            cnp.NPY_DOUBLE,
            self._correlator_cc.power_spectrum_k().data(),
            self)
        self._power_spectrum_counts = as_const_numpy(
            2,
            self._correlator_cc.power_spectrum_counts().shape().data(),
            cnp.NPY_INT,
            self._correlator_cc.power_spectrum_counts().data(),
            self)
        self._power_spectrum_histogram = as_const_numpy(
            2,
            self._correlator_cc.power_spectrum_histogram().shape().data(),
            cnp.NPY_DOUBLE,
            self._correlator_cc.power_spectrum_histogram().data(),
            self)

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