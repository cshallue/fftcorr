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
                  int maxell):
        self._correlator_cc = new Correlator_cc(
            dens.cc_grid()[0], rmax, dr, kmax, dk, maxell)
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

    def correlate_iso(self):
        self._correlator_cc.correlate_iso()

    def correlate_aniso(self, int wide_angle_exponent = 0, bool periodic = 0):
        self._correlator_cc.correlate_aniso(wide_angle_exponent, periodic)

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