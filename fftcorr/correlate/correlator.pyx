from fftcorr.array.numpy_adaptor cimport as_numpy
from fftcorr.grid cimport ConfigSpaceGrid
from fftcorr.histogram cimport HistogramList

# TODO: can numpy_adaptor take care of this? address boundary errors without it
cimport numpy as cnp
cnp.import_array()

import numpy as np
import astropy.table

cdef class PeriodicCorrelator:
    def __cinit__(self,
                  shape,
                  Float rmax,
                  Float dr,
                  Float kmax,
                  Float dk,
                  int maxell,
                  ConfigSpaceGrid dens2 = None,
                  # TODO: use enum?
                  unsigned fftw_flags = 0):
        shape = np.ascontiguousarray(shape, dtype=np.intc)
        cdef cnp.ndarray[int, ndim=1, mode="c"] cshape = shape

        self._periodic_correlator_cc = new PeriodicCorrelator_cc(
            (<array[int, Three] *> &cshape[0])[0], rmax, dr, kmax, dk, maxell, fftw_flags)
        # These are references to the internal C++ arrays and therefore should
        # be copied before being exposed to the user (they change with
        # subsequent correlation calls).
        self._correlation_r = as_numpy(self._periodic_correlator_cc.correlation_r())
        self._correlation_counts = as_numpy(self._periodic_correlator_cc.correlation_counts())
        self._correlation_histogram = as_numpy(self._periodic_correlator_cc.correlation_histogram())
        self._power_spectrum_k = as_numpy(self._periodic_correlator_cc.power_spectrum_k())
        self._power_spectrum_counts = as_numpy(self._periodic_correlator_cc.power_spectrum_counts())
        self._power_spectrum_histogram = as_numpy(self._periodic_correlator_cc.power_spectrum_histogram())

    def set_dens2(self, ConfigSpaceGrid dens2):
        self._periodic_correlator_cc.set_dens2(dens2.cc_grid()[0])

    def autocorrelate(self, ConfigSpaceGrid dens):
        self._periodic_correlator_cc.autocorrelate(dens.cc_grid()[0])

    def cross_correlate(self, ConfigSpaceGrid dens1, ConfigSpaceGrid dens2=None):
        if dens2 is None:
            self._periodic_correlator_cc.cross_correlate(dens1.cc_grid()[0])
        else:
            self._periodic_correlator_cc.cross_correlate(dens1.cc_grid()[0], dens2.cc_grid()[0])

    def correlations(self, squeeze=True):
        r = self._correlation_r
        xi = np.transpose(
            self._correlation_histogram / self._correlation_counts)
        histogram = np.transpose(self._correlation_histogram)
        count = self._correlation_counts[0, :]
        if squeeze:
            xi = np.squeeze(xi)
            histogram = np.squeeze(histogram)
        return astropy.table.Table(
            data=(r, xi, histogram, count),
            names=("r", "xi", "histogram", "count"),
            copy=True)

    def power_spectrum(self, squeeze=True):
        k = self._power_spectrum_k
        ps = np.transpose(
            self._power_spectrum_histogram / self._power_spectrum_counts)
        histogram = np.transpose(self._power_spectrum_histogram)
        count = self._power_spectrum_counts[0, :]
        if squeeze:
            ps = np.squeeze(ps)
            histogram = np.squeeze(histogram)
        return astropy.table.Table(
            data=(k, ps, histogram, count),
            names=("k", "ps", "histogram", "count"),
            copy=True)