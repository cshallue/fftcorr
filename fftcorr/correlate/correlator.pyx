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

        self._periodic_correlator_cc = new PeriodicCorrelator_cc(
            dens.cc_grid()[0], dens2.cc_grid()[0], rmax, dr, kmax, dk, maxell,
            fftw_flags)
        # These are references to the internal C++ arrays and therefore should
        # be copied before being exposed to the user (they change with
        # subsequent correlation calls).
        self._correlation_r = as_numpy(self._periodic_correlator_cc.correlation_r())
        self._correlation_counts = as_numpy(self._periodic_correlator_cc.correlation_counts())
        self._correlation_histogram = as_numpy(self._periodic_correlator_cc.correlation_histogram())
        self._power_spectrum_k = as_numpy(self._periodic_correlator_cc.power_spectrum_k())
        self._power_spectrum_counts = as_numpy(self._periodic_correlator_cc.power_spectrum_counts())
        self._power_spectrum_histogram = as_numpy(self._periodic_correlator_cc.power_spectrum_histogram())

    def correlate(self):
        self._periodic_correlator_cc.correlate()

    def correlations(self):
        return astropy.table.Table(
            data=(
                self._correlation_r,
                self._correlation_counts[0, :],
                np.transpose(self._correlation_histogram),
            ),
            names=("r", "count", "histogram"),
            copy=True)

    def power_spectrum(self):
        return astropy.table.Table(
            data=(
                self._power_spectrum_k,
                self._power_spectrum_counts[0, :],
                np.transpose(self._power_spectrum_histogram),
            ),
            names=("k", "count", "histogram"),
            copy=True)