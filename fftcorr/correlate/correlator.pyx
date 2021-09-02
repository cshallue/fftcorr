from fftcorr.array.numpy_adaptor cimport copy_to_numpy, as_RowMajorArrayPtr
from fftcorr.histogram cimport HistogramList
from fftcorr.particle_mesh.window_type cimport WindowType
from fftcorr.grid cimport ConfigSpaceGrid

# TODO: can numpy_adaptor take care of this? address boundary errors without it
cimport numpy as cnp
cnp.import_array()

import numpy as np
import astropy.table

cdef class PeriodicCorrelator:
    def __cinit__(self,
                  shape,
                  Float cell_size,
                  WindowType window_type,
                  Float rmax,
                  Float dr,
                  Float kmax,
                  Float dk,
                  int maxell,
                  # TODO: use enum?
                  unsigned fftw_flags = 0):
        shape = np.ascontiguousarray(shape, dtype=np.intc)
        cdef cnp.ndarray[int, ndim=1, mode="c"] cshape = shape

        self._periodic_correlator_cc = new PeriodicCorrelator_cc(
            (<array[int, Three] *> &cshape[0])[0], cell_size, window_type, rmax, dr, kmax, dk, maxell, fftw_flags)
        # These are references to the internal C++ arrays and therefore should
        # be copied before being exposed to the user (they change with
        # subsequent correlation calls).

    def __dealloc__(self):
        del self._periodic_correlator_cc

    # TODO: this method implies that we might want to have a ConfigSpaceGridSpec
    # class and the user would pass in grid.spec.
    @classmethod
    def from_grid_spec(cls,
                       ConfigSpaceGrid grid,
                       Float rmax,
                       Float dr,
                       Float kmax,
                       Float dk,
                       int maxell,
                       unsigned fftw_flags = 0):
        return cls(shape=grid.shape,
                cell_size=grid.cell_size,
                window_type=grid.window_type,
                rmax=rmax,
                dr=dr,
                kmax=kmax,
                dk=dk,
                maxell=maxell,
                fftw_flags=fftw_flags)

    cdef cnp.ndarray _correlation_r(self):
        return copy_to_numpy(self._periodic_correlator_cc.correlation_r())

    cdef cnp.ndarray _correlation_counts(self):
        return copy_to_numpy(self._periodic_correlator_cc.correlation_counts())

    cdef cnp.ndarray _correlation_histogram(self):
        return copy_to_numpy(self._periodic_correlator_cc.correlation_histogram())

    cdef cnp.ndarray _power_spectrum_k(self):
        return copy_to_numpy(self._periodic_correlator_cc.power_spectrum_k())

    cdef cnp.ndarray _power_spectrum_counts(self):
        return copy_to_numpy(self._periodic_correlator_cc.power_spectrum_counts())

    cdef cnp.ndarray _power_spectrum_histogram(self):
        return copy_to_numpy(self._periodic_correlator_cc.power_spectrum_histogram())

    def _correlation(self, squeeze=True):
        r = self._correlation_r()
        histogram = self._correlation_histogram()
        counts = self._correlation_counts()
        xi = histogram / counts
        if squeeze:
            xi = np.squeeze(xi)
            histogram = np.squeeze(histogram)
        return astropy.table.Table(
            data=(r, np.transpose(xi), np.transpose(histogram), counts[0, :]),
            names=("r", "xi", "histogram", "count"),
            copy=True)

    def _power_spectrum(self, squeeze=True):
        k = self._power_spectrum_k()
        histogram = self._power_spectrum_histogram()
        counts = self._power_spectrum_counts()
        ps = histogram / counts
        if squeeze:
            ps = np.squeeze(ps)
            histogram = np.squeeze(histogram)
        return astropy.table.Table(
            data=(k, np.transpose(ps), np.transpose(histogram), counts[0, :]),
            names=("k", "ps", "histogram", "count"),
            copy=True)

    def set_dens2(self, dens2):
        # TODO: dtype should be set somewhere global. Here and below.
        dens2 = np.ascontiguousarray(dens2, dtype=np.float64)
        self._periodic_correlator_cc.set_dens2(as_RowMajorArrayPtr[Float, Three](dens2))

    def autocorrelate(self, dens, squeeze=True):
        # TODO: as_RowMajorArrayPtr currently raises an error if the argument is
        # not writeable, but in this case it could be since the Correlator takes
        # a const RowMajorArrayPtr.
        dens = np.ascontiguousarray(dens, dtype=np.float64)
        self._periodic_correlator_cc.autocorrelate(as_RowMajorArrayPtr[Float, Three](dens))
        return self._power_spectrum(squeeze), self._correlation(squeeze)

    def cross_correlate(self, dens1, dens2=None, squeeze=True):
        dens1 = np.ascontiguousarray(dens1, dtype=np.float64)
        if dens2 is None:
            self._periodic_correlator_cc.cross_correlate(as_RowMajorArrayPtr[Float, Three](dens1))
        else:
            dens2 = np.ascontiguousarray(dens2, dtype=np.float64)
            self._periodic_correlator_cc.cross_correlate(
                as_RowMajorArrayPtr[Float, Three](dens1),
                as_RowMajorArrayPtr[Float, Three](dens2))

        return self._power_spectrum(squeeze), self._correlation(squeeze)
