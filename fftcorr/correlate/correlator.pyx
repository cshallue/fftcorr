from fftcorr.array.numpy_adaptor cimport copy_to_numpy, as_RowMajorArrayPtr
from fftcorr.histogram cimport HistogramList
from fftcorr.particle_mesh.window_type cimport CLOUD_IN_CELL
from fftcorr.grid cimport ConfigSpaceGrid

# TODO: can numpy_adaptor take care of this? address boundary errors without it
cimport numpy as cnp
cnp.import_array()

import numpy as np
import astropy.table

cdef class BaseCorrelator:
    def __cinit__(self):
        pass

    def __dealloc__(self):
        if self._correlator_cc is not NULL:
            del self._correlator_cc

    # @staticmethod
    # cdef BaseCorrelator from_ptr(BaseCorrelator_cc* correlator_cc):
    #     self._correlator_cc = correlator_cc

    cdef cnp.ndarray _correlation_r(self):
        return copy_to_numpy(self._correlator_cc.correlation_r())

    cdef cnp.ndarray _correlation_counts(self):
        return copy_to_numpy(self._correlator_cc.correlation_counts())

    cdef cnp.ndarray _correlation_histogram(self):
        return copy_to_numpy(self._correlator_cc.correlation_histogram())

    cdef cnp.ndarray _power_spectrum_k(self):
        return copy_to_numpy(self._correlator_cc.power_spectrum_k())

    cdef cnp.ndarray _power_spectrum_counts(self):
        return copy_to_numpy(self._correlator_cc.power_spectrum_counts())

    cdef cnp.ndarray _power_spectrum_histogram(self):
        return copy_to_numpy(self._correlator_cc.power_spectrum_histogram())

    def _correlation(self, squeeze=True):
        r = self._correlation_r()
        histogram = self._correlation_histogram()
        counts = self._correlation_counts()
        # TODO: we should really call xi something else, because it's only xi in
        # the periodic case. We do want it in the nonperiodic case though, as
        # it's what gets passed into compute_xi.
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
        # TODO: similar comment to xi above
        ps = histogram / counts
        if squeeze:
            ps = np.squeeze(ps)
            histogram = np.squeeze(histogram)
        return astropy.table.Table(
            data=(k, np.transpose(ps), np.transpose(histogram), counts[0, :]),
            names=("k", "ps", "histogram", "count"),
            copy=True)

    def set_grid2(self, grid2):
        # TODO: dtype should be set somewhere global. Here and below.
        grid2 = np.ascontiguousarray(grid2, dtype=np.float64)
        self._correlator_cc.set_grid2(as_RowMajorArrayPtr[Float, Three](grid2))

    def set_grid2_fft(self, grid2_fft):
        # TODO: dtype should be set somewhere global. Here and below.
        grid2_fft = np.ascontiguousarray(grid2_fft, dtype=np.complex128)
        self._correlator_cc.set_grid2_fft(as_RowMajorArrayPtr[Complex, Three](grid2_fft))

    def autocorrelate(self, grid, squeeze=True):
        # TODO: as_RowMajorArrayPtr currently raises an error if the argument is
        # not writeable, but in this case it could be since the Correlator takes
        # a const RowMajorArrayPtr.
        grid = np.ascontiguousarray(grid, dtype=np.float64)
        self._correlator_cc.autocorrelate(as_RowMajorArrayPtr[Float, Three](grid))
        return self._power_spectrum(squeeze), self._correlation(squeeze)

    def cross_correlate(self, grid1, grid2=None, squeeze=True):
        grid1 = np.ascontiguousarray(grid1, dtype=np.float64)
        if grid2 is None:
            self._correlator_cc.cross_correlate(as_RowMajorArrayPtr[Float, Three](grid1))
        else:
            grid2 = np.ascontiguousarray(grid2, dtype=np.float64)
            self._correlator_cc.cross_correlate(
                as_RowMajorArrayPtr[Float, Three](grid1),
                as_RowMajorArrayPtr[Float, Three](grid2))

        return self._power_spectrum(squeeze), self._correlation(squeeze)


cdef class PeriodicCorrelator(BaseCorrelator):
    def __cinit__(self,
                  shape,
                  Float cell_size,
                  WindowCorrection window_correct,
                  Float rmax,
                  Float dr,
                  Float kmax,
                  Float dk,
                  int maxell,
                  # TODO: use enum?
                  unsigned fftw_flags = 0):
        shape = np.ascontiguousarray(shape, dtype=np.intc)
        cdef cnp.ndarray[int, ndim=1, mode="c"] cshape = shape
        self._correlator_cc = <BaseCorrelator_cc *> new PeriodicCorrelator_cc(
            (<array[int, Three] *> &cshape[0])[0], cell_size, window_correct, rmax, dr, kmax, dk, maxell, fftw_flags)

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
                       WindowCorrection window_correct = NO_CORRECTION,
                       unsigned fftw_flags = 0):
        return cls(shape=grid.shape,
                   cell_size=grid.cell_size,
                   window_correct=window_correct,
                   rmax=rmax,
                   dr=dr,
                   kmax=kmax,
                   dk=dk,
                   maxell=maxell,
                   fftw_flags=fftw_flags)


cdef class Correlator(BaseCorrelator):
    def __cinit__(self,
                  shape,
                  Float cell_size,
                  posmin,
                  WindowCorrection window_correct,
                  Float rmax,
                  Float dr,
                  Float kmax,
                  Float dk,
                  int maxell,
                  # TODO: use enum?
                  unsigned fftw_flags = 0):
        shape = np.ascontiguousarray(shape, dtype=np.intc)
        cdef cnp.ndarray[int, ndim=1, mode="c"] cshape = shape
        posmin = np.ascontiguousarray(posmin, dtype=np.float64)
        cdef cnp.ndarray[Float, ndim=1, mode="c"] cposmin = posmin
        self._correlator_cc = <BaseCorrelator_cc *> new Correlator_cc(
            (<array[int, Three] *> &cshape[0])[0], cell_size, (<array[Float, Three] *> &cposmin[0])[0], window_correct, rmax, dr, kmax, dk, maxell, fftw_flags)

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
                       WindowCorrection window_correct = NO_CORRECTION,
                       unsigned fftw_flags = 0):
        return cls(shape=grid.shape,
                   cell_size=grid.cell_size,
                   posmin=grid.posmin,
                   window_correct=window_correct,
                   rmax=rmax,
                   dr=dr,
                   kmax=kmax,
                   dk=dk,
                   maxell=maxell,
                   fftw_flags=fftw_flags)