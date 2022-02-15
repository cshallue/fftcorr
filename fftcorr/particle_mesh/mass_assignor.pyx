from fftcorr.grid cimport ConfigSpaceGrid
from fftcorr.types cimport array
from fftcorr.array.numpy_adaptor cimport as_RowMajorArrayPtr

cimport numpy as cnp
cnp.import_array()

import numpy as np

from fftcorr.utils import Timer

# TODO: consider making MassAssignor a context manager, or wrapping it
# in a function that is a context manager
# https://book.pythontips.com/en/latest/context_managers.html
# when the context exits, flush() should be called. should it also
# deallocate the buffer?
cdef class MassAssignor:
    def __cinit__(self, ConfigSpaceGrid grid, bool periodic_wrap=False, int buffer_size=0):
        self._cc_ma = new MassAssignor_cc(grid.cc_grid()[0], periodic_wrap, buffer_size)
        

    # @staticmethod
    # cdef create(_ConfigSpaceGrid grid, WindowType window_type, int buffer_size):
    #     cdef _MassAssignor ma = _MassAssignor()
    #     ma.cc_ma = new MassAssignor_cc(grid.cc_grid, window_type, buffer_size)
    #     return ma

    # @staticmethod
    # def py_create(_ConfigSpaceGrid grid, WindowType window_type, int buffer_size):
    #     return _MassAssignor.create(grid, window_type, buffer_size)

    def __dealloc__(self):
        del self._cc_ma

    cpdef clear(self):
        self._cc_ma.clear()

    def add_particles(self, particles: np.ndarray, weight=None):
        self.add_particles_to_buffer(particles, weight)
        self.flush()

    def add_particles_to_buffer(self, particles: np.ndarray, weight=None):
        cdef RowMajorArrayPtr[Float, Two] pptr = as_RowMajorArrayPtr[Float, Two](particles)
        if particles.shape[1] == 4:
            if weight is not None:
                raise ValueError("Cannot pass weights twice")
            return self._cc_ma.add_particles_to_buffer(pptr)

        if particles.shape[1] != 3:
            raise ValueError(
                "Particles must have shape (n, 3) or (n, 4). "
                f"Got: {particles.shape}")

        if weight is None:
            weight = 1.0

        weight = np.asarray(weight)
        if not weight.shape:
            # Weight is a scalar.
            return self._cc_ma.add_particles_to_buffer(pptr, <Float> weight)

        # Weight is an array.
        if weight.shape != (particles.shape[0], ):
            raise ValueError(
                "Weight must be 1D with the same length as particles. "
                f"Got {weight.shape} and {particles.shape}")
        # TODO: turn this into a helper as_ArrayPtr1D, or wrap ArrayPtr1D separately?
        cdef RowMajorArrayPtr[Float, One] wptr = as_RowMajorArrayPtr[Float, One](weight)
        return self._cc_ma.add_particles_to_buffer(pptr, wptr)

    cpdef flush(self):
        self._cc_ma.flush()

    @property
    def num_added(self) -> int:
        return self._cc_ma.num_added()

    @property
    def num_skipped(self) -> int:
        return self._cc_ma.num_skipped()

    @property
    def totw(self) -> Float:
        return self._cc_ma.totw()

    @property
    def totwsq(self) -> Float:
        return self._cc_ma.totwsq()

    @property
    def sort_time(self) -> Float:
        return self._cc_ma.sort_time()

    @property
    def window_time(self) -> Float:
        return self._cc_ma.window_time()
