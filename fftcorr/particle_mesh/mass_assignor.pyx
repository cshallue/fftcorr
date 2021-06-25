from fftcorr.grid cimport ConfigSpaceGrid
from fftcorr.types cimport array
from fftcorr.array.row_major_array cimport as_RowMajorArrayPtr1D, as_RowMajorArrayPtr2D, as_RowMajorArrayPtr4D

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
    def __cinit__(self, ConfigSpaceGrid grid, bool periodic_wrap=False, int buffer_size=10000, Float[:, :, :, :] disp=None):
        self._posmin = grid.posmin
        self._posmax = grid.posmax
        
        cdef cnp.ndarray[cnp.npy_int] disp_shape
        cdef const RowMajorArrayPtr[Float, Four]* disp_ptr = NULL
        if disp is not None:
            # Validate dimensions.
            # TODO: is the intc thing necessary?
            expected_shape = np.concatenate((grid.shape, (3,))).astype(np.intc)
            # TODO: unify this with disp_shape?
            actual_shape = np.array(disp.shape, dtype=np.intc)[:4]
            if (np.any(expected_shape != actual_shape)):
                raise ValueError(
                    f"Expected disp to have shape: {expected_shape}. Got: {actual_shape}")

            # Make a contiguous copy.
            # TODO: since we're making a copy here, the input array doesn't have
            # to be contiguous.
            print("Copying the displacement vector field")
            with Timer() as copy_timer:
                self._disp_data = disp.copy()
            print("Copy time: {:.2g} sec".format(copy_timer.elapsed))

            self._disp = as_RowMajorArrayPtr4D(self._disp_data)
            disp_ptr = &self._disp

        self._cc_ma = new MassAssignor_cc(grid.cc_grid(), periodic_wrap, buffer_size, disp_ptr)
        

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

    cpdef add_particles(self, Float[:, ::1] particles, weight=None):
        self.add_particles_to_buffer(particles, weight)
        self.flush()

    cpdef add_particles_to_buffer(self, Float[:, ::1] particles, weight=None):
        particles = np.asarray(particles, order="C")  # TODO: ensure no copy
        cdef RowMajorArrayPtr[Float, Two] pptr = as_RowMajorArrayPtr2D(particles)
        if particles.shape[1] == 4:
            if weight is not None:
                raise ValueError("Cannot pass weights twice")
            return self._cc_ma.add_particles_to_buffer(pptr)

        if particles.shape[1] != 3:
            raise ValueError(
                "Expected particles to have shape (n, 3) or (n, 4). "
                "Got: {}".format(particles.shape))

        if weight is None:
            weight = 1.0

        weight = np.asarray(weight, order="C")
        if not weight.shape:
            # Weight is a scalar.
            return self._cc_ma.add_particles_to_buffer(pptr, <Float> weight)

        # Weight is an array.
        if weight.shape != (particles.shape[0], ):
            raise ValueError(
                "Expected weight to be 1D with the same length as "
                "particles. Got {} and {}".format(
                    weight.shape, particles.shape))
        # TODO: wrap Array1D instead?
        cdef RowMajorArrayPtr[Float, One] wptr = as_RowMajorArrayPtr1D(weight)
        return self._cc_ma.add_particles_to_buffer(pptr, wptr)


    cpdef add_particle_to_buffer(self, Float x, Float y, Float z, Float w):
        self._cc_ma.add_particle_to_buffer(x, y, z, w)

    cpdef flush(self):
        self._cc_ma.flush()

    # TODO: annotate return type
    @property
    def posmin(self):
        return self._posmin

    @property
    def posmax(self):
        return self._posmax

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
