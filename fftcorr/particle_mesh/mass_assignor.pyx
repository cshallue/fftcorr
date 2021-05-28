from fftcorr.grid cimport ConfigSpaceGrid
from fftcorr.types cimport array

cimport numpy as cnp
cnp.import_array()

import numpy as np

# TODO: consider making MassAssignor a context manager, or wrapping it
# in a function that is a context manager
# https://book.pythontips.com/en/latest/context_managers.html
# when the context exits, flush() should be called. should it also
# deallocate the buffer?
# TODO: formatting. 2 space indentation instead of 4?
cdef class MassAssignor:
    def __cinit__(self, ConfigSpaceGrid grid, bool periodic_wrap=False, int buffer_size=10000, Float[:, :, :, ::1] disp=None):
        cdef cnp.ndarray[cnp.npy_int] disp_shape
        cdef RowMajorArrayPtr[Float, Four] disp_rma
        if disp is not None:
            # TODO: wrap this, it's used in multiple places
            disp_shape =  np.array(disp.shape, dtype=np.intc)
            disp_rma.set_data(
                (<array[int, Four] *> &disp_shape[0])[0], &disp[0,0,0,0])
        self._cc_ma = new MassAssignor_cc(
            grid.cc_grid(), periodic_wrap, buffer_size, &disp_rma)
        

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
        particles = np.asarray(particles, order="C")
        # TODO: wrap this?
        cdef cnp.ndarray[cnp.npy_int] pshape = np.array(particles.shape, dtype=np.intc)
        cdef RowMajorArrayPtr[Float, Two] pptr = RowMajorArrayPtr[Float, Two](
            (<array[int, Two] *> &pshape[0])[0], &particles[0,0])
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
        cdef cnp.ndarray[cnp.npy_int] wshape = np.array(weight.shape, dtype=np.intc)
        cdef Float[::1] weight_arr = weight
        # TODO: wrap Array1D?
        cdef RowMajorArrayPtr[Float, One] wptr = RowMajorArrayPtr[Float, One](
            (<array[int, One] *> &wshape[0])[0], &weight_arr[0])
        return self._cc_ma.add_particles_to_buffer(pptr, wptr)


    cpdef add_particle_to_buffer(self, Float x, Float y, Float z, Float w):
        self._cc_ma.add_particle_to_buffer(x, y, z, w)

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