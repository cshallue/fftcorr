from config_space_grid cimport array, Three, Float, ConfigSpaceGrid as c_ConfigSpaceGrid

from cpython cimport Py_INCREF

# TODO: figure out these statements
import numpy as np
cimport numpy as np
np.import_array()

cdef class _ConfigSpaceGrid:
    # Allocate the grid on the heap; it would need to have a nullary
    # constructor to allocate it on the stack. TODO: consider this.
    cdef c_ConfigSpaceGrid *c_grid
    cdef np.ndarray data_arr

    # TODO: to allow the end user to call this more flexibly (i.e. with python
    # lists or numpy arrays of a different type), we can instead transform
    # the input args via np.ascontiguousarray(, dtype=).
    def __cinit__(self, const int[::1] ngrid, const Float[::1] posmin, Float cell_size):
        cdef array[int, Three] *ngrid_arr = <array[int, Three] *>(&ngrid[0])
        cdef array[Float, Three] *posmin_arr = <array[Float, Three] *>(&posmin[0])
        self.c_grid = new c_ConfigSpaceGrid(
            ngrid_arr[0],
            posmin_arr[0],
            cell_size)
        
        cdef np.npy_intp shape[3]
        for i in range(3):
            shape[i] = ngrid[i]
        cdef Float* data_ptr = self.c_grid.raw_data()
        # TODO: np.NPY_DOUBLE should be declared in the same place as Float.
        self.data_arr = np.PyArray_SimpleNewFromData(3, shape, np.NPY_DOUBLE, data_ptr)
        assert(np.PyArray_SetBaseObject(self.data_arr, self) == 0)
        Py_INCREF(self)

    def __dealloc__(self):
        del self.c_grid

    @property
    def data(self):
        return self.data_arr

    # TODO: everything below here can go; it's for testing only

    def add_scalar(self, Float s):
        self.c_grid.add_scalar(s)

    def multiply_by(self, Float s):
        self.c_grid.multiply_by(s)

    def sum(self) -> Float:
        return self.c_grid.sum()

    def sumsq(self) -> Float:
        return self.c_grid.sumsq()
