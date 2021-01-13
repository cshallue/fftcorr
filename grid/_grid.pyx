from config_space_grid cimport array, Three, Float, ConfigSpaceGrid as c_ConfigSpaceGrid

from cpython cimport Py_INCREF

# TODO: figure out these statements
import numpy as np
cimport numpy as np
np.import_array()

cdef class ConfigSpaceGrid:
    # Allocate the grid on the heap; it would need to have a nullary
    # constructor to allocate it on the stack. TODO: consider this.
    cdef c_ConfigSpaceGrid *_c_grid
    cdef Float* _data_ptr
    cdef np.ndarray _data

    # TODO: to allow the end user to call this more flexibly (i.e. with python
    # lists or numpy arrays of a different type), we can instead transform
    # the input args via np.ascontiguousarray(, dtype=).
    def __cinit__(self, int[::1] ngrid, Float[::1] posmin, Float cell_size):
        cdef array[int, Three] *ngrid_arr = <array[int, Three] *>(&ngrid[0])
        cdef array[Float, Three] *posmin_arr = <array[Float, Three] *>(&posmin[0])
        self._c_grid = new c_ConfigSpaceGrid(
            ngrid_arr[0],
            posmin_arr[0],
            cell_size)
        
        self._data_ptr = self._c_grid.raw_data()
        cdef np.npy_intp shape[3]
        for i in range(3):
            shape[i] = ngrid[i]
        # TODO: np.NPY_DOUBLE should be declared in the same place as Float.
        self._data = np.PyArray_SimpleNewFromData(3, shape, np.NPY_DOUBLE, self._data_ptr)
        print("{0:x}".format(<unsigned long> np.PyArray_BASE(self._data)))
        assert(np.PyArray_SetBaseObject(self._data, self) == 0)
        print("{0:x}".format(<unsigned long> np.PyArray_BASE(self._data)))
        Py_INCREF(self)


    def __dealloc__(self):
        del self._c_grid

    @property
    def cell_size(self):
        return self._c_grid.cell_size()

    @property
    def data(self):
        return self._data

    def add_scalar(self, Float s):
        self._c_grid.add_scalar(s)

    def multiply_by(self, Float s):
        self._c_grid.multiply_by(s)

    def sum(self) -> Float:
        return self._c_grid.sum()

    def sumsq(self) -> Float:
        return self._c_grid.sumsq()