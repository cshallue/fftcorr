# TODO: could import cc_ConfigSpaceGrid from fftcorr.grid since it's in the
# __init__.pxd file.
from fftcorr.grid.config_space_grid_cc cimport array, Three, ConfigSpaceGrid as cc_ConfigSpaceGrid
from fftcorr.types cimport Float

from cpython cimport Py_INCREF

cimport numpy as cnp
cnp.import_array()

cdef class _ConfigSpaceGrid:
    def __cinit__(self, const int[::1] ngrid, const Float[::1] posmin, Float cell_size):
        cdef array[int, Three] *ngrid_arr = <array[int, Three] *>(&ngrid[0])
        cdef array[Float, Three] *posmin_arr = <array[Float, Three] *>(&posmin[0])
        self.cc_grid = new cc_ConfigSpaceGrid(
            ngrid_arr[0],
            posmin_arr[0],
            cell_size)
        
        cdef cnp.npy_intp shape[3]
        for i in range(3):
            shape[i] = ngrid[i]
        cdef Float* data_ptr = self.cc_grid.raw_data()
        # TODO: cnp.NPY_DOUBLE should go in types.pxd
        self.data_arr = cnp.PyArray_SimpleNewFromData(
            3, shape, cnp.NPY_DOUBLE, data_ptr)
        assert(cnp.PyArray_SetBaseObject(self.data_arr, self) == 0)
        Py_INCREF(self)

    def __dealloc__(self):
        del self.cc_grid

    @property
    def data(self):
        return self.data_arr

    # TODO: everything below here can go; it's for testing only

    def add_scalar(self, Float s):
        self.cc_grid.add_scalar(s)

    def multiply_by(self, Float s):
        self.cc_grid.multiply_by(s)

    def sum(self) -> Float:
        return self.cc_grid.sum()

    def sumsq(self) -> Float:
        return self.cc_grid.sumsq()
