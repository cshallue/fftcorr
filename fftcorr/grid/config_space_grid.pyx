from cpython cimport Py_INCREF
cimport numpy as cnp
cnp.import_array()

import numpy as np


cdef class ConfigSpaceGrid:
    def __cinit__(self, shape, posmin, cell_size):
        # Convert input arrays to contiguous arrays of the correct data type.
        # Then check for errors, as python objects.
        # Insist posmin is copied since it will be retained as a class attribute.
        # TODO: np.double should be declared in the same place as Float.
        shape = np.ascontiguousarray(shape, dtype=np.intc)
        posmin = np.asarray(posmin, dtype=np.double).copy(order="C")
        posmin.setflags(write=False)

        if shape.shape != (3, ):
            raise ValueError(
                "Expected shape to have shape (3,), got: {}".format(
                    shape.shape))

        if np.any(shape <= 0):
            raise ValueError(
                "Expected shape to be positive, got: {}".format(shape))

        if posmin.shape != (3, ):
            raise ValueError(
                "Expected posmin to have shape (3,), got: {}".format(
                    posmin.shape))

        if cell_size <= 0:
            raise ValueError(
                "Expected cell_size to be positive, got: {}".format(cell_size))

        self._posmin = posmin
        self._cell_size = cell_size

        # Create the wrapped C++ ConfigSpaceGrid.
        cdef cnp.ndarray[int, ndim=1, mode="c"] cshape = shape
        cdef cnp.ndarray[double, ndim=1, mode="c"] cposmin = posmin
        self._cc_grid = new cc_ConfigSpaceGrid(
            (<array[int, Three] *> &cshape[0])[0],
            (<array[Float, Three] *> &cposmin[0])[0],
            cell_size)
        
        # Wrap the data array as a numpy array.
        cdef cnp.npy_intp shape_np[3]
        cdef int i
        for i in range(3):
            shape_np[i] = shape[i]
        cdef Float* data_ptr = self._cc_grid.raw_data()
        # TODO: cnp.NPY_DOUBLE should go in types.pxd
        self._data_arr = cnp.PyArray_SimpleNewFromData(
            3, shape_np, cnp.NPY_DOUBLE, data_ptr)
        assert(cnp.PyArray_SetBaseObject(self._data_arr, self) == 0)
        Py_INCREF(self)

    def __dealloc__(self):
        del self._cc_grid

    cdef cc_ConfigSpaceGrid* cc_grid(self):
        return self._cc_grid

    # TODO: make consistent shape or ngrid in python and C++?
    @property
    def shape(self):
        return self.data.shape

    @property
    def cell_size(self):
        return self._cell_size

    @property
    def data(self):
        return self._data_arr

    # TODO: everything below here can go; it's for testing only
    # However, look into whether these should be cdef, cpdef, or def

    def add_scalar(self, Float s):
        self._cc_grid.add_scalar(s)

    def multiply_by(self, Float s):
        self._cc_grid.multiply_by(s)

    def sum(self) -> Float:
        return self._cc_grid.sum()

    def sumsq(self) -> Float:
        return self._cc_grid.sumsq()

