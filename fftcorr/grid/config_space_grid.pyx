
from fftcorr.array cimport as_numpy
from fftcorr.particle_mesh.window_type cimport WindowType
from cpython cimport Py_INCREF
cimport numpy as cnp
cnp.import_array()

import numpy as np


cdef class ConfigSpaceGrid:
    def __cinit__(self,
                  shape,
                  posmin,
                  posmax=None,
                  cell_size=None,
                  padding=None,
                  window_type=0):
        # Convert input arrays to contiguous arrays of the correct data type.
        # Then check for errors, as python objects.
        # Insist posmin is copied since it will be retained as a class attribute.
        # TODO: np.double should be declared in the same place as Float.
        shape = np.ascontiguousarray(shape, dtype=np.intc)
        posmin = np.asarray(posmin, dtype=np.double).copy(order="C")

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

        if ((posmax is None) == (cell_size is None)):
            raise ValueError("Exactly one of posmax and cell_size is required")

        if padding is not None:
            if padding < 0:
                raise ValueError("Expected padding > 0, got {}".format(padding))

            print("Padding boundaries by {:.6g}".format(padding))
            posmin -= padding
        else:
            padding = 0

        if cell_size is not None:
            if cell_size <= 0:
                raise ValueError(
                    "Expected cell_size to be positive, got: {}".format(
                        cell_size))

            print("Requested cell size {:.6g}".format(cell_size))
        else:
            # Posmax is not None.
            posmax = np.asarray(posmax, dtype=np.double).copy(order="C")
            posmax += padding

            if posmax.shape != (3, ):
                raise ValueError(
                    "Expected posmax to have shape (3,), got: {}".format(
                        posmax.shape))

            if np.any(posmax <= posmin):
                raise ValueError(
                    "Expected posmin < posmax, got posmin={}, posmax={}".format(
                        posmin, posmax))

            cell_size = np.amax((posmax - posmin) / shape)

        posmax = posmin + shape * cell_size
        print("Adopting:")
        print("  ngrid = [" + ", ".join(["{}".format(x) for x in shape]) + "]")
        print("  cell_size = {:.6g}".format(cell_size))
        print("  posmin = [" + ", ".join(["{:.6g}".format(x) for x in posmin]) + "]")
        print("  posmax = [" + ", ".join(["{:.6g}".format(x) for x in posmax]) + "]")

        self._posmin = posmin
        self._posmax = posmax
        for arr in [posmin, posmax]:
            arr.setflags(write=False)
        self._cell_size = cell_size

        # Create the wrapped C++ ConfigSpaceGrid.
        cdef cnp.ndarray[int, ndim=1, mode="c"] cshape = shape
        cdef cnp.ndarray[double, ndim=1, mode="c"] cposmin = posmin
        cdef WindowType wt = window_type
        self._cc_grid = new ConfigSpaceGrid_cc(
            (<array[int, Three] *> &cshape[0])[0],
            (<array[Float, Three] *> &cposmin[0])[0],
            cell_size,
            wt)
        
        # Wrap the data array as a numpy array.
        self._data_arr = as_numpy(
            3,
            self._cc_grid.data().shape().data(),
            cnp.NPY_DOUBLE,
            self._cc_grid.data().data(),
            self)

    def __dealloc__(self):
        del self._cc_grid

    cdef ConfigSpaceGrid_cc* cc_grid(self):
        return self._cc_grid

    # TODO: make consistent shape or ngrid in python and C++?
    @property
    def shape(self):
        return self.data.shape

    @property
    def posmin(self):
        return self._posmin

    @property
    def posmax(self):
        return self._posmax

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

