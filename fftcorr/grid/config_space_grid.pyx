
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
                  posmax,
                  cell_size=None,
                  padding=0,  # padding already built in to posmin and posmax
                  extra_pad=0,  # additional padding to add
                  window_type=0):
        # Convert input arrays to contiguous arrays of the correct data type.
        # Then check for errors, as python objects.
        # Insist posmin is copied since it will be retained as a class attribute.
        # TODO: np.double should be declared in the same place as Float.
        shape = np.ascontiguousarray(shape, dtype=np.intc)
        posmin = np.asarray(posmin, dtype=np.double).copy(order="C")
        posmax = np.asarray(posmax, dtype=np.double).copy(order="C")

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

        if posmax.shape != (3, ):
            raise ValueError(
                "Expected posmax to have shape (3,), got: {}".format(
                    posmax.shape))

        if np.any(posmax <= posmin):
            raise ValueError(
                "Expected posmin < posmax, got posmin={}, posmax={}".format(
                    posmin, posmax))

        if cell_size is not None:
            if cell_size <= 0:
                raise ValueError(
                    "Expected cell_size to be positive, got: {}".format(
                        cell_size))

        if padding < 0:
            raise ValueError("Expected padding > 0, got {}".format(padding))

        if extra_pad < 0:
            raise ValueError("Expected extra_pad > 0, got {}".format(extra_pad))

        if extra_pad:
            print("Padding boundaries by {:.6g}".format(extra_pad))
            posmin -= extra_pad
            posmax += extra_pad

        minimum_cell_size = np.amax((posmax - posmin) / shape)
        if cell_size is not None:
            print("Requested cell size {:.6g}".format(cell_size))
            if cell_size < minimum_cell_size:
                raise ValueError(
                    "Requested cell size too small to cover entire region. "
                    "Need at least {:.6g}".format(minimum_cell_size))
        else:
            cell_size = minimum_cell_size

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
        self._padding = padding + extra_pad
        self._window_type = window_type

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
    def padding(self):
        return self._padding

    @property
    def window_type(self):
        return self._window_type

    @property
    def data(self):
        return self._data_arr

    def write(self, filename):
        import asdf  # TODO: move to top level?
        tree = {
            "header": {
                "shape": self.shape,
                "posmin": self.posmin,
                "posmax": self.posmax,
                "cell_size": self.cell_size,
                "padding": self.padding,
                "window_type": self.window_type,
            },
            "data": self.data,
        }
        af = asdf.AsdfFile(tree)
        af.write_to(filename)

    @classmethod
    def read(cls, filename):
        import asdf  # TODO: move to top level?
        af = asdf.open(filename)
        header = af.tree["header"]
        grid = cls(
            shape=header["shape"],
            posmin=header["posmin"],
            posmax=header["posmax"],
            padding=header["padding"],
            window_type=header["window_type"])
        assert np.allclose(grid.cell_size, header["cell_size"])
        np.copyto(grid.data, af.tree["data"])  # TODO: better way to do this?
        return grid

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

