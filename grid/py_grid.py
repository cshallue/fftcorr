from grid import _ConfigSpaceGrid  # TODO: perhaps _grid and ConfigSpaceGrid

import numpy as np


# TODO: rename these files
class ConfigSpaceGrid(object):
    def __init__(self, ngrid, posmin, cell_size):
        # Convert input arrays to contiguous arrays of the correct data type.
        # Insist posmin is copied since it will be retained as a class attribute.
        # TODO: np.double should be declared in the same place as Float.
        ngrid = np.ascontiguousarray(ngrid, dtype=np.intc)
        posmin = np.asarray(posmin, dtype=np.double).copy(order="C")
        posmin.setflags(write=False)

        if ngrid.shape != (3, ):
            raise ValueError(
                "Expected ngrid to have shape (3,), got: {}".format(
                    ngrid.shape))

        if np.any(ngrid <= 0):
            raise ValueError(
                "Expected ngrid to be positive, got: {}".format(ngrid))

        if posmin.shape != (3, ):
            raise ValueError(
                "Expected posmin to have shape (3,), got: {}".format(
                    posmin.shape))

        if cell_size <= 0:
            raise ValueError(
                "Expected cell_size to be positive, got: {}".format(cell_size))

        self._grid = _ConfigSpaceGrid(ngrid, posmin, cell_size)
        self._posmin = posmin
        self._cell_size = cell_size

    @property
    def data(self):
        return self._grid.data

    @property
    def posmin(self):
        return self._posmin

    @property
    def cell_size(self):
        return self._cell_size

    # TODO: everything below here can go; it's for testing only

    def add_scalar(self, s):
        self._grid.add_scalar(s)

    def multiply_by(self, s):
        self._grid.multiply_by(s)

    def sum(self):
        return self._grid.sum()

    def sumsq(self):
        return self._grid.sumsq()