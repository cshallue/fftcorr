from fftcorr.particle_mesh._mass_assignor import _MassAssignor

import numpy as np


# TODO: consider making MassAssignor a context manager, or wrapping it
# in a function that is a context manager
# https://book.pythontips.com/en/latest/context_managers.html
# when the context exits, flush() should be called. should it also
# deallocate the buffer?
class MassAssignor(object):
    def __init__(self, grid, window_type, buffer_size):
        self.ma = _MassAssignor(grid._grid, window_type, buffer_size)

    def add_particle(self, x, y, z, w):
        self.ma.add_particle(x, y, z, w)

    def flush(self):
        self.ma.flush()

    @property
    def count(self):
        return self.ma.count

    @property
    def totw(self):
        return self.ma.totw

    @property
    def totwsq(self):
        return self.ma.totwsq