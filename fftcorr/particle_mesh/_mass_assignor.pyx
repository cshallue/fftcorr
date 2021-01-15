from fftcorr.grid cimport _ConfigSpaceGrid
from fftcorr.particle_mesh cimport cc_MassAssignor
from fftcorr.particle_mesh cimport WindowType
from fftcorr.types cimport Float


# TODO: formatting. 2 space indentation instead of 4?
cdef class _MassAssignor:
    def __cinit__(self, _ConfigSpaceGrid grid, WindowType window_type, int buffer_size):
        self.cc_ma = new cc_MassAssignor(grid.cc_grid, window_type, buffer_size)

    # @staticmethod
    # cdef create(_ConfigSpaceGrid grid, WindowType window_type, int buffer_size):
    #     cdef _MassAssignor ma = _MassAssignor()
    #     ma.cc_ma = new cc_MassAssignor(grid.cc_grid, window_type, buffer_size)
    #     return ma

    # @staticmethod
    # def py_create(_ConfigSpaceGrid grid, WindowType window_type, int buffer_size):
    #     return _MassAssignor.create(grid, window_type, buffer_size)

    def __dealloc__(self):
        del self.cc_ma

    def add_particle(self, Float x, Float y, Float z, Float w):
        self.cc_ma.add_particle(x, y, z, w)

    def flush(self):
        self.cc_ma.flush()

    @property
    def count(self) -> int:
        return self.cc_ma.count()

    @property
    def totw(self) -> Float:
        return self.cc_ma.totw()

    @property
    def totwsq(self) -> Float:
        return self.cc_ma.totwsq()