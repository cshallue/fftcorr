from fftcorr.grid cimport ConfigSpaceGrid
from fftcorr.types cimport Float

# TODO: consider making MassAssignor a context manager, or wrapping it
# in a function that is a context manager
# https://book.pythontips.com/en/latest/context_managers.html
# when the context exits, flush() should be called. should it also
# deallocate the buffer?
# TODO: formatting. 2 space indentation instead of 4?
cdef class MassAssignor:
    def __cinit__(self, ConfigSpaceGrid grid, int buffer_size):
        self._cc_ma = new MassAssignor_cc(grid.cc_grid(), buffer_size)

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

    # TODO: make the below cdef?

    def add_particle(self, Float x, Float y, Float z, Float w):
        self._cc_ma.add_particle(x, y, z, w)

    def flush(self):
        self._cc_ma.flush()

    @property
    def count(self) -> int:
        return self._cc_ma.count()

    @property
    def totw(self) -> Float:
        return self._cc_ma.totw()

    @property
    def totwsq(self) -> Float:
        return self._cc_ma.totwsq()
