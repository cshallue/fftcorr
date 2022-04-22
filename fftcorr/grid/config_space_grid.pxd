from fftcorr.array cimport RowMajorArrayPtr
from fftcorr.types cimport Float, array, Three

from libcpp cimport bool

cimport numpy as cnp

cdef extern from "config_space_grid.h":
  cdef cppclass ConfigSpaceGrid_cc "ConfigSpaceGrid":
    ConfigSpaceGrid_cc(array[int, Three], array[Float, Three], Float) except +
    Float cell_size()
    RowMajorArrayPtr[Float, Three]& data()
    void clear()
    bool get_grid_coords(const Float*, bool, Float*)


cdef class ConfigSpaceGrid:
    # Allocate the grid on the heap; it would need to have a nullary
    # constructor to allocate it on the stack. TODO: consider this.
    cdef cnp.ndarray _posmin
    cdef cnp.ndarray _posmax
    cdef Float _cell_size
    cdef Float _padding
    cdef ConfigSpaceGrid_cc *_cc_grid
    cdef cnp.ndarray _data

    cdef ConfigSpaceGrid_cc* cc_grid(self)
    cdef bool get_grid_coords(self, const Float*, bool, Float*)