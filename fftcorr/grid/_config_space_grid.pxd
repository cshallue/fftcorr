# TODO: could import cc_ConfigSpaceGrid from fftcorr.grid since it's in the
# __init__.pxd file.
from fftcorr.grid.config_space_grid_cc cimport ConfigSpaceGrid as cc_ConfigSpaceGrid

cimport numpy as cnp

cdef class _ConfigSpaceGrid:
    # Allocate the grid on the heap; it would need to have a nullary
    # constructor to allocate it on the stack. TODO: consider this.
    cdef cc_ConfigSpaceGrid *cc_grid
    cdef cnp.ndarray data_arr