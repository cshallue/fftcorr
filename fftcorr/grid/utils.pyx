from fftcorr.grid cimport ConfigSpaceGrid
from fftcorr.types cimport Float

cimport cython
from libcpp cimport bool

cimport numpy as cnp
cnp.import_array()

import numpy as np

# TODO: assuming contiguous makes the generated C code simpler and therefore
# perhaps faster. But it necessitates us doing a contiguous copy below.
# Figure out whether it would be better to allow an arbitrary data layout.
# We can simply delete the "::1"'s from this function and it all will still
# work. Currently we're not requiring out to be contiguous.
# Turn off cython bounds checking and negative index support for performance.
# We're doing our own internal checks for index validity.
@cython.boundscheck(False)
@cython.wraparound(False)
cdef bool _apply_displacement_field(ConfigSpaceGrid grid,
                                    Float[:, ::1] pos,
                                    const Float[:, :, :, :] disp,
                                    bool periodic_wrap,
                                    Float[:, :] out):
    cdef const Float[::1] survey_coords
    cdef Float[:] out_coords
    cdef Float grid_coords[3]
    cdef const Float[:] dxyz
    cdef int i, j
    for i in range(pos.shape[0]):
        survey_coords = pos[i]
        out_coords = out[i]
        if not grid.get_grid_coords(&survey_coords[0], periodic_wrap, grid_coords):
            return False
        # TODO: cast to int okay because positive and not too big?
        dxyz = disp[<int> grid_coords[0], <int> grid_coords[1], <int> grid_coords[2]]
        for j in range(3):
            out_coords[j] = survey_coords[j] + dxyz[j]

    return True


def apply_displacement_field(ConfigSpaceGrid grid, pos, disp, periodic_wrap=False, out=None):
    if out is None:
        out = np.empty_like(pos)
    
    # TODO: validate dimensions

    if not _apply_displacement_field(grid, pos, disp, periodic_wrap, out):
        raise ValueError(
            "Failed to apply displacement field: coordinates out of bounds? "
            "Try setting periodic_wrap=True.")

    return out
