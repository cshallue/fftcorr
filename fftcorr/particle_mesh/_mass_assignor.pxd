from fftcorr.grid cimport _ConfigSpaceGrid
from fftcorr.particle_mesh cimport cc_MassAssignor
from fftcorr.particle_mesh cimport WindowType

cdef class _MassAssignor:
    # Allocate the cc_MassAssignor on the heap; it would need to have a nullary
    # constructor to allocate it on the stack. TODO: consider this.
    cdef cc_MassAssignor *cc_ma