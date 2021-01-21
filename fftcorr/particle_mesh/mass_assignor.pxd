from fftcorr.grid cimport cc_ConfigSpaceGrid
from fftcorr.particle_mesh cimport WindowType
from fftcorr.types cimport Float

cdef extern from "mass_assignor.h":
  cdef cppclass cc_MassAssignor "MassAssignor":
    cc_MassAssignor(cc_ConfigSpaceGrid* grid, int buffer_size) except +
    void add_particle(Float x, Float y, Float z, Float w)
    void flush()
    int count()
    Float totw()
    Float totwsq()

cdef class MassAssignor:
    # Allocate the cc_MassAssignor on the heap; it would need to have a nullary
    # constructor to allocate it on the stack. TODO: consider this.
    cdef cc_MassAssignor *_cc_ma