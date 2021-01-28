from fftcorr.array cimport RowMajorArrayPtr
from fftcorr.grid cimport ConfigSpaceGrid_cc
from fftcorr.particle_mesh cimport WindowType
from fftcorr.types cimport Float, Two

cdef extern from "mass_assignor.h":
  cdef cppclass MassAssignor_cc "MassAssignor":
    MassAssignor_cc(ConfigSpaceGrid_cc* grid, int buffer_size) except +
    void add_particle(Float x, Float y, Float z, Float w)
    void add_particles(const RowMajorArrayPtr[Float, Two]&)
    void flush()
    int count()
    Float totw()
    Float totwsq()

cdef class MassAssignor:
    # Allocate the MassAssignor_cc on the heap; it would need to have a nullary
    # constructor to allocate it on the stack. TODO: consider this.
    cdef MassAssignor_cc *_cc_ma
    cpdef add_particles(self, Float[:, ::1] particles)