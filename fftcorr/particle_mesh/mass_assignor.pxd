from fftcorr.array cimport RowMajorArrayPtr
from fftcorr.grid cimport ConfigSpaceGrid_cc
from fftcorr.particle_mesh cimport WindowType
from fftcorr.types cimport Float, One, Two

cdef extern from "mass_assignor.h":
  cdef cppclass MassAssignor_cc "MassAssignor":
    MassAssignor_cc(ConfigSpaceGrid_cc* grid, int buffer_size) except +
    void clear()
    void add_particles(const RowMajorArrayPtr[Float, Two]&)
    void add_particles(const RowMajorArrayPtr[Float, Two]&,
                       const RowMajorArrayPtr[Float, One]&)
    void add_particles(const RowMajorArrayPtr[Float, Two]&, Float)
    void add_particle_to_buffer(Float x, Float y, Float z, Float w)
    void flush()
    unsigned long long count()
    unsigned long long skipped()
    Float totw()
    Float totwsq()
    Float sort_time()
    Float window_time()

cdef class MassAssignor:
    # Allocate the MassAssignor_cc on the heap; it would need to have a nullary
    # constructor to allocate it on the stack. TODO: consider this.
    cdef MassAssignor_cc *_cc_ma
    cpdef add_particles(self, Float[:, ::1] particles, weight=*)
    cpdef clear(self)