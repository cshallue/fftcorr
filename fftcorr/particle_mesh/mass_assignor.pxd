from fftcorr.array cimport RowMajorArrayPtr
from fftcorr.grid cimport ConfigSpaceGrid_cc
from fftcorr.particle_mesh cimport WindowType
from fftcorr.types cimport Float, One, Two, Four

from libcpp cimport bool

cdef extern from "mass_assignor.h":
  cdef cppclass MassAssignor_cc "MassAssignor":
    MassAssignor_cc(
      ConfigSpaceGrid_cc* grid,
      bool periodic_wrap,
      int buffer_size,
      const RowMajorArrayPtr[Float, Four]* disp) except +
    void clear()
    void add_particles_to_buffer(const RowMajorArrayPtr[Float, Two]&)
    void add_particles_to_buffer(const RowMajorArrayPtr[Float, Two]&,
                                 const RowMajorArrayPtr[Float, One]&)
    void add_particles_to_buffer(const RowMajorArrayPtr[Float, Two]&, Float)
    void add_particle_to_buffer(Float x, Float y, Float z, Float w)
    void flush()
    unsigned long long num_added()
    unsigned long long num_skipped()
    Float totw()
    Float totwsq()
    Float sort_time()
    Float window_time()

cdef class MassAssignor:
    # Allocate the MassAssignor_cc on the heap; it would need to have a nullary
    # constructor to allocate it on the stack. TODO: consider this.
    cdef MassAssignor_cc *_cc_ma
    cpdef add_particles(self, Float[:, ::1] particles, weight=*)
    cpdef add_particles_to_buffer(self, Float[:, ::1] particles, weight=*)
    cpdef add_particle_to_buffer(self, Float, Float, Float, Float)
    cpdef flush(self)
    cpdef clear(self)