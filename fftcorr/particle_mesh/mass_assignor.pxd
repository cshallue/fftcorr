from fftcorr.array cimport RowMajorArrayPtr
from fftcorr.grid cimport ConfigSpaceGrid_cc
from fftcorr.particle_mesh cimport WindowType
from fftcorr.types cimport Float, One, Two, Four

from libcpp cimport bool
cimport numpy as cnp

cdef extern from "mass_assignor.h":
  cdef cppclass MassAssignor_cc "MassAssignor":
    MassAssignor_cc(
      ConfigSpaceGrid_cc& grid,
      bool periodic_wrap,
      int buffer_size) except +
    void clear()
    void add_particles_to_buffer(const RowMajorArrayPtr[Float, Two]&)
    void add_particles_to_buffer(const RowMajorArrayPtr[Float, Two]&,
                                 const RowMajorArrayPtr[Float, One]&)
    void add_particles_to_buffer(const RowMajorArrayPtr[Float, Two]&, Float)
    void add_particle_to_buffer(Float*, Float)
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
    # TODO: _cc_disp? _disp_rma? Consider naming conventions for arrays and
    # their shapes. Disp at all?
    cdef RowMajorArrayPtr[Float, Four] _disp

    # Keep track of these because they're useful for validation in functions 
    # that take a mass assignor but not the grid. Is this how we want to
    # structure thingss?
    cdef cnp.ndarray _posmin
    cdef cnp.ndarray _posmax

    cpdef flush(self)
    cpdef clear(self)