from fftcorr.types cimport Float
from fftcorr.grid cimport cc_ConfigSpaceGrid

cdef extern from "mass_assignor.h":
  cdef cppclass MassAssignor:
    # TODO: make window_type an enum
    MassAssignor(cc_ConfigSpaceGrid* grid, int window_type, int buffer_size) except +
    void add_particle(Float x, Float y, Float z, Float w)
    void flush()
    int count()
    Float totw()
    Float totwsq()