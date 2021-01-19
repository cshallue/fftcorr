from fftcorr.types cimport Float
cimport numpy as cnp

# TODO: declare this in a common place too? types.pxd?
cdef extern from "<array>" namespace "std" nogil:
    # TODO: consider alternative workarounds, such as a C++ typedef or
    # defining the class with template arguments.
    cdef cppclass Three "3":
        pass

    cdef cppclass array[T, N]:
      array()
      T& operator[](int)


cdef extern from "config_space_grid.h":
  cdef cppclass cc_ConfigSpaceGrid "ConfigSpaceGrid":
    cc_ConfigSpaceGrid(array[int, Three], array[Float, Three], Float) except +
    Float cell_size()
    void add_scalar(Float s)
    void multiply_by(Float s)
    Float sum()
    Float sumsq()
    Float* raw_data()


cdef class ConfigSpaceGrid:
    # Allocate the grid on the heap; it would need to have a nullary
    # constructor to allocate it on the stack. TODO: consider this.
    cdef cnp.ndarray _posmin
    cdef Float _cell_size
    cdef cc_ConfigSpaceGrid *_cc_grid
    cdef cnp.ndarray _data_arr

    cdef cc_ConfigSpaceGrid* cc_grid(self)