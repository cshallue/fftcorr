from fftcorr.types cimport Float

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
  cdef cppclass ConfigSpaceGrid:
    ConfigSpaceGrid(array[int, Three], array[Float, Three], Float) except +
    Float cell_size()
    void add_scalar(Float s)
    void multiply_by(Float s)
    Float sum()
    Float sumsq()
    Float* raw_data()