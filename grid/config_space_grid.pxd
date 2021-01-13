cdef extern from "<array>" namespace "std" nogil:
    # TODO: consider alternative workarounds, such as a C++ typedef or
    # defining the class with template arguments.
    cdef cppclass Three "3":
        pass

    cdef cppclass array[T, N]:
      array()
      T& operator[](int)

# cdef extern from "../types.h":
#   # TODO: ctypedef?
#   cdef cppclass Float:
#     pass

# TODO: this must be kept in sync with the C++ file. Ideally we'd
# import Float from the C++, but it's not smart enough to recognize
# that as a typedef itself.
ctypedef double Float
# NP_FLOAT = np.double

cdef extern from "config_space_grid.h":
  cdef cppclass ConfigSpaceGrid:
    ConfigSpaceGrid(array[int, Three], array[Float, Three], Float) except +
    Float cell_size()
    void add_scalar(Float s)
    void multiply_by(Float s)
    Float sum()
    Float sumsq()
    Float* raw_data()