# This must be kept in sync with the Float typedef in types.h. Ideally, we'd 
# import the alias directly from types.h, but we need to use it as a primitive
# type in cython (for memoryviews) and there's no way for Cython to know in
# advance that an extern alias is a primitive type.
ctypedef double Float
# TODO: is this right?
# https://github.com/pyFFTW/pyFFTW/blob/ff5e19f5aedeb52658f2ea6b3fa5541849d77e1e/pyfftw/pyfftw.pxd
ctypedef double complex Complex

ctypedef unsigned long long int uint64

# The numpy type enum corresponding to the Float typedef above.
# https://numpy.org/doc/stable/reference/c-api/dtype.html
#cimport numpy as cnp
#NP_FLOAT_ENUM = cnp.NPY_DOUBLE
# TODO: I don't think this is really possible in a pxd file; cnp.NPY_DOUBLE is
# a value, not a type, so we can't have an alias to it. We might have
# to make types an actual cython module with a pyx file.
# We also need an alias for np.double, the associated python type.

cdef extern from "<array>" namespace "std" nogil:
    # TODO: consider alternative workarounds, such as a C++ typedef or
    # defining the class with template arguments. We almost always use 3 as the
    # dimension, so in C++ we might be able to make a templated typedef
    # array3d<type>
    cdef cppclass One "1":
        pass

    cdef cppclass Two "2":
        pass

    cdef cppclass Three "3":
        pass

    cdef cppclass Four "4":
        pass

    cdef cppclass array[T, N]:
      array()
      T& operator[](int)
      T* data()