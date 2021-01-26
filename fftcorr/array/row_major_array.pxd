from fftcorr.types cimport Float, Complex, array, Three
cimport numpy as cnp

cdef extern from "row_major_array.h":
  cdef cppclass RowMajorArrayPtr[dtype, N]:
    RowMajorArrayPtr(array[int, N], dtype *data) except +
    dtype* data()
    const array[int, N]& shape()


# TODO: since RowMajorArrayPtr has a nullary constructor, we should be able to
# stack allocate the RowMajorArrayPtr[Float] object.
# https://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html
cdef class RowMajorArrayPtr3D_Float:
    cdef RowMajorArrayPtr[Float, Three]* _ptr
    cdef RowMajorArrayPtr[Float, Three]* ptr(self)


cdef class RowMajorArrayPtr3D_Complex:
    cdef RowMajorArrayPtr[Complex, Three]* _ptr
    cdef RowMajorArrayPtr[Complex, Three]* ptr(self)

cdef cnp.ndarray as_numpy(int ndim, const int* shape, Float* data, owner)