from fftcorr.array cimport Array
from fftcorr.types cimport Float, Complex, array, One, Two, Three, Four
cimport numpy as cnp

cdef extern from "row_major_array.h":
  cdef cppclass RowMajorArrayPtr[dtype, N](Array):
    RowMajorArrayPtr() except +
    RowMajorArrayPtr(array[int, N], dtype *data) except +
    dtype* data()
    const array[int, N]& shape()


# TODO: I think all of the below are obsoleted by numpy_adaptor.

# TODO: since RowMajorArrayPtr has a nullary constructor, we should be able to
# stack allocate the RowMajorArrayPtr[Float] object.
# https://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html
cdef class RowMajorArrayPtr3D_Float:
    cdef RowMajorArrayPtr[Float, Three]* _ptr
    cdef RowMajorArrayPtr[Float, Three]* ptr(self)


cdef class RowMajorArrayPtr3D_Complex:
    cdef RowMajorArrayPtr[Complex, Three]* _ptr
    cdef RowMajorArrayPtr[Complex, Three]* ptr(self)


cdef inline RowMajorArrayPtr[Float, One] as_RowMajorArrayPtr1D(Float[::1] arr):
  cdef array[int, One] shape
  shape[0] = arr.shape[0]
  return RowMajorArrayPtr[Float, One](shape, &arr[0])


cdef inline RowMajorArrayPtr[Float, Two] as_RowMajorArrayPtr2D(Float[:, ::1] arr):
  cdef array[int, Two] shape
  # TODO: std::copy instead, or something like that
  for i in range(2):
    shape[i] = arr.shape[i]
  return RowMajorArrayPtr[Float, Two](shape, &arr[0,0])


cdef inline RowMajorArrayPtr[Float, Three] as_RowMajorArrayPtr3D(Float[:, :, ::1] arr):
  cdef array[int, Three] shape
  # TODO: std::copy instead, or something like that
  for i in range(3):
    shape[i] = arr.shape[i]
  return RowMajorArrayPtr[Float, Three](shape, &arr[0,0,0])


cdef inline RowMajorArrayPtr[Float, Four] as_RowMajorArrayPtr4D(Float[:, :, :, ::1] arr):
  cdef array[int, Four] shape
  # TODO: std::copy instead, or something like that
  for i in range(4):
    shape[i] = arr.shape[i]
  return RowMajorArrayPtr[Float, Four](shape, &arr[0,0,0,0])