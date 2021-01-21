from fftcorr.types cimport Float, Complex, array, Three

cdef extern from "row_major_array.h":
  cdef cppclass RowMajorArrayPtr[dtype]:
    RowMajorArrayPtr(dtype *data, array[int, Three]) except +
    # TODO: not needed in Python (it has the numpy array); only for testing.
    dtype at(int ix, int iy, int iz)


# TODO: since RowMajorArrayPtr has a nullary constructor, we should be able to
# stack allocate the RowMajorArrayPtr[Float] object.
# https://cython.readthedocs.io/en/latest/src/userguide/wrapping_CPlusPlus.html
cdef class RowMajorArrayPtr_Float:
    cdef RowMajorArrayPtr[Float]* _ptr
    cdef RowMajorArrayPtr[Float]* ptr(self)
    # TODO: not needed in Python (it has the numpy array); only for testing.
    cpdef Float at(self, int ix, int iy, int iz)


cdef class RowMajorArrayPtr_Complex:
    cdef RowMajorArrayPtr[Complex]* _ptr
    cdef RowMajorArrayPtr[Complex]* ptr(self)
    # TODO: not needed in Python (it has the numpy array); only for testing.
    #cdef Complex at(self, int ix, int iy, int iz)