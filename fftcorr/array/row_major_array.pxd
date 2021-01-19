from fftcorr.types cimport Float, array, Three

cdef extern from "row_major_array.h":
  cdef cppclass RowMajorArray[dtype]:
    RowMajorArray(dtype *data, array[int, Three]) except +
    # TODO: not needed in Python (it has the numpy array); only for testing.
    dtype at(int ix, int iy, int iz)

# TODO: names. Is there a better way to deal with the templates?
ctypedef RowMajorArray[double] _FloatArray

# TODO: this is just used for testing now; figure out what to do with it.
cdef class Wrapper:
    cdef _FloatArray* _arr
    # TODO: not needed in Python (it has the numpy array); only for testing.
    cpdef double at(self, int ix, int iy, int iz)