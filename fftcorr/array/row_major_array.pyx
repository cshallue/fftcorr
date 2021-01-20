cimport numpy as cnp
cnp.import_array()

import numpy as np

# The only way to avoid code duplication here might be to generate the wrappers
# for each type automatically. But we only have two classes, so it's ok for now.
# https://stackoverflow.com/questions/31436593/cython-templates-in-python-class-wrappers

cdef class RowMajorArrayPtr_Float:
    def __cinit__(self, Float[:, :, ::1] arr):
        # TODO: I do this below and also in ConfigSpaceGrid too; parse out into
        # a templated function to_c_array()?
        # TODO: also parse out the cast of a memoryview to an array<dtype, N>*,
        # which is used in an overlapping but not identical set of circumstances.
        cdef int shape[3]
        cdef int i
        for i in range(3):
            shape[i] = arr.shape[i]
        self._ptr = new RowMajorArrayPtr[Float](
            &arr[0,0,0],
            (<array[int, Three] *> &shape[0])[0])

    def __dealloc__(self):
        del self._ptr

    cdef RowMajorArrayPtr[Float]* ptr(self):
        return self._ptr

    cpdef double at(self, int ix, int iy, int iz):
        return self._ptr.at(ix, iy, iz)


cdef class RowMajorArrayPtr_Complex:
    def __cinit__(self, Complex[:, :, ::1] arr):
        cdef int shape[3]
        cdef int i
        for i in range(3):
            shape[i] = arr.shape[i]
        self._ptr = new RowMajorArrayPtr[Complex](
            &arr[0,0,0],
            (<array[int, Three] *> &shape[0])[0])

    def __dealloc__(self):
        del self._ptr

    cdef RowMajorArrayPtr[Complex]* ptr(self):
        return self._ptr

    # TODO: only for testing. Can't return a Complex, that's an array
    def at(self, int ix, int iy, int iz):
        #cdef Float *val = self._arr.at(ix, iy, iz)
        #return val[0], val[1]
        return self._ptr.at(ix, iy, iz)

