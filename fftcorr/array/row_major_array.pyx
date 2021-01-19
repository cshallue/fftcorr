cimport numpy as cnp
cnp.import_array()

import numpy as np

cdef class Wrapper:
    # TODO: get the syntax right for a continuous 3d memoryview
    def __cinit__(self, double[:, :, ::1] arr):
        # TODO: I do this in ConfigSpaceGrid too; parse out into a templated
        # function to_c_array()?
        # TODO: also parse out the cast of a memoryview to an array<dtype, N>*,
        # which is used in an overlapping but not identical set of circumstances.
        cdef int shape[3]
        cdef int i
        for i in range(3):
            shape[i] = arr.shape[i]
        self._arr = new _FloatArray(
            &arr[0,0,0],
            (<array[int, Three] *> &shape[0])[0])

    def __dealloc__(self):
        del self._arr

    cpdef double at(self, int ix, int iy, int iz):
        return self._arr.at(ix, iy, iz)
