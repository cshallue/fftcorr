import numpy as np

# The only way to avoid code duplication here might be to generate the wrappers
# for each type automatically. But we only have two classes, so it's ok for now.
# https://stackoverflow.com/questions/31436593/cython-templates-in-python-class-wrappers

cdef class RowMajorArrayPtr3D_Float:
    def __cinit__(self, Float[:, :, ::1] arr):
        # TODO: I do this below and also in ConfigSpaceGrid too; parse out into
        # a templated function to_c_array()?
        # TODO: also parse out the cast of a memoryview to an array<dtype, N>*,
        # which is used in an overlapping but not identical set of circumstances.
        cdef cnp.ndarray[cnp.npy_int] shape = np.array(arr.shape, dtype=np.intc)
        self._ptr = new RowMajorArrayPtr[Float, Three](
            (<array[int, Three] *> &shape[0])[0], &arr[0,0,0])

    def __dealloc__(self):
        del self._ptr

    cdef RowMajorArrayPtr[Float, Three]* ptr(self):
        return self._ptr


cdef class RowMajorArrayPtr3D_Complex:
    def __cinit__(self, Complex[:, :, ::1] arr):
        cdef cnp.ndarray[cnp.npy_int] shape = np.array(arr.shape, dtype=np.intc)
        self._ptr = new RowMajorArrayPtr[Complex, Three](
            (<array[int, Three] *> &shape[0])[0], &arr[0,0,0])

    def __dealloc__(self):
        del self._ptr

    cdef RowMajorArrayPtr[Complex, Three]* ptr(self):
        return self._ptr

