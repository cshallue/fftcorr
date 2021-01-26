from cpython cimport PyObject, Py_INCREF
cimport numpy as cnp
cnp.import_array()

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
        cdef int shape[3]
        cdef int i
        for i in range(3):
            shape[i] = arr.shape[i]
        self._ptr = new RowMajorArrayPtr[Float, Three](
            (<array[int, Three] *> &shape[0])[0], &arr[0,0,0])

    def __dealloc__(self):
        del self._ptr

    cdef RowMajorArrayPtr[Float, Three]* ptr(self):
        return self._ptr


cdef class RowMajorArrayPtr3D_Complex:
    def __cinit__(self, Complex[:, :, ::1] arr):
        cdef int shape[3]
        cdef int i
        for i in range(3):
            shape[i] = arr.shape[i]
        self._ptr = new RowMajorArrayPtr[Complex, Three](
            (<array[int, Three] *> &shape[0])[0], &arr[0,0,0])

    def __dealloc__(self):
        del self._ptr

    cdef RowMajorArrayPtr[Complex, Three]* ptr(self):
        return self._ptr

cdef cnp.ndarray as_numpy(int ndim, const int* shape, Float* data, owner):
    cdef cnp.npy_intp shape_np[5]  # TODO: ndim
    cdef int i
    for i in range(3):  # TODO: ndim
        shape_np[i] = shape[i]
    # TODO: cnp.NPY_DOUBLE should go in types.pxd
    cdef cnp.ndarray arr = cnp.PyArray_SimpleNewFromData(
        3, shape_np, cnp.NPY_DOUBLE, data)
    assert(cnp.PyArray_SetBaseObject(arr, owner) == 0)
    Py_INCREF(owner)
    return arr
