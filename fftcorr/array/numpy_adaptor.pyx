from cpython cimport PyObject, Py_INCREF
cimport numpy as cnp
cnp.import_array()

import numpy as np

# It's tempting to template these functions, making data a fused type (e.g.
# cython.numeric *), and then infer typenum in here, making these functions even
# more convenient. Unfortunately, while fused_type * works fine, 
# const fused_type * does not: https://github.com/cython/cython/issues/1772
# Then the workarounds get a bit messy (we could wrap the const and
# non-const pointers inside our own fused types) as well as how to map from 
# cython type to typenum enum: cython.typeof(ptr) == cython.typeof(<int *> 0)
# does not seem to work for me, even when ptr is an int *.

cdef cnp.ndarray as_numpy(
        int ndim, const int* shape, int typenum, void* data, base_object):
    cdef cnp.ndarray[cnp.npy_intp] shape_np = np.empty(ndim, dtype=np.intp)
    cdef int i
    for i in range(ndim):
        shape_np[i] = shape[i]
    cdef cnp.ndarray arr = cnp.PyArray_SimpleNewFromData(
        ndim, &shape_np[0], typenum, data)
    assert(cnp.PyArray_SetBaseObject(arr, base_object) == 0)
    Py_INCREF(base_object)
    return arr

cdef cnp.ndarray as_const_numpy(
        int ndim, const int* shape, int typenum, const void* data, base_object):
    cdef cnp.ndarray arr = as_numpy(
        ndim, shape, typenum, <void *>data, base_object)
    arr.setflags(write=False)
    return arr
