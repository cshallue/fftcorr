cimport numpy as cnp

cdef cnp.ndarray as_numpy(
    int ndim, const int* shape, int typenum, void* data, base_object)

cdef cnp.ndarray as_const_numpy(
    int ndim, const int* shape, int typenum, const void* data, base_object)
