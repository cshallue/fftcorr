from fftcorr.types cimport uint64

cdef extern from "array.h":
  cdef cppclass Array[dtype]:
    uint64 size()
    dtype* data()