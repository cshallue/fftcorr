from fftcorr.array cimport as_const_numpy
cimport numpy as cnp
cnp.import_array()

import numpy as np

cdef class Histogram:
    def __cinit__(self, int n, Float minval, Float maxval, Float binsize):
        self._cc_hist = new Histogram_cc(n, minval, maxval, binsize)
        # TODO: could make a simpler function to wrap a 1D array
        self._bins = as_const_numpy(
            1,
            self._cc_hist.bins().shape().data(),
            cnp.NPY_DOUBLE,
            self._cc_hist.bins().data(),
            self)
        self._count = as_const_numpy(
            1,
            self._cc_hist.count().shape().data(),
            cnp.NPY_INT,
            self._cc_hist.count().data(),
            self)
        self._accum = as_const_numpy(
            2,
            self._cc_hist.accum().shape().data(),
            cnp.NPY_DOUBLE,
            self._cc_hist.accum().data(),
            self)

    def __dealloc__(self):
        del self._cc_hist

    @property
    def bins(self):
        return self._bins

    @property
    def count(self):
        return self._count

    @property
    def accum(self):
        return self._accum