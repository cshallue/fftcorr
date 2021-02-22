from fftcorr.array cimport as_const_numpy
cimport numpy as cnp
cnp.import_array()

import numpy as np

cdef class HistogramList:
    def __cinit__(self, int n, Float minval, Float maxval, Float binsize):
        self._cc_hist_list = new HistogramList_cc(n, minval, maxval, binsize)
        # TODO: could make a simpler function to wrap a 1D array
        self._bins = as_const_numpy(
            1,
            self._cc_hist_list.bins().shape().data(),
            cnp.NPY_DOUBLE,
            self._cc_hist_list.bins().data(),
            self)
        self._counts = as_const_numpy(
            2,
            self._cc_hist_list.counts().shape().data(),
            cnp.NPY_INT,
            self._cc_hist_list.counts().data(),
            self)
        self._hist_values = as_const_numpy(
            2,
            self._cc_hist_list.hist_values().shape().data(),
            cnp.NPY_DOUBLE,
            self._cc_hist_list.hist_values().data(),
            self)

    def __dealloc__(self):
        del self._cc_hist_list

    cdef HistogramList_cc* cc_hist_list(self):
        return self._cc_hist_list

    @property
    def bins(self):
        return self._bins

    @property
    def counts(self):
        return self._counts

    @property
    def hist_values(self):
        return self._hist_values