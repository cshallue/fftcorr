from fftcorr.array.numpy_adaptor cimport as_numpy

# TODO: does numpy_adaptor take care of this? 
cimport numpy as cnp
cnp.import_array()

import numpy as np

cdef class HistogramList:
    def __cinit__(self, int n, Float minval, Float maxval, Float binsize):
        self._cc_hist_list = new HistogramList_cc(n, minval, maxval, binsize)
        self._bins = as_numpy(self._cc_hist_list.bins())
        self._counts = as_numpy(self._cc_hist_list.counts())
        self._hist_values = as_numpy(self._cc_hist_list.hist_values())

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