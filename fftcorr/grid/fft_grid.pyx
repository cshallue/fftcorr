from fftcorr.array cimport RowMajorArrayPtr_Float
from cpython cimport Py_INCREF
cimport numpy as cnp
cnp.import_array()

import numpy as np


cdef class FftGrid:
    def __cinit__(self, shape):
        # Convert input arrays to contiguous arrays of the correct data type.
        # Then check for errors, as python objects.
        shape = np.ascontiguousarray(shape, dtype=np.intc)

        if shape.shape != (3, ):
            raise ValueError(
                "Expected shape to have shape (3,), got: {}".format(
                    shape.shape))

        if np.any(shape <= 0):
            raise ValueError(
                "Expected shape to be positive, got: {}".format(shape))

        # Create the wrapped C++ ConfigSpaceGrid.
        cdef cnp.ndarray[int, ndim=1, mode="c"] cshape = shape
        self._cc_grid = new cc_FftGrid((<array[int, Three] *> &cshape[0])[0])
        
        # TODO: we might choose not to even expose this, since it comes with
        # padding. We might just expose operations to do on the grid, and
        # hide the padding under the hood. Then we'd need to declare the array
        # operations in the FftGrid class.
        # Wrap the data array as a numpy array.
        # TODO: helper for this, since I do it multiple times?
        # TODO: really I'm wrapping a RowMajorArray here, which I might do
        # elsewhere, and possibly that can be shared.
        cdef cnp.npy_intp dshape[3]
        for i in range(3):
            dshape[i] = self._cc_grid.dshape(i)
        cdef Float* data_ptr = self._cc_grid.raw_data()
        # TODO: cnp.NPY_DOUBLE should go in types.pxd
        self._data_arr = cnp.PyArray_SimpleNewFromData(
            3, dshape, cnp.NPY_DOUBLE, data_ptr)
        assert(cnp.PyArray_SetBaseObject(self._data_arr, self) == 0)
        Py_INCREF(self)

    def __dealloc__(self):
        del self._cc_grid

    @property
    def data(self):
        return self._data_arr

    def setup_fft(self):
        self._cc_grid.setup_fft()

    def execute_fft(self):
        self._cc_grid.execute_fft()

    def execute_ifft(self):
        self._cc_grid.execute_ifft()

    cpdef extract_submatrix(self, out):
        # TODO: I'm creating a new (thin) wrapper class every time this is
        # called. Might want to accept the wrapper class instead.
        cdef RowMajorArrayPtr_Float out_wrap = RowMajorArrayPtr_Float(out)
        self._cc_grid.extract_submatrix(out_wrap.ptr())

    cpdef extract_submatrix_c2r(self, out):
        # TODO: I'm creating a new (thin) wrapper class every time this is
        # called. Might want to accept the wrapper class instead.
        cdef RowMajorArrayPtr_Float out_wrap = RowMajorArrayPtr_Float(out)
        self._cc_grid.extract_submatrix_C2R(out_wrap.ptr())



