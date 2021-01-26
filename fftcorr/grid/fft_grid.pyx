from fftcorr.array cimport as_numpy, RowMajorArrayPtr3D_Float
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
        self._data_arr = as_numpy(
            3,
            self._cc_grid.arr().shape().data(),
            self._cc_grid.arr().data(),
            self)

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
        cdef RowMajorArrayPtr3D_Float out_wrap = RowMajorArrayPtr3D_Float(out)
        self._cc_grid.extract_submatrix(out_wrap.ptr())

    cpdef extract_submatrix_c2r(self, out):
        # TODO: I'm creating a new (thin) wrapper class every time this is
        # called. Might want to accept the wrapper class instead.
        cdef RowMajorArrayPtr3D_Float out_wrap = RowMajorArrayPtr3D_Float(out)
        self._cc_grid.extract_submatrix_C2R(out_wrap.ptr())



