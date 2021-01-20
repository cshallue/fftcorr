from fftcorr.array cimport RowMajorArrayPtr
from fftcorr.types cimport Float, array, Three
cimport numpy as cnp


cdef extern from "fft_grid.h":
  cdef cppclass cc_FftGrid "FftGrid":
    cc_FftGrid(array[int, Three]) except +
    void setup_fft()
    void execute_fft()
    void execute_ifft()
    void extract_submatrix(RowMajorArrayPtr[Float]* out)
    # void extract_submatrix(RowMajorArray[Float]* out,
    #                        const RowMajorArray[Float]* mult)
    void extract_submatrix_C2R(RowMajorArrayPtr[Float]* out)
    # void extract_submatrix_C2R(RowMajorArrayPtr[Float]* out,
    #                            const RowMajorArrayPtr[Float]* mult)
    int rshape(int i)
    int dshape(int i)
    Float* raw_data()


cdef class FftGrid:
    # Allocate the grid on the heap it would need to have a nullary
    # constructor to allocate it on the stack. TODO: consider this.
    cdef cc_FftGrid *_cc_grid
    cdef cnp.ndarray _data_arr
    cpdef extract_submatrix(self, out)
    cpdef extract_submatrix_c2r(self, out)