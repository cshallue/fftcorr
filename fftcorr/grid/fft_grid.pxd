from fftcorr.array cimport RowMajorArrayPtr
from fftcorr.types cimport Float, array, Three
cimport numpy as cnp


cdef extern from "fft_grid.h":
  cdef cppclass FftGrid_cc "FftGrid":
    FftGrid_cc(array[int, Three]) except +
    void execute_fft()
    void execute_ifft()
    RowMajorArrayPtr[Float, Three]& arr()
    void convolve_with_gaussian(Float sigma)
    void extract_submatrix(RowMajorArrayPtr[Float, Three]* out)
    # void extract_submatrix(RowMajorArray[Float]* out,
    #                        const RowMajorArray[Float]* mult)
    void extract_submatrix_C2R(RowMajorArrayPtr[Float, Three]* out)
    # void extract_submatrix_C2R(RowMajorArrayPtr[Float]* out,
    #                            const RowMajorArrayPtr[Float]* mult)
    int rshape(int i)
    int dshape(int i)
    Float setup_time()
    Float plan_time()
    Float fft_time()
    Float extract_time()
    Float convolve_time()


cdef class FftGrid:
    # Allocate the grid on the heap it would need to have a nullary
    # constructor to allocate it on the stack. TODO: consider this.
    cdef FftGrid_cc *_cc_grid
    cdef cnp.ndarray _data_arr
    cpdef convolve_with_gaussian(self, Float sigma)
    cpdef extract_submatrix(self, Float[:, :, ::1] out)
    cpdef extract_submatrix_c2r(self, Float[:, :, ::1] out)