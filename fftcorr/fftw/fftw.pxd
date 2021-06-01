cdef extern from 'fftw3.h':

    int fftw_export_wisdom_to_filename(char *filename)
    int fftw_import_wisdom_from_filename(char *filename)
    void fftw_forget_wisdom()

# TODO: figure out what to do with these
cdef enum:
    FFTW_MEASURE = 0
    FFTW_DESTROY_INPUT = 1
    FFTW_UNALIGNED = 2
    FFTW_CONSERVE_MEMORY = 4
    FFTW_EXHAUSTIVE = 8
    FFTW_PRESERVE_INPUT = 16
    FFTW_PATIENT = 32
    FFTW_ESTIMATE = 64
    FFTW_WISDOM_ONLY = 2097152