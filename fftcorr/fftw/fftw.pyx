def export_wisdom_to_file(str filename):
    cdef bytes fn = filename.encode('utf-8')
    cdef bint success = fftw_export_wisdom_to_filename(fn)
    if not success:
        raise IOError("Failed to import wisdom file")

def import_wisdom_from_file(str filename):
    cdef bytes fn = filename.encode('utf-8')
    cdef bint success = fftw_import_wisdom_from_filename(fn)
    if not success:
        raise IOError("Failed to export wisdom file")

def forget_wisdom():
    fftw_forget_wisdom()