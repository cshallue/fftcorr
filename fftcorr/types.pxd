# This must be kept in sync with the Float typedef in types.h. Ideally, we'd 
# import the alias directly from types.h, but we need to use it as a primitive
# type in cython (for Float arrays) and there's no way for Cython to know in
# advance that an extern alias is a primitive type.
ctypedef double Float

# The numpy type enum corresponding to the Float typedef above.
# https://numpy.org/doc/stable/reference/c-api/dtype.html
#cimport numpy as cnp
#NP_FLOAT_ENUM = cnp.NPY_DOUBLE
# TODO: I don't think this is really possible in a pxd file; cnp.NPY_DOUBLE is
# a value, not a type, so we can't have an alias to it. We might have
# to make types an actual cython module with a pyx file.
# We also need an alias for np.double, the associated python type.