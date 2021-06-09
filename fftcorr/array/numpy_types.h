#ifndef NUMPY_TYPES_H
#define NUMPY_TYPES_H

#include "numpy/arrayobject.h"

// Base template. Specializations are defined below for supported types.
template <typename dtype>
struct TypeNum {
  static const int value = -1;  // Unknown type.
};

#define REGISTER_NPY_DTYPE(ctype, type_num) \
  template <>                               \
  struct TypeNum<ctype> {                   \
    static const int value = type_num;      \
  };

// Supported types are registered below.
// See https://numpy.org/doc/stable/reference/c-api/dtype.html
// Avoiding types that are aliases of other types on some platforms.

// (Un)signed integer.
REGISTER_NPY_DTYPE(npy_byte, NPY_BYTE);
REGISTER_NPY_DTYPE(npy_ubyte, NPY_UBYTE);
REGISTER_NPY_DTYPE(npy_short, NPY_SHORT);
REGISTER_NPY_DTYPE(npy_ushort, NPY_USHORT);
REGISTER_NPY_DTYPE(npy_int, NPY_INT);
REGISTER_NPY_DTYPE(npy_uint, NPY_UINT);
REGISTER_NPY_DTYPE(npy_longlong, NPY_LONGLONG);
REGISTER_NPY_DTYPE(npy_ulonglong, NPY_ULONGLONG);

// (Complex) Floating point.
REGISTER_NPY_DTYPE(npy_float, NPY_FLOAT);
REGISTER_NPY_DTYPE(npy_cfloat, NPY_CFLOAT);
REGISTER_NPY_DTYPE(npy_double, NPY_DOUBLE);
REGISTER_NPY_DTYPE(npy_cdouble, NPY_CDOUBLE);

#endif  // NUMPY_TYPES_H