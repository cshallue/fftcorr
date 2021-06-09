#ifndef NUMPY_ADAPTOR_H
#define NUMPY_ADAPTOR_H

#include <algorithm>
#include <array>

#include "Python.h"
#include "numpy/arrayobject.h"

#include "array/row_major_array.h"
#include "numpy_types.h"

// Returns a numpy array wrapper around data pointed to by a RowMajorArrayPtr.
// It is the responsibility of the caller to ensure that underlying memory is
// not freed as long as the returned array is in existence.
template <typename dtype, std::size_t N>
PyObject* as_numpy(RowMajorArrayPtr<dtype, N>& arr) {
  int typenum = TypeNum<dtype>::value;
  if (typenum < 0) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Unsupported data type. See numpy_types.h");
    return NULL;
  }
  std::array<npy_intp, N> shape;
  std::copy(arr.shape().begin(), arr.shape().end(), shape.begin());
  return PyArray_SimpleNewFromData(N, shape.data(), typenum, (void*)arr.data());
}

// As above, but for a const RowMajorArrayPtr.
template <typename dtype, std::size_t N>
PyObject* as_numpy(const RowMajorArrayPtr<dtype, N>& arr) {
  // First wrap as a writeable numpy array, then make it non-writeable.
  PyObject* py_arr = as_numpy(const_cast<RowMajorArrayPtr<dtype, N>&>(arr));
  PyArray_CLEARFLAGS((PyArrayObject*)py_arr, NPY_ARRAY_WRITEABLE);
  return py_arr;
}

#endif  // NUMPY_ADAPTOR_H