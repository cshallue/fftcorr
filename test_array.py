from fftcorr.array import RowMajorArrayPtr_Float
import numpy as np

print("Imported RowMajorArrayPtr_Float class")
# print(dir(Wrapper))

shape = [2, 3, 4]
realarr = np.arange(24, dtype=np.double).reshape(shape)
wrap = RowMajorArrayPtr_Float(realarr)

print(realarr[1, 1, 1], wrap.at(1, 1, 1))
print(realarr[1, 1, 2], wrap.at(1, 1, 2))

realarr[1, 1, 1] = 99
print(realarr[1, 1, 1], wrap.at(1, 1, 1))
print(realarr[1, 1, 2], wrap.at(1, 1, 2))

from fftcorr.array import RowMajorArrayPtr_Complex
print("Imported RowMajorArrayPtr_Complex class")

complexarr = np.arange(24, dtype=np.complex128).reshape(shape)
wrapc = RowMajorArrayPtr_Complex(complexarr)

print(complexarr[1, 1, 1], wrapc.at(1, 1, 1))
print(complexarr[1, 1, 2], wrapc.at(1, 1, 2))

complexarr[1, 1, 1] = 1234 + 5678j
print(complexarr[1, 1, 1], wrapc.at(1, 1, 1))
print(complexarr[1, 1, 2], wrapc.at(1, 1, 2))
