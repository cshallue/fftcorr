from fftcorr.array import RowMajorArrayPtr3D_Float
import numpy as np

print("Imported RowMajorArrayPtr3D_Float class")
# print(dir(Wrapper))

shape = [2, 3, 4]
realarr = np.arange(24, dtype=np.double).reshape(shape)
wrap = RowMajorArrayPtr3D_Float(realarr)

#print(realarr[1, 1, 1], wrap.at(1, 1, 1))
#print(realarr[1, 1, 2], wrap.at(1, 1, 2))

realarr[1, 1, 1] = 99
#print(realarr[1, 1, 1], wrap.at(1, 1, 1))
#print(realarr[1, 1, 2], wrap.at(1, 1, 2))

from fftcorr.array import RowMajorArrayPtr3D_Complex
print("Imported RowMajorArrayPtr3D_Complex class")

complexarr = np.arange(24, dtype=np.complex128).reshape(shape)
wrapc = RowMajorArrayPtr3D_Complex(complexarr)

#print(complexarr[1, 1, 1], wrapc.at(1, 1, 1))
#print(complexarr[1, 1, 2], wrapc.at(1, 1, 2))

complexarr[1, 1, 1] = 1234 + 5678j
#print(complexarr[1, 1, 1], wrapc.at(1, 1, 1))
#print(complexarr[1, 1, 2], wrapc.at(1, 1, 2))
