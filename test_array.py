from fftcorr.array import Wrapper
import numpy as np

print("Imported Wrapper class")
# print(dir(Wrapper))

shape = [2, 3, 4]
realarr = np.arange(24, dtype=np.double).reshape(shape)
wrap = Wrapper(realarr)

print(realarr[1, 1, 1], wrap.at(1, 1, 1))
print(realarr[1, 1, 2], wrap.at(1, 1, 2))

realarr[1, 1, 1] = 99
print(realarr[1, 1, 1], wrap.at(1, 1, 1))
print(realarr[1, 1, 2], wrap.at(1, 1, 2))