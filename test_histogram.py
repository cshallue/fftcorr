from fftcorr.histogram import Histogram
import numpy as np

print("Imported Histogram class")
# print(dir(Wrapper))

h = Histogram(2, 0, 10, 2)
print(dir(h))

bins = h.bins
count = h.count
accum = h.accum
print(bins)
print(count)
print(accum)
try:
    accum[0][0] = 5
except ValueError as e:
    print(e)
print(accum)
