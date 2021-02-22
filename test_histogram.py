from fftcorr.histogram import HistogramList
import numpy as np

print("Imported HistogramList class")
# print(dir(Wrapper))

h = HistogramList(2, 0, 10, 2)
print(dir(h))

bins = h.bins
counts = h.counts
hist_values = h.hist_values
print(bins)
print(counts)
print(hist_values)
try:
    hist_values[0][0] = 5
except ValueError as e:
    print(e)
print(hist_values)
