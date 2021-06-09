from fftcorr.histogram import HistogramList
import numpy as np

h = HistogramList(2, 0, 10, 2)

bins = h.bins
counts = h.counts
hist_values = h.hist_values
print(hist_values.flags)

print(bins)
print(counts)
print(hist_values)
try:
    hist_values[0][0] = 5
    print("Changed the value of hist_values[0][0]")
    print(hist_values)
except ValueError as e:
    print(e)
print(hist_values)
