import numpy as np

from fftcorr.histogram import HistogramList

h = HistogramList(2, -10, 10, 2)

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
print()

print("Adding some values")
x = np.array([[1, 7], [3, -5]], dtype=np.double)
y = np.array([[11, 77], [33, -55]], dtype=np.double)
h.accumulate(0, x, y)
print(counts)
print(hist_values)
print()

print("Adding some more values")
x = np.array([1, -7], dtype=np.double)
y = np.array([100, -700], dtype=np.double)
h.accumulate(1, x, y)
print(counts)
print(hist_values)
print()

print("Resetting")
h.reset()
print(counts)
print(hist_values)
