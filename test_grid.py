import fftcorr.grid
import numpy as np

print("Imported grid module")
print(dir(fftcorr.grid))

ngrid = [2, 2, 2]
posmin = [2, -5, 77]
posmax = [50, 45, 120]
cell_size = 25
g = fftcorr.grid.ConfigSpaceGrid(shape=ngrid,
                                 posmin=posmin,
                                 posmax=posmax,
                                 cell_size=cell_size,
                                 window_type=0)
print(g.cell_size)
print(g.posmin)
print(g.posmax)
print()

print("{0:x}".format(id(g)))
print(dir(g))
print("posmin =", posmin)
print("cell_size =", g.cell_size)

print("clearing")
g.clear()

print(g.data.flags)
print("data =", g.data.flatten())
import sys
print("g ref count =", sys.getrefcount(g))
d = g.data
print("type(d) =", type(d))
print("g ref count =", sys.getrefcount(g))
d += 7
print("data =", g.data.flatten())
d[0][0][0] = 2
d[1][1][1] = -5
print("data =", g.data.flatten())

d2 = g.data
print("{0:x}".format(id(d)))
print("{0:x}".format(id(d2)))

print(d.flatten())
print()

g2 = fftcorr.grid.ConfigSpaceGrid(shape=ngrid,
                                  posmin=posmin,
                                  posmax=[4, 5, 79],
                                  window_type=0)
print(g2.cell_size)
print(g2.posmin)
print(g2.posmax)
print()

try:
    fftcorr.grid.ConfigSpaceGrid(shape=[2, 2],
                                 posmin=posmin,
                                 posmax=posmax,
                                 cell_size=cell_size,
                                 window_type=0)
except ValueError as e:
    print(e)

try:
    fftcorr.grid.ConfigSpaceGrid(shape=ngrid,
                                 posmin=posmin,
                                 posmax=[4, 5, 79],
                                 cell_size=4,
                                 window_type=0)
except ValueError as e:
    print(e)

try:
    fftcorr.grid.ConfigSpaceGrid(shape=ngrid,
                                 posmin=posmin,
                                 posmax=[4, 5, 72],
                                 window_type=0)
except ValueError as e:
    print(e)

print()
ngrid = [4, 4, 4]
fg = fftcorr.grid.FftGrid(ngrid)
fgd = fg.data
print("type(fgd) =", type(fgd))
print(ngrid, fgd.shape)
fgd += np.arange(96).reshape(fgd.shape)
print(fg.data[0])

subm = np.zeros((2, 2, 2), dtype=np.double)
print(subm.flatten())
fg.extract_submatrix(subm)
print(subm.flatten())

subm = np.zeros((2, 2, 2), dtype=np.double)
print(subm.flatten())
fg.extract_submatrix_c2r(subm)
print(subm.flatten())

print()
print(fg.data[0])
fg.execute_fft()
print(fg.data[0])
fg.execute_ifft()
print(fg.data[0])
print(fg.data[0] / np.prod(ngrid))

print()
np.copyto(fgd, np.arange(96).reshape(fgd.shape))
print(fg.data[0])
print(np.sum(fg.data[:, :, :-2]))
fg.convolve_with_gaussian(1.0)
print(np.sum(fg.data[:, :, :-2]))
print(fg.data[0])
print("Setup time:", fg.setup_time)
print("Plan time:", fg.plan_time)
print("FFT time:", fg.fft_time)
print("Extract time:", fg.extract_time)
print("Convolve time:", fg.convolve_time)