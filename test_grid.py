import fftcorr.grid
import numpy as np

print("Imported grid module")
print(dir(fftcorr.grid))

ngrid = [2, 2, 2]
posmin = [2, -5, 77]
cell_size = 25
g = fftcorr.grid.ConfigSpaceGrid(ngrid, posmin, cell_size)
print("{0:x}".format(id(g)))
print(dir(g))
print("posmin =", posmin)
print("cell_size =", g.cell_size)
print("sum =", g.sum())
print("sumsq =", g.sumsq())
print("Adding 3")
g.add_scalar(3)
print("sum =", g.sum())
print("sumsq =", g.sumsq())
print()

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
print("sum =", g.sum())

d2 = g.data
print("{0:x}".format(id(d)))
print("{0:x}".format(id(d2)))

print(d.flatten())

try:
    fftcorr.grid.ConfigSpaceGrid([2, 2], posmin, cell_size)
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
fg.setup_fft()
print(fg.data[0])
fg.execute_fft()
print(fg.data[0])
fg.execute_ifft()
print(fg.data[0])
print(fg.data[0] / np.prod(ngrid))
