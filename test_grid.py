import py_grid
import numpy as np

print("Imported grid module")
print(dir(py_grid))

ngrid = [2, 2, 2]
posmin = [2, -5, 77]
cell_size = 25
g = py_grid.ConfigSpaceGrid(ngrid, posmin, cell_size)
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
    py_grid.ConfigSpaceGrid([2, 2], posmin, cell_size)
except ValueError as e:
    print(e)