import ctypes

import numpy as np

import fftcorr.grid

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

g_refcount = ctypes.c_long.from_address(id(g)).value
g_data_refcount = ctypes.c_long.from_address(id(g.data)).value
print(f"Address of g: {id(g):x}. Refcount: {g_refcount}")
print(f"Address of g.data: {id(g.data):x}. Refcount: {g_data_refcount}")

d = g.data
g_refcount = ctypes.c_long.from_address(id(g)).value
g_data_refcount = ctypes.c_long.from_address(id(g.data)).value
print(f"Address of d: {id(d):x}. g Refcount: {g_refcount}. g.data refcount: "
      f"{g_data_refcount}")
print()

print("type(d) =", type(d))
print("data =", g.data.flatten())
print(g.data.flags)

d2 = g.data
print("{0:x}".format(id(d)))
print("{0:x}".format(id(d2)))
print()

# Interact with g through the numpy array
print("data =", g.data.flatten())
d += 7
print("data =", g.data.flatten())
d[0][0][0] = 2
d[1][1][1] = -5
print("data =", g.data.flatten())
print()

print("clearing")
g.clear()
print("data =", g.data.flatten())

# Interact with g through its exposed inplace methods
d += -2
print("data =", g.data.flatten())
d *= -3
print("data =", g.data.flatten())
d /= 6
print("data =", g.data.flatten())
d -= 2
print("data =", g.data.flatten())

print("sum of g:", np.sum(g))
print("mean of g:", np.mean(g))

try:
    # This shouldn't work: we only implemented inplace add.
    x = g + 1
except TypeError as e:
    print(e)

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

#posmin = [2, -5, 77]
#posmax = [50, 45, 120]
gc = g.to_grid_coords([2, -4, 79])
print(gc)
try:
    g.to_grid_coords([1, 0, 77])
except ValueError as e:
    print(e)
gc = g.to_grid_coords([1, 0, 77], periodic_wrap=True)
print(gc)

# Try some displacements.
pos = np.array([[2, -5, 77], [49, 44, 119]], dtype=np.float64)
print(pos)
for p in pos:
    print(g.to_grid_coords(p))
disp = np.random.uniform(size=(ngrid + [3]))
print(disp)
fftcorr.grid.apply_displacement_field(g, pos, disp, out=pos)
print(pos)
