import fftcorr.grid

print("Imported grid module")
print(dir(fftcorr.grid))
print()

import fftcorr.particle_mesh

print("Imported particle_mesh module")
print(dir(fftcorr.particle_mesh))
print()

import numpy as np

ngrid = [2, 2, 2]
posmin = [2, -5, 77]
posmax = [50, 45, 120]
cell_size = 25
g = fftcorr.grid.ConfigSpaceGrid(ngrid, posmin, posmax, cell_size=cell_size)
print("created grid")
print("metadata =", g.metadata)
d = g.data
print(d.flatten())
print("sum =", np.sum(g))
print()

ma = fftcorr.particle_mesh.MassAssignor(g, window_type=1, buffer_size=100)
print("Created mass assignor")
print(dir(ma))
print("num_added =", ma.num_added)
print("totw =", ma.totw)
print("totwsq =", ma.totwsq)
print("sort time =", ma.sort_time)
print("window time =", ma.window_time)
print()

print("adding particle")
posw = np.array([[10, 10, 100, 12]], dtype=np.float64)
ma.add_particles(posw)
print("num_added =", ma.num_added)
print("totw =", ma.totw)
print("totwsq =", ma.totwsq)
print("we haven't flushed yet, so...")
print("sum =", np.sum(g))
print("sort time =", ma.sort_time)
print("window time =", ma.window_time)
print("metadata =", g.metadata)
print(d.flatten())
print()

print("flushing...")
ma.flush()
print("num_added =", ma.num_added)
print("totw =", ma.totw)
print("totwsq =", ma.totwsq)
print("sum =", np.sum(g))
print("sort time =", ma.sort_time)
print("window time =", ma.window_time)
print("metadata =", g.metadata)
print(d.flatten())
print()

n = 10
pos = np.random.uniform(posmin, posmax, size=(n, 3)).astype(np.float64)
weight = np.arange(n, dtype=np.float64)

# Add as a single posw array.
g.clear()
ma.clear()
posw = np.empty(shape=(n, 4), dtype=np.float64)
posw[:, :3] = pos
posw[:, 3] = weight
ma.add_particles(posw)
print("num_added =", ma.num_added)
print("totw =", ma.totw)
print("sum =", np.sum(g))
print()

# Add as a position and weight array.
g.clear()
ma.clear()
ma.add_particles(pos, weight)
print("num_added =", ma.num_added)
print("totw =", ma.totw)
print("sum =", np.sum(g))
print()

# Add as a position and constant weight.
g.clear()
ma.clear()
ma.add_particles(pos, 5.0)
print("num_added =", ma.num_added)
print("totw =", ma.totw)
print("sum =", np.sum(g))
print()

# Try a bunch of things that shouldn't work.

# Too many dimensions
pos = np.zeros(shape=(2, 3, 4), dtype=np.float64)
try:
    ma.add_particles(pos, 5.0)
except RuntimeError as e:
    print(e)

# Wrong type
pos = np.zeros(shape=(n, 3), dtype=np.float32)
try:
    ma.add_particles(pos, 5.0)
except RuntimeError as e:
    print(e)

# Not c-contiguous.
buf = np.zeros(shape=(n, 4), dtype=np.float64)
pos = buf[:, 0:3]
try:
    ma.add_particles(pos, 5.0)
except RuntimeError as e:
    print(e)

# Not writeable
pos = np.zeros(shape=(n, 4), dtype=np.float64)
pos.setflags(write=False)
try:
    print("adding particles")
    ma.add_particles(pos)
    print("hello")
except RuntimeError as e:
    print(e)
