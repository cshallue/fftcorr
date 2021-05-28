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
g = fftcorr.grid.ConfigSpaceGrid(ngrid,
                                 posmin,
                                 posmax,
                                 cell_size=cell_size,
                                 window_type=0)
print("created grid")
d = g.data
print(d.flatten())
print("sum =", np.sum(g))
print()

ma = fftcorr.particle_mesh.MassAssignor(g, buffer_size=100)
print("Created mass assignor")
print(dir(ma))
print("num_added =", ma.num_added)
print("totw =", ma.totw)
print("totwsq =", ma.totwsq)
print("sort time =", ma.sort_time)
print("window time =", ma.window_time)
print()

print("adding particle")
ma.add_particle_to_buffer(10, 10, 100, 12)
print("num_added =", ma.num_added)
print("totw =", ma.totw)
print("totwsq =", ma.totwsq)
print("we haven't flushed yet, so...")
print("sum =", np.sum(g))
print("sort time =", ma.sort_time)
print("window time =", ma.window_time)
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

# Test displacement field.
g.clear()
disp = np.random.uniform(-1.0, 1.0, size=ngrid + [3])
ma = fftcorr.particle_mesh.MassAssignor(g, buffer_size=100, disp=disp)
ma.add_particles(pos, 1.0)
print("num_added =", ma.num_added)
print("totw =", ma.totw)
print("sum =", np.sum(g))
print()

# Test displacement field with large displacements that push outside the boundary.
g.clear()
ma.clear()
disp = np.random.uniform(-1000.0, 1000.0, size=ngrid + [3])
ma = fftcorr.particle_mesh.MassAssignor(g, buffer_size=100, disp=disp)
ma.add_particles(pos, 1.0)
print("num_added =", ma.num_added)
print("totw =", ma.totw)
print("sum =", np.sum(g))