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
cell_size = 25
g = fftcorr.grid.ConfigSpaceGrid(ngrid, posmin, cell_size, window_type=0)
print("created grid")
d = g.data
print(d.flatten())
print("sum =", g.sum())
print("sumsq =", g.sumsq())
print()

ma = fftcorr.particle_mesh.MassAssignor(g, buffer_size=100)
print("Created mass assignor")
print(dir(ma))
print("count =", ma.count)
print("totw =", ma.totw)
print("totwsq =", ma.totwsq)
print()

print("adding particle")
ma.add_particle(10, 10, 100, 12)
print("count =", ma.count)
print("totw =", ma.totw)
print("totwsq =", ma.totwsq)
print("we haven't flushed yet, so...")
print("sum =", g.sum())
print("sumsq =", g.sumsq())
print(d.flatten())
print()

print("flushing...")
ma.flush()
print("count =", ma.count)
print("totw =", ma.totw)
print("totwsq =", ma.totwsq)
print("sum =", g.sum())
print("sumsq =", g.sumsq())
print(d.flatten())
print()