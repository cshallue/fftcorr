import struct
import time

import numpy as np

from fftcorr.correlate import Correlator
from fftcorr.grid import ConfigSpaceGrid
from fftcorr.particle_mesh import MassAssignor

DATA_FILE = "./test_data/abacus/smallmass/corrDD.dat"

with open(DATA_FILE, "rb") as f:
    header = struct.unpack_from("dddddddd", f.read(8 * 8))
    posmin = np.array(header[0:3])
    posmax = np.array(header[3:6])
    print(posmin, posmax)

ngrid = [256, 256, 256]
cell_size = 100
window_type = 0
dens = ConfigSpaceGrid(shape=ngrid,
                       posmin=posmin,
                       posmax=posmax,
                       cell_size=cell_size,
                       window_type=window_type)
print("created grid")
print()

ma = MassAssignor(dens, buffer_size=100)
print("Created mass assignor")
print("num_added =", ma.num_added)
print("totw =", ma.totw)
print("totwsq =", ma.totwsq)
print()

galaxies = np.fromfile(DATA_FILE, dtype=np.float64, offset=8 * 8).reshape(
    (-1, 4))
print("Read {} galaxies from {}".format(len(galaxies), DATA_FILE))
print()

start = time.time()
ma.add_particles(galaxies)
print("Mass assignment took {:.2f} seconds".format(time.time() - start))

print("num_added = {} vs {}".format(ma.num_added, len(galaxies)))
print("totw = {} vs {}".format(ma.totw, np.sum(galaxies[:, 3])))
print("totwsq = {} vs {}".format(ma.totwsq, np.sum(galaxies[:, 3]**2)))
print()

# Normalize
mean = np.mean(dens)
dens -= mean
dens /= mean
print("totw = {} vs {}".format(ma.totw, np.sum(galaxies[:, 3])))
print("totwsq = {} vs {}".format(ma.totwsq, np.sum(galaxies[:, 3]**2)))
print()

rmax = 250.0
dr = 10
kmax = 0.4
dk = 0.02
maxell = 2
c = Correlator.from_grid_spec(dens, rmax, dr, kmax, dk, maxell)
print("Computing autocorrelation!")
ps, corr = c.autocorrelate(dens.data)
print("Done correlating!")
print(corr)
print(ps)
print()

dens2 = ConfigSpaceGrid(shape=ngrid,
                        posmin=posmin,
                        posmax=posmax,
                        cell_size=cell_size,
                        window_type=window_type)
np.copyto(dens2.data, np.random.uniform(size=dens2.shape))
print("Computing cross correlation!")
ps, corr = c.cross_correlate(dens.data, dens2.data)
print("Done correlating!")
print(corr)
print(ps)
print()

print("Computing cross correlation!")
c.set_dens2(dens2.data)
ps, corr = c.cross_correlate(dens.data)
print("Done correlating!")
print(corr)
print(ps)
print()
