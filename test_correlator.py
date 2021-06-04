from fftcorr.grid import ConfigSpaceGrid
from fftcorr.particle_mesh import MassAssignor
from fftcorr.correlate import Correlator
from fftcorr.histogram import HistogramList

import struct
import numpy as np
import time

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
d = dens.data
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
mean = np.mean(dens.data)
d -= mean  # TODO: dens.data -= doesn't work - expected?
d /= mean
print("totw = {} vs {}".format(ma.totw, np.sum(galaxies[:, 3])))
print("totwsq = {} vs {}".format(ma.totwsq, np.sum(galaxies[:, 3]**2)))
print()

rmax = 250.0
dr = 5
kmax = 0.4
dk = 0.002
maxell = 2
c = Correlator(dens, rmax, dr, kmax, dk, maxell)
c.correlate_periodic()
print("Done correlating!")
print(c.correlation_r)
print(c.correlation_counts)

print()
