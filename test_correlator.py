from fftcorr.grid import ConfigSpaceGrid
from fftcorr.particle_mesh import MassAssignor
from fftcorr.correlate import Correlator
from fftcorr.histogram import Histogram

import struct
import numpy as np

DATA_FILE = "./test_data/abacus/smallmass/corrDD.dat"

with open(DATA_FILE, "rb") as f:
    header = struct.unpack_from("dddddddd", f.read(8 * 8))
    posmin = np.array(header[0:3])
    posmax = np.array(header[3:6])
    print(posmin, posmax)

ngrid = [256, 256, 256]
cell_size = 100
window_type = 0
dens = ConfigSpaceGrid(ngrid, posmin, cell_size, window_type)
d = dens.data
print("created grid")
print()

ma = MassAssignor(dens, buffer_size=100)
print("Created mass assignor")
print("count =", ma.count)
print("totw =", ma.totw)
print("totwsq =", ma.totwsq)
print()

galaxies = np.fromfile(DATA_FILE, dtype=np.float64, offset=8 * 8).reshape(
    (-1, 4))
print("Read {} galaxies from {}".format(len(galaxies), DATA_FILE))
print()

# TODO: make this fast
for posw in galaxies:
    ma.add_particle(*posw)
ma.flush()

print("count = {} vs {}".format(ma.count, len(galaxies)))
print("totw = {} vs {}".format(ma.totw, np.sum(galaxies[:, 3])))
print("sum(dens) = {} vs {}".format(dens.sum(), np.sum(d)))
print("totwsq = {} vs {}".format(ma.totwsq, np.sum(galaxies[:, 3]**2)))
print("sumsq(dens) = {} vs {}".format(dens.sumsq(), np.sum(d**2)))
print()

# Normalize
mean = np.mean(dens.data)
d -= mean  # TODO: dens.data -= doesn't work - expected?
d /= mean
print("totw = {} vs {}".format(ma.totw, np.sum(galaxies[:, 3])))
print("sum(dens) = {} vs {}".format(dens.sum(), np.sum(d)))
print("totwsq = {} vs {}".format(ma.totwsq, np.sum(galaxies[:, 3]**2)))
print("sumsq(dens) = {} vs {}".format(dens.sumsq(), np.sum(d**2)))
print()

rmax = 250.0
dr = 5
kmax = 0.4
dk = 0.002
c = Correlator(dens, rmax, kmax)
h = Histogram(n=0, minval=0, maxval=rmax, binsize=dr)
kh = Histogram(n=0, minval=0, maxval=kmax, binsize=dk)
zerolag = c.correlate_iso(h, kh)
print("Done correlating! Zerolag = {:.6e}".format(zerolag))
print(h.bins)
print(h.count)
print(h.accum)
print(kh.bins)
print(kh.count)
print(kh.accum)

print()
