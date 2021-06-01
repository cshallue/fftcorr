from fftcorr.grid import ConfigSpaceGrid
from fftcorr.correlate import Correlator
from fftcorr.fftw import import_wisdom_from_file, export_wisdom_to_file, forget_wisdom

import struct
import numpy as np
import time

ngrid = [64] * 3
posmin = [-100] * 3
posmax = [100] * 3
window_type = 0
dens = ConfigSpaceGrid(shape=ngrid,
                       posmin=posmin,
                       posmax=posmax,
                       window_type=window_type)
print("created grid")
print()

rmax = 250.0
dr = 5
kmax = 0.4
dk = 0.002
maxell = 2
export_wisdom_to_file("/tmp/wisdom0")
c = Correlator(dens, rmax, dr, kmax, dk, maxell)
export_wisdom_to_file("/tmp/wisdom1")
c = Correlator(dens, rmax, dr, kmax, dk, maxell)
export_wisdom_to_file("/tmp/wisdom2")
forget_wisdom()
export_wisdom_to_file("/tmp/wisdom3")

c = Correlator(dens, rmax, dr, kmax, dk, maxell)
forget_wisdom()

import_wisdom_from_file("/tmp/wisdom1")
c = Correlator(dens, rmax, dr, kmax, dk, maxell)

try:
    import_wisdom_from_file("/tmp/wisdom-idontexist")
except IOError as e:
    print(e)