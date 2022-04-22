import struct
import time

import numpy as np

from fftcorr.correlate import Correlator
from fftcorr.fftw import (export_wisdom_to_file, forget_wisdom,
                          import_wisdom_from_file)
from fftcorr.grid import ConfigSpaceGrid

ngrid = [64] * 3
posmin = [-100] * 3
posmax = [100] * 3
dens = ConfigSpaceGrid(shape=ngrid, posmin=posmin, posmax=posmax)
print("created grid")
print()

rmax = 250.0
dr = 5
kmax = 0.4
dk = 0.002
maxell = 2
export_wisdom_to_file("/tmp/wisdom0")
c = Correlator.from_grid_spec(dens, rmax, dr, kmax, dk, maxell)
export_wisdom_to_file("/tmp/wisdom1")
c = Correlator.from_grid_spec(dens, rmax, dr, kmax, dk, maxell)
export_wisdom_to_file("/tmp/wisdom2")
forget_wisdom()
export_wisdom_to_file("/tmp/wisdom3")

c = Correlator.from_grid_spec(dens, rmax, dr, kmax, dk, maxell)
forget_wisdom()

import_wisdom_from_file("/tmp/wisdom1")
c = Correlator.from_grid_spec(dens, rmax, dr, kmax, dk, maxell)

try:
    import_wisdom_from_file("/tmp/wisdom-idontexist")
except IOError as e:
    print(e)
