import os.path
import glob

import asdf
import numpy as np

from fftcorr.grid import ConfigSpaceGrid
from fftcorr.particle_mesh import MassAssignor
from fftcorr.utils import Timer


def read_abacus_halos(file_pattern, ngrid, window_type):
    filenames = glob.glob(file_pattern)
    box_size_notches = None
    # box_size_hmpc = None
    total_halos = 0
    max_halos = 0

    with Timer() as setup_timer:
        # First, open all files and figure out the max number of halos in a single file.
        for filename in filenames:
            with asdf.open(filename, lazy_load=True) as af:
                if box_size_notches is None:
                    box_size_notches = af.tree["header"]["BoxSize"]
                    # box_size_hmpc = af.tree["header"]["BoxSizeHMpc"]
                n = af.tree["data"]["N"].shape[0]  # Doesn't load data.
                total_halos += n
                max_halos = max(max_halos, n)
        print("Found {} halos in {} files".format(total_halos, len(filenames)))

        # TODO: doing this for now to avoid multiplying the positions by the boxsize
        # but need to figure out a better solution
        box_size_notches = 1.0

        print("Box size:", box_size_notches)
        xmax = box_size_notches / 2
        posmin = [-xmax] * 3
        posmax = [xmax] * 3
        grid = ConfigSpaceGrid(shape=ngrid,
                               posmin=posmin,
                               posmax=posmax,
                               window_type=window_type)
        print()

        # Space for the halos in each file.
        # TODO: the halos are actually float32s, so we could make the mass assignor
        # accept that type as well.
        buffer = np.empty((max_halos, 4), dtype=np.float64, order="C")

    with Timer() as work_timer:
        ma = MassAssignor(grid, buffer_size=10000)
        halos_added = 0
        io_time = 0.0
        ma_time = 0.0
        for filename in filenames:
            print("Reading", os.path.basename(filename))
            with asdf.open(filename, lazy_load=True) as af:
                pos = af.tree["data"]["x_com"]
                weight = af.tree["data"]["N"]
                n = pos.shape[0]
                posw = buffer[:n]
                with Timer() as io_timer:
                    np.copyto(posw[:, :3], pos)
                    np.copyto(posw[:, 3], weight)
                io_time += io_timer.elapsed
                # TODO: this causes a crash for some reason on one particular file!?
                # Might just be OOMing? Is there a better way to do this anyway?
                if box_size_notches != 1.0:
                    assert False  # Not doing this right now.
                    posw[:, :3] *= box_size_notches

                with Timer() as ma_timer:
                    ma.add_particles(posw)
                ma_time += ma_timer.elapsed
                halos_added += n

    assert total_halos == halos_added
    assert total_halos == ma.count + ma.skipped

    print()
    print("Setup time: {:2f} sec".format(setup_timer.elapsed))
    print("Work time: {:2f} sec".format(work_timer.elapsed))
    print("  IO time: {:2f} sec".format(io_time))
    print("  Mass assignor time: {:2f} sec".format(ma_time))
    print("    Sort time: {:2f} sec".format(ma.sort_time))
    print("    Window time: {:2f} sec".format(ma.window_time))
    return grid
