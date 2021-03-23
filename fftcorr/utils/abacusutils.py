import os.path
import glob

import asdf
import numpy as np

from fftcorr.grid import ConfigSpaceGrid
from fftcorr.particle_mesh import MassAssignor
from fftcorr.utils import Timer


def read_abacus_halos(file_pattern,
                      ngrid,
                      window_type,
                      convert_units=True,
                      wrap_boundaries=False):
    filenames = glob.glob(file_pattern)
    box_size = None
    total_halos = 0
    max_halos = 0

    with Timer() as setup_timer:
        # First, open all files and figure out the max number of halos in a
        # single file.
        for filename in filenames:
            with asdf.open(filename, lazy_load=True) as af:
                if box_size is None:
                    box_size = af.tree["header"]["BoxSize"]
                assert box_size == af.tree["header"]["BoxSize"]
                n = af.tree["data"]["N"].shape[0]  # Doesn't load data.
                total_halos += n
                max_halos = max(max_halos, n)
        print("Found {} halos in {} files".format(total_halos, len(filenames)))

        if not convert_units:
            box_size = 1.0

        print("Box size:", box_size)
        xmax = box_size / 2
        posmin = [-xmax] * 3
        posmax = [xmax] * 3
        grid = ConfigSpaceGrid(shape=ngrid,
                               posmin=posmin,
                               posmax=posmax,
                               window_type=window_type)
        print()

        # Space for the halos in each file.
        # TODO: the halos are actually float32s, so we could make the mass
        # assignor accept that type as well.
        buffer = np.empty((max_halos, 4), dtype=np.float64, order="C")

    with Timer() as work_timer:
        ma = MassAssignor(grid, buffer_size=10000)
        halos_added = 0
        io_time = 0.0
        unit_time = 0.0
        wrap_time = 0.0
        copy_time = 0.0
        ma_time = 0.0
        for filename in filenames:
            print("Reading", os.path.basename(filename))
            with asdf.open(filename, lazy_load=True) as af:
                pos = af.tree["data"]["x_com"]
                weight = af.tree["data"]["N"]
                with Timer() as io_timer:
                    # Force the arrays to load now so we can track IO time.
                    _ = pos[0][0]
                    _ = weight[0][0]
                io_time += io_timer.elapsed
                if convert_units:
                    with Timer() as unit_timer:
                        pos *= box_size
                    unit_time += unit_timer.elapsed
                if wrap_boundaries:
                    with Timer() as wrap_timer:
                        pos += xmax
                        np.mod(pos, xmax, out=pos)
                        pos -= xmax
                    wrap_time += wrap_timer.elapsed
                n = pos.shape[0]
                posw = buffer[:n]
                with Timer() as copy_timer:
                    np.copyto(posw[:, :3], pos)
                    np.copyto(posw[:, 3], weight)
                copy_time += copy_timer.elapsed
                with Timer() as ma_timer:
                    ma.add_particles(posw)
                ma_time += ma_timer.elapsed
                halos_added += n
                total_weight += np.sum(weight)

    assert total_halos == halos_added
    assert total_halos == ma.count + ma.skipped
    if wrap_boundaries:
        assert ma.skipped == 0

    print()
    print("Setup time: {:2f} sec".format(setup_timer.elapsed))
    print("Work time: {:2f} sec".format(work_timer.elapsed))
    print("  IO time: {:2f} sec".format(io_time))
    if convert_units:
        print("  Convert units time: {:2f} sec".format(unit_time))
    if wrap_boundaries:
        print("  Wrap time: {:2f} sec".format(wrap_time))
    print("  Copy time: {:2f} sec".format(copy_time))
    print("  Mass assignor time: {:2f} sec".format(ma_time))
    print("    Sort time: {:2f} sec".format(ma.sort_time))
    print("    Window time: {:2f} sec".format(ma.window_time))
    return grid
