import os.path
import glob

import asdf
import numpy as np

from fftcorr.grid import ConfigSpaceGrid
from fftcorr.particle_mesh import MassAssignor
from fftcorr.utils import Timer


def read_abacus_halos(file_pattern,
                      grid,
                      convert_units=True,
                      wrap_boundaries=False,
                      verbose=True):
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
        if verbose:
            print("Found {} halos in {} files\n".format(
                total_halos, len(filenames)))

        if not convert_units:
            box_size = 1.0

        xmax = box_size / 2
        xmin = -xmax
        if np.any(grid.posmin > xmin) or np.any(grid.posmax < xmax):
            raise ValueError(
                "Grid does not cover halos: grid posmin = {}, halos posmin = {}"
                ", grid posmax = {}, halos posmax ={}".format(
                    grid.posmin, xmin, grid.posmax, xmax))

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
                n = pos.shape[0]
                posw = buffer[:n]
                with Timer() as io_timer:
                    # Force the arrays to load now so we can track IO time.
                    _ = pos[0][0]
                    _ = weight[0]
                io_time += io_timer.elapsed
                with Timer() as copy_timer:
                    np.copyto(posw[:, :3], pos)
                    np.copyto(posw[:, 3], weight)
                copy_time += copy_timer.elapsed
                if convert_units:
                    with Timer() as unit_timer:
                        posw[:, :3] *= box_size
                    unit_time += unit_timer.elapsed
                if wrap_boundaries:
                    with Timer() as wrap_timer:
                        # Most files don't spill the boundary.
                        if (posw[:, :3].min() < -xmax
                                or posw[:, :3].max() >= xmax):
                            posw[:, :3] += xmax
                            np.mod(posw[:, :3], xmax, out=posw[:, :3])
                            posw[:, :3] -= xmax
                    wrap_time += wrap_timer.elapsed
                with Timer() as ma_timer:
                    ma.add_particles(posw)
                ma_time += ma_timer.elapsed
                halos_added += n

    assert total_halos == halos_added
    assert total_halos == ma.count + ma.skipped
    if wrap_boundaries:
        assert ma.skipped == 0

    if verbose:
        print("Setup time: {:.2f} sec".format(setup_timer.elapsed))
        print("Work time: {:.2f} sec".format(work_timer.elapsed))
        print("  IO time: {:.2f} sec".format(io_time))
        if convert_units:
            print("  Convert units time: {:.2f} sec".format(unit_time))
        if wrap_boundaries:
            print("  Wrap time: {:.2f} sec".format(wrap_time))
        print("  Copy time: {:.2f} sec".format(copy_time))
        print("  Mass assignor time: {:.2f} sec".format(ma_time))
        print("    Sort time: {:.2f} sec".format(ma.sort_time))
        print("    Window time: {:.2f} sec".format(ma.window_time))

    return total_halos
