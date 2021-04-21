import os.path
import glob

from abacusnbody.data.read_abacus import read_asdf
import asdf
import numpy as np

from fftcorr.grid import ConfigSpaceGrid
from fftcorr.particle_mesh import MassAssignor
from fftcorr.utils import Timer


def read_abacus_halos(file_pattern, grid, verbose=True, buffer_size=10000):
    filenames = sorted(glob.glob(file_pattern))
    if not filenames:
        raise ValueError("Found no files matching {}".format(file_pattern))

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
            print("Found {:,} halos in {:,} files\n".format(
                total_halos, len(filenames)))

        assert box_size > 0
        xmax = box_size / 2
        xmin = -xmax
        if np.any(grid.posmin > xmin) or np.any(grid.posmax < xmax):
            raise ValueError(
                "Grid does not cover halos: grid posmin = {}, halos posmin = {}"
                ", grid posmax = {}, halos posmax = {}".format(
                    grid.posmin, xmin, grid.posmax, xmax))

        # Space for the halos in each file.
        # TODO: the halos are actually float32s, so we could make the mass
        # assignor accept that type without needing to cast.
        pos_buf = np.empty((max_halos, 3), dtype=np.float64, order="C")
        weight_buf = np.empty((max_halos, ), dtype=np.float64, order="C")

    with Timer() as work_timer:
        ma = MassAssignor(grid, buffer_size=buffer_size)
        halos_seen = 0
        halos_skipped = 0
        io_time = 0.0
        unit_time = 0.0
        copy_time = 0.0
        ma_time = 0.0
        for filename in filenames:
            if verbose:
                print("Reading", os.path.basename(filename))
            with asdf.open(filename, lazy_load=True) as af:
                raw_pos = af.tree["data"]["x_com"]
                raw_weight = af.tree["data"]["N"]
                n = raw_pos.shape[0]
                with Timer() as io_timer:
                    # Force the arrays to load now so we can track IO time.
                    _ = raw_pos[0][0]
                    _ = raw_weight[0]
                io_time += io_timer.elapsed
                # Copy into buffers.
                pos = pos_buf[:n]
                weight = weight_buf[:n]
                with Timer() as copy_timer:
                    np.copyto(pos, raw_pos)
                    np.copyto(weight, raw_weight)
                copy_time += copy_timer.elapsed
            with Timer() as unit_timer:
                pos *= box_size
            unit_time += unit_timer.elapsed
            if not grid.is_periodic and (pos.min() < xmin or pos.max() >= xmax):
                num_skip = np.sum(np.logical_or(pos < xmin, pos >= xmax))
                if verbose:
                    print("{:,g} halos falling outside the grid will be "
                          "skipped".format(num_skip))
                halos_skipped += num_skip
            with Timer() as ma_timer:
                ma.add_particles_to_buffer(pos, weight)
                if filename == filenames[-1]:
                    ma.flush()  # Last file.
            ma_time += ma_timer.elapsed
            halos_seen += n

    assert halos_seen == total_halos
    assert ma.count + ma.skipped == halos_seen
    assert ma.skipped = halos_skipped

    if verbose:
        print("Setup time: {:.2f} sec".format(setup_timer.elapsed))
        print("Work time: {:.2f} sec".format(work_timer.elapsed))
        print("  IO time: {:.2f} sec".format(io_time))
        print("  Convert units time: {:.2f} sec".format(unit_time))
        print("  Copy time: {:.2f} sec".format(copy_time))
        print("  Mass assignor time: {:.2f} sec".format(ma_time))
        print("    Sort time: {:.2f} sec".format(ma.sort_time))
        print("    Window time: {:.2f} sec".format(ma.window_time))

    return ma.count


def read_abacus_particles(file_pattern,
                          grid,
                          verbose=True,
                          buffer_size=10000):
    filenames = sorted(glob.glob(file_pattern))
    if not filenames:
        raise ValueError("Found no files matching {}".format(file_pattern))

    with Timer() as work_timer:
        ma = MassAssignor(grid, buffer_size=buffer_size)
        particles_seen = 0
        particles_skipped = 0
        io_time = 0.0
        ma_time = 0.0
        box_size = None
        for filename in filenames:
            if verbose:
                print("Reading", os.path.basename(filename))
            with Timer() as io_timer:
                table = read_asdf(filename,
                                  load_pos=True,
                                  load_vel=False,
                                  dtype=np.float64)  # TODO: support float32
            io_time += io_timer.elapsed

            if box_size is None:
                box_size = table.meta["BoxSize"]
                assert box_size > 0
                xmax = box_size / 2
                xmin = -xmax
            assert box_size == table.meta["BoxSize"]

            if np.any(grid.posmin > xmin) or np.any(grid.posmax < xmax):
                raise ValueError(
                    "Grid does not cover particles: grid posmin = {}, "
                    "particles posmin = {}, grid posmax = {}, particles "
                    "posmax = {}".format(grid.posmin, xmin, grid.posmax, xmax))

            pos = table["pos"].data
            if not grid.is_periodic and (pos.min() < xmin or pos.max() >= xmax):
                num_skip = np.sum(np.logical_or(pos < xmin, pos >= xmax))
                if verbose:
                    print("{:,g} particles falling outside the grid will be "
                          "skipped".format(num_skip))
                particles_skipped += num_skip
            with Timer() as ma_timer:
                ma.add_particles(pos, weight=1.0)
            ma_time += ma_timer.elapsed
            particles_seen += pos.shape[0]

    assert ma.count + ma.skipped == particles_seen
    assert ma.skipped += particles_skipped

    if verbose:
        print("Work time: {:.2f} sec".format(work_timer.elapsed))
        print("  IO time: {:.2f} sec".format(io_time))
        print("  Mass assignor time: {:.2f} sec".format(ma_time))
        print("    Sort time: {:.2f} sec".format(ma.sort_time))
        print("    Window time: {:.2f} sec".format(ma.window_time))

    return ma.count