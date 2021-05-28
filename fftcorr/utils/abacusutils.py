import os.path
import glob

from abacusnbody.data.bitpacked import unpack_rvint
import asdf
import numpy as np

from fftcorr.particle_mesh import MassAssignor
from fftcorr.utils import Timer


def read_density_field(file_pattern,
                       grid,
                       file_type=None,
                       redshift_distortion=False,
                       displacement_field=None,
                       periodic_wrap=True,
                       bounds_error="warn",
                       verbose=True,
                       buffer_size=10000):
    filenames = sorted(glob.glob(file_pattern))
    if not filenames:
        raise ValueError("Found no files matching {}".format(file_pattern))

    # Infer and/or validate the file type: halos or particles.
    for filename in filenames:
        basename = os.path.basename(filename)
        if basename.startswith("halo_info"):
            ft = "halos"
        elif basename.startswith("field_rv"):
            ft = "particles"
        else:
            raise ValueError("Unrecognized file type: '{}'".format(basename))
        if file_type is None:
            file_type = ft
        elif file_type != ft:
            raise ValueError("Inconsistent file types")

    if bounds_error not in ["raise", "warn", "none", None]:
        raise ValueError(
            "Unrecognized bounds_error: '{}'".format(bounds_error))

    gridmin = grid.posmin
    gridmax = grid.posmax
    box_size = None
    total_items = 0
    max_items = 0

    with Timer() as setup_timer:
        # First, open all files and figure out the max number of items in a
        # single file.
        for filename in filenames:
            with asdf.open(filename, lazy_load=True) as af:
                if box_size is None:
                    box_size = af.tree["header"]["BoxSize"]
                assert box_size == af.tree["header"]["BoxSize"]
                key = "N" if file_type == "halos" else "rvint"
                n = af.tree["data"][key].shape[0]  # Doesn't load data.
                total_items += n
                max_items = max(max_items, n)
        if verbose:
            print("Found {:,} items in {:,} files".format(
                total_items, len(filenames)))

        assert box_size > 0
        if np.any(gridmin > -box_size / 2) or np.any(gridmax < box_size / 2):
            raise ValueError(
                "Grid does not cover {}: grid posmin = {}, file posmin = {},"
                "grid posmax = {}, file posmax = {}".format(
                    file_type, gridmin, -box_size / 2, gridmax, box_size / 2))

        # Space for the items in each file.
        # TODO: the items are actually float32s, so we could make the mass
        # assignor accept that type without needing to cast. But we'd still need
        # to copy because asdf arrays are read only.
        pos_buf = np.empty((max_items, 3), dtype=np.float64, order="C")
        vel_buf = None
        if redshift_distortion:
            vel_buf = np.empty((max_items, 3), dtype=np.float64, order="C")
        weight_buf = None
        if file_type == "halos":
            weight_buf = np.empty((max_items, ), dtype=np.float64, order="C")

    with Timer() as work_timer:
        ma = MassAssignor(grid, periodic_wrap, buffer_size, displacement_field)
        items_seen = 0
        items_skipped = 0
        io_time = 0.0
        ma_time = 0.0
        for filename in filenames:
            if verbose:
                print("Reading", os.path.basename(filename))

            # Load positions and weights.
            with asdf.open(filename, lazy_load=True) as af:
                key = "N" if file_type == "halos" else "rvint"
                n = af.tree["data"][key].shape[0]
                pos = pos_buf[:n]
                vel = vel_buf[:n] if redshift_distortion else False
                scale_factor = af.tree["header"]["ScaleFactor"]
                with Timer() as io_timer:
                    if file_type == "halos":
                        np.copyto(pos, af.tree["data"]["x_com"])
                        pos *= box_size
                        weight = weight_buf[:n]
                        np.copyto(weight, af.tree["data"]["N"])
                        if redshift_distortion:
                            np.copyto(vel, af.tree["data"]["v_com"])
                    else:  # file_type == "particles"
                        npos, nvel = unpack_rvint(af.tree["data"]["rvint"],
                                                  box_size,
                                                  float_dtype=np.float64,
                                                  posout=pos,
                                                  velout=vel)
                        weight = 1.0
                        assert npos == n
                        assert nvel == (n if redshift_distortion else 0)
                io_time += io_timer.elapsed

            # Apply redshift distortions.
            if redshift_distortion:
                assert vel.shape == (n, 3)
                # Apply redshift distortions in the z direction because that is
                # the direction from which the polar angle is defined in the
                # correlator. If desired in the future, we could accept any
                # arbitrary direction vector and apply redshift distortion in
                # that direction.
                pos[:, 2] += vel[:, 2] / (100 * scale_factor)

            # Check if any items fall outside the grid.
            if (np.any(pos.min(axis=0) < gridmin)
                    or np.any(pos.max(axis=0) >= gridmax)):
                num_outside = np.sum(
                    np.logical_or(np.any(pos < gridmin, axis=1),
                                  np.any(pos >= gridmax, axis=1)))
                msg = "{:,g} {} falling outside the grid".format(
                    num_outside, file_type)
                if bounds_error == "raise":
                    raise ValueError(msg)
                elif bounds_error == "warn":
                    print(msg)
                if not periodic_wrap:
                    items_skipped += num_outside

            # Add items to the density field.
            with Timer() as ma_timer:
                ma.add_particles_to_buffer(pos, weight)
                if filename == filenames[-1]:
                    ma.flush()  # Last file.
            ma_time += ma_timer.elapsed
            items_seen += n

    assert items_seen == total_items
    assert ma.num_added + ma.num_skipped == items_seen
    if displacement_field is None:
        assert ma.num_skipped == items_skipped

    if verbose:
        print("Setup time: {:.2f} sec".format(setup_timer.elapsed))
        print("Work time: {:.2f} sec".format(work_timer.elapsed))
        print("  IO time: {:.2f} sec".format(io_time))
        print("  Mass assignor time: {:.2f} sec".format(ma_time))
        print("    Sort time: {:.2f} sec".format(ma.sort_time))
        print("    Window time: {:.2f} sec".format(ma.window_time))

    return ma.num_added, ma.num_skipped
