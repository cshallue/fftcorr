import numpy as np

from fftcorr.particle_mesh import MassAssignor
from fftcorr.utils import Timer


def add_random_particles(grid,
                         n,
                         particle_weight=None,
                         total_weight=None,
                         transform_coords_fn=None,
                         periodic_wrap=False,
                         bounds_error="warn",
                         verbose=True,
                         batch_size=10000000,
                         buffer_size=10000):
    if total_weight is not None:
        if particle_weight is not None:
            raise ValueError(
                "particle_weight and total_weight cannot both be set")

        particle_weight = total_weight / n

    if particle_weight is None:
        particle_weight = 1.0

    if verbose:
        print("Particle weight: {:.6g}".format(particle_weight))

    if bounds_error not in ["raise", "warn", "none", None]:
        raise ValueError(
            "Unrecognized bounds_error: '{}'".format(bounds_error))

    gridmin = grid.posmin
    gridmax = grid.posmax
    particles_skipped = 0

    with Timer() as setup_timer:
        pos_buf = np.empty((batch_size, 3), dtype=np.float64, order="C")

    with Timer() as work_timer:
        ma = MassAssignor(grid, periodic_wrap, buffer_size)
        particles_added = 0
        rng_time = 0.0
        copy_time = 0.0
        transform_time = 0.0
        ma_time = 0.0
        while particles_added < n:
            nbatch = int(min(batch_size, n - particles_added))
            if verbose:
                print("Adding particles [{:,.6g}, {:,.6g}]".format(
                    particles_added + 1, particles_added + nbatch))

            pos = pos_buf[:nbatch]

            with Timer() as rng_timer:
                rnd = np.random.uniform(gridmin, gridmmax, (nbatch, 3))
            rng_time += rng_timer.elapsed

            with Timer() as copy_timer:
                np.copyto(pos, rnd)
            copy_time += copy_timer.elapsed

            if transform_coords_fn is not None:
                with Timer() as transform_timer:
                    transform_coords_fn(pos)
                transform_time += transform_timer.elapsed

            if (np.any(pos.min(axis=0) < gridmin)
                    or np.any(pos.max(axis=0) >= gridmax)):
                num_outside = np.sum(
                    np.logical_or(np.any(pos < gridmin, axis=1),
                                  np.any(pos >= gridmax, axis=1)))
                msg = "{:,g} particles falling outside the grid".format(
                    num_outside)
                if bounds_error == "raise":
                    raise ValueError(msg)
                elif bounds_error == "warn":
                    print(msg)
                if not periodic_wrap:
                    particles_skipped += num_outside

            with Timer() as ma_timer:
                ma.add_particles_to_buffer(pos, particle_weight)
                particles_added += nbatch
                if particles_added == n:
                    ma.flush()  # Last batch.
            ma_time += ma_timer.elapsed

    assert ma.num_added + ma.num_skipped == n
    assert ma.num_skipped == particles_skipped

    if verbose:
        print("Setup time: {:.2f} sec".format(setup_timer.elapsed))
        print("Work time: {:.2f} sec".format(work_timer.elapsed))
        print("  RNG time: {:.2f} sec".format(rng_time))
        print("  Copy time: {:.2f} sec".format(copy_time))
        if transform_coords_fn is not None:
            print("  Transform coords time: {:.2f} sec".format(transform_time))
        print("  Mass assignor time: {:.2f} sec".format(ma_time))
        print("    Sort time: {:.2f} sec".format(ma.sort_time))
        print("    Window time: {:.2f} sec".format(ma.window_time))

    return ma.totw