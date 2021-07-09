import numpy as np
from fftcorr.particle_mesh import MassAssignor
from fftcorr.utils import Timer, apply_displacement_field


def add_random_particles(n,
                         grid,
                         particle_weight=None,
                         total_weight=None,
                         periodic_wrap=False,
                         disp=None,
                         batch_size=int(1e8),
                         buffer_size=10000,
                         verbose=True):
    if total_weight is not None:
        if particle_weight is not None:
            raise ValueError(
                "particle_weight and total_weight cannot both be set")

        particle_weight = total_weight / n

    if particle_weight is None:
        particle_weight = 1.0

    if verbose:
        print("Particle weight: {:.6g}".format(particle_weight))

    gridmin = grid.posmin
    gridmax = grid.posmax

    with Timer() as setup_timer:
        pos_buf = np.empty((batch_size, 3), dtype=np.float64, order="C")

    ma = MassAssignor(grid, periodic_wrap, buffer_size)
    with Timer() as work_timer:
        particles_added = 0
        rng_time = 0.0
        disp_time = 0.0
        ma_time = 0.0
        while particles_added < n:
            nbatch = int(min(batch_size, n - particles_added))
            if verbose:
                print("Adding particles [{:,.6g}, {:,.6g}]".format(
                    particles_added + 1, particles_added + nbatch))

            pos = pos_buf[:nbatch]

            with Timer() as rng_timer:
                rnd = np.random.uniform(gridmin, gridmax, (nbatch, 3))
                np.copyto(pos, rnd)
            rng_time += rng_timer.elapsed

            if disp is not None:
                with Timer() as disp_timer:
                    apply_displacement_field(grid, pos, disp, out=pos)
                disp_time += disp_timer()

            with Timer() as ma_timer:
                ma.add_particles_to_buffer(pos, particle_weight)
                particles_added += nbatch
                if particles_added == n:
                    ma.flush()  # Last batch.
            ma_time += ma_timer.elapsed

    assert particles_added == n
    assert ma.num_added + ma.num_skipped == n

    if verbose:
        print("Setup time: {:.2f} sec".format(setup_timer.elapsed))
        print("Work time: {:.2f} sec".format(work_timer.elapsed))
        print("  RNG time: {:.2f} sec".format(rng_time))
        if disp is not None:
            print("  Displacement field time: {:.2f} sec".format(disp_time))
        print("  Mass assignor time: {:.2f} sec".format(ma_time))
        print("    Sort time: {:.2f} sec".format(ma.sort_time))
        print("    Window time: {:.2f} sec".format(ma.window_time))

    return ma.num_added, ma.num_skipped
