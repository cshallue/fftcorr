import numpy as np

from fftcorr.particle_mesh import MassAssignor
from fftcorr.utils import Timer


def add_random_particles(n,
                         ma,
                         particle_weight=None,
                         total_weight=None,
                         verbose=True,
                         batch_size=int(1e8)):
    if total_weight is not None:
        if particle_weight is not None:
            raise ValueError(
                "particle_weight and total_weight cannot both be set")

        particle_weight = total_weight / n

    if particle_weight is None:
        particle_weight = 1.0

    if verbose:
        print("Particle weight: {:.6g}".format(particle_weight))

    gridmin = ma.posmin
    gridmax = ma.posmax

    with Timer() as setup_timer:
        pos_buf = np.empty((batch_size, 3), dtype=np.float64, order="C")

    with Timer() as work_timer:
        ma.clear()
        particles_added = 0
        rng_time = 0.0
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
        print("  Mass assignor time: {:.2f} sec".format(ma_time))
        print("    Sort time: {:.2f} sec".format(ma.sort_time))
        print("    Window time: {:.2f} sec".format(ma.window_time))
