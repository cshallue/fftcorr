import numpy as np
from absl import logging
from fftcorr.grid import apply_displacement_field
from fftcorr.particle_mesh import MassAssignor
from fftcorr.utils import Timer


def add_random_particles(n,
                         grid,
                         window_type,
                         particle_weight=None,
                         total_weight=None,
                         periodic_wrap=False,
                         disp=None,
                         seed=None,
                         batch_size=int(1e8),
                         buffer_size=int(1e7)):
    if total_weight is not None:
        if particle_weight is not None:
            raise ValueError(
                "particle_weight and total_weight cannot both be set")

        particle_weight = total_weight / n

    if particle_weight is None:
        particle_weight = 1.0

    logging.info(f"Particle weight: {particle_weight:.6g}")

    if disp is not None:
        disp = np.ascontiguousarray(disp, dtype=np.float64)

    gridmin = grid.posmin
    gridmax = grid.posmax

    with Timer() as setup_timer:
        pos_buf = np.empty((batch_size, 3), dtype=np.float64, order="C")

    rng = np.random.default_rng(seed)
    ma = MassAssignor(grid, window_type, periodic_wrap, buffer_size)
    with Timer() as work_timer:
        particles_added = 0
        rng_time = 0.0
        disp_time = 0.0
        ma_time = 0.0
        while particles_added < n:
            nbatch = int(min(batch_size, n - particles_added))
            logging.info(f"Adding particles [{particles_added + 1:,.6g}, "
                         f"{particles_added + nbatch:,.6g}]")

            pos = pos_buf[:nbatch]

            with Timer() as rng_timer:
                rnd = rng.uniform(gridmin, gridmax, (nbatch, 3))
                np.copyto(pos, rnd)
            rng_time += rng_timer.elapsed

            if disp is not None:
                with Timer() as disp_timer:
                    apply_displacement_field(grid,
                                             pos,
                                             disp,
                                             periodic_wrap=periodic_wrap,
                                             out=pos)
                disp_time += disp_timer.elapsed

            with Timer() as ma_timer:
                ma.add_particles_to_buffer(pos, particle_weight)
                particles_added += nbatch
                if particles_added == n:
                    ma.flush()  # Last batch.
            ma_time += ma_timer.elapsed

    assert particles_added == n
    assert ma.num_added + ma.num_skipped == n
    logging.info(
        f"Added {ma.num_added:,} randoms ({ma.num_skipped:,} skipped). Total "
        f"weight: {ma.totw:.4g}")

    logging.info(f"Setup time: {setup_timer.elapsed:.2f} sec")
    logging.info(f"Work time: {work_timer.elapsed:.2f} sec")
    logging.info(f"  RNG time: {rng_time:.2f} sec")
    if disp is not None:
        logging.info(f"  Displacement field time: {disp_time:.2f} sec")
    logging.info(f"  Mass assignor time: {ma_time:.2f} sec")
    logging.info(f"    Sort time: {ma.sort_time:.2f} sec")
    logging.info(f"    Window time: {ma.window_time:.2f} sec")
