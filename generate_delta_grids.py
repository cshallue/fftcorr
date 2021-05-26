import glob
import json
import os
import shutil

from absl import app
from absl import flags

import asdf
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate

import ml_collections
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags

from fftcorr.grid import ConfigSpaceGrid
from fftcorr.utils import read_density_field, add_random_particles
from fftcorr.correlate import Correlator

DATA_BASE_DIR = "/mnt/marvin2/bigsims/AbacusSummit/"

BASE_CONFIG = ConfigDict(
    dict(ngrid=576,
         xmin=-1000,
         xmax=1000,
         window_type=1,
         redshift_distortion=False,
         gaussian_sigma=10,
         nrandom=1e9))
BASE_CONFIG.lock()  # Prevent new fields from being accidentally added.

flags.DEFINE_string(
    "sim_name",
    None,
    f"Subdirectory of {DATA_BASE_DIR} containing the input files",
    required=True)
ALLOWED_DATA_TYPES = ["field_A", "field_B", "field_AB", "halos"]
flags.DEFINE_enum("data_type",
                  None,
                  ALLOWED_DATA_TYPES,
                  "Type of data to process",
                  required=True)
flags.DEFINE_string("output_dir",
                    None,
                    "Base directory for the output",
                    required=True)
ALLOWED_REDSHIFTS = [
    "0.100", "0.200", "0.300", "0.500", "0.800", "1.100", "1.400", "1.700",
    "2.000", "2.500", "3.000"
]
flags.DEFINE_multi_enum("z",
                        None,
                        ALLOWED_REDSHIFTS,
                        "Redshifts to process",
                        required=True)
flags.DEFINE_bool(
    "overwrite_all", False,
    "Whether to overwrite the existing output directory, if it exists")
flags.DEFINE_bool(
    "use_existing_config", False,
    "Whether to use the existing config in the output directory, if it exists")
config_flags.DEFINE_config_dict("config", BASE_CONFIG)

FLAGS = flags.FLAGS


def _normalize(grid, mean):
    grid -= mean
    grid /= mean


def _ensure_dir_exists(path):
    if not os.path.isdir(path):
        raise ValueError(f"Directory does not exist: {path}")


def process_redshift(config, sim_name, data_type, redshift, output_dir):
    # Get input file pattern.
    data_dir = os.path.join(DATA_BASE_DIR, sim_name, f"halos/z{redshift}/")
    if data_type == "field_A":
        _ensure_dir_exists(os.path.join(data_dir, "field_rv_A"))
        file_pattern = os.path.join(data_dir, "field_rv_A/field_rv_A_*.asdf")
    elif FLAGS.data_type == "field_B":
        _ensure_dir_exists(os.path.join(data_dir, "field_rv_B"))
        file_pattern = os.path.join(data_dir, "field_rv_B/field_rv_B_*.asdf")
    elif FLAGS.data_type == "field_AB":
        _ensure_dir_exists(os.path.join(data_dir, "field_rv_A"))
        _ensure_dir_exists(os.path.join(data_dir, "field_rv_B"))
        file_pattern = os.path.join(data_dir,
                                    "field_rv_[AB]/field_rv_[AB]_*.asdf")
    if FLAGS.data_type == "halos":
        _ensure_dir_exists(os.path.join(data_dir, "halo_info"))
        file_pattern = os.path.join(data_dir, "halo_info/halo_info_*.asdf")

    shape = [config.ngrid] * 3
    posmin = [config.xmin] * 3
    posmax = [config.xmax] * 3
    grid = ConfigSpaceGrid(shape,
                           posmin=posmin,
                           posmax=posmax,
                           window_type=config.window_type)
    # Create density field.
    print("\nReading density field")
    nparticles = read_density_field(
        file_pattern,
        grid,
        redshift_distortion=config.redshift_distortion,
        periodic_wrap=True)
    print(f"Added {nparticles:,} particles to density field\n")
    dens_mean = np.mean(grid.data)
    _normalize(grid.data, dens_mean)
    dens_filename = os.path.join(output_dir, f"delta-z{redshift}.asdf")
    grid.write(dens_filename)
    print(f"Wrote density field to {dens_filename}\n")

    # Compute correlations for validation.
    correlations = []
    rmax = 150.0
    dr = 5
    kmax = 0.4
    dk = 0.002
    maxell = 2
    c = Correlator(grid, rmax, dr, kmax, dk, maxell)
    print("Computing density field correlations")
    c.correlate_periodic()
    correlations.append((f"Density field at z = {redshift}", c.correlation_r,
                         c.correlation_histogram / c.correlation_counts))

    # Now start reconstruction.
    print("\nStarting reconstruction")

    # Cartesian coordinates of the box.
    xmin, ymin, zmin = posmin
    xmax, ymax, zmax = posmax
    nx, ny, nz = shape
    x, deltax = np.linspace(xmin, xmax, nx, retstep=True, endpoint=False)
    y, deltay = np.linspace(ymin, ymax, ny, retstep=True, endpoint=False)
    z, deltaz = np.linspace(zmin, zmax, nz, retstep=True, endpoint=False)
    # indexing="ij" means that X, Y, Z and f are indexed f[xi][yi][zi]
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")
    X_COORDS = np.stack((X, Y, Z), axis=-1)

    # Fourier space coordinates.
    kx = np.fft.fftfreq(nx, deltax)
    ky = np.fft.fftfreq(ny, deltay)
    kz = np.fft.fftfreq(nz, deltaz)
    deltakx = 1 / (nx * deltax)
    deltaky = 1 / (ny * deltay)
    deltakz = 1 / (ny * deltaz)
    # indexing="ij" means that X, Y, Z and f are indexed f[xi][yi][zi]
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K_COORDS = np.stack((KX, KY, KZ), axis=-1)

    # Convolve with Gaussian in Fourier space.
    print("Convolving with Gaussian, sigma =", config.gaussian_sigma)
    sigmax, sigmay, sigmaz = [config.gaussian_sigma] * 3
    kgaussian = np.exp(-2 * np.pi**2 * ((KX * sigmax)**2 + (KY * sigmay)**2 +
                                        (KZ * sigmaz)**2))
    kdelta = np.fft.fftn(grid.data)
    kconv = kdelta * kgaussian

    # Set all frequencies on the boundary to zero.
    assert config.ngrid % 2 == 0, "ngrid must be even"
    imid = int(config.ngrid / 2)
    kconv_masked = kconv.copy()
    kconv_masked[imid, :, :] = 0
    kconv_masked[:, imid, :] = 0
    kconv_masked[:, :, imid] = 0

    # Compute displacement field.
    print("Solving for displacement field")
    ksq = KX**2 + KY**2 + KZ**2
    with np.errstate(divide='ignore', invalid='ignore'):
        iksq = (1J / (2 * np.pi)) / ksq  # Divide by zero at zero
    # It's the mean of the displacement field, which we'll assume to be zero.
    iksq[0][0][0] = 0
    kdisp = K_COORDS * np.expand_dims(iksq * kconv_masked, -1)
    # Displacement field
    disp = np.fft.ifftn(kdisp, axes=range(3))

    # Print displacement field info.
    for i, name in enumerate(["x", "y", "z"]):
        dslice = disp.real[:, :, :, i]
        print("{} displacement: min: {:.2f}, max: {:.2f}, RMS: {:.2f}".format(
            name, dslice.min(), dslice.max(), np.sqrt(np.mean(dslice**2))))
    print("Mean displacement magnitude: {:.2f}\n".format(
        np.mean(np.sqrt(np.sum(disp.real**2, axis=-1)))))

    # Nearest neighbor interpolation.
    compute_disp = scipy.interpolate.RegularGridInterpolator(
        (x + deltax / 2, y + deltay / 2, z + deltaz / 2),
        disp.real,
        method="nearest",
        bounds_error=False,
        fill_value=None)

    def transform_coords_fn(pos):
        pos -= compute_disp(pos)

    # Now generate the reconstructed delta grid.
    print("Reading shifted density field")
    grid.clear()
    nparticles = read_density_field(
        file_pattern,
        grid,
        redshift_distortion=config.redshift_distortion,
        transform_coords_fn=transform_coords_fn,
        periodic_wrap=True,
        buffer_size=10000)
    print(f"Added {nparticles:,} particles\n")

    print("Adding shifted random particles")
    random_weight = -dens_mean * np.prod(shape)
    totw = add_random_particles(grid,
                                config.nrandom,
                                total_weight=random_weight,
                                transform_coords_fn=transform_coords_fn,
                                periodic_wrap=True)
    print(
        "Added {:,} randoms. Total weight: {:.4g} ({:.4g}) ({:.4g})\n".format(
            config.nrandom, totw, np.sum(grid.data), random_weight))
    d = grid.data
    d /= dens_mean
    recon_dens_filename = os.path.join(
        output_dir, f"delta-z{redshift}-reconstructed.asdf")
    grid.write(recon_dens_filename)
    print(f"Wrote reconstructed density field to {recon_dens_filename}\n")
    print("Computing reconstructed density field correlations")
    c.correlate_periodic()
    correlations.append = (f"Reconstructed density field at z = {redshift}",
                           c.correlation_r,
                           c.correlation_histogram / c.correlation_counts)

    # Compute initial correlations.
    ic_file = os.path.join(DATA_BASE_DIR, "ic", sim_name,
                           f"id_dens_N{config.ngrid}.asdf")
    with asdf.open(ic_file) as af:
        np.copyto(grid.data, af.tree["data"]["density"])
    print("\nComputing initial density field correlations")
    c.correlate_periodic()
    correlations.append = (f"Initial density field", c.correlation_r,
                           c.correlation_histogram / c.correlation_counts)

    # Save correlation plots.
    print("\nGenerating correlation plots")
    nrows = int(maxell / 2) + 1
    ncols = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(8 * ncols, 6 * nrows))
    for i in range(nrows):
        ell = 2 * i
        for j in range(ncols):
            if j == 0:
                imin = 9
                imax = len(r)
            else:
                imin = 14
                imax = 36

            ax = axes[i][j]
            for label, r, xi in zip(correlations):
                ax.plot(r[imin:imax], xi[imin:imax], "o-", label=label)
                ax.set_title("$\ell =$ {}".format(ell))
                ax.set_xlabel("r [Mpc/h]")
                ax.set_ylabel("$\\xi(r)$")
                ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "corr.pdf"))


def main(unused_argv):
    config = FLAGS.config

    # Ensure the output directory exists.
    output_dir = os.path.join(FLAGS.output_dir, FLAGS.sim_name,
                              FLAGS.data_type)
    print(output_dir)
    if os.path.exists(output_dir):
        print("Output directory already exists:", output_dir)
        if FLAGS.overwrite_all:
            print("Contents will be overwritten")
            shutil.rmtree(output_dir)
            os.makedirs(output_dir)
        elif FLAGS.use_existing_config:
            print("Using the existing config file")
            existing_config = ConfigDict(
                json.load(os.path.join(output_dir, "config.json")))
            existing_config.lock()
            config.update(existing_config)
            # This makes sure that the keys are identical.
            existing_config.update(config)
        else:
            print(
                "One of --overwrite_all and --use_existing_config must be set "
                "when the output directory already exists")
            return 1
    else:
        print("making output dir")
        os.makedirs(output_dir)

    # Save the config.
    config_json = config.to_json(indent=2)
    print("Adopting config:")
    print(config_json)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        f.write(config_json)

    for redshift in FLAGS.z:
        try:
            process_redshift(config, FLAGS.sim_name, FLAGS.data_type, redshift,
                             output_dir)
        except ValueError as e:
            print(f"Failed to process redshift {redshift}:\n", e)


if __name__ == '__main__':
    app.run(main)
