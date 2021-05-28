import glob
import json
from multiprocessing import Value
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
from fftcorr.particle_mesh import MassAssignor
from fftcorr.utils import Timer, read_density_field, add_random_particles

DATA_BASE_DIR = "/mnt/marvin2/bigsims/AbacusSummit/"

BASE_CONFIG = ConfigDict(
    dict(ngrid=576,
         xmin=-1000,
         xmax=1000,
         window_type=1,
         redshift_distortion=False,
         gaussian_sigma=10,
         nrandom=int(1e9)))
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
    "overwrite", False,
    "Whether to overwrite an existing output directory, if it exists")
config_flags.DEFINE_config_dict("config", BASE_CONFIG)

FLAGS = flags.FLAGS


def ensure_dir_exists(path):
    if not os.path.isdir(path):
        raise ValueError(f"Directory does not exist: {path}")


def get_file_pattern(data_dir, data_type):
    if data_type == "field_A":
        ensure_dir_exists(os.path.join(data_dir, "field_rv_A"))
        return os.path.join(data_dir, "field_rv_A/field_rv_A_*.asdf")
    if data_type == "field_B":
        ensure_dir_exists(os.path.join(data_dir, "field_rv_B"))
        return os.path.join(data_dir, "field_rv_B/field_rv_B_*.asdf")
    if data_type == "field_AB":
        ensure_dir_exists(os.path.join(data_dir, "field_rv_A"))
        ensure_dir_exists(os.path.join(data_dir, "field_rv_B"))
        return os.path.join(data_dir, "field_rv_[AB]/field_rv_[AB]_*.asdf")
    if data_type == "halos":
        ensure_dir_exists(os.path.join(data_dir, "halo_info"))
        return os.path.join(data_dir, "halo_info/halo_info_*.asdf")
    raise ValueError(f"Unrecognized data_type: {data_type}")


def compute_displacement_field(deltagrid, gaussian_sigma):
    # Cartesian coordinates of the box.
    xmin, ymin, zmin = deltagrid.posmin
    xmax, ymax, zmax = deltagrid.posmax
    nx, ny, nz = deltagrid.shape
    x, deltax = np.linspace(xmin, xmax, nx, retstep=True, endpoint=False)
    y, deltay = np.linspace(ymin, ymax, ny, retstep=True, endpoint=False)
    z, deltaz = np.linspace(zmin, zmax, nz, retstep=True, endpoint=False)
    # indexing="ij" means that X, Y, Z and f are indexed f[xi][yi][zi]
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    # Fourier space coordinates.
    kx = np.fft.fftfreq(nx, deltax)
    ky = np.fft.fftfreq(ny, deltay)
    kz = np.fft.fftfreq(nz, deltaz)
    # indexing="ij" means that X, Y, Z and f are indexed f[xi][yi][zi]
    KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
    K_COORDS = np.stack((KX, KY, KZ), axis=-1)

    # Convolve with Gaussian in Fourier space.
    print("Convolving with Gaussian, sigma =", gaussian_sigma)
    sigmax, sigmay, sigmaz = [gaussian_sigma] * 3
    kgaussian = np.exp(-2 * np.pi**2 * ((KX * sigmax)**2 + (KY * sigmay)**2 +
                                        (KZ * sigmaz)**2))
    kdelta = np.fft.fftn(deltagrid.data)
    kconv = kdelta * kgaussian

    # Set all frequencies on the boundary to zero.
    kconv[int(nx / 2), :, :] = 0
    kconv[:, int(ny / 2), :] = 0
    kconv[:, :, int(nz / 2)] = 0

    # Compute displacement field.
    print("Solving for displacement field")
    ksq = KX**2 + KY**2 + KZ**2
    with np.errstate(divide='ignore', invalid='ignore'):
        iksq = (1J / (2 * np.pi)) / ksq  # Divide by zero at zero
    # It's the mean of the displacement field, which we'll assume to be zero.
    iksq[0][0][0] = 0
    kdisp = K_COORDS * np.expand_dims(iksq * kconv, -1)
    # Displacement field
    disp = np.fft.ifftn(kdisp, axes=range(3)).real

    # Print displacement field info.
    for i, name in enumerate(["x", "y", "z"]):
        dslice = disp[:, :, :, i]
        print("{} displacement: min: {:.2f}, max: {:.2f}, RMS: {:.2f}".format(
            name, dslice.min(), dslice.max(), np.sqrt(np.mean(dslice**2))))
    print("Mean displacement magnitude: {:.2f}\n".format(
        np.mean(np.sqrt(np.sum(disp**2, axis=-1)))))

    return disp


def process(config, input_file_pattern, output_dir, overwrite):
    # Make the output directory.
    if os.path.exists(output_dir):
        print("Output directory already exists:", output_dir)
        if overwrite:
            print("Contents will be overwritten")
            shutil.rmtree(output_dir)
        else:
            raise ValueError(
                "Output directory already exists and --overrwrite is False")
    os.makedirs(output_dir)

    # Save the config.
    config_json = config.to_json(indent=2)
    print("Cconfig:")
    print(config_json)
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        f.write(config_json)

    shape = [config.ngrid] * 3
    posmin = [config.xmin] * 3
    posmax = [config.xmax] * 3
    grid = ConfigSpaceGrid(shape,
                           posmin=posmin,
                           posmax=posmax,
                           window_type=config.window_type)
    mass_assignor = MassAssignor(grid, periodic_wrap=True)
    # Create density field.
    print("\nReading density field")
    nparticles = read_density_field(
        input_file_pattern,
        grid,
        mass_assignor,
        redshift_distortion=config.redshift_distortion,
        periodic_wrap=True)
    print(f"Added {nparticles:,} particles to density field\n")
    dens_mean = np.mean(grid)
    grid -= dens_mean
    grid /= dens_mean
    dens_filename = os.path.join(output_dir, "delta.asdf")
    grid.write(dens_filename)
    print(f"Wrote density field to {dens_filename}\n")

    # Start reconstruction.
    print("\nStarting reconstruction")
    with Timer() as recon_timer:
        disp = compute_displacement_field(grid, config.gaussian_sigma)
    print("Reconstruction time: {:.2f} sec".format(recon_timer.elapsed))

    # Generate the reconstructed delta grid.
    print("Reading shifted density field")
    grid.clear()
    mass_assignor = MassAssignor(grid, periodic_wrap=True, disp=-disp)
    nparticles = read_density_field(
        input_file_pattern,
        grid,
        mass_assignor,
        redshift_distortion=config.redshift_distortion)
    print(f"Added {nparticles:,} particles\n")

    print("Adding shifted random particles")
    random_weight = -dens_mean * np.prod(shape)
    add_random_particles(config.nrandom,
                         grid,
                         mass_assignor,
                         total_weight=random_weight)
    print("Added {:,} randoms. Total weight: {:.4g} ({:.4g})\n".format(
        config.nrandom, mass_assignor.totw, random_weight))
    grid /= dens_mean
    recon_dens_filename = os.path.join(output_dir, f"delta-reconstructed.asdf")
    grid.write(recon_dens_filename)
    print(f"Wrote reconstructed density field to {recon_dens_filename}\n")


def main(unused_argv):
    config = FLAGS.config

    for redshift in FLAGS.z:
        print("Processing redshift", redshift)
        data_dir = os.path.join(DATA_BASE_DIR, FLAGS.sim_name, "halos",
                                f"z{redshift}")
        input_file_pattern = get_file_pattern(data_dir, FLAGS.data_type)
        output_dir = os.path.join(FLAGS.output_dir, FLAGS.sim_name,
                                  f"z{redshift}", FLAGS.data_type)
        try:
            process(config, input_file_pattern, output_dir, FLAGS.overwrite)
        except ValueError as e:
            print(f"Failed to process redshift {redshift}:\n", e)


if __name__ == '__main__':
    app.run(main)
