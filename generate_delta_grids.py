import glob
import os

from absl import app
from absl import flags

import matplotlib as mpl
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
    dict(
        data_dir="",  # Relative to DATA_BASE_DIR
        data_type="field_rv_A",  # field_rv_{A,B} or halos
        ngrid=576,
        xmin=-1000,
        xmax=1000,
        window_type=1,
        redshift_distortion=False,
        dens_mean=0.0,
        gaussian_sigma=10,
        nrandom=1e9))

flags.DEFINE_string("data_dir", None, "Directory containing the input files")
flags.DEFINE_string("output_dir", None, "Directory to write the output")
config_flags.DEFINE_config_dict("config", BASE_CONFIG)

FLAGS = flags.FLAGS


def _normalize(grid, mean):
    grid -= mean
    grid /= mean


def run(config, output_base_dir):
    data_dir = os.path.join(DATA_BASE_DIR, config.data_dir)
    if config.data_type.startswith("field_rv_"):
        input_file_pattern = os.path.join(data_dir, config.data_type,
                                          config.data_type + "*.asdf")
    else:  # config.data_type == "halos"
        input_file_pattern = os.path.join(data_dir,
                                          "halo_info/halo_info*.asdf")
    if not glob.glob(input_file_pattern):
        print("Found no files matching:", input_file_pattern)
        return 1

    output_dir = os.path.join(output_base_dir, config.data_dir)
    if os.path.exists(output_dir):
        print("Output directory alraedy exists:", output_dir)
        return 1
    os.makedirs(output_dir)

    shape = [config.ngrid] * 3
    posmin = [config.xmin] * 3
    posmax = [config.xmax] * 3

    deltagrid = ConfigSpaceGrid(shape,
                                posmin=posmin,
                                posmax=posmax,
                                window_type=config.window_type)

    # Create grid in redshift space.
    nparticles = read_density_field(
        input_file_pattern,
        deltagrid,
        redshift_distortion=config.redshift_distortion,
        periodic_wrap=True)
    print("Added {:,} particles.".format(nparticles))
    if config.dens_mean != 0.0:
        print("dens_mean is set automatically")
        return 1
    config.dens_mean = np.mean(deltagrid.data)
    _normalize(deltagrid.data, config.dens_mean)
    deltagrid.write(os.path.join(output_dir, "delta.asdf"))
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        f.write(config.to_json())

    # Now start reconstruction.

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
    sigmax, sigmay, sigmaz = [config.gaussian_sigma] * 3
    kgaussian = np.exp(-2 * np.pi**2 * ((KX * sigmax)**2 + (KY * sigmay)**2 +
                                        (KZ * sigmaz)**2))
    kdelta = np.fft.fftn(deltagrid.data)
    kconv = kdelta * kgaussian

    # Set all frequencies on the boundary to zero.
    assert config.ngrid % 2 == 0, "ngrid must be even"
    imid = int(config.ngrid / 2)
    kconv_masked = kconv.copy()
    kconv_masked[imid, :, :] = 0
    kconv_masked[:, imid, :] = 0
    kconv_masked[:, :, imid] = 0

    # Compute displacement field.
    ksq = KX**2 + KY**2 + KZ**2
    with np.errstate(invalid='ignore'):
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
    print("Mean displacement magnitude: {:.2f}".format(
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

    # Reconstructed delta grid.
    rdeltagrid = ConfigSpaceGrid(shape,
                                 posmin=posmin,
                                 posmax=posmax,
                                 window_type=config.window_type)
    nparticles = read_density_field(
        input_file_pattern,
        rdeltagrid,
        redshift_distortion=config.redshift_distortion,
        transform_coords_fn=transform_coords_fn,
        periodic_wrap=True,
        buffer_size=10000)
    print("Added {:,} particles.".format(nparticles))

    random_weight = -config.dens_mean * np.prod(shape)
    totw = add_random_particles(rdeltagrid,
                                config.nrandom,
                                total_weight=random_weight,
                                transform_coords_fn=transform_coords_fn,
                                periodic_wrap=True)
    print("Added {:,} randoms. Total weight: {:.4g} ({:.4g}) ({:.4g})".format(
        config.nrandom, totw, np.sum(rdeltagrid.data), random_weight))
    rdeltagrid.data /= config.dens_mean
    rdeltagrid.write(os.path.join(output_dir, "delta-reconstructed.asdf"))

    # Make some plots for validation.
    rmax = 150.0
    dr = 5
    kmax = 0.4
    dk = 0.002
    maxell = 2
    correlations = []
    for label, grid in [("Density field", deltagrid),
                        ("Reconstructed density field", rdeltagrid)]:
        print("Correlating:", label)
        c = Correlator(grid, rmax, dr, kmax, dk, maxell)
        c.correlate_periodic()
        print("Done correlating!\n")
        correlations.append = (label, c.correlation_r,
                               c.correlation_histogram / c.correlation_counts)

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
    config.data_dir = FLAGS.data_dir
    return run(config, FLAGS.output_dir)


if __name__ == '__main__':
    flags.mark_flag_as_required('data_dir')
    flags.mark_flag_as_required('output_dir')
    app.run(main)
