import abc
import copy
import glob
import os.path

import asdf
import numpy as np
from abacusnbody.data.bitpacked import unpack_rvint
from absl import logging
from fftcorr.grid import apply_displacement_field
from fftcorr.particle_mesh import MassAssignor
from fftcorr.utils import Timer


def _apply_redshift_distortion(s, pos, vel, conversion):
    if s == 0 or s == "x":
        s = [1, 0, 0]
    elif s == 1 or s == "y":
        s = [0, 1, 0]
    elif s == 2 or s == "z":
        s = [0, 0, 1]

    s = np.asarray(s, dtype=np.float64)
    if s.shape != (3, ):
        raise ValueError("direction should be an array of length 3")

    vs = np.dot(vel, s)  # Component of velocity in direction s.
    dpos = np.transpose(vs * np.expand_dims(s, -1))
    pos += dpos / conversion


def _load_file(reader, filename, redshift_distortion):
    applying_rsd = (redshift_distortion is not False
                    and redshift_distortion is not None)
    data = reader.read(filename, load_velocity=applying_rsd)
    if applying_rsd:
        conversion = data.header["VelZSpace_to_kms"] / data.header["BoxSize"]
        logging.info("Converting velocities to redshift-space comoving "
                     f"displacements with factor {conversion}")
        _apply_redshift_distortion(redshift_distortion, data.pos, data.vel,
                                   conversion)

    return data


class AbacusData:
    def __init__(self, header, pos, weight, vel=None):
        self.header = copy.deepcopy(header)
        self.pos = np.array(pos, dtype=np.float64, order="C")
        self.weight = np.array(weight, dtype=np.float64, order="C")
        if vel is not None:
            self.vel = np.array(vel, dtype=np.float64, order="C")


class AbacusFileReader(abc.ABC):
    @abc.abstractmethod
    def read(self, filename, load_velocity=False):
        pass


class HaloFileReader(AbacusFileReader):
    def read(self, filename, load_velocity=False):
        with asdf.open(filename, lazy_load=True) as af:
            data = AbacusData(
                header=af.tree["header"],
                pos=af.tree["data"]["x_com"],
                weight=af.tree["data"]["N"],
                vel=af.tree["data"]["v_com"] if load_velocity else None)
        data.pos *= data.header["BoxSize"]
        return data


class ParticleFileReader(AbacusFileReader):
    def read(self, filename, load_velocity=False):
        with asdf.open(filename, lazy_load=True) as af:
            velout = None if load_velocity else False
            posvel = unpack_rvint(af.tree["data"]["rvint"],
                                  boxsize=af.tree["header"]["BoxSize"],
                                  float_dtype=np.float64,
                                  velout=velout)
            data = AbacusData(header=af.tree["header"],
                              pos=posvel[0],
                              weight=1.0,
                              vel=posvel[1] if load_velocity else None)
        return data


def read_density_field(file_patterns,
                       grid,
                       reader=None,
                       periodic_wrap=False,
                       redshift_distortion=None,
                       disp=None,
                       buffer_size=10000):
    if isinstance(file_patterns, (str, bytes)):
        file_patterns = [file_patterns]

    filenames = []
    for file_pattern in file_patterns:
        matches = sorted(glob.glob(file_pattern))
        if not matches:
            raise ValueError(f"Found no files matching {file_pattern}")
        logging.info(f"Found {len(matches):,} files matching {file_pattern}")
        filenames.extend(matches)
    logging.info(f"Reading density field from {len(filenames):,} files")

    # Create the file reader if necessary.
    if reader is None:
        # Infer the type of files.
        file_type = None
        for filename in filenames:
            basename = os.path.basename(filename)
            if basename.startswith("halo_info"):
                ft = "halos"
            elif (basename.startswith("field_rv")
                  or basename.startswith("halo_rv")):
                ft = "particles"
            else:
                raise ValueError(f"Could not infer file type: '{basename}'")
            if file_type is None:
                file_type = ft
            elif file_type != ft:
                raise ValueError(
                    f"Inconsistent file types: {ft} vs {file_type}")
        # Create the appropriate reader.
        if file_type == "halos":
            reader = HaloFileReader()
        elif file_type == "particles":
            reader = ParticleFileReader()
        else:
            raise ValueError(f"Unrecognized file_type: {file_type}")

    if disp is not None:
        disp = np.ascontiguousarray(disp, dtype=np.float64)

    ma = MassAssignor(grid, periodic_wrap, buffer_size)
    with Timer() as work_timer:
        items_seen = 0
        io_time = 0.0
        disp_time = 0.0
        ma_time = 0.0
        for filename in filenames:
            logging.info(f"Reading {os.path.basename(filename)}")
            with Timer() as io_timer:
                data = _load_file(reader, filename, redshift_distortion)
            io_time += io_timer.elapsed

            # Apply displacement field.
            if disp is not None:
                with Timer() as disp_timer:
                    apply_displacement_field(grid,
                                             data.pos,
                                             disp,
                                             periodic_wrap=periodic_wrap,
                                             out=data.pos)
                disp_time += disp_timer.elapsed

            # Add items to the density field.
            with Timer() as ma_timer:
                ma.add_particles_to_buffer(data.pos, data.weight)
                if filename == filenames[-1]:
                    ma.flush()  # Last file.
            ma_time += ma_timer.elapsed
            items_seen += data.pos.shape[0]

    assert ma.num_added + ma.num_skipped == items_seen

    logging.debug(f"Work time: {work_timer.elapsed:.2f} sec")
    logging.debug(f"  IO time: {io_time:.2f} sec")
    if disp is not None:
        logging.debug(f"  Displacement field time: {disp_time:.2f} sec")
    logging.debug(f"  Mass assignor time: {ma_time:.2f} sec")
    logging.debug(f"    Sort time: {ma.sort_time:.2f} sec")
    logging.debug(f"    Window time: {ma.window_time:.2f} sec")

    logging.info(f"Particles added: {ma.num_added}")
    logging.info(f"Particles skipped: {ma.num_skipped}")
    return ma.num_added, ma.num_skipped
