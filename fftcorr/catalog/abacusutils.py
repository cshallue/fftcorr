import abc
import copy
import glob
import os.path
from re import S

import asdf
import numpy as np
from abacusnbody.data.bitpacked import unpack_rvint
from fftcorr.grid import apply_displacement_field
from fftcorr.particle_mesh import MassAssignor
from fftcorr.utils import Timer


def _apply_redshift_distortion(s, pos, vel, scale_factor):
    if s == 0 or s == "x":
        s = [1, 0, 0]
    elif s == 1 or s == "y":
        s = [0, 1, 0]
    elif s == 2 or s == "z":
        s = [0, 0, 1]

    s = np.asarray(s)
    if s.shape != (3, ):
        raise ValueError("direction should be an array of length 3")

    vs = np.dot(vel, s)  # Component of velocity in direction s.
    dpos = np.transpose(vs * np.expand_dims(s, -1))
    pos += dpos / (100 * scale_factor)


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
                       buffer_size=10000,
                       verbose=True):
    if isinstance(file_patterns, (str, bytes)):
        file_patterns = [file_patterns]

    filenames = []
    for file_pattern in file_patterns:
        matches = sorted(glob.glob(file_pattern))
        if not matches:
            raise ValueError(f"Found no files matching {file_pattern}")
        filenames.extend(matches)
    if verbose:
        print("Reading density field from {:,} files".format(len(filenames)))

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
            if verbose:
                print("Reading", os.path.basename(filename))
            with Timer() as io_timer:
                want_redshift_distort = (redshift_distortion is not None)
                data = reader.read(filename, load_vel=want_redshift_distort)
                if want_redshift_distort:
                    _apply_redshift_distortion(redshift_distortion, data.pos,
                                               data.vel,
                                               data.header["ScaleFactor"])
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

    if verbose:
        print("Work time: {:.2f} sec".format(work_timer.elapsed))
        print("  IO time: {:.2f} sec".format(io_time))
        if disp is not None:
            print("  Displacement field time: {:.2f} sec".format(disp_time))
        print("  Mass assignor time: {:.2f} sec".format(ma_time))
        print("    Sort time: {:.2f} sec".format(ma.sort_time))
        print("    Window time: {:.2f} sec".format(ma.window_time))

    return ma.num_added, ma.num_skipped
