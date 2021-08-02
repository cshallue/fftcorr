import abc
import glob
import os.path
from multiprocessing import Value

import asdf
import numpy as np
from abacusnbody.data.bitpacked import unpack_rvint
from fftcorr.grid import apply_displacement_field
from fftcorr.particle_mesh import MassAssignor
from fftcorr.utils import Timer


class CatalogReader(abc.ABC):
    @abc.abstractmethod
    def read(self, filename, redshift_distortion=False):
        pass

    def apply_redshift_distortion(self, pos, vel, scale_factor):
        # Apply redshift distortions in the z direction. If desired in the
        # future, we could accept any arbitrary direction vector and apply
        # redshift distortion in that direction.
        pos[:, 2] += vel[:, 2] / (100 * scale_factor)


class HaloReader(CatalogReader):
    def read(self, filename, redshift_distortion=False):
        with asdf.open(filename, lazy_load=True) as af:
            pos = np.ascontiguousarray(af.tree["data"]["x_com"],
                                       dtype=np.float64)
            pos *= af.tree["header"]["BoxSize"]
            weight = np.ascontiguousarray(af.tree["data"]["N"],
                                          dtype=np.float64)

            if redshift_distortion:
                vel = np.ascontiguousarray(af.tree["data"]["v_com"],
                                           dtype=np.float64)
                self.apply_redshift_distortion(
                    pos, vel, af.tree["header"]["ScaleFactor"])

        return pos, weight


class ParticleReader(CatalogReader):
    def read(self, filename, redshift_distortion=False):
        with asdf.open(filename, lazy_load=True) as af:
            posvel = unpack_rvint(af.tree["data"]["rvint"],
                                  boxsize=af.tree["header"]["BoxSize"],
                                  float_dtype=np.float64,
                                  posout=True,
                                  velout=redshift_distortion)
            if redshift_distortion:
                pos, vel = posvel
                self.apply_redshift_distortion(
                    pos, vel, af.tree["header"]["ScaleFactor"])
            else:
                pos = posvel

        weight = 1.0
        return pos, weight


def read_density_field(file_patterns,
                       grid,
                       reader=None,
                       periodic_wrap=False,
                       redshift_distortion=False,
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
            reader = HaloReader()
        elif file_type == "particles":
            reader = ParticleReader()
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
                pos, weight = reader.read(filename, redshift_distortion)
            io_time += io_timer.elapsed

            # Apply displacement field.
            if disp is not None:
                with Timer() as disp_timer:
                    apply_displacement_field(grid,
                                             pos,
                                             disp,
                                             periodic_wrap=periodic_wrap,
                                             out=pos)
                disp_time += disp_timer.elapsed

            # Add items to the density field.
            with Timer() as ma_timer:
                ma.add_particles_to_buffer(pos, weight)
                if filename == filenames[-1]:
                    ma.flush()  # Last file.
            ma_time += ma_timer.elapsed
            items_seen += pos.shape[0]

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
