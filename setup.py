from distutils.core import setup, Extension
import numpy
from Cython.Distutils import build_ext

# TODO: can we put a separate setup.py for each module?
ext_modules = [
    # TODO: combine numpy_adaptor with another module?
    Extension("fftcorr.array.numpy_adaptor",
              sources=["fftcorr/array/numpy_adaptor.pyx"],
              include_dirs=[
                  numpy.get_include(),
              ]),
    Extension("fftcorr.array.row_major_array",
              sources=["fftcorr/array/row_major_array.pyx"],
              include_dirs=[numpy.get_include(), "cc/array/"]),
    Extension(
        "fftcorr.grid.config_space_grid",
        sources=[
            "fftcorr/grid/config_space_grid.pyx", "cc/array/array_ops.cc"
        ],
        # TODO: just add the includes to all extensions?
        include_dirs=[
            numpy.get_include(), "cc/array", "cc/grid/", "cc/particle_mesh/"
        ]),
    # TODO: this should inherit from the row_major_array includes
    Extension("fftcorr.grid.fft_grid",
              sources=[
                  "fftcorr/grid/fft_grid.pyx", "cc/array/array_ops.cc",
                  "cc/grid/fft_grid.cc"
              ],
              include_dirs=[numpy.get_include(), "cc/grid/", "cc/array/"]),
    Extension(
        "fftcorr.particle_mesh.mass_assignor",
        sources=["fftcorr/particle_mesh/mass_assignor.pyx"],
        # TODO: this should inherit from the config_space_grid includes
        include_dirs=[
            numpy.get_include(), "cc/particle_mesh/", "cc/grid/", "cc/array/"
        ]),
    Extension("fftcorr.histogram.histogram",
              sources=[
                  "fftcorr/histogram/histogram.pyx",
              ],
              include_dirs=[numpy.get_include(), "cc/histogram/",
                            "cc/array/"]),
    Extension("fftcorr.correlate.correlator",
              sources=[
                  "fftcorr/correlate/correlator.pyx",
                  "cc/grid/fft_grid.cc",
                  "cc/array/array_ops.cc",
              ],
              include_dirs=[
                  numpy.get_include(), "cc/histogram/", "cc/grid/",
                  "cc/array/", "cc/correlate/"
              ]),
]

for e in ext_modules:
    e.language = "c++"
    e.cython_directives = {'language_level': "3"}  # Python 3
    e.extra_compile_args.append("-std=c++11")  # Use C++ 11
    # https://github.com/cython/cython/issues/3474
    # TODO: remove this when the issue is fixed.
    e.extra_compile_args.append("-Wno-deprecated-declarations")
    # Compilation disables assert()s, which makes variables only used in asserts
    # appear unused.
    # TODO: consider alternate workarounds.
    e.extra_compile_args.append("-Wno-unused-variable")
    # http://docs.cython.org/en/latest/src/userguide/source_files_and_compilation.html#configuring-the-c-build
    e.define_macros.append(('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'))

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
)
