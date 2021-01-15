from distutils.core import setup, Extension
import numpy
from Cython.Distutils import build_ext

ext_modules = [
    # TODO: can we put a separate setup.py for each module?
    Extension("fftcorr.grid._config_space_grid",
              sources=["fftcorr/grid/_config_space_grid.pyx", "cc/array3d.cc"],
              include_dirs=[numpy.get_include(), "cc/grid/"],
              define_macros=[]),
    Extension(
        "fftcorr.particle_mesh._mass_assignor",
        sources=["fftcorr/particle_mesh/_mass_assignor.pyx"],
        # TODO: this should inherit from the config_space_grid includes
        include_dirs=[numpy.get_include(), "cc/particle_mesh/", "cc/grid/"])
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
