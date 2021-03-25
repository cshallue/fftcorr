from distutils.core import setup, Extension
import numpy
from Cython.Distutils import build_ext

import os.path

OPENMP = False


class CcLibrary(object):
    def __init__(self, hdr, srcs=None, deps=None, ext_libs=None):
        self.name = os.path.splitext(hdr)[0]
        self.hdr = hdr
        self.srcs = srcs if srcs else []
        self.deps = deps if deps else []
        self.ext_libs = ext_libs if ext_libs else []


class CcLibraries(object):
    def __init__(self, libs):
        self.libs = {}
        for lib in libs:
            self.add(lib)

    def add(self, lib):
        name = lib.name
        if name in self.libs:
            raise ValueError("Duplicate library: {}".format(name))
        self.libs[name] = lib

    def find_deps(self, names):
        hdrs = []
        srcs = []
        ext_libs = []
        self._recursive_add_deps(names, hdrs, srcs, ext_libs)
        return hdrs, srcs, ext_libs

    def _recursive_add_deps(self, names, hdrs, srcs, ext_libs):
        for name in names:
            lib = self.libs[name]
            if lib.hdr in hdrs:
                continue
            hdrs.append(lib.hdr)
            srcs.extend(lib.srcs)
            ext_libs.extend(lib.ext_libs)
            self._recursive_add_deps(lib.deps, hdrs, srcs, ext_libs)


class CythonLibrary(object):
    def __init__(self, name, pxd_file, srcs=None, cc_deps=None, pyx_deps=None):
        self.name = name
        if pxd_file[-4:] != ".pxd":
            raise ValueError("Expected .pxd file, got: {}".format(pxd_file))
        self.pxd_file = pxd_file
        self.srcs = srcs if srcs else []
        self.cc_deps = cc_deps if cc_deps else []
        self.pyx_deps = pyx_deps if pyx_deps else []


class CythonLibraries(object):
    def __init__(self, libs):
        self.libs = {}
        for lib in libs:
            self.add(lib)

    def add(self, lib):
        name = lib.name
        if name in self.libs:
            raise ValueError("Duplicate library: {}".format(name))
        self.libs[name] = lib


cc_libs = CcLibraries([
    CcLibrary("cc/types.h"),
    CcLibrary("cc/profiling/timer.h", srcs=["cc/profiling/timer.cc"]),
    CcLibrary("cc/array/row_major_array.h", deps=["cc/types"]),
    CcLibrary("cc/array/array_ops.h",
              srcs=["cc/array/array_ops.cc"],
              deps=[
                  "cc/types",
                  "cc/array/row_major_array",
              ]),
    CcLibrary("cc/particle_mesh/window_functions.h",
              deps=[
                  "cc/types",
                  "cc/array/row_major_array",
              ]),
    CcLibrary("cc/particle_mesh/mass_assignor.h",
              deps=[
                  "cc/types",
                  "cc/array/row_major_array",
                  "cc/grid/config_space_grid",
                  "cc/particle_mesh/window_functions",
                  "cc/profiling/timer",
              ]),
    CcLibrary("cc/grid/config_space_grid.h",
              deps=[
                  "cc/types",
                  "cc/array/row_major_array",
                  "cc/array/array_ops",
                  "cc/particle_mesh/window_functions",
              ]),
    CcLibrary(
        "cc/grid/fft_grid.h",
        srcs=["cc/grid/fft_grid.cc"],
        deps=[
            "cc/types",
            "cc/array/array_ops",
            "cc/array/row_major_array",
            "cc/profiling/timer",
        ],
        ext_libs=(["fftw3", "fftw3_threads", "m"] if OPENMP else ["fftw3"])),
    CcLibrary("cc/histogram/histogram_list.h",
              deps=[
                  "cc/types",
                  "cc/array/row_major_array",
              ]),
    CcLibrary("cc/correlate/correlator.h",
              deps=[
                  "cc/types",
                  "cc/array/row_major_array",
                  "cc/array/array_ops",
                  "cc/profiling/timer",
                  "cc/grid/config_space_grid",
                  "cc/particle_mesh/window_functions",
                  "cc/profiling/timer",
                  "cc/grid/fft_grid",
                  "cc/histogram/histogram_list",
              ]),
])

cython_libs = CythonLibraries([
    # TODO: combine numpy_adaptor with another module?
    CythonLibrary("fftcorr.array.numpy_adaptor",
                  pxd_file="fftcorr/array/numpy_adaptor.pxd",
                  srcs=["fftcorr/array/numpy_adaptor.pyx"]),
    CythonLibrary("fftcorr.types", pxd_file="fftcorr/types.pxd"),
    CythonLibrary("fftcorr.array.row_major_array",
                  pxd_file="fftcorr/array/row_major_array.pxd",
                  srcs=["fftcorr/array/row_major_array.pyx"],
                  cc_deps=["cc/array/row_major_array"],
                  pyx_deps=["fftcorr.types"]),
    CythonLibrary("fftcorr.particle_mesh.window_type",
                  pxd_file="fftcorr/particle_mesh/window_type.pxd",
                  cc_deps=["cc/particle_mesh/window_functions"]),
    CythonLibrary("fftcorr.particle_mesh.mass_assignor",
                  pxd_file="fftcorr/particle_mesh/mass_assignor.pxd",
                  srcs=["fftcorr/particle_mesh/mass_assignor.pyx"],
                  cc_deps=["cc/particle_mesh/mass_assignor"],
                  pyx_deps=[
                      "fftcorr.types", "fftcorr.array.row_major_array",
                      "fftcorr.grid.config_space_grid",
                      "fftcorr.particle_mesh.window_type"
                  ]),
    CythonLibrary("fftcorr.grid.config_space_grid",
                  pxd_file="fftcorr/grid/config_space_grid.pxd",
                  srcs=["fftcorr/grid/config_space_grid.pyx"],
                  cc_deps=["cc/grid/config_space_grid"],
                  pyx_deps=[
                      "fftcorr.types", "fftcorr.array.row_major_array",
                      "fftcorr.particle_mesh.window_type"
                  ]),
    CythonLibrary("fftcorr.grid.fft_grid",
                  pxd_file="fftcorr/grid/fft_grid.pxd",
                  srcs=["fftcorr/grid/fft_grid.pyx"],
                  cc_deps=["cc/grid/fft_grid"],
                  pyx_deps=[
                      "fftcorr.types",
                      "fftcorr.array.row_major_array",
                  ]),
    CythonLibrary("fftcorr.histogram.histogram_list",
                  pxd_file="fftcorr/histogram/histogram_list.pxd",
                  srcs=["fftcorr/histogram/histogram_list.pyx"],
                  cc_deps=["cc/histogram/histogram_list"],
                  pyx_deps=[
                      "fftcorr.types",
                      "fftcorr.array.row_major_array",
                  ]),
    CythonLibrary("fftcorr.correlate.correlator",
                  pxd_file="fftcorr/correlate/correlator.pxd",
                  srcs=["fftcorr/correlate/correlator.pyx"],
                  cc_deps=["cc/correlate/correlator"],
                  pyx_deps=[
                      "fftcorr.types",
                      "fftcorr.array.row_major_array",
                      "fftcorr.grid.config_space_grid",
                      "fftcorr.histogram.histogram_list",
                  ]),
])

ext_modules = []
for cython_lib in cython_libs.libs.values():
    if not cython_lib.srcs:
        continue  # Pxd only library for use by other cython libraries.
    name = cython_lib.name
    sources = cython_lib.srcs.copy()
    include_dirs = set([numpy.get_include()])
    libraries = []
    cc_deps = set(cython_lib.cc_deps)
    for pyx_dep in cython_lib.pyx_deps:
        cc_deps.update(cython_libs.libs[pyx_dep].cc_deps)
    print("Finding dependencies for", name)
    cc_hdrs, cc_src, cc_ext_libs = cc_libs.find_deps(cc_deps)
    sources.extend(cc_src)
    include_dirs.update([os.path.dirname(hdr) for hdr in cc_hdrs])
    libraries.extend(cc_ext_libs)
    print("sources:", sources)
    print("include_dirs:", include_dirs)
    print("libraries:", libraries)
    print()
    e = Extension(name,
                  sources=sources,
                  include_dirs=include_dirs,
                  libraries=libraries,
                  language="c++",
                  extra_compile_args=["-std=c++11"])
    if OPENMP:
        e.extra_compile_args.append("-march=native")
        e.extra_compile_args.append("-fopenmp")
        e.extra_compile_args.append("-lgomp")
        e.extra_compile_args.append("-O3")
        e.define_macros.append(("OPENMP", None))
        # e.define_macros.append(("SLAB", None))
        # e.define_macros.append(("FFTSLAB", None))
    else:
        e.extra_compile_args.append("-O2")
        e.extra_compile_args.append("-Wno-unknown-pragmas")
    # https://github.com/cython/cython/issues/3474
    # TODO: remove this when the issue is fixed.
    e.extra_compile_args.append("-Wno-deprecated-declarations")
    # Compilation disables assert()s, which makes variables only used in asserts
    # appear unused.
    # TODO: consider alternate workarounds.
    e.extra_compile_args.append("-Wno-unused-variable")
    # http://docs.cython.org/en/latest/src/userguide/source_files_and_compilation.html#configuring-the-c-build
    e.define_macros.append(('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'))
    e.cython_directives = {'language_level': "3"}  # Python 3
    ext_modules.append(e)

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
)
