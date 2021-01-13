from distutils.core import setup, Extension
import numpy
from Cython.Distutils import build_ext

# The following removes compiler warnings, but causes an error:
# define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
ext_modules = [
    Extension("grid",
              sources=["_grid.pyx", "../array3d.cc"],
              include_dirs=[numpy.get_include()],
              language="c++",
              extra_compile_args=["-std=c++0x"])
]

for e in ext_modules:
    e.cython_directives = {'language_level': "3"}  # all are Python-3

setup(
    cmdclass={'build_ext': build_ext},
    ext_modules=ext_modules,
)
