from distutils.core import setup, Extension
import sysconfig
import numpy

# the c++ extension module
extension_mod = Extension("fwg", ["fwgmodule.cpp", "fwg.cpp"],
language="c++", extra_compile_args=['-std=c++11', '-O2'], include_dirs=[numpy.get_include()])

setup(
    name = "fwg",
    version="0.2.0",
    author="Thomas Ricatte",
    description="Fast sliced wasserstein distance matrix computation",
    ext_modules=[extension_mod]
)