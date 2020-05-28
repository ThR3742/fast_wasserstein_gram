import setuptools
from distutils.core import setup, Extension
import sysconfig
import os
import platform

setuptools.dist.Distribution().fetch_build_eggs(['numpy'])
import numpy

extra_compile_args = ["-std=c++11", "-O2"]
extra_link_args = ["-std=c++11"]

if "BOOST_ROOT" in os.environ:
    extra_compile_args += [f"-I{os.environ['BOOST_ROOT']}"]

if platform.system() == "Darwin":
    extra_compile_args += ["-stdlib=libc++", "-mmacosx-version-min=10.9"]
    extra_link_args += ["-stdlib=libc++", "-mmacosx-version-min=10.9"]

# the c++ extension module
extension_mod = Extension(
    name="fwg",
    sources=["fwgmodule.cpp", "fwg.cpp"],
    language="c++",
    extra_compile_args=extra_compile_args,
    extra_link_args=extra_link_args,
    include_dirs=[".", numpy.get_include()],
)

setuptools.setup(
    name = "fwg",
    version="0.2.11",
    author="Thomas Ricatte",
    install_requires=['numpy'],
    description="Fast sliced wasserstein distance matrix computation",
    ext_modules=[extension_mod]
)