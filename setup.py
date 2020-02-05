from distutils.core import setup, Extension
import sysconfig

# the c++ extension module
extension_mod = Extension("fwg", ["fwgmodule.cpp", "fwg.cpp"],
language="c++", extra_compile_args=['-std=c++11', '-O2'])

setup(name = "fwg", ext_modules=[extension_mod])