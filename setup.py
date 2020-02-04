from distutils.core import setup, Extension
import sysconfig

# the c++ extension module
extension_mod = Extension("fwg", ["fwgmodule.cpp", "fwg.cpp"], language="c++")

setup(name = "fwg", ext_modules=[extension_mod])