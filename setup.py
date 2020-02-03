from distutils.core import setup, Extension

# the c++ extension module
extension_mod = Extension("fwg", ["fwgmodule.c", "fwg.c"])

setup(name = "fwg", ext_modules=[extension_mod])