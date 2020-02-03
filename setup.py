from distutils.core import setup, Extension
import sysconfig
from sys import platform

extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
extra_compile_args += ["-O3"]

print(platform)
if platform != "darwin":
    extra_compile_args.append("-fopenmp")

# the c++ extension module
extension_mod = Extension("fwg", ["fwgmodule.c", "fwg.c"], extra_compile_args=extra_compile_args)

setup(name = "fwg", ext_modules=[extension_mod])