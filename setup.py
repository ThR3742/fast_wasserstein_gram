from distutils.core import setup, Extension
import sysconfig

extra_compile_args = sysconfig.get_config_var('CFLAGS').split()
extra_compile_args += ["-Og", "-g"]

# the c++ extension module
extension_mod = Extension("fwg", ["fwgmodule.c", "fwg.c"], extra_compile_args=extra_compile_args)

setup(name = "fwg", ext_modules=[extension_mod])