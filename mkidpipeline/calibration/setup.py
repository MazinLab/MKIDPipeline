from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_module = Extension("ts_binner",
                       ['ts_binner.pyx'],
                       )

setup(
    name = 'ts_binner',
    cmdclass = {'build_ext': build_ext},
    ext_modules = [ext_module],
)
