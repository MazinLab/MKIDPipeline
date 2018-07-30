from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_module = Extension("photonstats_utils", 
                       ['photonstats_utils.pyx'],
                       )

setup(
    name = 'photonstats_utils',
    cmdclass = {'build_ext': build_ext},
    ext_modules = [ext_module],
)
