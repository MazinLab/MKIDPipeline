from distutils.core import setup

from Cython.Build import cythonize

'''Cython example from http://docs.cython.org/en/latest/src/quickstart/build.html#building-a-cython-module-using-distutils
to compile parsePacketDump2's cython version, parsePacketDumpC.pyx.

Run in command line:
python compilePPDC.py build_ext --inplace
'''

'''
setup(
  name = 'Parse Packet Dump Cython',
  ext_modules = cythonize("parsePacketDumpC.pyx"),
)
'''

setup(
  name = 'binFile Cython',
  ext_modules = cythonize("binFileC.pyx"),
)

