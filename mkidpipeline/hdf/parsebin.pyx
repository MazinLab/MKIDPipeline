'''
parsebin.pyx
Created by KD and JB Sept 2018

This cython module will take an input .bin file and send it to be parsed to
with a .c file and return the data as a python struct. The parsing is
done in a program binlib.c, and finction variables are declared in binlib.h
The parsed data has the full timestamp created from adding the header and
data packet timestamps in the .bin files.

File is called by:
p = parsebin.parse(fname.bin)

Returns
p.base = baseline
p.phase = phase
p.tstamp = timestamp
p.y = ycoord
p.x = xcoord
p.roach = roachnum

IF the .c code has been changed, you can recompile it
gcc -c binlib.c -O3

When updating this code:
You must also recompile the setup.py file located in the current directory
which containes the information to make(compile) the cythonized file.
python setup.py build_ext --inplace

'''

import os
import numpy as np
cimport numpy as np

###############################################################################
# Cythonized .h file
# Declare value types in Cython that the binlib.h file needs

cdef extern from "binlib.h":
    long cparsebin(const char *fName, unsigned long max_len, int* baseline, int* wavelength, unsigned long* timestamp, unsigned int* ycoord, unsigned int* xcoord, unsigned int* roach)

###############################################################################
# Calling binlib and passing it stuff
#####################################

def parse(file, n=0):
    # Creating pointers to memory bocks that the .c code will fill
    n = int(max(os.stat(file).st_size/8, n))
    baseline   = np.empty(n, dtype=np.int)
    wavelength = np.empty(n, dtype=np.int)
    timestamp  = np.empty(n, dtype=np.uint64)
    y = np.empty(n, dtype=np.uint32)
    x = np.empty(n, dtype=np.uint32)
    roachnum = np.empty(n, dtype=np.uint32)

    # Calling parsebin from binlib.c
    r = cparsebin(file.encode('UTF-8'), n,
                 <int*> np.PyArray_DATA(baseline),
                 <int*> np.PyArray_DATA(wavelength),
                 <unsigned long*> np.PyArray_DATA(timestamp),
                 <unsigned int*> np.PyArray_DATA(y),
                 <unsigned int*> np.PyArray_DATA(x),
                 <unsigned int*> np.PyArray_DATA(roachnum))


    # Raising Errors
    if r>n:
        return parse(file,abs(r))
    elif r<0:
        errors = {-1:'Data not found'}
        raise RuntimeError(errors.get(r,'Unknown Error: {}'.format(r)))

    # Create new struct from output from binlib.c
    # What this does is only grab the elements of the arrays that are real valued
    # This essentially clips the data since we declared it to be as long as the .bin
    #  file, but some of those lines were headers, which are now empty in the returned arrays
    ret = np.array([baseline[:r],wavelength[:r],timestamp[:r],y[:r],x[:r],roachnum[:r]],
                   dtype=[('base', int),('phase', int),
                          ('tstamp', np.float64),('y', int), ('x', int),
                          ('roach', int)])

    return ret.view(np.recarray)