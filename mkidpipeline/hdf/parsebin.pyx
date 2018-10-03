'''
parsebin.pyx
Created by KD and JB Sept 2018

This cython module will take an input .bin file and send it to be parsed to
with a .c file and return the data as a python struct. The parsing is
done in a program binlib.c, and finction variables are declared in binlib.h
The parsed data has the full timestamp created from adding the header and
data packet timestamps in the .bin files.

The input .bin file to the function contains both header packets and
"fake photon" data, which is a photon created by the firmware to end an
ethernet frame. The firmware is set to end a fframe either when there are
100 photons in the frame or the clock reaches the next 0.5 ms. In the latter
casse, the firmware MUST generate a fake photon to end the frame. These
fake photons are not removed in the binlib.c code (they could be. We remove
them here in the cython file by creating a new numpy array through the
np.delete function. We have to create a new array, even though it is memory
intensive, because you cannot change a numpy array's shape (easily).

File is called by:
p = parsebin.parse("fname.bin",n)

Returns
p.base = baseline
p.phase = phase
p.tstamp = timestamp
p.y = ycoord
p.x = xcoord
p.roach = roachnum

IF the .c code has been changed, you can recompile it
$ gcc -c binlib.c -O3

When updating this code:
You must also recompile the setup.py file located in the current directory
which containes the information to make(compile) the cythonized file.
(in current repository, in .../MKIDPipeline dir)
$ python setup.py build_ext --inplace

'''

import os
import numpy as np
cimport numpy as np


###############################################################################
# Cythonized .h file
# Declare value types in Cython that the binlib.h file needs
# The namespace used here is only relevant to the C files; you do
#  not need to match the namespace in the rest of the .pyx file

cdef extern from "binlib.h":
    long cparsebin(const char *fName, unsigned long max_len, int* baseline, int* wavelength, unsigned long* timestamp, unsigned int* ycoord, unsigned int* xcoord, unsigned int* roach)

###############################################################################
# Calling binlib and passing it stuff
#####################################

def parse(file,_n=0):
    # Creating pointers to memory bocks that the binlib.c code will fill
    n = int(max(os.stat(file).st_size/8, _n))
    baseline   = np.empty(n, dtype=np.int)
    wavelength = np.empty(n, dtype=np.int)
    timestamp  = np.empty(n, dtype=np.uint64)
    y = np.empty(n, dtype=np.uint32)
    x = np.empty(n, dtype=np.uint32)
    roachnum = np.empty(n, dtype=np.uint32)

    # Calling parsebin from binlib.c
    # npackets is the number of real+fake photons processed by binlib.c
    npackets = cparsebin(file.encode('UTF-8'), n,
                 <int*>np.PyArray_DATA(baseline),
                 <int*>np.PyArray_DATA(wavelength),
                 <unsigned long*>np.PyArray_DATA(timestamp),
                 <unsigned int*>np.PyArray_DATA(y),
                 <unsigned int*>np.PyArray_DATA(x),
                 <unsigned int*>np.PyArray_DATA(roachnum))
    print("number of photons = {}".format(npackets))
    #r = cparsebin(file.encode('UTF-8'), n, &baseline[0], &wavelength[0], &timestamp[0],&y[0], &x[0], &roachnum[0])

    # Raising Errors
    if npackets>n:
        return parse(file,abs(npackets))
    elif npackets<0:
        errors = {-1:'Data not found'}
        raise RuntimeError(errors.get(npackets,'Unknown Error: {}'.format(npackets)))

    # Create new struct from output from binlib.c
    # What this does is only grab the elements of the arrays that are real valued
    # This essentially clips the data since we declared it to be as long as the .bin
    #  file, but some of those lines were headers, which are now empty in the returned arrays
    square = list(zip(baseline[:npackets],wavelength[:npackets],timestamp[:npackets],y[:npackets],x[:npackets],roachnum[:npackets]))
    parsed = np.array(square, dtype=[('base', int),('phase', int), ('tstamp', np.float64),('y', int), ('x', int),('roach', int)])
    parsed = parsed.view(np.recarray)

    # Remove Fake Photons
    #   fake photons translate to x=511 when you read the bitvalues as numbers
    #   we just throw them away since they aren't useful to anybody
    ret = np.delete(parsed,(parsed.x==511).nonzero())

    return ret