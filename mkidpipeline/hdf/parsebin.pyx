
import os
import numpy as np
cimport numpy as np

cdef extern from "binlib.h":
    long parsebin(const char *file, unsigned long max_len, int* baseline, int* wavelength, unsigned long* timestamp, unsigned int* ycoord, unsigned int* xcoord, unsigned int* roach)
    #long parsebin(const char *file, unsigned long max_len,
     #             double* a1, double *a2, double* a3);

def parse(file, n=0):
    n = max(os.stat(file).st_size/8, n)
    baselne = np.empty(n, dtype=np.int)
    phase = np.empty(n, dtype=np.int)
    tstamp = np.empty(n, dtype=np.float64)
    y = np.empty(n, dtype=np.int)
    x = np.empty(n, dtype=np.int)

    r = parsebin(file.encode('UTF-8'), n,
                 <double*> np.PyArray_DATA(baselne),
                 <double*> np.PyArray_DATA(phase))
                 <double*> np.PyArray_DATA(tstamp))
                 <double*> np.PyArray_DATA(y))
                 <double*> np.PyArray_DATA(x))

    if r>n:
        return parse(file,abs(r))

    ret = np.array([baselne[:r],phase[:r]], dtype=[('x', float), ('y', int)])
    return ret.view(np.recarray)