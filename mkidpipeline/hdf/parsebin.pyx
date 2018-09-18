
import os
import numpy as np
cimport numpy as np

cdef extern from "binlib.h":
    long parsebin(const char *file, unsigned long max_len,
                  double* a1, double *a2, double* a3);

def parse(file, n=0):
    n = max(os.stat(file).st_size/64, n)
    a1 = np.empty(n, dtype=np.float64)
    a2 = np.empty(n, dtype=np.float64)

    r = parsebin(file.encode('UTF-8'), n,
                 <double*> np.PyArray_DATA(a1), <double*> np.PyArray_DATA(a1))

    if r<0:
        return parse(file,abs(r))

    ret = np.array([a1,a2], dtype=[('x', float), ('y', int)])
    return ret.view(np.recarray)