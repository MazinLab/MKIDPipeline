import cython
import numpy as np

@cython.boundscheck(False)
@cython.wraparound(False)

def removedeadtime(double [:] t, double deadtime):

    n = t.shape[0]
    keep_np = np.ones((n), np.int)
    cdef long [:] keep = keep_np

    cdef int i, j

    for i in range(n):
        for j in range(i - 1, -1, -1):
            if t[i] - t[j] > deadtime:
                break
            elif keep[j]:
                keep[i] = 0
                break
            
    return keep_np
    
def recursion(double [:] r, double [:] g, double f, double sqrt1mf2, int n):

    cdef int i
    for i in range(1, n):
        r[i] = r[i - 1]*f + g[i]*sqrt1mf2

    return r

    
