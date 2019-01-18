from __future__ import division
import numpy as np

cimport numpy as np



def tsBinner(np. ndarray[np.uint64_t, ndim = 1] tstamps, np.ndarray[np.int_t, ndim = 1] bins):
"""
bin the values of the tstamps array into the bins array.

tstamps array is of type np.uint64

bins array is of type int
"""
cdef int nTStamps = tstamps.shape[0]
cdef int iTStamp
for iTStamp in range(nTStamps):
    bins[tstamps[iTStamp]] += 1
return


def tsBinner32(np.ndarray[np.uint32_t, ndim = 1] tstamps, np.ndarray[np.int_t, ndim = 1] bins):
"""
bin the values of the tstamps array into the bins array.

tstamps array is of type np.uint32

bins array is of type int
"""
cdef int nTStamps = tstamps.shape[0]
cdef int iTStamp
for iTStamp in range(nTStamps):
    bins[tstamps[iTStamp]] += 1
return
