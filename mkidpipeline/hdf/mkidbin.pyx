import numpy as np
cimport numpy as np
import os
from mkidcore.corelog import getLogger

PHOTON_BIN_SIZE_BYTES = 8
PHOTON_SIZE_BYTES = 5*4


cdef extern from "binprocessor.h":
    struct photon
    long extract_photons(const char *dname, unsigned long start, unsigned long inttime, const char *bmap,
                         unsigned int x, unsigned int y,unsigned long n_max_photons, photon* photons)
    long extract_photons_dummy(const char *dname, unsigned long start, unsigned long inttime, const char *bmap,
                               unsigned int x, unsigned int y, unsigned long n_max_photons, photon* photons)


np_photon = np.dtype([('resID',np.uint32),
                      ('timestamp', np.uint32),
                      ('wvl', np.float32),
                      ('wSpec', np.float32),
                      ('wNoise', np.float32)], align=True)


def extract(directory, start, inttime, beamfile, x, y):
    files = [os.path.join(directory, '{}.bin'.format(t)) for t in range(start-1, start+inttime)]
    n_max_photons = int(np.ceil(sum([os.stat(f).st_size for f in files])/PHOTON_BIN_SIZE_BYTES))
    getLogger(__name__).debug('Calling C to extract ~{:g} photons, will require ~{:.1f}GB of RAM'.format(n_max_photons,
                                                                                                         n_max_photons*PHOTON_SIZE_BYTES/1024/1024/1024))
    photons = np.zeros(n_max_photons, dtype=np_photon)
    photons = np.ascontiguousarray(photons)
    nphotons = extract_photons(directory.encode('UTF-8'), start, inttime, beamfile.encode('UTF-8'), x, y,
                               n_max_photons, <photon*> np.PyArray_DATA(photons))
    getLogger(__name__).debug('C code returned {} photons'.format(nphotons))
    photons = photons[:nphotons]
    return photons[photons['timestamp']>=start]


def extract_fake(nphotons, start=1547683242, intt=150, nres=20000):
    photons = np.zeros(nphotons, dtype=np_photon)
    photons['resID'] = np.random.randint(0, nres, nphotons, np.uint32)
    photons['timestamp'] = np.random.randint(start,start+intt, nphotons, np.uint32)
    photons['wvl'] = np.random.random(nphotons)
    photons['wSpec'] = np.random.random(nphotons)
    photons['wNoise'] = np.random.random(nphotons)
    return photons

def test(nphot=10):
    #see https://stackoverflow.com/questions/17239091/cython-memoryviews-from-array-of-structs
    #https://cython.readthedocs.io/en/latest/src/userguide/memoryviews.html

    # np_photon = np.dtype([('resID',np.uint32),
    #                       ('timestamp', np.uint32),
    #                       ('wvl', np.float32),
    #                       ('wSpec', np.float32),
    #                       ('wNoise', np.float32)], align=True)

    photons = np.zeros(nphot, dtype=np_photon)
    photons = np.ascontiguousarray(photons)

    getLogger(__name__).debug('Calling C to extract ~{:g} photons, will require ~{:.1f}GB of RAM'.format(nphot,
                                                                                                         nphot*PHOTON_SIZE_BYTES/1024/1024/1024))
    photons['resID'] = np.arange(nphot)
    photons['timestamp'] = np.arange(nphot)*2
    photons['wvl'] = np.ones(nphot)
    photons['wSpec'] = np.ones(nphot)*2
    photons['wNoise'] = np.ones(nphot)*3

    ret = extract_photons_dummy('/a/test/dir/'.encode('UTF-8'), 123456789, 54321, '/a/test/beammap'.encode('UTF-8'),
                                111, 222, nphot, <photon*> np.PyArray_DATA(photons))
    getLogger(__name__).debug('C code returned {} photons'.format(ret))

    return photons