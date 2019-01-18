from mkidcore.headers import ObsFileCols
import numpy as np
import tables
import time
np_photon = np.dtype([('resid',np.uint32),
                      ('timestamp', np.uint32),
                      ('wavelength', np.float32),
                      ('wSpec', np.float32),
                      ('wNoise', np.float32)], align=True)



""" Test 1 generate a basic photon table: ~10s for 375 mphotons"""
nphotons=375000000
np.random.randint(0,20000, nphotons, np.uint32)
photons = np.zeros(nphotons, dtype=np_photon)
photons['resid'] = np.random.randint(0,20000, nphotons, np.uint32)
photons['timestamp'] = np.random.randint(1547683242,1547683242+150, nphotons, np.uint32)
photons['wavelength'] = np.random.random(nphotons)
photons['wSpec'] = np.random.random(nphotons)
photons['wNoise'] = np.random.random(nphotons)

tic=time.time()
filter = tables.Filters(complevel=1, complib='blosc', shuffle=True, bitshuffle=False, fletcher32=False)
h5file = tables.open_file("test.h5", mode="w", title="Test file")
group = h5file.create_group("/", 'Photons', 'Photon Information')
table = h5file.create_table(group, name='PhotonTable', description=ObsFileCols,
                            title="Photon Table", expectedrows=nphotons, filters=filter)
table.append(photons)
h5file.close()
toc=time.time()
print(toc-tic)