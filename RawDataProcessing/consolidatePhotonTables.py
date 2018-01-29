import tables
import numpy as np
import os, sys
from Headers.ObsFileHeaders import ObsFileCols

if(len(sys.argv)<2):
    print("Usage: python consolidatePhotonTables.py <filename>")
    exit(1)

fn = sys.argv[1]
hfile = tables.open_file(fn, mode='a')

nRows = 0
for pixTable in hfile.iter_nodes('/Photons'):
    nRows += pixTable.shape[0]

photonTable = hfile.create_table('/Photons', 'PhotonTable', ObsFileCols, 'Photon Table', expectedrows=nRows)

beamMap = hfile.get_node('/BeamMap/Map').read()
resIDList = np.sort(beamMap.flatten())

for resID in resIDList:
    pixelTable = hfile.get_node('/Photons/' + str(resID)).read()
    photonTable.append(pixelTable)
    photonTable.flush()
    hfile.remove_node('/Photons/' + str(resID))

hfile.close()
