"""
This script corrects a firmware timing bug present through PAL2017b. It directly modifies the 
timestamps of the provided HDF5 file, so it only needs to be run once.

usage: python correctUnsortedTimestamps.py <path to h5 file>
"""

import tables
import os, sys, struct
import matplotlib.pyplot as plt
import numpy as np

def correctTimeStamps(timestamps):
    """
    Corrects errors in timestamps due to firmware bug present
    through PAL2017b.

    Parameters
    ----------
    timestamps: numpy array of integers
        List of timestamps from photon list. Must be in original, unsorted order.

    Returns
    -------
    Array of corrected timestamps, dtype is uint32
    """
    timestamps = np.array(timestamps, dtype=np.int64) #convert timestamps to signed values
    photonTimestamps = timestamps%500
    hdrTimestamps = timestamps - photonTimestamps

    unsortedInds = np.where(np.diff(timestamps)<0)[0]+1 #mark locations n where T(n)<T(n-1)

    for ind in unsortedInds:
        indsToIncrement = np.where(hdrTimestamps==hdrTimestamps[ind])[0]
        indsToIncrement = indsToIncrement[indsToIncrement>=ind]
        hdrTimestamps[indsToIncrement] += 500

    correctedTimestamps = hdrTimestamps + photonTimestamps

    if(np.any(np.diff(correctedTimestamps)<0)):
        correctedTimestamps = correctTimeStamps(correctedTimestamps)

    return np.array(correctedTimestamps, dtype=np.uint32)


if __name__=='__main__':
    if len(sys.argv)<2:
        print("Must specify h5 filename")
        exit(1)
    noResIDFlag = 2**32-1
    filename = sys.argv[1]
    hfile = tables.open_file(filename, mode='a')
    beamMap = hfile.root.BeamMap.Map.read()

    imShape = np.shape(beamMap)
    
    for x in range(imShape[0]):
        for y in range(imShape[1]):
            #print('Correcting pixel', x, y, ', resID =', obsfl.beamImage[x,y])
            resID = beamMap[x,y] 
            if resID == noResIDFlag:
                print('Table not found for pixel', x, ',', y)
                continue
            photonTable = hfile.get_node('/Photons/' + str(resID))
            photonList = photonTable.read()
            timeList = photonList['Time']
            correctedTimeList = correctTimeStamps(timeList)

            assert len(photonTable)==len(timeList), 'Timestamp list does not match length of photon list!'
            photonTable.modify_column(column=timeList, colname='Time')
            photonTable.flush()
                            
