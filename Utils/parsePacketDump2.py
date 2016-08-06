"""
File:      parsePacketDump.py
Author:    Matt Strader

"""

import matplotlib, time, struct
import matplotlib.pyplot as plt
import numpy as np
from Utils import binTools
import sys
from Utils.arrayPopup import plotArray
import os

nRows = 125
nCols = 80

def parsePacketData(words,verbose=False):

    nWords = len(words)

    fakePhotonWord = 2**63-1
    headerFirstByte = 0xff

    firstBytes = words >> (64-8)
    if verbose:
        print nWords,' words parsed'
    headerIdx = np.where(firstBytes == headerFirstByte)[0]
    headers = words[firstBytes == headerFirstByte]
    if verbose:
        print len(headerIdx),'headers'

    if verbose:
        fig,ax = plt.subplots(1,1)
        ax.plot(np.diff(headerIdx))
        ax.set_title('frame size')
        print np.max(np.diff(headerIdx)),'max frame size'

    fakeIdx = np.where(words == fakePhotonWord)[0]
    if verbose:
        print len(fakeIdx),'fake photons'

    #header format: 8 bits all ones, 8 bits roach num, 12 bits frame num, ufix36_1 bit timestamp
    nBitsHdrTstamp = 36
    binPtHdrTstamp = 1
    nBitsHdrNum = 12
    nBitsHdrRoach = 8

    roachNumMask = binTools.bitmask(nBitsHdrRoach)
    roachNums = (headers >> (nBitsHdrNum+nBitsHdrTstamp)) & roachNumMask
    roachList = np.unique(roachNums)
    if verbose:
        print np.unique(roachNums)

    frameNumMask = binTools.bitmask(nBitsHdrNum)
    frameNums = (headers >> nBitsHdrTstamp) & frameNumMask
    frameNumDiff = np.diff(frameNums)

    #for roach in roachList:
    #    frameNumsByRoach.append(frameNums[roachNums==roach])

    nMissedFrames = np.sum(np.logical_and(frameNumDiff != 1,frameNumDiff != -((2**nBitsHdrNum) - 1)))
    if verbose:
        fig,ax = plt.subplots(1,1)
        ax.plot(np.diff(frameNums))
        ax.set_title('frame nums')
        print nMissedFrames,'missed frames'


    realIdx = np.where(np.logical_and(firstBytes != headerFirstByte, words != fakePhotonWord))[0]
    realPhotons = words[realIdx]
    nRealPhotons = len(realPhotons)
    if verbose:
        print nRealPhotons,'real photons parsed'
    #photon format: 20 bits id, 9 bits ts, fix18_15 phase, fix17_14 base
    nBitsPhtId = 20
    nBitsXCoord = 10
    nBitsYCoord = 10
    nBitsPhtTstamp = 9
    nBitsPhtPhase = 18
    binPtPhtPhase = 15
    nBitsPhtBase = 17
    binPtPhtBase = 14

    #find each photon's corresponding header
    photonIdMask = binTools.bitmask(nBitsPhtId)
    pixelIds = (realPhotons >> (nBitsPhtTstamp+nBitsPhtPhase+nBitsPhtBase)) & photonIdMask
    xMask = binTools.bitmask(nBitsXCoord)
    yMask = binTools.bitmask(nBitsYCoord)

    xCoords = (pixelIds >> nBitsYCoord) & xMask
    yCoords = pixelIds & yMask
    #selectId = 34
    #selectMask = pixelIds==selectId
    #realIdx = realIdx[selectMask]
    #realPhotons = realPhotons[selectMask]

    photonsHeaderIdx = np.searchsorted(headerIdx,realIdx)-1

    photonsHeader = headers[photonsHeaderIdx]
    #now get the timestamp from this
    headerTimestampBitmask = int('1'*nBitsHdrTstamp,2)
    headerTimestamps = ((headerTimestampBitmask & photonsHeader)/2.)

    headerFrameNumBitmask = int('1'*nBitsHdrNum,2)
    headerFrameNums = headerFrameNumBitmask & (photonsHeader>>nBitsHdrTstamp)

    photonTimestampMask = binTools.bitmask(nBitsPhtTstamp)
    photonTimestamps = (realPhotons >> (nBitsPhtPhase+nBitsPhtBase)) & photonTimestampMask

    photonTimestamps = photonTimestamps*1.e-3 + headerTimestamps #convert us to ms
    dt = np.diff(photonTimestamps)
    if verbose:
        fig,ax = plt.subplots(1,1)
        ax.plot(photonTimestamps)
        ax.set_title('timestamps')

    phtPhaseMask = int('1'*nBitsPhtPhase,2)
    phases = (realPhotons >> nBitsPhtBase) & phtPhaseMask
    phasesDeg = 180./np.pi * binTools.reinterpretBin(phases,nBits=nBitsPhtPhase,binaryPoint=binPtPhtPhase)

    phtBaseMask = int('1'*nBitsPhtBase,2)
    bases = (realPhotons) & phtPhaseMask
    basesDeg = 180./np.pi * binTools.reinterpretBin(bases,nBits=nBitsPhtBase,binaryPoint=binPtPhtBase)

    image = np.zeros((nRows,nCols))
    for x,y in zip(xCoords,yCoords):
        image[y,x] += 1


    return {'basesDeg':basesDeg,'phasesDeg':phasesDeg,'photonTimestamps':photonTimestamps,'pixelIds':pixelIds,'photons':realPhotons,'headers':headers,'roachNums':roachNums,'image':image,'xCoords':xCoords,'yCoords':yCoords}

if __name__=='__main__':
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = 'photonDump.bin'

    pathTstamp = os.path.splitext(os.path.basename(path))[0]
    with open(path,'rb') as dumpFile:
        data = dumpFile.read()

    nBytes = len(data)
    nWords = nBytes/8 #64 bit words
    #break into 64 bit words
    words = np.array(struct.unpack('>{:d}Q'.format(nWords), data),dtype=object)

    parseDict = parsePacketData(words,verbose=True)

    phasesDeg = parseDict['phasesDeg']
    basesDeg = parseDict['basesDeg']
    pixelIds = parseDict['pixelIds']
    image = parseDict['image']

    #selPixelId = 0#(1<<10)+23
    selPixelId = (30<<10)+46
    print 'selected pixel',selPixelId
    print len(np.where(pixelIds==selPixelId)),'photons for selected pixel'

    print 'phase',phasesDeg[0:10]
    print 'base',basesDeg[0:10]
    print 'IDs',pixelIds[0:10]

    fig,ax = plt.subplots(1,1)
    ax.plot(phasesDeg[np.where(pixelIds==selPixelId)])
    ax.plot(basesDeg[np.where(pixelIds==selPixelId)])
    ax.set_title('phases (deg)')
    #ax.plot(pixelIds)
    plotArray(image,origin='upper')

    np.savez('/mnt/data0/test2/{}.npz'.format(pathTstamp),**parseDict)


    plt.show()
