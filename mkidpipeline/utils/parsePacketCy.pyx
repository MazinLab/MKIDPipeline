"""
File:      parsePacketDump.pyx
Author:    Matt Strader

Cythonized Sept 2018
KD Comments:
    Seems to be a QXcbConnection error when running main() as verbose

The code is parsing a single .bin file consisting of many frames. each frame starts with a header, followed by a number of photon events. The frame ends when there are either 100 photons or the next 0.5 millisecond is reached.
Both the header line and the photon event are saved as 'words', where each word is a 64 bit number. Each bit of the work corresponds to some information about the frame (header) or photon event.
The .bin file is a packed file, meaning that the words were converted into a tighter packed format for transfer. The Words were packed into strings in a specific translation mechanism
The code firsts unpacks the data and re-translates the words back into their original 64 bit format. This code then makes arrays of each information type contained within the header and photon data, and returns them.

Note on 'real' and 'fake' photons:
    The 'fake' photons are the result of a firmware technicality that requires data to be written when closing the frame and sending it to memory. The End of File can't be empty, so it generages garbage data that shouldn't be read.

Once unpacked the format of each word type is
The header format is:
  8 bits all ones, 8 bits roach num, 12 bits frame num, ufix36_1 bit timestamp

The photon format is:
  20 bits id, 9 bits ts, fix18_15 phase, fix17_14 base
    Assuming here ID is the pixel ID, and base is ??


"""

import os
import struct
import sys

import matplotlib.pyplot as plt
import numpy as np

from mkidpipeline.utils import binTools
from mkidpipeline.utils.arrayPopup import plotArray

# For debugging
from time import time

nRows = 125
nCols = 80

def parsePacketData(words,verbose=False):
    nWords = len(words)

    #######################################################
    # Processing Header Data
    headerFirstByte = 0xff  # =255, is a 8 bit number of all 1's

    firstBytes = words >> (64-8) # bitwise operator to shift all words by 56 bits to the right. Effectively, this cuts the bit length down to just 8. Then firstbytes can be read as 8bit numbers (ie firstbytes[0]=255, => 0xff)
    headerIdx = np.where(firstBytes == headerFirstByte)[0]
    headers = words[firstBytes == headerFirstByte] # pulls all headers out

    #header format: 8 bits all ones, 8 bits roach num, 12 bits frame num, ufix36_1 bit timestamp
    nBitsHdrTstamp = 36
    binPtHdrTstamp = 1
    nBitsHdrNum = 12
    nBitsHdrRoach = 8

    # Pulling out Header Roach Number Bitstring
    roachNumMask = binTools.bitmask(nBitsHdrRoach) # Returns an int, here 255. Code returns:int('1'*nBits,2)
    roachNums = (headers >> (nBitsHdrNum+nBitsHdrTstamp)) & roachNumMask
    roachList = np.unique(roachNums)

    # Pulling out Header Frame Number(timestamp?) Bitstring
    frameNumMask = binTools.bitmask(nBitsHdrNum)
    frameNums = (headers >> nBitsHdrTstamp) & frameNumMask
    frameNumDiff = np.diff(frameNums)


    nMissedFrames = np.sum(np.logical_and(frameNumDiff != 1,frameNumDiff != -((2**nBitsHdrNum) - 1)))

    ###############################################
    # Processing Photon Data

    # Finding 'Real' Photons
    #   See code doc ^^ for more info about what fake photons are, we only care about dem real 'uns
    fakePhotonWord = 2 ** 63 - 1 # just an int? has 63 bits, so not full word?
    realIdx = np.where(np.logical_and(firstBytes != headerFirstByte, words != fakePhotonWord))[0]
    realPhotons = words[realIdx]
    nRealPhotons = len(realPhotons)

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
    pixelIds = (realPhotons >> (nBitsPhtTstamp+nBitsPhtPhase+nBitsPhtBase)) & photonIdMask # chop off all but the first 20 bits and mask them
    xMask = binTools.bitmask(nBitsXCoord)
    yMask = binTools.bitmask(nBitsYCoord)

    # Extracting X,Y coordinates from Pixel ID
    #  PhtID is 20 bits, first 10 are X and last 10 are Y
    xCoords = (pixelIds >> nBitsYCoord) & xMask
    yCoords = pixelIds & yMask

    # Coordinate Photon Timestamp with Header Timestamp
    photonsHeaderIdx = np.searchsorted(headerIdx,realIdx)-1 # Grab the index location of the previous header (-1) in the .bin file for each photon
    photonsHeader = headers[photonsHeaderIdx] # then grab all the headers that have real photons associated with them
    #now get the timestamp from the header file
    headerTimestampBitmask = int('1'*nBitsHdrTstamp,2) # ?not calling binTools.bitmask for this?
    headerTimestamps = ((headerTimestampBitmask & photonsHeader)/2.)

    # Get the Header Frame number (did we already do this?)
    headerFrameNumBitmask = int('1'*nBitsHdrNum,2)
    headerFrameNums = headerFrameNumBitmask & (photonsHeader>>nBitsHdrTstamp)

    # Get Photon Timestamp
    photonTimestampMask = binTools.bitmask(nBitsPhtTstamp)
    photonTimestamps = (realPhotons >> (nBitsPhtPhase+nBitsPhtBase)) & photonTimestampMask

    # Get Full Photon Timestamp by Combining Header+Photon Tstamps
    photonTimestamps = photonTimestamps*1.e-3 + headerTimestamps #convert us to ms
    #dt = np.diff(photonTimestamps)

    # Get Photon Phase
    phtPhaseMask = int('1'*nBitsPhtPhase,2)
    phases = (realPhotons >> nBitsPhtBase) & phtPhaseMask
    phasesDeg = 180./np.pi * binTools.reinterpretBin(phases,nBits=nBitsPhtPhase,binaryPoint=binPtPhtPhase)

    # Get Photon Base
    phtBaseMask = int('1'*nBitsPhtBase,2)
    bases = (realPhotons) & phtPhaseMask
    basesDeg = 180./np.pi * binTools.reinterpretBin(bases,nBits=nBitsPhtBase,binaryPoint=binPtPhtBase)

    # Make Image In a Very Difficult Way
    image = np.zeros((nRows,nCols))
    for x,y in zip(xCoords,yCoords):
        image[y,x] += 1


    return {'basesDeg':basesDeg,'phasesDeg':phasesDeg,'photonTimestamps':photonTimestamps,'pixelIds':pixelIds,'photons':realPhotons,'headers':headers,'roachNums':roachNums,'image':image,'xCoords':xCoords,'yCoords':yCoords}

if __name__=='__main__':
    import pyximport; pyximport.install()
    startT = time()

    # Getting Data path from imput arguments
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = 'photonDump.bin'

    # Using the .bin file name as timestamp and saving it
    pathTstamp = os.path.splitext(os.path.basename(path))[0]

    # Opening the File
    try:
        with open(path,'rb') as dumpFile:
            data = dumpFile.read()
    except (IOError, ValueError) as e:
        print(e)

    nBytes = len(data)
    nWords = nBytes//8 #64 bit words

    #break into 64 bit words
    # Next line of code gets ugly
    # '>{:d}Q'.format(nwords) : this is mostly doing the new python way of converting numbers to string format
    #   the 'stuff{foo}'.format(bar) replaces {foo} with the value of bar in the format specified by foo
    #   {stuff in curly braces} has the format of {[field name][!conversion][:field_spec]}
    #   so :d means field_spec=d, where d is decimal integer
    #   so in this case, bar = int(nWords) == 1413468; and the data type is specified as decimal integer by foo
    #   then we get '>barQ' being passed as the format type into struct.upack

    # Now then, struct.unpack(fmt,string)
    #   fmt here is '>barQ'
    #   where > means byteorder=little-endian, size=standard, alignment=none
    #   and again bar=some number with datatype of decimal integer
    #   it is confusing what Q means here (repeat format type?), but assuming it is the same for the struct call then:
    #   Q means ctype=unsigned long long, pytype=float, std size=8
    #   string=data (where we confusingly call a variable data)
    #   data is something that looks like: b'\xffwj\x86\x00\xf6\x86C.........
    #   struct.upack(fmt,string) returns a tuple

    # Finally, we save this into a numpy ndarray of tupes with datatype=object
    #   The final result is that words is a numpy ndarray
    #   The array has 1 row, nWords columns
    #   each element is a big ass number. words[0].bit_length=64 and type(words[0])=int
    words = np.array(struct.unpack('>{:d}Q'.format(nWords), data),dtype=object)

    # Calling the module
    parseDict = parsePacketData(words,verbose=False)

    # Separating Returned Data into Readable Arrays
    phasesDeg = parseDict['phasesDeg']
    basesDeg = parseDict['basesDeg']
    pixelIds = parseDict['pixelIds']
    image = parseDict['image']
'''
    #selPixelId = 0#(1<<10)+23
    selPixelId = (30<<10)+46
    print('selected pixel',selPixelId)
    print(len(np.where(pixelIds==selPixelId)),'photons for selected pixel')

    print('phase',phasesDeg[0:10])
    print('base',basesDeg[0:10])
    print('IDs',pixelIds[0:10])

    fig,ax = plt.subplots(1,1)
    ax.plot(phasesDeg[np.where(pixelIds==selPixelId)])
    ax.plot(basesDeg[np.where(pixelIds==selPixelId)])
    ax.set_title('phases (deg)')
    #ax.plot(pixelIds)
    plotArray(image,origin='upper')

    np.savez('/mnt/data0/test2/{}.npz'.format(pathTstamp),**parseDict)


    plt.show()
'''