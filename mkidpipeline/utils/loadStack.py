import os
import struct

import numpy as np
import tables

from mkidpipeline.utils.parsePacketDump import parsePacketData

"""
Utilities for loading sets of image stacks from either .IMG or .bin files
"""

def loadIMGStack(dataDir, start, stop, nCols=80, nRows=125, verbose=True):
    useImg = True
    frameTimes = np.arange(start, stop+1)
    frames = []
    for iTs,ts in enumerate(frameTimes):
        try:
            if useImg==False:
                imagePath = os.path.join(dataDir,str(ts)+'.bin')
                print(imagePath)
                with open(imagePath,'rb') as dumpFile:
                    data = dumpFile.read()

                nBytes = len(data)
                nWords = nBytes/8 #64 bit words

                #break into 64 bit words
                words = np.array(struct.unpack('>{:d}Q'.format(nWords), data),dtype=object)
                parseDict = parsePacketData(words,verbose=False)
                image = parseDict['image']

            else:
                imagePath = os.path.join(dataDir,str(ts)+'.img')
                if verbose:
                    print(imagePath)
                image = np.fromfile(open(imagePath, mode='rb'),dtype=np.uint16)
                image = np.transpose(np.reshape(image, (nCols, nRows)))

        except (IOError, ValueError):
            print("Failed to load ", imagePath)
            image = np.zeros((nRows, nCols),dtype=np.uint16)
        frames.append(image)
    stack = np.array(frames)
    return stack
"""
def loadBINStack(dataDir, start, stop, nCols=80, nRows=125):
    useImg = False
    frameTimes = np.arange(start, stop+1)
    frames = []
    for iTs,ts in enumerate(frameTimes):
        try:
            if useImg==False:
                imagePath = os.path.join(dataDir,str(ts)+'.bin')
                print(imagePath)
                with open(imagePath,'rb') as dumpFile:
                    data = dumpFile.read()

                nBytes = len(data)
                nWords = nBytes/8 #64 bit words

                #break into 64 bit words
                words = np.array(struct.unpack('>{:d}Q'.format(nWords), data),dtype=object)
                parseDict = parsePacketData(words,verbose=False)
                image = parseDict['image']

            else:
                imagePath = os.path.join(dataDir,str(ts)+'.img')
                print(imagePath)
                image = np.fromfile(open(imagePath, mode='rb'),dtype=np.uint16)
                image = np.transpose(np.reshape(image, (nCols, nRows)))

        except (IOError, ValueError):
            print("Failed to load ", imagePath)
            image = np.zeros((nRows, nCols),dtype=np.uint16)
        frames.append(image)
    stack = np.array(frames)
    return stack
"""
def loadBINStack(dataDir, start, stop, nCols=80, nRows=125):
    useImg = False
    frameTimes = np.arange(start, stop+1)
    frames = []
    for iTs,ts in enumerate(frameTimes):
            if useImg==False:
                imagePath = os.path.join(dataDir,str(ts)+'.bin')
                print(imagePath)
                with open(imagePath,'rb') as dumpFile:
                    data = dumpFile.read()

                nBytes = len(data)
                nWords = nBytes/8 #64 bit words

                #break into 64 bit words
                words = np.array(struct.unpack('>{:d}Q'.format(nWords), data),type=object)
                parseDict = parsePacketData(words,verbose=False)
                image = parseDict['image']

            else:
                imagePath = os.path.join(dataDir,str(ts)+'.img')
                print(imagePath)
                image = np.fromfile(open(imagePath, mode='rb'),type=np.uint16)
                image = np.transpose(np.reshape(image, (nCols, nRows)))

        
            frames.append(image)
    stack = np.array(frames)
    return stack


def loadH5Stack(h5Path, verbose=True):
    if (not os.path.exists(h5Path)):
        msg='file does not exist: %s'%h5Path
        if verbose:
            print(msg)
        raise Exception(msg)
    f = tables.open_file(h5Path, mode='r')

    stackParams = f.get_node('/imageStack/params').read()

    imageStack = np.array(f.get_node('/imageStack/rawImgs').read(),dtype=float)
    timeStamps = np.array(f.get_node('/imageStack/timestamps').read())
    ditherArray = np.array(f.get_node('/imageStack/dithers').read())
    hotArray = np.array(f.get_node('/imageStack/hpms').read())
    coldArray = np.array(f.get_node('/imageStack/cpms').read())
    deadArray = np.array(f.get_node('/imageStack/dpms').read())
    aperArray = np.array(f.get_node('/imageStack/ams').read())
    roughXArray = np.array(f.get_node('/imageStack/roughX').read())
    roughYArray = np.array(f.get_node('/imageStack/roughY').read())
    fineXArray = np.array(f.get_node('/imageStack/fineX').read())
    fineYArray = np.array(f.get_node('/imageStack/fineY').read())
    centXArray = np.array(f.get_node('/imageStack/centX').read())
    centYArray = np.array(f.get_node('/imageStack/centY').read())
    dark = np.array(f.get_node('/imageStack/dark').read())
    flat = np.array(f.get_node('/imageStack/flat').read())
    final = np.array(f.get_node('/imageStack/finalImg').read())

    f.close()
    return {"stack":imageStack,"times":timeStamps,"params":stackParams,"final":final,"centX":centXArray,
            "centY":centYArray,"dark":dark,"hpm":hotArray}


def loadCubeStack(npzPath,verbose=True):
    """
    Load up times, cubes, and wvlBinEdges from npz output of makeCubeTimestream
    """
    npzfile = np.load(npzPath)
    times = npzfile['times']
    cubes = npzfile['cubes']
    wbe = npzfile['wvlBinEdges']
    npzfile.close()
    return {"times":times,"cubes":cubes,"wvlBinEdges":wbe}
