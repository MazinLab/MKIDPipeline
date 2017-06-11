import numpy as np
import os, struct

'''
Utilities for loading sets of image stacks from either .IMG or .bin files
'''

def loadIMGStack(dataDir, start, stop, useImg = True, nCols=80, nRows=125):
    frameTimes = np.arange(start, stop+1)
    frames = []
    for iTs,ts in enumerate(frameTimes):
        try:
            if useImg==False:
                imagePath = os.path.join(dataDir,str(ts)+'.bin')
                print imagePath
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
                print imagePath
                image = np.fromfile(open(imagePath, mode='rb'),dtype=np.uint16)
                image = np.transpose(np.reshape(image, (nCols, nRows)))

        except (IOError, ValueError):
            print "Failed to load ", imagePath
            image = np.zeros((nRows, nCols),dtype=np.uint16)  
        frames.append(image)
    stack = np.array(frames)
    return stack
