"""
Author: Seth Meeker        Date: November 9, 2016

function to load up darkness .bin file and output a .img file
in the format they would have been written to ramdisk in

"""

import glob
import os
import struct

import matplotlib
import numpy as np

matplotlib.rcParams['backend.qt4']='PyQt4'
#from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from parsePacketDump2 import parsePacketData


def bin2img(ts, binPath = '/mnt/data0/ScienceData/', imgPath = '/mnt/data0/ScienceDataIMGs/'):
    imagePath = os.path.join(binPath,str(ts)+'.bin')
    
    with open(imagePath,'rb') as dumpFile:
        data = dumpFile.read()

    nBytes = len(data)
    nWords = nBytes/8 #64 bit words
                
    #break into 64 bit words
    words = np.array(struct.unpack('>{:d}Q'.format(nWords), data),dtype=object)

    parseDict = parsePacketData(words,verbose=False)

    image = parseDict['image']
    
    image = np.transpose(image)
    
    #formattedImage = np.array(image, dtype=np.uint16)
    image.flatten()
    
    imagePath = os.path.join(imgPath,str(ts)+'.img')
    image.tofile(imagePath)

    print "Wrote IMG to ", imagePath
    
    return
    
if __name__=='__main__':

    binPath = '/mnt/data0/ScienceData/20161107/'
    imgPath = '/mnt/data0/ScienceDataIMGs/20161107/'
    
    #startTstamp = 1478650710
    #endTstamp = 1478650715

    timestampList = []
    fileName = '*.bin'

    for bf in sorted(glob.glob(os.path.join(binPath,fileName))):
        pathTstamp = os.path.splitext(os.path.basename(bf))[0]
        timestampList.append(pathTstamp)
    
    print timestampList
    
    #timestampList = np.arange(startTstamp,endTstamp+1)
    timestampList = np.array(timestampList)

    for iTs,ts in enumerate(timestampList):
        bin2img(ts, binPath, imgPath)
    
