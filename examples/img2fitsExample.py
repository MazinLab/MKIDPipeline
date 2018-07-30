"""
Author: Isabel Lipartito        Date: August 10, 2016
Loads darkness image stack given start and stop timestamps, then writes your input slice range to a fits file.  
Name of the fits file is in the format -starttime- to -endtime-.fits
"""

import os
import sys

import numpy as np
from astropy.io import fits as pyfits

from mkidpipeline.utils.arrayPopup import plotArray

beammap = '/mnt/data0/Darkness/20160722/filledBeammap20160722.txt'
verbose = True
mf=None

dataPath = '/mnt/data0/ScienceDataIMGs/'
imageShape = {'nRows':125,'nCols':80}

outpath = '/mnt/data0/ProcessedData/FITS/'
outfile = 'junk.FITS'


def loadImageStack(startTstamp, endTstamp, dark=None):
    timestampList = np.arange(startTstamp,endTstamp+1)
    images = []
    
    if beammap is not None:
        bmData = np.loadtxt(beammap)
        pixIndex = bmData[:,0]
        bmFlag = bmData[:,1]
        bmX = bmData[:,2]
        bmY = bmData[:,3]
        bmRoach = bmData[:,4]
    
    
    for iTs,ts in enumerate(timestampList):
        try:
            imagePath = os.path.join(dataPath,str(ts)+'.img')
            image = np.fromfile(open(imagePath, mode='rb'),dtype=np.uint16)
            image = np.transpose(np.reshape(image, (imageShape['nCols'], imageShape['nRows'])))
            if beammap is not None:
                newImage = np.zeros(image.shape,dtype=np.uint16)
                for y in range(len(newImage)):
                    for x in range(len(newImage[0])):
                        if bmFlag[np.where((y==bmY) & (x==bmX))[0]] == 0:
                            newImage[y,x] = image[y,x]
                image = newImage
        except IOError:
            image = np.zeros((imageShape['nRows'], imageShape['nCols']),dtype=np.uint16)
            
        if not dark==None:
            zeroes = np.where(dark>image)
            image-=dark
            image[zeroes]=0
        images.append(image)

    imageStack = np.array(images)
    return imageStack
    
def writeFits(imageStack, path):
    hdu = pyfits.PrimaryHDU(imageStack)
    hdulist = pyfits.HDUList([hdu])
    #hdulist.writeto('/mnt/data0/ProcessedData/FITS/new.fits')
    hdulist.writeto(path)

def medianFrame(imageStack):
    mf = np.median(imageStack, axis=0)
    return mf

if __name__ == "__main__":
    kwargs = {}
    if len(sys.argv) != 3:
        print('Usage: {} tstampStart tstampEnd'.format(sys.argv[0]))
        exit(0)
    else:
        startTstamp = int(sys.argv[1])
        endTstamp = int(sys.argv[2])
    
    darkStart = 1469354906
    darkEnd = 1469354926 
    darkStack = loadImageStack(darkStart, darkEnd)
    darkFrame = medianFrame(darkStack)
    
    imageStack = loadImageStack(startTstamp, endTstamp)
    #option to flatten stack to median combined frame
    mf = medianFrame(imageStack)
    
    FITSfile = str(startTstamp)+'_to_'+str(endTstamp)+'_goodBeammapOnly.fits'
    path = outpath+FITSfile
    writeFits(imageStack, path)
    
    if not mf==None:
        medianFile = str(startTstamp)+'_to_'+str(endTstamp)+'_goodBeammapOnly_median.fits'
        path = outpath+medianFile
        writeFits(mf, path)
        if verbose:
            plotArray(image=mf)
            #form = PopUp(showMe=False,title='B')
            #form.plotArray(np.arange(9).reshape(3,3))
            #form.show()

