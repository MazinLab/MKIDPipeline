'''
Author Seth Meeker 2017-06-11

New PtSi devices do not seem to display hot pixel behavior from the same mechanism as
old TiN devices. As such, the full time masking technique used in the ARCONS pipeline
may not be necessary. Bad pixels largely remain "bad" with little switching behavior. 
A small population have a sawtooth pattern in their lightcurve, possibly due to constant, 
slow loop rotation. These may need the full time masking routine for cleaning, but are 
ignored for now.

This routine provides a temporary (though quite good) means of determining which pixels
are hot using the dark frames taken around an observation.

We first apply a hard cut on anything over 2450 counts, since the firmware has a soft
2500 cps limit to stop buffers from overflowing with hot pixels' data. Anything over ~2450
cps in the dark is cut outright. We then take the mean cps on the remaining pixels,
and further cut anything with cps > 5-sigma over the mean.

functions:
makeMask(run=None, date=None, startTimeStamp=None, stopTimeStamp=None, verbose=False, sigma=5, maxCut=2450)
saveMask(mask, timeStamp=None, date=None)
loadMask(path)
plotMask(mask)

'''

import sys, os, time, struct
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from functools import partial
from arrayPopup import plotArray
import utils
from loadStack import loadIMGStack


#ultimately want the path to be flexible: use .bin files if they exist,
#and fall back on .img files if necessary (typically we didn't record
#.bin files for dark data in 2017a)
basePath = '/mnt/data0/ScienceDataIMGs/'
nCols=80
nRows=125


def makeMask(run=None, date=None, startTimeStamp=None, stopTimeStamp=None, verbose=False, sigma=5, maxCut=2450):
    dataPath = basePath+str(run)+os.path.sep+str(date)+os.path.sep

    stack = loadIMGStack(dataPath, startTimeStamp, stopTimeStamp, useImg = True, nCols=nCols, nRows=nRows)
    medStack = utils.medianStack(stack)

    if verbose:
        try:
            plotArray(medStack,title='Median Dark Stack')
        except:
            plt.matshow(medStack)
            plt.show()

    mask = np.zeros(np.shape(medStack),dtype=int)
    mask[np.where(medStack>=maxCut)]=1

    if verbose:
        try:
            plotArray(mask, title='cps>%i == 1'%maxCut)
        except:
            plt.matshow(mask)
            plt.show()

    medStack[np.where(mask==1)]=np.nan
    medStack[np.where(medStack==0)]=np.nan
    if verbose:
        try:
            plotArray(medStack, title='Median Stack with mask 1')
        except:
            plt.matshow(medStack)
            plt.show()

    mask2 = np.zeros(np.shape(medStack),dtype=int)
    mask2[np.where(medStack>=sigma*np.nanmean(medStack))]=1

    if verbose:
        try:
            plotArray(mask2, title='cps>mean+%i-sigma == 1'%sigma)
        except:
            plt.matshow(mask2)
            plt.show()

    medStack[np.where(mask2==1)]=np.nan
    if verbose:
        try:
            plotArray(medStack, title='Median Stack with mask 2')
        except:
            plt.matshow(medStack)
            plt.show()

    finalMask = mask+mask2
    if verbose:
        try:
            plotArray(finalMask, title='Final mask')
        except:
            plt.matshow(finalMask)
            plt.show()
        print "Number masked pixels = ", len(np.array(np.where(finalMask==1)).flatten())

    return finalMask

def saveMask(mask, timeStamp=None, date=None):
    '''
     Write hot pixel mask to darkness calibration files directory
    '''
    scratchDir = os.getenv('MKID_PROC_PATH', '/')
    hpMaskPath = os.path.join(scratchDir, 'darkHotPixMasks')
    datePath = os.path.join(hpMaskPath,date)
    fileName = "%i.npz"%timeStamp
    fullMaskFileName = os.path.join(datePath, fileName)

    np.savez(fullMaskFileName, mask=mask)

    return

def loadMask(path):
    '''
    Load and return a mask
    '''
    npzfile = np.load(path)
    mask = npzfile['mask']
    npzfile.close()
    return mask

def plotMask(mask):
    try:
        plotArray(mask, title='hp Mask')
    except:
        plt.matshow(mask)
        plt.show()


if __name__ == "__main__":
    if len(sys.argv)<2:
        print "No arguments provided, running on test files"
        run="PAL2017a"
        date = "20170410"
        startTS = 1491870115
        stopTS = 1491870135
        verb=True

    elif len(sys.argv) != 6:
        print 'Usage: {} run date startTimeStamp stopTimeStamp verbose'.format(sys.argv[0])
        exit(0)

    else:
        run = str(sys.argv[1])
        date = str(sys.argv[2])
        startTS = str(sys.argv[3])
        stopTS = str(sys.argv[4])
        verb = sys.argv[5]

    mask = makeMask(run=run, date=date, startTimeStamp=startTS, stopTimeStamp=stopTS, verbose=True)
    saveMask(mask, timeStamp = startTS, date=date)


    
