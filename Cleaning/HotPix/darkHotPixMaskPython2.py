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
from DarknessPipeline.Utils.popup import plotArray
import DarknessPipeline.Utils
import warnings
from DarknessPipeline.ImageReg.loadStackPython2 import loadIMGStack
import image_registration as ir
import DarknessPipeline.ImageReg.irUtilsPython2 as irUtils


#ultimately want the path to be flexible: use .bin files if they exist,
#and fall back on .img files if necessary (typically we didn't record
#.bin files for dark data in 2017a)
#basePath = '/mnt/data0/ScienceDataIMGs/'
nCols=80
nRows=125
#maxCut=2450
def makeDHPMask(stack=None, outputFileName=None, verbose=False, sigma=3, maxCut=2450, coldCut=False, manualArray=None):
    '''
    MaxCut sets level for initial hot pixel cut. Everything with cps>maxCut -> np.nan
    Sigma determines level for second cut. Everything with cps>mean+sigma*stdev -> np.nan
    If coldCut=True, third cut where everything with cps<mean-sigma*stdev -> np.nan
    manualArray is a way to give a list of bad pixels manually, in format [[row,col],[row,col],...]


    '''
    medStack = P3Utils.utils.medianStack(stack)

    if verbose:
        try:
            plotArray(medStack,title='Median Dark Stack')
        except:
            plt.matshow(medStack)
            plt.show()

    #initial masking, take out anything with cps > maxCut
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

    #second round of masking, where cps > mean+sigma*std
    mask2 = np.zeros(np.shape(medStack),dtype=int)
    with warnings.catch_warnings():
        # nan values will give an unnecessary RuntimeWarning
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mask2[np.where(medStack>=np.nanmean(medStack)+sigma*np.nanstd(medStack))]=1

    #if coldCut is true, also mask cps < mean-sigma*std
    if coldCut==True:
        with warnings.catch_warnings():
            # nan values will give an unnecessary RuntimeWarning
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mask2[np.where(medStack<=np.nanmean(medStack)-sigma*np.nanstd(medStack))]=1

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

    # provide an easy means to pipe in an array of manually identified pixels
    if manualArray is not None:
        for pix in manualArray:
            finalMask[pix[0],pix[1]]=1


    if verbose:
        try:
            plotArray(finalMask, title='Final mask')
        except:
            plt.matshow(finalMask)
            plt.show()
        print("Number masked pixels = ", len(np.array(np.where(finalMask==1)).flatten()))

    return finalMask


def makeMask(run=None, date=None, basePath=None,startTimeStamp=None, stopTimeStamp=None, verbose=False, sigma=3, maxCut=2450, coldCut=False, manualArray=None):
    '''
    MaxCut sets level for initial hot pixel cut. Everything with cps>maxCut -> np.nan
    Sigma determines level for second cut. Everything with cps>mean+sigma*stdev -> np.nan
    If coldCut=True, third cut where everything with cps<mean-sigma*stdev -> np.nan
    manualArray is a way to give a list of bad pixels manually, in format [[row,col],[row,col],...]


    '''

    try:
        dataPath = basePath+str(run)+os.path.sep+str(date)+os.path.sep
        stack = loadIMGStack(dataPath, startTimeStamp, stopTimeStamp, nCols=nCols, nRows=nRows)
    except:
        print("Could not find dark data in ScienceData path, checking ramdisk")
        dataPath = '/mnt/ramdisk/'
        stack = loadIMGStack(dataPath, startTimeStamp, stopTimeStamp, nCols=nCols, nRows=nRows)

    medStack = irUtils.medianStack(stack) #####

    if verbose:
        try:
            plotArray(medStack,title='Median Dark Stack')
        except:
            plt.matshow(medStack)
            plt.show()

    #initial masking, take out anything with cps > maxCut
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

    #second round of masking, where cps > mean+sigma*std
    mask2 = np.zeros(np.shape(medStack),dtype=int)
    mask2[np.where(medStack>=np.nanmean(medStack)+sigma*np.nanstd(medStack))]=1

    #if coldCut is true, also mask cps < mean-sigma*std
    if coldCut==True:
        mask2[np.where(medStack<=np.nanmean(medStack)-sigma*np.nanstd(medStack))]=1

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

    # provide an easy means to pipe in an array of manually identified pixels
    if manualArray is not None:
        for pix in manualArray:
            finalMask[pix[0],pix[1]]=1


    if verbose:
        try:
            plotArray(finalMask, title='Final mask')
        except:
            plt.matshow(finalMask)
            plt.show()
        print("Number masked pixels = ", len(np.array(np.where(finalMask==1)).flatten()))

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
        print("No arguments provided, running on test files")
        run="PAL2017a"
        date = "20170410"
        startTS = 1491894755
        stopTS = 1491894805
        verb=True
        coldCut=True
        #manual hot pix selection for 20170409 end of night (pi Her and HD91782)
        #manHP = [[0,3],[2,27],[8,63],[5,76],[10,78],[27,15],[27,20],[22,78],[37,52],[48,4],[54,20],[54,34],[63,33],[67,39],[63,61],[58,70],[62,74],[66,71],[75,24],[76,32],[78,62],[102,24],[110,75],[116,18],[114,28],[115,59],[116,72],[122,73],[88,73],[109,4],[75,21],[64,5],[16,63],[4,23],[42,53],[42,41],[16,51],[76,34],[117,60],[51,57]]

        #manual hot pix for 20170408 tauBoo data
        #manHP = [[0,1],[9,24],[39,11],[37,15],[35,39],[38,57],[42,56],[50,15],[65,18],[71,11],[75,13],[73,72],[82,42],[83,47],[80,75],[82,72],[89,36],[110,12],[103,21],[103,31],[106,30],[107,37],[117,36],[121,47],[117,60],[113,60],[78,62],[73,19],[88,38],[87,38],[93,75]]

        #manual hot pix for 20170410 HD91782 data
        manHP = None


    elif len(sys.argv) != 7:
        print('Usage: {} run date startTimeStamp stopTimeStamp verbose'.format(sys.argv[0]))
        exit(0)

    else:
        run = str(sys.argv[1])
        date = str(sys.argv[2])
        startTS = str(sys.argv[3])
        stopTS = str(sys.argv[4])
        verb = sys.argv[5]
        coldCut = sys.argv[6]
        manHP=None

    mask = makeMask(run=run, date=date, startTimeStamp=startTS, stopTimeStamp=stopTS, verbose=verb,
                    coldCut=coldCut,manualArray=manHP)
    saveMask(mask, timeStamp = startTS, date=date)
