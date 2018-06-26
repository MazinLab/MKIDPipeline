'''
Author: Isabel Lipartito        Date: Sept 30 2017

Quick routine to load in dark and flat img files.  For a given set of dark and flat timestamps, this code will:
-Load in dark and flat img files
-Make and save a hot pixel mask to a .npz
-Make a master dark calibration file and save to a .npz
-Dark subtract the flat and make a master flat calibration weight file and save both to a .npz

Adapted from SR Meeker's quickStack and makeSimpleFlatNPZ code

load params from a cfg file

Here is a sample:

#your file can have comments as long as it includes these parameters (they can be out of order, just make sure you name them correctly):
darkSpan = [1506549665, 1506549750]
flatSpan = [1506549910, 1506549980]
useImg = True
target = 'AOLabFlat'
numRows = 125
numCols = 80
date = 20170927
run = 'PAL2017b'
binDir = '/mnt/data0/ScienceData/'
imgDir = '/mnt/data0/ScienceDataIMGs/'
outputDir = '/mnt/data0/CalibrationFiles/imageStacks/'
beammapFile = '/mnt/data0/Darkness/20170924/Beammap/finalMap_20170924.txt'

#dark/flatSpan:  timespan range of dark/flat files
#useImg: boolean true/false, specify if you are using img or bin files
#target:  give your target a name
#numRows/numCols: should always be 125/80 for DARKNESS
#date: YYYYMMDD, the date your data was taken
#run: PALYYYa/b or LabData
#binDir/imgDir:  location of your bin/img files.  Code will append the run+date to your bin/img paths
#outputDir:  Where you want your output data to go.  Code will append the run+date to your output path.
#beammapFile: Directory and filename of the beammap you want to use.  This will be used to exclude the unbeammaped pixels from the flat median calculation.
'''

import glob
import sys, os, time, struct
import numpy as np
import tables
from scipy import ndimage
from scipy import signal
import astropy
import cPickle as pickle
from PyQt4 import QtCore
from PyQt4 import QtGui
import matplotlib
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from functools import partial
from parsePacketDump2 import parsePacketData
from arrayPopup import plotArray
from readDict import readDict

from img2fitsExample import writeFits
from readFITStest import readFITS
import HotPix.darkHotPixMask as dhpm

import image_registration as ir
from loadStack import loadIMGStack, loadBINStack
import irUtils


#def makeHPM(run=None, date=None, outputDir=None,darkSpan=None, numRows=None,numCols=None):
def makeHPM(configFileName):
    params=main(configFileName)
    darkSpan=params[0]
    flatSpan=params[1]
    darkStack=params[2]
    flatStack=params[3]
    numRows=params[4]
    numCols=params[5]
    target=params[6]
    run=params[7]
    date=params[8]
    outputDir=params[9]
    beammapFile=params[10]
    dataDir=params[11]
    
    if darkSpan[0]!='0':
        print "Making dark hot pixel mask"
        darkHPM = dhpm.makeMask(run=run, date=date, basePath=outputDir,startTimeStamp=darkSpan[0], stopTimeStamp=darkSpan[1], coldCut=False, manualArray=None)
        hpFN='darkHPM_'+target+'.npz'  #Save the hot pixel mask into the output directory
        runOutDir = os.path.join(outputDir,run)
        outPath = os.path.join(runOutDir,date)
        hpPath = os.path.join(outPath,hpFN)
        np.savez(hpPath,darkHPM = darkHPM)
    else:
       print "No dark hot pixel mask saved"
       darkHPM = np.zeros((numRows, numCols),dtype=int)
    return darkHPM
   

def makeDark(stackForDark=None, outPutFileName=None,verbose=True)
    """
    takes a list or array of arrays and makes a dark. Return an array with the dark. If outPutFileName is provided, it saves the dark in an npz
    """
    try: 
        dark = irUtils.medianStack(stackForDark)
        print "Making dark hot pixel mask"
        if verbose:
            plotArray(dark,title='Dark',origin='upper')
    except:
        print "####    Error with the stack of dark. makeDark takes a list of array of arrays to make the dark   ####"
        sys.exit(0)
    try:
        np.savez(darkPath,dark=dark)
    except: 
        if outPutFileName==None:
            print "No output file name provided. Dark not saved"
        else:
            print "Wrong output path. Dark not saved"
    return dark

def maskBadPixels(flatStack, beamMapMask=None, darkHotPixelMask=None)



def makeFlat(flatStack=None, dark=None, outPutFileName=None, crop=None, cropColsForMedian=([0,19],[60,79]), cropRowsForMedian=([25:49]), verbose=True):
        """
        flatStack: array or list of arrays to make the flat
        dark: dark. If none the flat is not dark subtracteed
        outPutFileName: if not provided returns the flat array without saving it
        crop: if None uses the entire array to calculate the median, otherwise it crops the columns in the range specified by cropForMedian
        cropColsForMedian, cropRowsForMedian: are tuples with each element being the range of columns/rows to be cropped when taking the median for the flat
        10/1/2017: currently crops the high frequency boards (bad performances) and FL2 (not working on Faceless)

        """
        print "Loading flat frame"
        flat = irUtils.medianStack(flatStack)

        if verbose:
            plotArray(flat,title='NotdarkSubFlat' 
        
        if dark!=None:
            #dark subtract the flat
            print "Subtracting dark"
            flatSub=flat-dark
        else:
            print "Dark=None, will make a flat without dark subtraction"
        
        flatSub[np.where(flatSub < 0.0)]=np.nan
      
        #Apply SR Meeker's feedline cropping script when we calculate the flat weights
        ########
        # This cropping assumes poor FL2 performance and
        # poor high frequency performance, a la Faceless from 2017a,b.
        # This is a good place to play around to change crappy weights
        # at fringes of the array.
        #######
        
        croppedFrame = np.copy(flatSub)
        if crop: 
            if cropColsForMedian:
                for iCrop, cropInd in enumerate(cropColsForMedian):
                    croppedFrame[:,cropInd[0]:cropInd[1]+1]=np.nan
            if cropRowsForMedian:
                for iCrop, cropInd in enumerate(cropRowsForMedian):
                    croppedFrame[cropInd[0]:cropInd[1]+1,:]=np.nan
            

        #Apply Neelay's beammap script which loads in the most recent beammap and removes out of bound pixels from the median calculation
        resID, flag, xCoords, yCoords = np.loadtxt(beammapFile, unpack=True)
        badInds = np.where(flag==1)[0]
        badXCoords = np.array(xCoords[badInds], dtype=np.int32)
        badYCoords = np.array(yCoords[badInds], dtype=np.int32)
        outOfBoundsMask = np.logical_or(badXCoords>=80, badYCoords>=125)
        outOfBoundsInds = np.where(outOfBoundsMask)[0]
        badXCoords = np.delete(badXCoords, outOfBoundsInds)
        badYCoords = np.delete(badYCoords, outOfBoundsInds)
        croppedFrame[badYCoords, badXCoords] = np.nan
        
        if verbose:
            plotArray(croppedFrame,title='Cropped flat frame') #Plot the cropped flat frame

        med = np.nanmedian(croppedFrame.flatten())
        print med
        weights = med/flatSub #calculate the weights by dividing the median by the dark-subtracted flat frame

        plotArray(weights,title='Weights') #Plot the weights

        flatFN='flat_'+target+'.npz'  #Save the dark-subtracted flat file and weights into the output directory
        flatPath =  os.path.join(outPath,flatFN)
        np.savez(flatPath,weights = weights,flat=flatSub)
    else:
        print "No flat provided, no flat file saved"
        flat = np.ones((numRows, numCols),dtype=int)
        weights = np.ones((numRows, numCols),dtype=int)
    return(flat,weights)

def readConfig(configFileName=configFileName):
    if len(configFileName)<2:
         #grab most recent .cfg file if one has not been provided
         print "No .cfg file provided, trying to grab most recent one from Params..."
         try:
              configFileName = max(glob.iglob('./Params/*.cfg'), key=os.path.getctime)
         except:
              print "Failed to load appropriate .cfg file. Please provide path as argument"
              sys.exit(0)
    else:
         configFileName = configFileName
    configData = readDict()
    configData.read_from_file(configFileName)

    # Extract parameters from config file
    darkSpan = np.array(configData['darkSpan'], dtype=int)
    flatSpan = np.array(configData['flatSpan'], dtype=int)
    numRows = int(configData['numRows'])
    numCols = int(configData['numCols'])
    target = str(configData['target'])
    run = str(configData['run'])
    date = str(configData['date'])
    imgDir = str(configData['imgDir'])
    binDir = str(configData['binDir'])
    outputDir = str(configData['outputDir'])
    useImg = bool(configData['useImg'])
    beammapFile=str(configData['beammapFile'])

    runOutDir = os.path.join(outputDir,run)
    outPath = os.path.join(runOutDir,date)

    if imgDir!='/mnt/ramdisk/':
        runDir = os.path.join(imgDir,run)
        imgPath = os.path.join(runDir,date)
    else:
        imgPath=imgDir

    binRunDir = os.path.join(binDir,run)
    binPath = os.path.join(binRunDir,date)

    if useImg == True:
        dataDir = imgPath
        print "Loading data from .img files"
    else:
        dataDir = binPath
        print "Loading data from .bin files"

    print "This is the span of Dark Frames"
    print darkSpan
    print "This is the span of Flat Frames"
    print flatSpan

    if darkSpan[0]!='0':
        print "Loading dark frame"
        if useImg == True:
           darkStack = loadIMGStack(dataDir, darkSpan[0], darkSpan[1], nCols=numCols, nRows=numRows)
        else:
           darkStack = loadBINStack(dataDir, darkSpan[0], darkSpan[1], nCols=numCols, nRows=numRows)
    if flatSpan[0]!='0':
        print "Loading flat frame"
        if useImg == True:
            flatStack = loadIMGStack(dataDir, flatSpan[0], flatSpan[1], nCols=numCols, nRows=numRows)
        else:
            flatStack = loadBINStack(dataDir, flatSpan[0], flatSpan[1], nCols=numCols, nRows=numRows)

return[(darkSpan,flatSpan,darkStack,flatStack,numRows,numCols,target,run,date,outputDir,beammapFile,dataDir)]




if __name__ == "__main__":
    readConfig(configFileName)
    makeHPM(configFileName)
    dark=makeDark(configFileName)
    
    makeFlat(dark=dark)
    
