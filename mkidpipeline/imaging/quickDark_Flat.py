"""
Quick routine to load in dark and flat img files.  For a given set of dark and flat timestamps, this code will:
-Load in dark and flat img files
-Make and save a hot pixel mask to a .npz
-Make a master dark calibration file and save to a .npz
-Dark subtract the flat and make a master flat calibration weight file and save both to a .npz

Adapted from SR Meeker's quickStack and makeSimpleFlatNPZ

load params from a cfg file. The functions makeFlat and makeDark can be used without a cfg file

Here is a sample:

#Comments using #
#The cfg need the following parameters (order deos not matter):
darkSpan = [1506549665, 1506549750]
flatSpan = [1506549910, 1506549980]
useImg = True 
target = 'AOLabFlat' #Optional
numRows = 125
numCols = 80
date = 20170927
run = 'PAL2017b'
binDir = '/mnt/data0/ScienceData/' #not used
imgDir = '/mnt/data0/ScienceDataIMGs/' #not used
outputDir = '/mnt/data0/CalibrationFiles/imageStacks/' 
beammapFile = '/mnt/data0/Darkness/20170924/Beammap/finalMap_20170924.txt'
"""

import glob
import os
import sys

import numpy as np

import mkidpipeline.hotpix.generatebadpixmask as gbpm
import mkidpipeline.utils.irUtils as irUtils
from mkidpipeline.utils.plottingTools import plot_array
from mkidpipeline.utils.loadStack import loadBINStack, loadIMGStack
from mkidcore.utils.readdict import ReadDict


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
        print("Making dark hot pixel mask")
        dark_stack = loadIMGStack(dataDir=dataDir, start=darkSpan[0], stop=darkSpan[1])
        len_dark_stack=int(darkSpan[1]-darkSpan[0])
        darkHPM = gbpm.quick_check_stack(stack=dark_stack, len_stack=len_dark_stack)
        hpFN='darkHPM_'+target+'.npz'  #Save the hot pixel mask into the output directory
        runOutDir = os.path.join(outputDir,run)
        outPath = os.path.join(runOutDir,date)
        hpPath = os.path.join(outPath,hpFN)
        np.savez(hpPath,darkHPM = darkHPM)
    else:
       print("No dark hot pixel mask saved")
       darkHPM = np.zeros((numRows, numCols),dtype=int)
    return darkHPM
   

def makeDark(stackForDark=None, outPutFileName=None,verbose=True):
    """
    takes a list or array of arrays and makes a dark. Return an array with the dark. If outPutFileName is provided, it saves the dark in an npz
    """
    try: 
        dark = irUtils.medianStack(stackForDark)
        print("Making dark hot pixel mask")
        if verbose:
            plot_array(dark,title='Dark',origin='upper')
    except:
        print("####    Error with the stack of dark. makeDark takes a list of array of arrays to make the dark   ####")
        sys.exit(0)
    try:
        np.savez(darkPath,dark=dark)
    except: 
        if outPutFileName==None:
            print("No output file name provided. Dark not saved")
        else:
            print("Wrong output path. Dark not saved")
    return dark

def makeBadPixelMask(beamMapFile=None, darkHotPixelMask=None):
        resID, flag, xCoords, yCoords = np.loadtxt(beammapFile, unpack=True)
        badInds = np.where(flag==1)[0]
        badXCoords = np.array(xCoords[badInds], dtype=np.int32)
        badYCoords = np.array(yCoords[badInds], dtype=np.int32)
        outOfBoundsMask = np.logical_or(badXCoords>=80, badYCoords>=125)
        outOfBoundsInds = np.where(outOfBoundsMask)[0]
        badXCoords = np.delete(badXCoords, outOfBoundsInds)
        badYCoords = np.delete(badYCoords, outOfBoundsInds)



def makeFlat(flatStack=None, dark=None, outPutFileName=None, badPixelMask=None, crop=False, cropColsForMedian=[(0,19),(60,79)], cropRowsForMedian=[(25,49)], verbose=True):
    """
    Input:
    flatStack: array or list of arrays to make the flat
    dark: dark. If none the flat is not dark subtracteed
    outPutFileName: if not provided returns the flat array without saving it
    crop: if False uses the entire array to calculate the median, otherwise it crops the columns in the range specified by cropForMedian
    cropColsForMedian, cropRowsForMedian: are tuples with each element being the range of columns/rows to be cropped when taking the median for the flat
    10/1/2017: currently crops the high frequency boards (bad performances) and FL2 (not working on Faceless)
    Returns:
    dictionary with 'weights' and 'flat'
    """
    
    try:
        flat = irUtils.medianStack(flatStack)
        print("Loading flat frame")
    except:
        print("No valid flat stack provided. Exiting")
        sys.exit(0)
    
    if verbose:
        plot_array(flat,title='NotdarkSubFlat')
    
    if darkSpan[0]!='0':
        #dark subtract the flat
        print("Subtracting dark")
        flatSub=flat-dark
    else:
        print("Dark=None, will make a flat without dark subtraction")
        flatSub=flat
    
    flatSub[np.where(flatSub < 0.0)]=np.nan
    croppedFrame = np.copy(flatSub)
    if crop: 
        if cropColsForMedian!=None:
            cropColsForMedian=np.array(cropColsForMedian)
            for iCrop, cropInd in enumerate(cropColsForMedian):
                croppedFrame[:,cropInd[0]:cropInd[1]+1]=np.nan
        if cropRowsForMedian!=None:
            cropRowsForMedian=np.array(cropRowsForMedian)
            for iCrop, cropInd in enumerate(cropRowsForMedian):
                croppedFrame[cropInd[0]:cropInd[1]+1,:]=np.nan
    #Apply bad pixel mask 
    if badPixelMask!=None:
         print("Applying bad pixel mask")
         croppedFrame[badPixelMask!=0] = np.nan
    if verbose:
        plot_array(croppedFrame,title='Cropped flat frame') #Plot the cropped flat frame
    med = np.nanmedian(croppedFrame.flatten())
    print('median', med)
    
    flatSub[flatSub==0]=1
    weights = med/flatSub #calculate the weights by dividing the median by the dark-subtracted flat frame
    if verbose: 
        plot_array(weights,title='Weights') #Plot the weights
   
    try: 
        np.savez(outPutFileName,weights = weights,flat=flatSub)
    except:
        if outPutFileName==None:
            print('No output file name provided. Not saving flat')
        else:
            print('Output file name not valid. Not saving flat')
    dict={}
    dict['flat']=flat
    dict['weights']=weights
    return dict

def readConfig(configFileName=None):
    return[(darkSpan,flatSpan,darkStack,flatStack,numRows,numCols,target,run,date,outputDir,beammapFile,dataDir)]




if __name__ == "__main__":
    if len(sys.argv)<2:
         #grab most recent .cfg file if one has not been provided
         print("No .cfg file provided, trying to grab most recent one from Params...")
         try:
              configFileName = max(glob.iglob('./Params/*.cfg'), key=os.path.getctime)
              print("Using cfg file", configFileName)
         except:
              print("Failed to load appropriate .cfg file. Please provide path as argument")
              sys.exit(0)
    else:
         configFileName = sys.argv[1]
    configData = ReadDict(file=configFileName)

    useImg=True
     
    ##use this to make flats or darks with files that are not in the standard ScienceDir or /mnt/ramdisk
    imgPath=None


    imgDir='/mnt/ramdisk/'
    binDir='/mnt/data0/ScienceData/'

    # Extract parameters from config file
    darkSpan = np.array(configData['darkSpan'], dtype=int)
    flatSpan = np.array(configData['flatSpan'], dtype=int)
    numRows = int(configData['numRows'])
    numCols = int(configData['numCols'])
    run = str(configData['run'])
    date = str(configData['date'])
    
    try:
        target = str(configData['target']) 
    except:
        target=None
    
    #imgDir = str(configData['imgDir'])
    #binDir = str(configData['binDir'])
    
    outputDir = str(configData['outputDir'])
    useImg = bool(configData['useImg'])
    beammapFile=str(configData['beammapFile'])

    runOutDir = os.path.join(outputDir,run)
    outPath = os.path.join(runOutDir,date)
    
    if imgPath==None:    
        if useImg:
            runDir=os.path.join(imgDir,run)
            #imgPath=os.path.join(runDir,date)
            imgPath=imgDir
        else:
            runDir=os.path.join(binDir,run)
            binPath=os.path.join(runDir,date)

    if useImg == True:
        dataDir = imgPath
        print("Loading data from .img files")
    else:
        dataDir = binPath
        print("Loading data from .bin files")

    print("This is the span of Dark Frames")
    print(darkSpan)
    print("This is the span of Flat Frames")
    print(flatSpan)

    if darkSpan[0]!='0':
        print("Loading dark frame")
        if useImg == True:
           darkStack = loadIMGStack(dataDir, darkSpan[0], darkSpan[1], nCols=numCols, nRows=numRows)
        else:
           darkStack = loadBINStack(dataDir, darkSpan[0], darkSpan[1], nCols=numCols, nRows=numRows)
    if flatSpan[0]!='0':
        print("Loading flat frame")
        if useImg == True:
            flatStack = loadIMGStack(dataDir, flatSpan[0], flatSpan[1], nCols=numCols, nRows=numRows)
        else:
            flatStack = loadBINStack(dataDir, flatSpan[0], flatSpan[1], nCols=numCols, nRows=numRows)

    #makeHPM(configFileName)
    
    if darkSpan[0]!='0':
        dark=makeDark(stackForDark=darkStack, outPutFileName=None,verbose=True)
    else:
        dark=None
    if flatSpan[0]!='0':
        makeFlat(flatStack=flatStack, dark=dark, outPutFileName=None, badPixelMask=None, crop=True, cropColsForMedian=[(0,19),(60,79)], cropRowsForMedian=[(100,125)], verbose=True)
