"""
Author: Isabel Lipartito        Date: Sept 27 2017

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
"""

import glob
import os
import sys

import numpy as np

import mkidpipeline.badpix as gbpm
import mkidpipeline.utils.irUtils as irUtils
from mkidpipeline.utils.plottingTools import plot_array
from mkidpipeline.utils.loadStack import loadBINStack, loadIMGStack
from mkidpipeline.utils.readDict import readDict

if len(sys.argv)<2:
    #grab most recent .cfg file if one has not been provided
    print("No .cfg file provided, trying to grab most recent one from Params...")
    try:
        configFileName = max(glob.iglob('./Params/*.cfg'), key=os.path.getctime)
    except:
        print("Failed to load appropriate .cfg file. Please provide path as argument")
        sys.exit(0)
else:
    configFileName = sys.argv[1]
    
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
    print("Loading data from .img files")
else:
    dataDir = binPath
    print("Loading data from .bin files")

print("This is the span of Dark Frames")
print(darkSpan)
print("This is the span of Flat Frames")
print(flatSpan)

#load dark frames
if darkSpan[0]!='0':
    print("Loading dark frame")
    if useImg == True:
        darkStack = loadIMGStack(dataDir, darkSpan[0], darkSpan[1], nCols=numCols, nRows=numRows)
    else:
        darkStack = loadBINStack(dataDir, darkSpan[0], darkSpan[1], nCols=numCols, nRows=numRows)
    dark = irUtils.medianStack(darkStack)

else:
    print("No dark provided")
    dark = np.zeros((numRows, numCols),dtype=int)

#if dark frames are provided, generate a hot pixel mask using badpix.py
if darkSpan[0]!='0':
    darkHPM = gbpm.quick_check_img(image=dark)['bad_mask']
else:
    print("Failed to generate dark mask. Turning off hot pixel masking")

#Apply the hot pixel mask and plot the hot pixel masked dark frame
dark[np.where(darkHPM==1)]=np.nan
plot_array(dark,title='Dark',origin='upper')

#load flat frames
if flatSpan[0]!='0':
    print("Loading flat frame")
    if useImg == True:
        flatStack = loadIMGStack(dataDir, flatSpan[0], flatSpan[1], nCols=numCols, nRows=numRows)
    else:
        flatStack = loadBINStack(dataDir, flatSpan[0], flatSpan[1], nCols=numCols, nRows=numRows)
    flat = irUtils.medianStack(flatStack)
    plot_array(flat,title='NotdarkSubFlat')
    #dark subtract the flat
    flatSub=flat-dark
    flatSub[np.where(flatSub < 0.0)]=np.nan
else:
    print("No flat provided")
    flat = np.ones((numRows, numCols),dtype=int)

#Apply SR Meeker's feedline cropping script when we calculate the flat weights
########
# This cropping assumes poor FL2 performance and
# poor high frequency performance, a la Faceless from 2017a,b.
# This is a good place to play around to change crappy weights
# at fringes of the array.
#######
croppedFrame = np.copy(flatSub)

#Could add a flag for feedline cropping.  Right now we'll make it a given
#if cropFLs:
croppedFrame[25:50,::] = np.nan
#if cropHF:
croppedFrame[::,:20] = np.nan
croppedFrame[::,59:] = np.nan

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

plot_array(croppedFrame,title='Cropped flat frame') #Plot the cropped flat frame

med = np.nanmedian(croppedFrame.flatten())
print(med)
weights = med/flatSub #calculate the weights by dividing the median by the dark-subtracted flat frame

plot_array(weights,title='Weights') #Plot the weights

hpFN='darkHPM_'+target+'.npz'  #Save the hot pixel mask into the output directory
hpPath = os.path.join(outPath,hpFN)
np.savez(hpPath,darkHPM = darkHPM)

flatFN='flat_'+target+'.npz'  #Save the dark-subtracted flat file and weights into the output directory
flatPath =  os.path.join(outPath,flatFN)
np.savez(flatPath,weights = weights,flat=flatSub)

darkFN='dark_'+target+'.npz'  #Save the dark into the output directory
darkPath = os.path.join(outPath,darkFN)
np.savez(darkPath,dark = dark)


