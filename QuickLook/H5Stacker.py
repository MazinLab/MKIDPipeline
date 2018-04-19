#!/bin/python
"""
Author:  Isabel Lipartito    Date 4/18/2018
Use if you've fot a bunch of H5s that are wavelength calibrated (and part of a single Dither stack)
Program reads in those H5s and use getPixelCountImage to make an image for each
Uses the same stacking and rotation code in quickStack to align them (irUtils)
Makes and Plots final Image

Will work for non-wavelength calibrated images, but you must set the wvlStart and wvlStop to 0
"""

import glob
import sys, os, time, struct
import numpy as np
import tables
from scipy import ndimage
from scipy import signal
import astropy
import _pickle as pickle
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5 import QtCore
import matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from functools import partial
from parsePacketDump2 import parsePacketData
from arrayPopup import plotArray
from readDict import readDict

from img2fitsExample import writeFits
from readFITStest import readFITS

from Headers.CalHeaders import FlatCalSoln_Description
from Headers import pipelineFlags
from RawDataProcessing.darkObsFile import ObsFile as obs
import HotPix.darkHotPixMask as dhpm

import image_registration as ir
from loadStack import loadIMGStack, loadBINStack
import irUtils


if len(sys.argv)<2:
    #grab most recent .cfg file
    print("No .cfg file provided, trying to grab most recent one from Params...")
    try:
        configFileName = max(glob.iglob('./Params/*.cfg'), key=os.path.getctime)
    except:
        print("Failed to load appropriate .cfg file. Please provide path as argument")
        sys.exit(0)
else:
    configFileName = sys.argv[1]

print("Using config", configFileName)


configData = readDict()
configData.read_from_file(configFileName)

# Extract parameters from config file
nPos = int(configData['nPos'])
startTimes = np.array(configData['startTimes'], dtype=int)
stopTimes = np.array(configData['stopTimes'], dtype=int)
xPos = np.array(configData['xPos'], dtype=int)
yPos = np.array(configData['yPos'], dtype=int)
numRows = int(configData['numRows'])
numCols = int(configData['numCols'])
upSample = int(configData['upSample'])
padFraction = float(configData['padFraction'])
coldCut = int(configData['coldCut'])
target = str(configData['target'])
run = str(configData['run'])
date = str(configData['date'])
wvlStart=int(configData['wvlStart'])
wvlStop=int(configData['wvlStop'])
h5dir = str(configData['h5dir'])
outputDir = str(configData['outputDir'])
outfileName=str(configData['outfileName'])

ObsFNList =glob.glob(h5dir+'*.h5')  

rawImgs=[]
roughShiftsX=[]
roughShiftsY=[]
centroidsX=[]
centroidsY=[]

for ObsFN in ObsFNList:
    obsfile=obs(ObsFN, mode='write')

#starting point centroid guess for first frame is where all subsequent frames will be aligned to
refPointX = xPos[0]
refPointY = yPos[0]
print('xPos',xPos[0],'yPos',yPos[0])

#determine the coarse x and y shifts that subsequent frames must be moved to align with first frame
dXs = refPointX-xPos
dYs = refPointY-yPos


#initialize hpDict so we can just check if it exists and only make it once
hpDict=None

#to ensure stacking is weighting all dither positions evenly, we need to know the one with the shortest integration
intTime = min(stopTimes-startTimes)
firstSec=0
print("Shortest Integration time = ", intTime)

#load dithered science frames
ditherFrames = []
for i in range(nPos):
        #load stack for entire data set to be saved to H5, even though only intTime number of frames will
        #be used from each position
        obsfile=obs(ObsFNList[i], mode='write')
        if wvlStart != 0 and wvlStop!=0:
           img = obsfile.getPixelCountImage(firstSec =0, integrationTime=intTime,applyWeight=False,flagToUse = 0,wvlRange = (wvlStart,wvlStop))
           print('Running getPixelCountImage on ',firstSec,'seconds to ',intTime,'seconds of data from wavelength ',wvlStart,'to ',wvlStop)
        else:
           img = obsfile.getPixelCountImage(firstSec =0, integrationTime=intTime,applyWeight=False,flagToUse = 0,wvlRange = None)
           print('Running getPixelCountImage on ',firstSec,'seconds to ',intTime,'seconds of data on all wavelengths')
        processedIm = np.transpose(img['image'])/intTime

        roughShiftsX.append(dXs[i])
        roughShiftsY.append(dYs[i])
        centroidsX.append(refPointX-dXs[i])
        centroidsY.append(refPointY-dYs[i])

       #plot an example of the UNmasked image for inspection
        if i==0:
            plotArray(processedIm,title='Dither Pos %i'%i,origin='upper',vmin=0)
        #cut out cold/dead pixels
        processedIm[np.where(processedIm<=coldCut)]=np.nan

        #plot an example of the masked image for inspection
        if i==0:
            plotArray(processedIm,title='Dither Pos %i ColdPix Masked'%i,origin='upper',vmin=0)

        #pad frame with margin for shifting and stacking
        paddedFrame = irUtils.embedInLargerArray(processedIm,frameSize=padFraction)

        #apply rough dX and dY shift to frame
        print("Shifting dither %i, frame %i by x=%i, y=%i"%(i,0,dXs[i], dYs[i]))
        shiftedFrame = irUtils.rotateShiftImage(paddedFrame,0,dXs[i],dYs[i])

        #upSample frame for sub-pixel registration with fitting code
        upSampledFrame = irUtils.upSampleIm(shiftedFrame,upSample)
        #conserve flux. Break each pixel into upSample^2 sub pixels, 
        #each subpixel should have 1/(upSample^2) the flux of the original pixel
        upSampledFrame/=float(upSample*upSample) 

        #append upsampled, padded frame to array for storage into next part of code
        ditherFrames.append(upSampledFrame)

        print("Loaded dither position %i"%i)

shiftedFrames = np.array(ditherFrames)

#take median stack of all shifted frames
finalImage = irUtils.medianStack(shiftedFrames)

plotArray(finalImage,title='final',origin='upper')
outfile=h5dir+outfileName
np.save(outfile, finalImage)
