#!/bin/python
"""
Author:  Isabel Lipartito    Date 4/18/2018
Use if you've got a bunch of H5s that are wavelength calibrated (and part of a single Dither stack)
Program reads in those H5s and use getPixelCountImage to make an image for each
Uses the same stacking and rotation code in quickStack to align them (irUtils)
Makes and Plots final Image

Will work for non-wavelength calibrated images, but you must set the wvlStart and wvlStop to None
"""

"""
Config Example:

[Data]
#Sample Configuration File to run Dither2HDF.py

#XPIX (integer)
XPIX = 140

#YPIX (integer)
YPIX = 146

#Path to the Bin Files (string)
binPath = '/mnt/data0/ScienceData/Subaru/20180829'

#Path to write the output h5 files (string)
outPath = '/mnt/data0/isabel/highcontrastimaging/GuyonMECProposal/'

#Path and filename of the beammap file (string)
beamFile = '/mnt/data0/isabel/pipelinetest/mecwavecalanalyses/bin2hdfdebug/finalMapV2_1.txt'

#mapFlag (int)
mapFlag = 1

#Flag to use img files or pre-made numpy arrays instead of h5 files (bool)
useImg = False

#Path to the Bin2HDF code (string)
b2hPath = '/home/isabel/src/mkidpipeline/mkidpipeline/hdf'

#List of start times of each dither position (list of ints OR floats)
startTimes = [1535642304.761, 1535642336.032, 1535642367.254, 1535642398.524, 1535642430.626, 1535642462.424, 1535642493.998, 1535642524.964, 1535642556.059, 1535642588.208, 1535642619.879, 1535642651.293, 1535642682.212, 1535642713.994, 1535642745.233, 1535642776.998, 1535642808.493, 1535642839.604, 1535642870.698, 1535642901.952, 1535642934.198, 1535642965.213, 1535642996.515, 1535643027.706, 1535643058.992]

#List of stop times of each dither position (list of ints OR floats)
stopTimes = [1535642334.793, 1535642366.065, 1535642397.287, 1535642428.557, 1535642460.658, 1535642492.457, 1535642524.03, 1535642554.998, 1535642586.092, 1535642618.24, 1535642649.912, 1535642681.325, 1535642712.245, 1535642744.027, 1535642775.265, 1535642807.032, 1535642838.525, 1535642869.636, 1535642900.73, 1535642931.984, 1535642964.231, 1535642995.245, 1535643026.546, 1535643057.738, 1535643089.025]

#x location of each dither position (list of ints OR floats)
#xPos = [-0.76, -0.76, -0.76, -0.76, -0.76, -0.57, -0.57, -0.57, -0.57, -0.57, -0.38, -0.38, -0.38, -0.38, -0.38, -0.19, -0.19, -0.19, -0.19, -0.19, 0.0, 0.0, 0.0, 0.0, 0.0]
#          0       1     2      3      4      5      6       7     8      9      10     11      12     13    14     15     16     17     18      19   20   21   22   23   24

xPos=[    108,   108,   108,   108,   108,   95,    95,    95,     95,    95,    83,    83,    83,    83,    83,    70,   70,    70,     70,    70,  58,  58,   58,   58,  58]
#y location of each dither position (list of ints OR floats)
#yPos = [-0.76, -0.38, 0.0, 0.38, 0.76, -0.76, -0.38, 0.0, 0.38, 0.76, -0.76, -0.38, 0.0, 0.38, 0.76, -0.76, -0.38, 0.0, 0.38, 0.76, -0.76, -0.38, 0.0, 0.38, 0.76]

yPos  = [20,    45,    70,   95,  120,   20,    45,    70,   95,  120,   20,    45,    70,   95,  120, 20,    45,    70,   95,  120,  20,    45,    70,   95,  120]

#Integration time of each dither position (int)
intTime = 30

#Total number of dither positions (int)
nPos = 25
"""


#TODO TFD. Merge heart with image formation step of pipeline. Automate centroid.

import glob
import ast
import argparse
import sys


import numpy as np
from mkidpipeline.hdf.photontable import ObsFile as obs
from mkidpipeline.utils.plottingTools import plot_array
from configparser import ConfigParser
from mkidpipeline.utils import irUtils

parser = argparse.ArgumentParser(description='Process a dither stack into a final rough-stacked image.')
parser.add_argument('ditherName', metavar='file', nargs=1,
                    help='filename of the dither config file')

args = parser.parse_args()

config_directory = sys.argv[1]
config = ConfigParser()
config.read(config_directory)
XPIX = ast.literal_eval(config['Data']['XPIX'])
YPIX = ast.literal_eval(config['Data']['YPIX'])
binPath = ast.literal_eval(config['Data']['binPath'])  # path to raw .bin data
outPath = ast.literal_eval(config['Data']['outPath'])   # path to output data
beamFile = ast.literal_eval(config['Data']['beamFile'])  # path and filename to beam map file
mapFlag = ast.literal_eval(config['Data']['mapFlag'])
filePrefix = ast.literal_eval(config['Data']['filePrefix'])
b2hPath = ast.literal_eval(config['Data']['b2hPath'])
nPos = ast.literal_eval(config['Data']['nPos'])
intTime =  ast.literal_eval(config['Data']['intTime'])
startTimes = ast.literal_eval(config['Data']['startTimes'])
stopTimes =  ast.literal_eval(config['Data']['stopTimes'])
xPos = ast.literal_eval(config['Data']['xPos'])
yPos = ast.literal_eval(config['Data']['yPos'])
useImg = ast.literal_eval(config['Data']['useImg'])


upSample = 2
padFraction = 0.4
wvlStart=0
wvlStop=0

h5dir = outPath
print(h5dir)
outputDir = outPath
outfileName='HR8799StackWaveCal'

ObsFNList =glob.glob(h5dir+'15*.h5')
print(ObsFNList)

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
dXs = (np.zeros(len(xPos))+refPointX)-xPos
dYs = (np.zeros(len(yPos))+refPointY)-yPos


#initialize hpDict so we can just check if it exists and only make it once
hpDict=None

#to ensure stacking is weighting all dither positions evenly, we need to know the one with the shortest integration
intTime = min(np.array(stopTimes)-np.array(startTimes))
firstSec=0
print("Shortest Integration time = ", intTime)

#load dithered science frames
ditherFrames = []
for i in range(nPos):
        #load stack for entire data set to be saved to H5, even though only intTime number of frames will
        #be used from each position
        obsfile=obs(ObsFNList[i], mode='write')
        if wvlStart != 0 and wvlStop!=0:
            img = obsfile.getPixelCountImage(firstSec =0, integrationTime=intTime,applyWeight=False,flagToUse = 0,wvlStart=wvlStart,wvlStop=wvlStop)
            print('Running getPixelCountImage on ',firstSec,'seconds to ',intTime,'seconds of data from wavelength ',wvlStart,'to ',wvlStop)
        else:
            img = obsfile.getPixelCountImage(firstSec =0, integrationTime=intTime,applyWeight=False,flagToUse = 0,wvlStart=None,wvlStop=None)
            print('Running getPixelCountImage on ',firstSec,'seconds to ',intTime,'seconds of data on all wavelengths')

        saveim = np.transpose(img['image'])
        outfile = h5dir + outfileName + 'DitherPosition%i'%i
        np.save(outfile, saveim)

        processedIm = np.transpose(img['image'])/intTime
        print(np.shape(processedIm))
        #processedIm=processedIm[50:124, 0:80]

        roughShiftsX.append(dXs[i])
        roughShiftsY.append(dYs[i])
        centroidsX.append(refPointX-dXs[i])
        centroidsY.append(refPointY-dYs[i])

       #plot an example of the UNmasked image for inspection
        if i==0:
            plot_array(processedIm,title='Dither Pos %i'%i,origin='upper',vmin=0)
        #cut out cold/dead pixels
        processedIm[np.where(processedIm>=2400)]=np.nan

        #plot an example of the masked image for inspection
        if i==0:
            plot_array(processedIm,title='Dither Pos %i ColdPix Masked'%i,origin='upper',vmin=0)

        #pad frame with margin for shifting and stacking
        paddedFrame = irUtils.embedInLargerArray(processedIm,frameSize=padFraction)
        outfile=h5dir+outfileName+str(i)
        np.save(outfile, paddedFrame)

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

plot_array(finalImage,title='final',origin='upper')
outfile=h5dir+outfileName+'AlignedWaveCal'
np.save(outfile, finalImage)
