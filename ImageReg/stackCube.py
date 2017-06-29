'''
Author: Seth Meeker        Date: Jun 15, 2017

stackCube:
load npz file from makeCubeTimestream and h5 file from quickStack
offset each spectral frames of cube by associated centroids for each time step
return a stacked cube

makeCubeTimestream:
Uses darkObsFile getSpectralCube to make a stacked cube, with 1s frames, stacked by wvl bin,
organized into dictionary that follows the timestamps/centroids saved by quickstack to the associated
.h5 file.

Takes same .cfg as quickStack, since it will be synced with the centroids that quickStack produces

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
#from matplotlib.backends.backend_qt4agg import NavigationToolbar2QTAgg as NavigationToolbar
from matplotlib.figure import Figure
from functools import partial
#import mpfit
from parsePacketDump2 import parsePacketData
from arrayPopup import plotArray
from readDict import readDict
from FileName import FileName
from darkObsFile import darkObsFile

from img2fitsExample import writeFits
from readFITStest import readFITS

#import hotpix.hotPixels as hp
#import headers.TimeMask as tm
import HotPix.darkHotPixMask as dhpm

import image_registration as ir
from loadStack import *
import irUtils


def makeCubeTimestream(configFileName):
    
    configData = readDict()
    configData.read_from_file(configFileName)

    # Extract parameters from config file
    nPos = int(configData['nPos'])
    startTimes = np.array(configData['startTimes'], dtype=int)
    stopTimes = np.array(configData['stopTimes'], dtype=int)
    darkSpan = np.array(configData['darkSpan'], dtype=int)
    flatSpan = np.array(configData['flatSpan'], dtype=int)
    xPos = np.array(configData['xPos'], dtype=int)
    yPos = np.array(configData['yPos'], dtype=int)
    numRows = int(configData['numRows'])
    numCols = int(configData['numCols'])
    upSample = int(configData['upSample'])
    padFraction = float(configData['padFraction'])
    coldCut = int(configData['coldCut'])
    fitPos = bool(configData['fitPos'])
    target = str(configData['target'])
    run = str(configData['run'])
    date = str(configData['date'])
    outputDir = str(configData['outputDir'])
    useImg = bool(configData['useImg'])
    doHPM = bool(configData['doHPM'])
    subtractDark = bool(configData['subtractDark'])
    divideFlat = bool(configData['divideFlat'])
    refFile = str(configData['refFile'])

    #hard coded for now to the daytime wvl cal we did with the WL data
    wvlCalTS = '1491870376'

    calPath = os.getenv('MKID_PROC_PATH', '/')
    timeMaskPath = os.path.join(calPath,"darkHotPixMasks")
    hpPath = os.path.join(timeMaskPath,date)


    #################################################           
    # Create empty arrays to save to npz file later
    timeStamps=[]
    cubes = []
    #################################################


    #get wvl cal soln file name
    cfn = FileName(run=run,date=date,tstamp=wvlCalTS).calSoln()

    #loop through obs files and each second within each file to make 1-s cubes
    for ts in startTimes:
        #load obs file
        obsFN = FileName(run=run, date=date, tstamp=str(ts)).obs()
        print obsFN
        obs = darkObsFile(obsFN)
        totalIntTime = obs.totalIntegrationTime
        #load wvlSoln file
        obs.loadWvlCalFile(cfn)
        for i in range(totalIntTime):
            fullTS = ts+i
            timeStamps.append(fullTS)
            print i, fullTS

            #get spectral cube for this second
            cDict = obs.getSpectralCube(i,1,weighted=False,fluxWeighted=False,energyBinWidth=0.07)
            cube = cDict['cube']
            cubes.append(cube)

    wvlBinEdges = np.array(cDict['wvlBinEdges'])
    cubes = np.array(cubes)
    times = np.array(timeStamps)

    #################################################
    # Setup npz file to save imageStack intermediate/cal files, parameter, and output
    stackPath = os.path.join(calPath,"imageStacks")
    npzPath = os.path.join(stackPath,date)

    #if configFileName.split('/')[0] is not 'Params':
    #    print "Config file not in Params! Output stack will have wrong filename!"
    #    print configFileName.split('/')[0]
    #    npzBaseName = target
    #else:
    npzBaseName = configFileName.split('/')[1].split('.')[0]
    npzFileName = npzPath+'/%s.npz'%npzBaseName
    print npzFileName

    np.savez(npzFileName, times=times, cubes=cubes, wvlBinEdges = wvlBinEdges)
    return {'cubes':cubes,'times':times,'wvlBinEdges':wvlBinEdges}


def stackCube(h5File,npzFile, verbose=True):
    #load npz file with cubes and timestamps
    npzDict = loadCubeStack(npzFile)
    cubeTimes = npzDict['times']
    cubeCubes = npzDict['cubes']
    cubeWBEs = npzDict['wvlBinEdges']
    #define bin center wavelengths (in nm)
    wbWidths = np.diff(cubeWBEs)
    centers = cubeWBEs[:-1]+wbWidths/2.0
    nWvlBins = len(centers)
    if verbose: print "Loaded npz file..."

    #load h5 file with centroids, image resampling info, dark frames, and timestamps
    h5Dict = loadH5Stack(h5File)
    h5Times = h5Dict['times']
    centXs = h5Dict['centX']
    centYs = h5Dict['centY']
    hotPix = h5Dict['hpm']
    #get image stacking info from params dictionary
    paramsDict = h5Dict['params']
    padFraction = np.float(paramsDict['padFraction'][0])
    upSample = paramsDict['upSample'][0]
    doHPM = bool(paramsDict['doHPM'][0])
    coldCut = paramsDict['coldCut'][0]
    if verbose: print "Loaded h5 file..."

    if cubeTimes.all() == h5Times.all():
        print "Timestamps match. Carrying on with stacking..."
    else:
        print "Timestamp mismatch between two files!!"
        print cubeTimes
        print h5Times
        sys.exit(0)

    if doHPM:
        hpMask = hotPix[0]

    cubeStack = []
    for i in range(nWvlBins): cubeStack.append([])
    finalCube = []
    finalTimes = []

    for t in np.arange(len(cubeTimes)):
        time = cubeTimes[t]
        cube = cubeCubes[t]
        centX = centXs[t]
        centY = centYs[t]
        finalTimes.append(cubeTimes[t])
        for w in np.arange(nWvlBins):
            im = np.array(cube[:,:,w],float)
            im = np.transpose(im)
            
            #apply hp mask to image
            if doHPM:
                im[np.where(hpMask==1)]=np.nan

            #cut out cold/dead pixels
            im[np.where(im<=coldCut)]=np.nan

            #pad frame with margin for shifting and stacking
            paddedFrame = irUtils.embedInLargerArray(im,frameSize=padFraction)

            #upSample frame for sub-pixel registration with fitting code
            upSampledFrame = irUtils.upSampleIm(paddedFrame,upSample)
            #conserve flux. Break each pixel into upSample^2 sub pixels, 
            #each subpixel should have 1/(upSample^2) the flux of the original pixel
            upSampledFrame/=float(upSample*upSample)

            ### UPDATE WITH DX AND DY DETERMINED FROM ACTUAL STARTING POINT
            ### SHIFTS ALL TO 0 RIGHT NOW
            #apply dX and dY shift to frame
            dX = centX*-1.*upSample
            dY = centY*-1.*upSample
            if verbose:
                print "Shifting timestamp %i, wvl %i by x=%2.2f, y=%2.2f"%(t,w, dX, dY)
            shiftedFrame = irUtils.rotateShiftImage(upSampledFrame,0,dX,dY)

            cubeStack[w].append(shiftedFrame)

    cubeStack = np.array(cubeStack)
    for n in np.arange(nWvlBins):
        finalCube.append(irUtils.medianStack(cubeStack[n]))
        if verbose:
            plotArray(finalCube[n],title="%i nm"%centers[n])
    finalCube = np.array(finalCube)

    return {'finalCube':finalCube,'wvls':centers, 'cubeStack':cubeStack, 'times':finalTimes}




if __name__ == "__main__":
    
    if len(sys.argv)<2:
        #grab most recent .cfg file
        print "No .cfg file provided, trying to grab most recent one from Params..."
        try:
            configFileName = max(glob.iglob('./Params/*.cfg'), key=os.path.getctime)
        except:
            print "Failed to load appropriate .cfg file. Please provide path as argument"
            sys.exit(0)
    else:
        configFileName = sys.argv[1]

    makeCubeTimestream(configFileName)
