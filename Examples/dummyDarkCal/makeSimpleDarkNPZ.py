'''
Author: Seth Meeker        Date: Sept 15, 2017

Dark .npz part added on 9/27 by Isabel Lipartito
Takes stack of dark frames saved by quickStack_v2 and turns them into a temporary dark cal .npz file.

makeSimpleFlatNPZ.py adapted to make a mean dark frame.
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

from img2fitsExample import writeFits
from readFITStest import readFITS

#import hotpix.hotPixels as hp
#import headers.TimeMask as tm
import HotPix.darkHotPixMask as dhpm

import image_registration as ir
from loadStack import loadIMGStack, loadBINStack
import irUtils


# These flags determine if the HF margins and FLs 2 are ignored when making dark frame.
# Currently set to FL 2 for Faceless (Pal 2017 a,b array)
# These flags and selection of FLs should be made more flexible, and command line options
cropHF = True
cropFLs = True

if len(sys.argv)<4:
    print "syntax: python makeSimpleDarkNPZ.py <run name> <sunset date> <Dark stack .npz>"
    sys.exit(0)
else:
    run = str(sys.argv[1])
    date = str(sys.argv[2])
    darkFileName = str(sys.argv[3])

basePath = '/mnt/data0/CalibrationFiles/tempDarkCal'
baseFramePath = '/mnt/data0/CalibrationFiles/imageStacks'
rPath = os.path.join(baseFramePath,run)
dPath = os.path.join(rPath,date)
darkPath = os.path.join(dPath,darkFileName)

baseFN = darkFileName.split('.')[0]
outFN = baseFN+'_mean_dark.npz'
rPath = os.path.join(basePath, run)
dPath = os.path.join(rPath, date)
outPath = os.path.join(dPath,outFN)


try:
    darkData = np.load(darkPath)
except:
    print "couldn't find dark stack .npz file at path:"
    print darkPath
    sys.exit(0)

darkFrame = darkData['final']
plotArray(darkFrame,title='dark frame')

beammapFile = '/mnt/data0/Darkness/20170917/Beammap/finalMap_20170914.txt'
resID, flag, xCoords, yCoords = np.loadtxt(beammapFile, unpack=True)
badInds = np.where(flag==1)[0]
badXCoords = np.array(xCoords[badInds], dtype=np.int32)
badYCoords = np.array(yCoords[badInds], dtype=np.int32)
outOfBoundsMask = np.logical_or(badXCoords>=80, badYCoords>=125)
outOfBoundsInds = np.where(outOfBoundsMask)[0]
print np.shape(darkFrame)
badXCoords = np.delete(badXCoords, outOfBoundsInds)
badYCoords = np.delete(badYCoords, outOfBoundsInds)
darkFrame[badYCoords, badXCoords] = np.nan

########
# This cropping assumes poor FL2 performance and
# poor high frequency performance, a la Faceless from 2017a,b.
# This is a good place to play around to change crappy weights
# at fringes of the array.
#######
croppedFrame = np.copy(darkFrame)

if cropFLs:
    croppedFrame[25:50,::] = np.nan
if cropHF:
    croppedFrame[::,:20] = np.nan
    croppedFrame[::,59:] = np.nan

plotArray(croppedFrame,title='Cropped dark frame')

mean = np.nanmean(croppedFrame.flatten())

plotArray(mean,title='Mean Dark Array')

#write mean dark array .npz file
np.savez(outPath,mean = mean)





