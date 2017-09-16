'''
Author: Seth Meeker        Date: Sept 15, 2017

Quick routine to take a stack of flat frames saved by quickStack_v2
and turn them into a temporary flat cal .npz file.

Mostly useful when doing plain old J-band photometry with DARKNESS
when we have a J-band dome flat available.

Very tentative experience shows that a flat from one night is fairly
useable on subsequent (or previous) night as the bulk of the correction
seems to be difference in QE between pixels, which is likely related
to their sub-optimal readout powers. Those powers should not change 
from one observing night to another, so it would follow that the 
flat correction should be mostly the same from one night's setup to
the next.


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


if len(sys.argv)<4:
    print "syntax: python makeSimpleFlatNPZ.py <run name> <sunset date> <flat stack .npz>"
    sys.exit(0)
else:
    run = str(sys.argv[1])
    date = str(sys.argv[2])
    flatFileName = str(sys.argv[3])

basePath = '/mnt/data0/CalibrationFiles/tempFlatCal'
rPath = os.path.join(basePath,run)
dPath = os.path.join(rPath,date)
flatPath = os.path.join(dPath,flatFileName)

baseFN = flatFileName.split('.')[0]
outFN = baseFN+'_weights.npz'
outPath = os.path.join(dPath,outFN)

try:
    flatData = np.load(flatPath)
except:
    print "couldn't find flat stack .npz file at path:"
    print flatPath
    sys.exit(0)

flatFrame = flatData['final']
plotArray(flatFrame,title='flat frame')

########
# This cropping assumes poor FL2 performance and
# poor high frequency performance, a la Faceless from 2017a,b.
# This is a good place to play around to change crappy weights
# at fringes of the array.
#######
croppedFrame = np.copy(flatFrame)
croppedFrame[:50,::] = np.nan
croppedFrame[::,:20] = np.nan
croppedFrame[::,59:] = np.nan

plotArray(croppedFrame,title='Cropped flat frame')

med = np.nanmedian(croppedFrame.flatten())
weights = med/flatFrame

plotArray(weights,title='Weights')

#write flat weight .npz file
np.savez(outPath,weights = weights, flat = flatFrame)





