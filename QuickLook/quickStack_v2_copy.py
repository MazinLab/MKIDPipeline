'''
Author: Seth Meeker        Date: Jun 14, 2017

Quick routine to take a series of files from multiple dither positions,
align them, and median add them

Updated from v1 with newly re-compartmentalized image registration code

load params from quickStack.cfg

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


def StackCalSoln_Description(nPos=1):
    strLength = 100  
    description = {
            "intTime"       : tables.UInt16Col(),          # integration time used for each dither position
            "nPos"          : tables.UInt16Col(),          # number of dither positions
            "startTimes"    : tables.UInt32Col(nPos),      # list of data start times for each position
            "stopTimes"     : tables.UInt32Col(nPos),      # physical y location
            "darkSpan"      : tables.UInt32Col(2),         # start and stop time for dark data
            "flatSpan"      : tables.UInt32Col(2),         # start and stop time for flat data
            "xPos"          : tables.UInt16Col(nPos),      # rough guess at X centroid for each position
            "yPos"          : tables.UInt16Col(nPos),      # rough guess at Y centroid for each position
            "numRows"       : tables.UInt16Col(),          # y-dimension of image
            "numCols"       : tables.UInt16Col(),          # x-dimension of image
            "upSample"      : tables.UInt16Col(),          # divide each pixel into upSample^2 for subpix registration
            "padFraction"   : tables.Float64Col(),         # x-dimension of image
            "coldCut"       : tables.UInt8Col(),           # any pixels with counts<coldCut is set to NAN during regis.
            "fitPos"        : tables.BoolCol(),            # boolean flag to perform fitting for registration
            "target"        : tables.StringCol(strLength), # name of target object
            "run"           : tables.StringCol(strLength), # observation run (eg. PAL2017a)
            "date"          : tables.StringCol(strLength), # date of observation
            "imgDir"        : tables.StringCol(strLength), # location of .IMG files
            "binDir"        : tables.StringCol(strLength), # location of .bin files
            "outputDir"     : tables.StringCol(strLength), # location where stack was originally saved
            "fitsPath"      : tables.StringCol(strLength), # full path (dir+filename) for FITS file with stacked image
            "refFile"       : tables.StringCol(strLength), # path to reference file for image registration
            "useImg"        : tables.BoolCol(),            # boolean flag to use .IMG or .bin data files in stacking
            "doHPM"         : tables.StringCol(strLength), # string controlling manner that hot pix masks are used
            "subtractDark"  : tables.BoolCol(),            # boolean flag to apply dark
            "divideFlat"    : tables.BoolCol(),            # boolean flag to apply flat
            "apMaskRadPrim" : tables.UInt16Col(),          # radius of aperture mask around primary object
            "apMaskRadSec"  : tables.UInt16Col()}          # radius of aperture mask around secondary
    return description

#configFileName = 'quickStack_test.cfg'
#configFileName = 'quickStack_coron_20161119.cfg'
#configFileName = 'quickStack_20161122g_hp.cfg'
#configFileName = 'quickStack_20161121b.cfg'
#configFileName = 'quickStack_20161122h.cfg'

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
    
#configFileName = 'ditherStack_1491773225.cfg'


configData = readDict()
configData.read_from_file(configFileName)

# Extract parameters from config file
nPos = int(configData['nPos'])
startTimes = np.array(configData['startTimes'], dtype=int)
stopTimes = np.array(configData['stpTimes'], dtype=int)
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
imgDir = str(configData['imgDir'])
binDir = str(configData['binDir'])
outputDir = str(configData['outputDir'])
useImg = bool(configData['useImg'])
doHPM = bool(configData['doHPM'])
subtractDark = bool(configData['subtractDark'])
divideFlat = bool(configData['divideFlat'])
refFile = str(configData['refFile'])

apMaskRadPrim = 300
apMaskRadSec = 300

if imgDir!='/mnt/ramdisk':
    runDir = os.path.join(imgDir,run)
    imgPath = os.path.join(runDir,date)
else:
    imgPath=imgDir

#could update this to use env var $MKID_RAW_PATH
binRunDir = os.path.join(binDir,run)
binPath = os.path.join(binRunDir,date)

calPath = os.getenv('MKID_PROC_PATH', '/')
timeMaskPath = os.path.join(calPath,"darkHotPixMasks")
hpPath = os.path.join(timeMaskPath,date)

#manual hot pixel array
manHP = None

#kludge for now, as identified in the 20170410 WL data
#manHP = [[1,14],[15,8],[16,16],[42,13],[51,2],[52,9],[50,20],[51,57],[55,62],[51,67],[50,70],[53,70],[55,69],[77,1],[77,58],[76,67],[76,77],[115,4],[117,13],[51,74],[84,31],[16,2],[7,11],[10,60],[14,69],[33,55],[50,18],[75,8],[78,12],[62,59],[76,60],[15,71],[50,17],[35,21],[15,68],[33,66],[58,72],[88,71],[75,21],[123,65],[116,17],[106,17],[110,41],[77,7],[60,70],[51,31],[103,21],[62,48],[65,16],[50,4],[83,17],[41,57],[41,25],[79,3],[75,0],[40,74],[108,72],[41,25],[50,42],[49,53],[70,14],[73,65],[115,20],[117,36],[40,64],[17,54],[77,65],[77,67],[65,67],[66,73],[80,75],[89,63],[87,54],[83,70],[58,71],[87,68],[81,61],[79,57],[77,58],[54,60],[78,55],[78,56]]

#kludge for 20170409 SAO65890 data manual hot pix
???LINES MISSING
???LINES MISSING
???LINES MISSING
???LINES MISSING
stackPath = os.path.join(calPath,"imageStacks")

h5Path = os.path.join(stackPath,run)
h5Path = os.path.join(h5Path, date)
h5baseName = configFileName.split('/')[1].split('.')[0]
h5FileName = h5Path+'/%s.h5'%h5baseName
print h5FileName
print 'h5Path, baseName', h5Path, h5baseName
quit()

np.savez(h5Path+'/%s.npz'%h5baseName,stack = shiftedFrames, final = finalImage)

h5file = tables.open_file(h5FileName,mode='w')
stackgroup = h5file.create_group(h5file.root, 'imageStack', 'Table of images, centroids, and parameters used to create a final stacked image')

#################################################
timeArray = tables.Array(stackgroup,'timestamps',timeStamps,title='Timestamps')
ditherArray = tables.Array(stackgroup,'dithers',ditherNums,title='Dither positions')
rawArray = tables.Array(stackgroup,'rawImgs',rawImgs,title='Raw Images')
hotArray = tables.Array(stackgroup,'hpms',hotPixMasks,title='Hot Pixel Masks')
coldArray = tables.Array(stackgroup,'cpms',coldPixMasks,title='Cold Pixel Masks')
deadArray = tables.Array(stackgroup,'dpms',deadPixMasks,title='Dead Pixel Masks')
aperArray = tables.Array(stackgroup,'ams',aperMasks,title='Aperture Masks')
roughXArray = tables.Array(stackgroup,'roughX',roughShiftsX,title='Rough X Shifts')
roughYArray = tables.Array(stackgroup,'roughY',roughShiftsY,title='Rough Y Shifts')
fineXArray = tables.Array(stackgroup,'fineX',fineShiftsX,title='Fine X Shifts')
fineYArray = tables.Array(stackgroup,'fineY',fineShiftsY,title='Fine Y Shifts')
centXArray = tables.Array(stackgroup,'centX',centroidsX,title='Centroid X Locations')
centYArray = tables.Array(stackgroup,'centY',centroidsY,title='Centroid Y Locations')
darkArray = tables.Array(stackgroup,'dark',dark,title='Dark Frame')
flatArray = tables.Array(stackgroup,'flat',flat,title='Flat Frame')
finalArray = tables.Array(stackgroup,'finalImg',finalImage,title='Final Image')

#Write parameter table with all settings used from cfg file to make stack
descriptionDict = StackCalSoln_Description(nPos)

paramTable = h5file.create_table(stackgroup, 'params', descriptionDict,title='Stack Parameter Table')

entry = paramTable.row
entry['intTime'] = intTime
entry['nPos'] = nPos
entry['startTimes'] = startTimes
entry['stopTimes'] = stopTimes
entry['darkSpan'] = darkSpan
entry['flatSpan'] = flatSpan
entry['xPos'] = xPos
entry['yPos'] = yPos
entry['numRows'] = numRows
entry['numCols'] = numCols
entry['upSample'] = upSample
entry['padFraction'] = padFraction
entry['coldCut'] = coldCut
entry['fitPos'] = fitPos
entry['target'] = target
entry['run'] = run
entry['date'] = date
entry['imgDir'] = imgDir
entry['binDir'] = binDir
entry['outputDir'] = outputDir
entry['useImg'] = useImg
entry['doHPM'] = doHPM
entry['subtractDark'] = subtractDark
entry['divideFlat'] = divideFlat
entry['refFile'] = refFile
entry['apMaskRadPrim'] = apMaskRadPrim
entry['apMaskRadSec'] = apMaskRadSec
entry['fitsPath'] = fitsFileName
entry.append()

############################################
h5file.flush()
h5file.close()
############################################


