'''
Author: Seth Meeker        Date: Jun 14, 2017

Quick routine to take a series of files from multiple dither positions,
align them, and median add them

Updated from v1 with newly re-compartmentalized image registraction code

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
#manHP = [[0,3],[2,27],[8,63],[5,76],[10,78],[27,15],[27,20],[22,78],[37,52],[48,4],[54,20],[54,34],[63,33],[67,39],[63,61],[58,70],[62,74],[66,71],[75,24],[76,32],[78,62],[102,24],[110,75],[116,18],[114,28],[115,59],[116,72],[122,73],[88,73],[109,4],[75,21],[64,5],[16,63],[4,23],[42,53],[42,41],[16,51],[76,34],[117,60],[51,57],[85,33],[88,10],[82,73],[122,72],[118,63],[110,59],[104,54],[102,48],[99,22],[89,23],[88,23],[90,19],[89,13],[71,54],[65,36],[65,30],[60,65],[73,65],[75,8],[51,25],[57,72],[60,51],[57,5],[50,42],[50,20],[54,60],[79,3],[84,1]]

#kludge for 20170408 tau Boo manual hot pix
#manHP = [[0,1],[9,24],[39,11],[37,15],[35,39],[38,57],[42,56],[50,15],[65,18],[71,11],[75,13],[73,72],[82,42],[83,47],[80,75],[82,72],[89,36],[110,12],[103,21],[103,31],[106,30],[107,37],[117,36],[121,47],[117,60],[113,60],[78,62],[73,19],[88,38],[87,38],[93,75],[42,41],[32,13],[83,17]]

#kludge for 20170410 HD91782 data
#manHP = [[3,4],[7,11],[1,26],[8,28],[5,57],[16,39],[21,5],[13,55],[16,57],[15,61],[14,63],[34,11],[37,15],[36,18],[33,67],[38,72],[40,73],[48,57],[54,2],[61,0],[57,5],[63,4],[63,6],[61,11],[62,11],[64,18],[60,25],[52,28],[57,54],[59,47],[64,49],[67,53],[60,58],[59,72],[63,74],[70,38],[73,68],[73,72],[76,78],[77,3],[82,27],[83,65],[84,79],[84,22],[88,13],[88,32],[89,9],[89,8],[85,33],[89,36],[88,37],[87,68],[89,63],[94,69],[97,49],[103,21],[102,62],[105,58],[101,78],[107,69],[108,72],[114,74],[110,21],[116,47],[116,58],[117,69],[122,72],[124,72],[15,57],[38,21],[42,36],[38,77],[52,15],[54,16],[52,44],[52,51],[73,19],[86,36],[106,17],[118,11],[117,59],[41,42],[66,27],[75,31],[84,8],[87,0],[6,6],[32,21],[16,63],[16,31],[38,57],[29,69],[31,75],[57,72],[52,19],[50,15],[65,13],[75,18],[57,72],[63,68],[87,36],[102,29],[107,37],[115,39],[124,55],[124,60],[123,78],[111,50],[86,30],[80,44],[65,16],[76,32],[73,48],[116,70],[75,75],[73,48],[50,43],[78,17],[90,9],[88,6],[62,34],[64,42],[64,31],[64,30],[65,29],[66,39],[66,30],[66,27],[67,26],[67,30],[67,39],[69,35],[69,36],[70,38],[61,26],[66,29],[67,34],[56,51],[82,49],[85,24],[83,17],[80,14],[76,12]]

#kludge for 20170410 HD148112
#manHP = [[13,3],[17,69],[30,52],[35,14],[35,77],[40,52],[42,53],[42,18],[54,34],[59,47],[58,70],[63,13],[65,18],[64,26],[76,32],[90,3],[85,21],[84,62],[97,49],[106,3],[110,21],[105,58],[107,69],[107,37],[109,31],[109,53],[109,72],[113,55],[113,28],[110,21],[114,21],[123,52],[124,60],[124,67],[122,74],[123,78],[4,23],[17,72],[39,41],[31,75],[39,41],[41,78],[50,15],[62,48],[73,48],[69,48],[72,51],[102,24],[117,36],[123,65],[115,79],[88,14],[61,5],[65,67],[76,23],[82,69],[116,18],[66,17],[90,19],[63,9],[64,67],[102,29],[118,28]]

#kludge for 20170410 Dome flat data
manHP = [[0,51],[3,4],[5,4],[8,28],[28,3],[32,13],[30,24],[31,78],[35,14],[36,19],[37,15],[35,19],[39,11],[42,10],[33,67],[34,67],[41,42],[42,53],[38,72],[40,78],[50,31],[50,32],[50,50],[54,20],[54,32],[53,58],[57,54],[53,77],[59,72],[61,73],[62,76],[61,0],[64,5],[65,10],[64,18],[63,61],[64,67],[65,67],[66,71],[69,73],[70,38],[69,49],[71,62],[73,65],[76,72],[78,62],[77,55],[77,50],[75,8],[75,13],[84,62],[79,10],[82,7],[85,19],[81,61],[89,9],[90,9],[92,18],[102,24],[106,3],[106,17],[105,58],[109,72],[112,18],[118,10],[118,28],[115,53],[123,78]]


if useImg == True:
    dataDir = imgPath
    print "Loading data from .img files"
else:
    dataDir = binPath
    print "Loading data from .bin files"


print startTimes
print stopTimes
print darkSpan
print flatSpan

#################################################           
# Create empty arrays to save to h5 file later
coldPixMasks=[]
hotPixMasks=[]
deadPixMasks=[]
aperMasks=[]
ditherNums=[]
timeStamps=[]
rawImgs=[]
roughShiftsX=[]
roughShiftsY=[]
fineShiftsX=[]
fineShiftsY=[]
centroidsX=[]
centroidsY=[]
#################################################


#load dark frames

if darkSpan[0]!='0':
    print "Loading dark frame"
    if useImg == True:
        darkStack = loadIMGStack(dataDir, darkSpan[0], darkSpan[1], nCols=numCols, nRows=numRows)
    else:
        darkStack = loadBINStack(dataDir, darkSpan[0], darkSpan[1], nCols=numCols, nRows=numRows)
    dark = irUtils.medianStack(darkStack)

else:
    print "No dark provided"
    dark = np.zeros((numRows, numCols),dtype=int)
#plotArray(dark,title='Dark',origin='upper')
#dark = signal.medfilt(dark,3)
#plotArray(dark,title='Med Filt Dark',origin='upper')

#load flat frames
if flatSpan[0]!='0':
    print "Loading flat frame"
    if useImg == True:
        flatStack = loadIMGStack(dataDir, flatSpan[0], flatSpan[1], nCols=numCols, nRows=numRows)
    else:
        flatStack = loadBINStack(dataDir, flatSpan[0], flatSpan[1], nCols=numCols, nRows=numRows)
    flat = irUtils.medianStack(flatStack)

else:
    print "No flat provided"
    flat = np.ones((numRows, numCols),dtype=int)
#plotArray(flat,title='Flat',origin='upper')


if subtractDark == True:
    flatSub = flat-dark
else:
    flatSub = flat


#try to load up hot pixel mask. Make it if it doesn't exist yet
hpFile = os.path.join(hpPath,"%s.npz"%(darkSpan[0]))
if doHPM:
    if not os.path.exists(hpFile):
        print "Could not find existing hot pix mask, generating one from dark..."
        try:
            darkHPM = dhpm.makeMask(run=run, date=date, startTimeStamp=darkSpan[0], stopTimeStamp=darkSpan[1], coldCut=True, manualArray=manHP)
            dhpm.saveMask(darkHPM, timeStamp=darkSpan[0], date=date)
        except:
            print "Failed to generate dark mask. Turning off hot pixel masking"
            doHPM = False
    else:
        darkHPM = dhpm.loadMask(hpFile)
        print "Loaded hot pixel mask from dark data %s"%hpFile


#starting point centroid guess for first frame is where all subsequent frames will be aligned to
refPointX = xPos[0]
refPointY = yPos[0]

#determine the coarse x and y shifts that subsequent frames must be moved to align with first frame
dXs = refPointX-xPos
dYs = refPointY-yPos

#initialize hpDict so we can just check if it exists and only make it once
hpDict=None

#to ensure stacking is weighting all dither positions evenly, we need to know the one with the shortest integration
intTime = min(stopTimes-startTimes)
print "Shortest Integration time = ", intTime

#load dithered science frames
ditherFrames = []
for i in range(nPos):
    #load stack for entire data set to be saved to H5, even though only intTime number of frames will
    #be used from each position
    if useImg == True:
        stack = loadIMGStack(dataDir, startTimes[i], stopTimes[i]-1, nCols=numCols, nRows=numRows)
    else:
        stack = loadBINStack(dataDir, startTimes[i], stopTimes[i]-1, nCols=numCols, nRows=numRows)
    
    for f in range(len(stack)):
        timeStamp = startTimes[i]+f

        if subtractDark == True:
            darkSub = stack[f]-dark
        else:
            darkSub = stack[f]

        if divideFlat == True:

            processedIm = np.array(darkSub/flatSub,dtype=float)
        else:
            processedIm = np.array(darkSub,dtype=float)
            
        #plotArray(med,title='Dither Pos %i'%i,origin='upper')
        #plotArray(embedInLargerArray(processedIm),title='Dither Pos %i - dark'%i,origin='upper')

        #Append the raw frames and rough dx, dy shifts to arrays for h5
        ditherNums.append(i)
        timeStamps.append(timeStamp)
	rawImgs.append(stack[f])
	roughShiftsX.append(dXs[i])
	roughShiftsY.append(dYs[i])
        centroidsX.append(refPointX-dXs[i])
        centroidsY.append(refPointY-dYs[i])

        '''
        if i==0 and f==0:
            #plotArray(processedIm,title='Dither Pos %i - dark'%i,origin='upper')#,vmin=0)
            print np.shape(processedIm)
            print np.shape(dark)
            print sum(processedIm)
            print np.where(processedIm==np.nan)
        '''

        if doHPM:
	    hpm = darkHPM
            hpMask = hpm
	    cpm = np.zeros((numRows, numCols),dtype=int)
            dpm = np.zeros((numRows, numCols),dtype=int)

        else:
            print "No hot pixel masking specified in config file"
            hpMask = np.zeros((numRows, numCols),dtype=int)

            hpm = np.zeros((numRows, numCols),dtype=int)
            cpm = np.zeros((numRows, numCols),dtype=int)
            dpm = np.zeros((numRows, numCols),dtype=int)
	    #print(hpMask)

        #Append the individual masks used on each frame to arrays that will go in h5 
	hotPixMasks.append(hpm)
	coldPixMasks.append(cpm)
	deadPixMasks.append(dpm)
        
        #apply hp mask to image
        if doHPM:
            processedIm[np.where(hpMask==1)]=np.nan

        #cut out cold/dead pixels
        processedIm[np.where(processedIm<=coldCut)]=np.nan

        #plot an example of the masked image for inspection
        if f==0 and i==0:
            plotArray(processedIm,title='Dither Pos %i HP Masked'%i,origin='upper',vmin=0)

        #pad frame with margin for shifting and stacking
        paddedFrame = irUtils.embedInLargerArray(processedIm,frameSize=padFraction)

        #apply rough dX and dY shift to frame
        print "Shifting dither %i, frame %i by x=%i, y=%i"%(i,f, dXs[i], dYs[i])
        shiftedFrame = irUtils.rotateShiftImage(paddedFrame,0,dXs[i],dYs[i])

        #upSample frame for sub-pixel registration with fitting code
        upSampledFrame = irUtils.upSampleIm(shiftedFrame,upSample)
        #conserve flux. Break each pixel into upSample^2 sub pixels, 
        #each subpixel should have 1/(upSample^2) the flux of the original pixel
        upSampledFrame/=float(upSample*upSample) 

        #append upsampled, padded frame to array for storage into next part of code
        ditherFrames.append(upSampledFrame)

    print "Loaded dither position %i"%i

shiftedFrames = np.array(ditherFrames)
#if fitPos==True, do second round of shifting using mpfit correlation
#using x and y pos from earlier as guess
if fitPos==True:
    reshiftedFrames=[]
    
    if refFile!=None and os.path.exists(refFile):
        refIm = readFITS(refFile)
        print "Loaded %s for fitting"%refFile
        plotArray(refIm,title='loaded reference FITS',origin='upper',vmin=0)
    else:
        refIm=shiftedFrames[0]

    cnt=0
    for im in shiftedFrames:
        print "\n\n------------------------------\n"
        print "Fitting frame %i of %i"%(cnt,len(shiftedFrames))
        pGuess=[0,1,1]
        pLowLimit=[-1,(dXs.min()-5)*upSample,(dYs.min()-5)*upSample]
        pUpLimit=[1,(dXs.max()+5)*upSample,(dYs.max()+5)*upSample]
        print "guess", pGuess, "limits", pLowLimit, pUpLimit

        '''
        #mask out background structure, only fit on known object location
        maskRad=apMaskRadPrim
        pMask = aperture(xPos[0]*upSample,yPos[0]*upSample,maskRad*upSample, numRows*upSample, numCols*upSample)
        pMask = embedInLargerArray(pMask,frameSize=padFraction,padValue = 0)
        m1 = np.ma.make_mask(pMask)

        #mask parameters for SAO binary secondary
        maskRad=apMaskRadSec
        sMask = aperture((xPos[0]-7)*upSample,(yPos[0]-20)*upSample,maskRad*upSample, numRows*upSample, numCols*upSample)
        #sMask = aperture((xPos[0])*upSample,(yPos[0])*upSample,maskRad*upSample, numRows*upSample, numCols*upSample)
        sMask = embedInLargerArray(sMask,frameSize=padFraction,padValue = 0)
        m2 = np.ma.make_mask(sMask)

        #aperture mask with secondary
        apMask = np.ma.mask_or(m1,m2)

        #aperture mask for no primary
        #apMask = m2

        #aperture mask for no secondary
        #apMask = m1

        #append aperture mask to
        aperMasks.append(apMask)

        maskedRefIm = np.copy(refIm)
        maskedRefIm[np.where(~apMask)]=np.nan
        maskedIm = np.copy(im)
        maskedIm[np.where(~apMask)]=np.nan
        '''

        maskedRefIm = np.copy(refIm)
        maskedIm = np.copy(im)

        maskedRefIm[:,400:]=np.nan
        maskedRefIm[:,:290]=np.nan
        maskedRefIm[:383,:]=np.nan
        maskedRefIm[662:,:]=np.nan



        if cnt==0:
            #plotArray(sMask, title='aperture mask', origin='upper',vmin=0,vmax=1)
            #plot array with secondary mask as well
            #plotArray(pMask+sMask, title='aperture mask', origin='upper',vmin=0,vmax=1)

            plotArray(maskedRefIm, title='masked Reference', origin='upper')
            plotArray(maskedIm, title='masked Image to be aligned', origin='upper') 

        #use mpfit and correlation fitting from Giulia's M82 code
        #mp = alignImages(maskedRefIm, maskedIm, parameterGuess=pGuess, parameterLowerLimit=pLowLimit, 				parameterUpperLimit=pUpLimit)

        # image registration by FFT returning all zeros for translation...
        #trans = imRegFFT.translation(refIm, im)
        #im2, scale, angle, trans = imRegFFT.similarity(np.nan_to_num(refIm),np.nan_to_num(im))
        #mp = [angle, trans[0],trans[1]]

        #try using keflavich image_registation repository
        dx, dy, ex, ey = ir.chi2_shifts.chi2_shift(maskedRefIm, maskedIm, zeromean=True)#,upsample_factor='auto')
        mp = [0,-1*dx,-1*dy]

        print "fitting output: ", mp

        newShiftedFrame = irUtils.rotateShiftImage(im,mp[0],mp[1],mp[2])
        reshiftedFrames.append(newShiftedFrame)
	fineShiftX=mp[1]
	fineShiftY=mp[2]
	fineShiftsX.append(float(fineShiftX/upSample))
	fineShiftsY.append(float(fineShiftY/upSample))
        centroidsX[cnt]-=float(fineShiftX/upSample)
        centroidsY[cnt]-=float(fineShiftY/upSample)
	cnt+=1
	
    shiftedFrames = np.array(reshiftedFrames)

#take median stack of all shifted frames
finalImage = irUtils.medianStack(shiftedFrames)# / 3.162277 #adjust for OD 0.5 difference between occulted/unocculted files
plotArray(finalImage,title='final',origin='upper')

nanMask = np.zeros(np.shape(finalImage))
nanMask[np.where(np.isnan(finalImage))]=1
#plotArray(nanMask,title='good=0, nan=1', origin='upper')

fitsFileName = outputDir+'%s_%sDithers_%sxSamp_%sHPM_%s.fits'%(target,nPos,upSample,doHPM,date)
try:
    writeFits(finalImage, fitsFileName)
    print "Wrote to FITS: ", fitsFileName
except:
    print "FITS file already present, did not overwrite"

#################################################
# Setup h5 file to save imageStack intermediate/cal files, parameter, and output
stackPath = os.path.join(calPath,"imageStacks")

h5Path = os.path.join(stackPath,date)        
h5baseName = configFileName.split('/')[1].split('.')[0]
h5FileName = h5Path+'/%s.h5'%h5baseName
print h5FileName

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


