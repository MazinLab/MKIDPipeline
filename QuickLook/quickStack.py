'''
Author: Seth Meeker        Date: Nov 20, 2016

Quick routine to take a series of files from multiple dither positions,
align them, and median add them

load params from quickStack.cfg

TODO:
Pipe unocculted centroids into occulted version

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
import hotpix.hotPixels as hp
import headers.TimeMask as tm
from utilsM82 import *
from readFITStest import readFITS
#import imRegFFT
import image_registration as ir

def aperture(startpx,startpy,radius, nRows, nCols):
        r = radius
        length = 2*r 
        height = length
        allx = xrange(startpx-int(np.ceil(length/2.0)),startpx+int(np.floor(length/2.0))+1)
        ally = xrange(startpy-int(np.ceil(height/2.0)),startpy+int(np.floor(height/2.0))+1)
        mask=np.zeros((nRows,nCols))
        
        for x in allx:
            for y in ally:
                if (np.abs(x-startpx))**2+(np.abs(y-startpy))**2 <= (r)**2 and 0 <= y and y < nRows and 0 <= x and x < nCols:
                    mask[y,x]=1.
        return mask

def loadStack(dataDir, start, stop, useImg = False, nCols=80, nRows=125):
    frameTimes = np.arange(start, stop+1)
    frames = []
    for iTs,ts in enumerate(frameTimes):
        try:
            if useImg==False:
                imagePath = os.path.join(dataDir,str(ts)+'.bin')
                print imagePath
                with open(imagePath,'rb') as dumpFile:
                    data = dumpFile.read()

                nBytes = len(data)
                nWords = nBytes/8 #64 bit words
                
                #break into 64 bit words
                words = np.array(struct.unpack('>{:d}Q'.format(nWords), data),dtype=object)
                parseDict = parsePacketData(words,verbose=False)
                image = parseDict['image']

            else:
                imagePath = os.path.join(dataDir,str(ts)+'.img')
                print imagePath
                image = np.fromfile(open(imagePath, mode='rb'),dtype=np.uint16)
                image = np.transpose(np.reshape(image, (nCols, nRows)))

        except (IOError, ValueError):
            print "Failed to load ", imagePath
            image = np.zeros((nRows, nCols),dtype=np.uint16)  
        frames.append(image)
    stack = np.array(frames)
    return stack

def medianStack(stack):
    return np.nanmedian(stack, axis=0)

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
<<<<<<< HEAD
#configFileName = 'quickStack_20161122h.cfg'

if len(sys.argv)<2:
    #grab most recent .cfg file
    configFileName = max(glob.iglob('*.cfg'), key=os.path.getctime)
else:
    configFileName = sys.argv[1]
    
#configFileName = 'ditherStack_1491773225.cfg'

=======
configFileName = 'quickStack_20161122h.cfg'
>>>>>>> 094d0556e7c743154de499d73bfa5cbdc4f99f4f
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
date = str(configData['date'])
imgDir = str(configData['imgDir'])
binDir = str(configData['binDir'])
outputDir = str(configData['outputDir'])
useImg = bool(configData['useImg'])
doHPM = str(configData['doHPM'])
subtractDark = bool(configData['subtractDark'])
divideFlat = bool(configData['divideFlat'])
refFile = str(configData['refFile'])

<<<<<<< HEAD
apMaskRadPrim = 300
apMaskRadSec = 300

if imgDir!='/mnt/ramdisk' and imgDir!='/mnt/ramdisk/':
    imgPath = os.path.join(imgDir,date)
else:
    imgPath=imgDir
=======
apMaskRadPrim = 18
apMaskRadSec = 24

imgPath = os.path.join(imgDir,date)
>>>>>>> 094d0556e7c743154de499d73bfa5cbdc4f99f4f
binPath = os.path.join(binDir,date)

calPath = "/mnt/data0/darknessCalFiles"
timeMaskPath = os.path.join(calPath,"timeMasks")
hpPath = os.path.join(timeMaskPath,date)

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
<<<<<<< HEAD
=======
print "Loading dark frame"
darkStack = loadStack(dataDir, darkSpan[0], darkSpan[1],useImg = useImg, nCols=numCols, nRows=numRows)
dark = medianStack(darkStack)
#plotArray(dark,title='Dark',origin='upper')
>>>>>>> 094d0556e7c743154de499d73bfa5cbdc4f99f4f

if darkSpan[0]!='0':
    print "Loading dark frame"
    darkStack = loadStack(dataDir, darkSpan[0], darkSpan[1],useImg = useImg, nCols=numCols, nRows=numRows)
    dark = medianStack(darkStack)
else:
    print "No dark provided"
    dark = np.zeros((numRows, numCols),dtype=int)
#plotArray(dark,title='Dark',origin='upper')
#dark = signal.medfilt(dark,3)
#plotArray(dark,title='Med Filt Dark',origin='upper')

#load flat frames
<<<<<<< HEAD
if flatSpan[0]!='0':
    print "Loading flat frame"
    flatStack = loadStack(dataDir, flatSpan[0], flatSpan[1],useImg = useImg, nCols=numCols, nRows=numRows)
    flat = medianStack(flatStack)
else:
    print "No flat provided"
    flat = np.ones((numRows, numCols),dtype=int)
#plotArray(flat,title='Flat',origin='upper')

if subtractDark == True:
    flatSub = flat-dark
else:
    flatSub = flat

=======
print "Loading flat frame"
flatStack = loadStack(dataDir, flatSpan[0], flatSpan[1],useImg = useImg, nCols=numCols, nRows=numRows)
flat = medianStack(flatStack)
#plotArray(flat,title='Flat',origin='upper')
if subtractDark == True:
    flatSub = flat-dark
else:
    flatSub = flat

>>>>>>> 094d0556e7c743154de499d73bfa5cbdc4f99f4f
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
    stack = loadStack(dataDir, startTimes[i], stopTimes[i],useImg = useImg, nCols=numCols, nRows=numRows)
    
    for f in range(len(stack)):
        timeStamp = startTimes[i]+f
        hpPklFile = os.path.join(hpPath,"%s.pkl"%(timeStamp))
        
        if subtractDark == True:
            darkSub = stack[f]-dark
        else:
            darkSub = stack[f]

        if divideFlat == True:
<<<<<<< HEAD
            processedIm = np.array(darkSub/flatSub,dtype=float)
        else:
            processedIm = np.array(darkSub,dtype=float)
        print darkSub
=======
            processedIm = darkSub/flatSub
        else:
            processedIm = darkSub
>>>>>>> 094d0556e7c743154de499d73bfa5cbdc4f99f4f
            
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

        if i==0 and f==0:
<<<<<<< HEAD
            #plotArray(processedIm,title='Dither Pos %i - dark'%i,origin='upper')#,vmin=0)
            print np.shape(processedIm)
            print np.shape(dark)
            print sum(processedIm)
            print np.where(processedIm==np.nan)
=======
            plotArray(processedIm,title='Dither Pos %i - dark'%i,origin='upper',vmin=0)
>>>>>>> 094d0556e7c743154de499d73bfa5cbdc4f99f4f

        if (doHPM == 'one' and hpDict==None) or (doHPM=='all'):
            if not os.path.exists(hpPklFile):
                print "Creating time mask: %s"%(hpPklFile)
                hpDict = hp.checkInterval(image=processedIm,fwhm=2.5, boxSize=5, nSigmaHot=3.0, dispToPickle=hpPklFile)
                hpMask = hpDict['mask']

                #open pkl file to grab masks in format that h5 file expects
                pklDict = pickle.load(open(hpPklFile,"rb"))
		hpm=pklDict["hotMask"]
	        cpm=pklDict["coldMask"]
	        dpm=pklDict["deadMask"]
                #plotArray(hpMask,title='12 = hot, 19 = OK', origin='upper',vmin=0, vmax=13)

            else:
                print "Loading time mask: %s"%(hpPklFile)
                pklDict = pickle.load(open(hpPklFile,"rb"))
                hpMask = np.empty((numRows, numCols),dtype=int)
                hpMask.fill(tm.timeMaskReason['none'])
<<<<<<< HEAD
=======
		
>>>>>>> 094d0556e7c743154de499d73bfa5cbdc4f99f4f
                
                hpMask[pklDict["hotMask"]] = tm.timeMaskReason['hot pixel']
                hpMask[pklDict["coldMask"]] = tm.timeMaskReason['cold pixel']
                hpMask[pklDict["deadMask"]] = tm.timeMaskReason['dead pixel']

		hpm=pklDict["hotMask"]
	        cpm=pklDict["coldMask"]
	        dpm=pklDict["deadMask"]

        else:
            print "No hot pixel masking specified in config file"
            hpMask = np.zeros((numRows, numCols),dtype=int)
<<<<<<< HEAD
            hpm = np.zeros((numRows, numCols),dtype=int)
            cpm = np.zeros((numRows, numCols),dtype=int)
            dpm = np.zeros((numRows, numCols),dtype=int)
=======
>>>>>>> 094d0556e7c743154de499d73bfa5cbdc4f99f4f
	    #print(hpMask)

        #Append the individual masks used on each frame to arrays that will go in h5 
	hotPixMasks.append(hpm)
	coldPixMasks.append(cpm)
	deadPixMasks.append(dpm)
<<<<<<< HEAD
        
        #apply hp mask to image
        if (doHPM == 'all') or (doHPM =='one'):
            processedIm[np.where(hpMask==tm.timeMaskReason['hot pixel'])]=np.nan

        #cut out cold/dead pixels
        processedIm[np.where(processedIm<=coldCut)]=np.nan
        
        #plot an example of the masked image for inspection
        #if f==0 and i==0:
            #plotArray(processedIm,title='Dither Pos %i HP Masked'%i,origin='upper',vmin=0)
=======

        #apply hp mask to image
        processedIm[np.where(hpMask==tm.timeMaskReason['hot pixel'])]=np.nan

        #cut out cold/dead pixels
        processedIm[np.where(processedIm<=coldCut)]=np.nan
        
        #plot an example of the masked image for inspection
        if f==0 and i==0:
            plotArray(processedIm,title='Dither Pos %i HP Masked'%i,origin='upper',vmin=0)
>>>>>>> 094d0556e7c743154de499d73bfa5cbdc4f99f4f

        #pad frame with margin for shifting and stacking
        paddedFrame = embedInLargerArray(processedIm,frameSize=padFraction)

        #apply rough dX and dY shift to frame
        print "Shifting dither %i, frame %i by x=%i, y=%i"%(i,f, dXs[i], dYs[i])
        shiftedFrame = rotateShiftImage(paddedFrame,0,dXs[i],dYs[i])

        #upSample frame for sub-pixel registration with fitting code
        upSampledFrame = upSampleIm(shiftedFrame,upSample)

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

        if cnt==0:
            #plotArray(sMask, title='aperture mask', origin='upper',vmin=0,vmax=1)
            #plot array with secondary mask as well
            plotArray(pMask+sMask, title='aperture mask', origin='upper',vmin=0,vmax=1)

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

        newShiftedFrame = rotateShiftImage(im,mp[0],mp[1],mp[2])
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
finalImage = medianStack(shiftedFrames)# / 3.162277 #adjust for OD 0.5 difference between occulted/unocculted files
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
h5baseName = configFileName.split('.')[0]
h5FileName = h5Path+'/%s.h5'%h5baseName
<<<<<<< HEAD
h5file = tables.open_file(h5FileName,mode='w')
stackgroup = h5file.create_group(h5file.root, 'imageStack', 'Table of images, centroids, and parameters used to create a final stacked image')
=======
h5file = tables.openFile(h5FileName,mode='w')
stackgroup = h5file.createGroup(h5file.root, 'imageStack', 'Table of images, centroids, and parameters used to create a final stacked image')
>>>>>>> 094d0556e7c743154de499d73bfa5cbdc4f99f4f
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
<<<<<<< HEAD
paramTable = h5file.create_table(stackgroup, 'params', descriptionDict,title='Stack Parameter Table')
=======
paramTable = h5file.createTable(stackgroup, 'params', descriptionDict,title='Stack Parameter Table')
>>>>>>> 094d0556e7c743154de499d73bfa5cbdc4f99f4f
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


