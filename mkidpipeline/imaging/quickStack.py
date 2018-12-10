"""
Original Author: Seth Meeker        Date: Jun 14, 2017

Quick routine to take a series of files from multiple dither positions,
align them, and median add them
Updated from v1 with newly re-compartmentalized image registraction code
load params from quickStack.cfg

Modification Author: Isabel Lipartito     Date: Sept 29, 2017

Modifications: searches for pre-existing dark, flat, and hpm from quickImgCal.py and uses them.  If no dark/flat/hpm found, code will make them.

"""

import glob
import os
import sys

import matplotlib
import numpy as np
import tables

#from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
matplotlib.use('Qt5agg')
import matplotlib.pyplot as plt
from mkidpipeline.utils.plottingTools import plot_array
from mkidpipeline.utils.readDict import readDict

import mkidpipeline.hotpix.badpix as gbpm

import image_registration as ir
from mkidpipeline.utils.loadStack import loadIMGStack, loadBINStack
import mkidpipeline.utils.irUtils as irUtils


def StackCalSoln_Description(nPos=1):
    strLength = 100  
    description = {
            "intTime"       : tables.UInt16Col(),          # integration time used for each dither position
            "nPos"          : tables.UInt16Col(),          # number of dither positions
            "startTimes"    : tables.UInt32Col(nPos),      # list of data start times for each position
            "stopTimes"     : tables.UInt32Col(nPos),      # physical y location
            "darkSpanImg"   : tables.UInt32Col(2),         # start and stop time for dark data for the image (object)
            "darkSpanFlat"  : tables.UInt32Col(2),         # start and stop time for dark data for the flat
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
            "refFile"       : tables.StringCol(strLength), # path to reference file for image registration
            "useImg"        : tables.BoolCol(),            # boolean flag to use .IMG or .bin data files in stacking
            "doHPM"         : tables.StringCol(strLength), # string controlling manner that hot pix masks are used
            "calibPath"     : tables.StringCol(strLength), # location of dark, flat, and hpm .npz file if one exists 
            "subtractDark"  : tables.BoolCol(),            # boolean flag to apply dark
            "divideFlat"    : tables.BoolCol(),            # boolean flag to apply flat
            "apMaskRadPrim" : tables.UInt16Col(),          # radius of aperture mask around primary object
            "apMaskRadSec"  : tables.UInt16Col(),          # radius of aperture mask around secondary
            "beammapFile"   : tables.StringCol(strLength)}  #Directory and filename of the beammap you want to use. 
    return description

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
darkSpanImg = np.array(configData['darkSpanImg'], dtype=int)
darkSpanFlat = np.array(configData['darkSpanFlat'], dtype=int)
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
calibPath = str(configData['calibPath'])
useImg = bool(configData['useImg'])
doHPM = bool(configData['doHPM'])
subtractDark = bool(configData['subtractDark'])
divideFlat = bool(configData['divideFlat'])
refFile = str(configData['refFile'])
beammapFile=str(configData['beammapFile'])

apMaskRadPrim = 300
apMaskRadSec = 300

if imgDir!='/mnt/ramdisk/':
    runDir = os.path.join(imgDir,run)
    imgPath = os.path.join(runDir,date)
else:
    imgPath=imgDir

calibPath = os.path.join(calibPath,run)
calibPath = os.path.join(calibPath,date)

#could update this to use env var $MKID_RAW_PATH
binRunDir = os.path.join(binDir,run)
binPath = os.path.join(binRunDir,date)

maxCut=1500
sigma=3

#manual hot pixel array
manHP = None

if useImg == True:
    dataDir = imgPath
    print("Loading data from .img files")
else:
    dataDir = binPath
    print("Loading data from .bin files", dataDir)

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


if doHPM:

   print("Making HPM")

   dark_stack_img = loadIMGStack(dataDir=dataDir, start=darkSpanImg[0], stop=darkSpanImg[1])
   len_dark_stack_img = int(darkSpanImg[1] - darkSpanImg[0])
   darkHPMImg = gbpm.quick_check_stack(stack=dark_stack_img, len_stack=len_dark_stack_img)
   hpFN='darkHPMImg_'+target+'.npz'  #Save the hot pixel mask into the output directory
   hpPath = os.path.join(outputDir,hpFN)
   np.savez(hpPath,darkHPMImg = darkHPMImg)
   plot_array(darkHPMImg, title='Hot Pixel Mask Image', origin='upper')

   dark_stack_flat = loadIMGStack(dataDir=dataDir, start=darkSpanFlat[0], stop=darkSpanFlat[1])
   len_dark_stack_flat = int(darkSpanFlat[1] - darkSpanFlat[0])
   darkHPMFlat = gbpm.quick_check_stack(stack=dark_stack_flat, len_stack=len_dark_stack_flat)
   hpFN='darkHPMFlat_'+target+'.npz'  #Save the hot pixel mask into the output directory
   hpPath = os.path.join(outputDir,hpFN)
   np.savez(hpPath,darkHPMFlat = darkHPMFlat)
   plot_array(darkHPMFlat, title='Hot Pixel Mask Flat', origin='upper')

else:

   print("No HPM made")
   darkHPMImg = np.zeros((numRows, numCols),dtype=int)
   darkHPMFlat = np.zeros((numRows, numCols),dtype=int)


#load dark frames

if subtractDark:

   print("Loading dark frame to make Dark and Saving It")
   if useImg == True:
      darkStackImg = loadIMGStack(dataDir, darkSpanImg[0], darkSpanImg[1], nCols=numCols, nRows=numRows, verbose=False)
   else:
      darkStackImg = loadBINStack(dataDir, darkSpanImg[0], darkSpanImg[1], nCols=numCols, nRows=numRows,verbose=False)
   darkImg = irUtils.medianStack(darkStackImg)
   plot_array(darkImg,title='DarkImg',origin='upper')
   darkImg[np.where(darkHPMImg==1)]=np.nan
   plot_array(darkImg,title='DarkImg HP Masked',origin='upper')
   darkFN='darkImg_'+target+'.npz'  #Save the dark into the output directory
   darkPath = os.path.join(outputDir,darkFN)
   np.savez(darkPath,darkImg = darkImg)

   if useImg == True:
      darkStackFlat = loadIMGStack(dataDir, darkSpanFlat[0], darkSpanFlat[1], nCols=numCols, nRows=numRows,verbose=False)
   else:
      darkStackFlat = loadBINStack(dataDir, darkSpanFlat[0], darkSpanFlat[1], nCols=numCols, nRows=numRows,verbose=False)
   darkFlat = irUtils.medianStack(darkStackFlat)
   darkFlat[np.where(darkHPMFlat==1)]=np.nan
   darkFN='darkFlat_'+target+'.npz'  #Save the dark into the output directory
   darkPath = os.path.join(outputDir,darkFN)
   np.savez(darkPath,darkFlat = darkFlat)


else:

   print("No dark made")
   darkImg = np.zeros((numRows, numCols),dtype=int)
   darkFlat = np.zeros((numRows, numCols),dtype=int)


if divideFlat:

   #load flat frames
   print("Loading flat frame to make Flat and Saving it")
   if useImg == True:
      flatStack = loadIMGStack(dataDir, flatSpan[0], flatSpan[1], nCols=numCols, nRows=numRows,verbose=False)
   else:
      flatStack = loadBINStack(dataDir, flatSpan[0], flatSpan[1], nCols=numCols, nRows=numRows,verbose=False)
   flat = irUtils.medianStack(flatStack)
   flat[np.where(darkHPMFlat==1)]=np.nan

   #dark subtract the flat
   flatSub=flat-darkFlat
   flatSub[np.where(flatSub <= 0.0)]=np.nan
   flatSub[np.where(flatSub<=np.nanmean(flatSub)-sigma*np.nanstd(flatSub))]=np.nan
   flatSub[np.where(flatSub>=np.nanmean(flatSub)+ sigma*np.nanstd(flatSub))]=np.nan

   croppedFrame = np.copy(flatSub)

   plot_array(flatSub, title='FLAT',origin='upper')
   """
   Apply SR Meeker's feedline cropping script when we calculate the flat weights
   This cropping assumes poor FL2 performance and
   poor high frequency performance, a la Faceless from 2017a,b.
   This is a good place to play around to change deviant weights
   at fringes of the array.
   """
   croppedFrame = np.copy(flatSub)

   croppedFrame[25:50,::] = np.nan
   croppedFrame[::,:20] = np.nan
   croppedFrame[::,59:] = np.nan
   
   #Apply Neelay's beammap script which loads in the most recent beammap and removes out of bound pixels from the    
   #median calculation
   resID, flag, xCoords, yCoords = np.loadtxt(beammapFile, unpack=True)
   badInds = np.where(flag==1)[0]
   badXCoords = np.array(xCoords[badInds], dtype=np.int32)
   badYCoords = np.array(yCoords[badInds], dtype=np.int32)
   outOfBoundsMask = np.logical_or(badXCoords>=80, badYCoords>=125)
   outOfBoundsInds = np.where(outOfBoundsMask)[0]
   badXCoords = np.delete(badXCoords, outOfBoundsInds)
   badYCoords = np.delete(badYCoords, outOfBoundsInds)
   croppedFrame[badYCoords, badXCoords] = np.nan

   med = np.nanmean(croppedFrame.flatten())
   weights = med/flatSub #calculate the weights by dividing the median by the dark-subtracted flat frame

   med = np.nanmedian(weights.flatten())
   plot_array(weights,title='Weights',origin='upper') #Plot the weights
   flatFN='flat_'+target+'.npz'  #Save the dark-subtracted flat file and weights into the output directory
   flatPath =  os.path.join(outputDir,flatFN)
   np.savez(flatPath,weights = weights,flat=flatSub)


else: 
   print("No flat provided")
   flat = np.zeros((numRows, numCols),dtype=int)
   weights = np.zeros((numRows, numCols),dtype=int)

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
print("Shortest Integration time = ", intTime)

#load dithered science frames
ditherFrames = []
for i in range(nPos):
    #load stack for entire data set to be saved to H5, even though only intTime number of frames will
    #be used from each position
    if useImg == True:
        stack = loadIMGStack(dataDir, startTimes[i], stopTimes[i]-1, nCols=numCols, nRows=numRows,verbose=False)
    else:
        stack = loadBINStack(dataDir, startTimes[i], stopTimes[i]-1, nCols=numCols, nRows=numRows,verbose=False)
    
    for f in range(len(stack)):
        timeStamp = startTimes[i]+f
        if f==0 and i==0:
               plot_array(stack[f],title='Raw Image Dither Pos %i'%i,origin='upper',vmin=0)

        if subtractDark == True:
            darkSubImg = stack[f]-darkImg
        else:
            darkSubImg = stack[f]

        if divideFlat == True:

            processedIm = np.array(darkSubImg * weights,dtype=float)
        else:
            processedIm = np.array(darkSubImg,dtype=float)
            
        #Append the raw frames and rough dx, dy shifts to arrays for h5
        ditherNums.append(i)
        timeStamps.append(timeStamp)
        rawImgs.append(stack[f])
        roughShiftsX.append(dXs[i])
        roughShiftsY.append(dYs[i])
        centroidsX.append(refPointX-dXs[i])
        centroidsY.append(refPointY-dYs[i])

        if doHPM:
            hpm = darkHPMImg
            hpMask = hpm
            cpm = np.zeros((numRows, numCols),dtype=int)
            dpm = np.zeros((numRows, numCols),dtype=int)

        else:
            print("No hot pixel masking specified in config file")
            hpMask = np.zeros((numRows, numCols),dtype=int)
            hpm = np.zeros((numRows, numCols),dtype=int)
            cpm = np.zeros((numRows, numCols),dtype=int)
            dpm = np.zeros((numRows, numCols),dtype=int)

        #Append the individual masks used on each frame to arrays that will go in h5 
        hotPixMasks.append(hpm)
        coldPixMasks.append(cpm)
        deadPixMasks.append(dpm)
        
        #apply hp mask to image
        if doHPM:
            processedIm[np.where(hpMask==1)]=np.nan
            processedIm[np.where(processedIm>=maxCut)]=np.nan

        #cut out cold/dead pixels
        processedIm[np.where(processedIm<=coldCut)]=np.nan

        #plot an example of the masked image for inspection
        if f==0 and i==0:
            plot_array(processedIm,title='Dither Pos %i Processing Applied'%i,origin='upper',vmin=0)

        #pad frame with margin for shifting and stacking
        paddedFrame = irUtils.embedInLargerArray(processedIm, frameSize=padFraction)

        #apply rough dX and dY shift to frame
        #print("Shifting dither %i, frame %i by x=%i, y=%i"%(i,f, dXs[i], dYs[i]))
        shiftedFrame = irUtils.rotateShiftImage(paddedFrame, 0, dXs[i], dYs[i])

        #upSample frame for sub-pixel registration with fitting code
        upSampledFrame = irUtils.upSampleIm(shiftedFrame, upSample)
        #conserve flux. Break each pixel into upSample^2 sub pixels, 
        #each subpixel should have 1/(upSample^2) the flux of the original pixel
        upSampledFrame/=float(upSample*upSample) 

        #append upsampled, padded frame to array for storage into next part of code
        ditherFrames.append(upSampledFrame)

    print("Loaded dither position %i"%i)

shiftedFrames = np.array(ditherFrames)
#if fitPos==True, do second round of shifting using mpfit correlation
#using x and y pos from earlier as guess
if fitPos==True:
    reshiftedFrames=[]
    
    if refFile!=None and os.path.exists(refFile):
        refIm = readFITS(refFile)
        print("Loaded %s for fitting"%refFile)
        plot_array(refIm,title='loaded reference FITS',origin='upper',vmin=0)
    else:
        refIm=shiftedFrames[0]

    cnt=0
    for im in shiftedFrames:
        print("\n\n------------------------------\n")
        print("Fitting frame %i of %i"%(cnt,len(shiftedFrames)))
        pGuess=[0,1,1]
        pLowLimit=[-1,(dXs.min()-5)*upSample,(dYs.min()-5)*upSample]
        pUpLimit=[1,(dXs.max()+5)*upSample,(dYs.max()+5)*upSample]
        print("guess", pGuess, "limits", pLowLimit, pUpLimit)

        maskedRefIm = np.copy(refIm)
        maskedIm = np.copy(im)

        maskedRefIm[:,400:]=np.nan
        maskedRefIm[:,:290]=np.nan
        maskedRefIm[:383,:]=np.nan
        maskedRefIm[662:,:]=np.nan

        if cnt==0:

            plot_array(maskedRefIm, title='masked Reference', origin='upper')
            plot_array(maskedIm, title='masked Image to be aligned', origin='upper')

        #try using keflavich image_registation repository
        dx, dy, ex, ey = ir.chi2_shifts.chi2_shift(maskedRefIm, maskedIm, zeromean=True)#,upsample_factor='auto')
        mp = [0,-1*dx,-1*dy]

        print("fitting output: ", mp)

        newShiftedFrame = irUtils.rotateShiftImage(im, mp[0], mp[1], mp[2])
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
print('Shape',shiftedFrames.shape)

plot_array(finalImage,title='final',origin='upper')
img=plt.imshow(finalImage,cmap=plt.cm.spectral)
#cbar = plt.colorbar(format='%05.2f')
#plt.clim(0,np.nanmedian(finalImage))
#cbar.set_norm(MyNormalize(vmin=0,vmax=800,stretch='linear'))
#cbar = DraggableColorbar(cbar,img)
#cbar.connect()
plt.title('Final Image')
plt.show()

outfile=outputDir+target
np.save(outfile, finalImage)

nanMask = np.zeros(np.shape(finalImage))
nanMask[np.where(np.isnan(finalImage))]=1

h5Path = outputDir
h5baseName = configFileName.split('/')[1].split('.')[0]
h5FileName = h5Path+'/%s.h5'%h5baseName
print(h5FileName)

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
darkArrayImg = tables.Array(stackgroup,'darkImg',darkImg,title='Dark Frame Img')
darkArrayFlat = tables.Array(stackgroup,'darkFlat',darkImg,title='Dark Frame Flat')
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
entry['darkSpanImg'] = darkSpanImg
entry['darkSpanFlat'] = darkSpanFlat
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
entry['calibPath'] = calibPath
entry['beammapFile'] = beammapFile
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


