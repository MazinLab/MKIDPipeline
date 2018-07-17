"""
Seth Meeker 2017-06-14 for DarknessPipeline
Based on utils-M82.py by Giulia Collura

Handful of utilities for making padded image canvases and aligning images
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage


secondsPerDay = 24*60*60


def medianStack(stack):
    return np.nanmedian(stack, axis=0)


def aperture(startpx,startpy,radius, nRows, nCols):
        r = radius
        length = 2*r 
        height = length
        allx = np.xrange(startpx-int(np.ceil(length/2.0)),startpx+int(np.floor(length/2.0))+1)
        ally = np.xrange(startpy-int(np.ceil(height/2.0)),startpy+int(np.floor(height/2.0))+1)
        mask=np.zeros((nRows,nCols))
        
        for x in allx:
            for y in ally:
                if (np.abs(x-startpx))**2+(np.abs(y-startpy))**2 <= (r)**2 and 0 <= y and y < nRows and 0 <= x and x < nCols:
                    mask[y,x]=1.
        return mask


def embedInLargerArray(obsArray,frameSize=1,padValue = -1):
    """
    gets a numpy array and -1-pads it. The frame size gives the dimension of the frame in units of the
    biggest dimension of the array (if obsArray.shape gives (2,4), then 4 rows of -1s will be added before and
    after the array and 4 columns of -1s will be added before and after. The final size of the array will be (10,12))
    It is padding with -1 and not 0s to distinguish the added pixels from valid pixels that have no photons. Masked pixels
    (dead or hot) are nan
    It returns a numpy array
    """
    framePixSize = int(max(obsArray.shape)*frameSize)
    paddedArray = np.pad(obsArray,framePixSize,'constant',constant_values=padValue)
    return paddedArray

def upSampleIm(obsArray,nPix):
    """
    upsamples the array so that the rotation and shift can be done to subpixel precision
    each pixel will be converted in a square of nPix*nPix
    """
    upSampled=obsArray.repeat(nPix, axis=0).repeat(nPix, axis=1)
    return upSampled

def convertNegativeValuesToNAN(obsArray):
    obsArray=obsArray.astype(float)
    obsArray[obsArray<0]=np.nan
    return obsArray

def rotateShiftImage(obsArray,degree,xShift,yShift):
    """
    rotates the image counterclockwise and shifts it in x and y
    When shifting, the pixel that exit one side of the array get in from the other side. Make sure that 
    the padding is large enough so that only -1s roll and not real pixels
    """
    ###makes sure that the shifts are integers
    xShift=int(round(xShift))
    yShift=int(round(yShift))
    rotatedIm = ndimage.rotate(obsArray,degree,order=0,cval=-1,reshape=False)
    rotatedIm=convertNegativeValuesToNAN(rotatedIm)

    xShifted = np.roll(rotatedIm,xShift,axis=1)
    rotatedShifted=np.roll(xShifted,yShift,axis=0)
    return rotatedShifted

def correlation(p,fjac=None,refImage=None,imToAlign=None, err=None,returnCorrelation=False):
    """
    this is the model for mpfit.py
    p[0]=rotation angle
    p[1]=x shift
    p[2]=y shift
    """
    #print("pars", p)
    ###rotates and shifts the image
    rot=rotateShiftImage(imToAlign,p[0],p[1],p[2])
    #the figure of merit is calculated by multiplying each pixel in the two images and dividing by the maximum of the two pixels squared.
    #This gives a number between 0 and 1 for each pixel
    corrIm=refImage*rot/np.maximum(refImage,rot)**2

    
    plt.figure(0)
    plt.imshow(corrIm,origin="lower",interpolation='none')
    plt.figure(1)
    plt.imshow(refImage,origin="lower",interpolation='none')
    #plt.show()
    ###calculates the sum of all the pixels that are not nan and are >0 (i.e. all the valid pixels)
    corr=np.sum(corrIm[corrIm>0])
    ###calculates the number of valid pixels so that the correlation can be normalized
    nValidPixel=np.sum(corrIm>0)
    ###deviates is the quantity that mpfit will minimize. The normalized correlation is nCorr=corr/nValidPixel
    ###so deviates is 1/nCorr
    #print("corr test", corr)
    deviatesValue=nValidPixel/corr
    if returnCorrelation:
        return([corrIm,nValidPixel,deviatesValue])
    ###deviates needs to be an array or mpfit won't work
    deviates=np.empty(len(corrIm))
    deviates.fill(deviatesValue)
    status=0
    #print(deviates[0])
    return([status,deviates])

def alignImages(refImage,imToAlign,parameterGuess,parameterLowerLimit,parameterUpperLimit,parameterMinStep=[0.05,2,2],model=correlation):
    """
    Runs mpfit.py
    Inputs:
        refImage: image used as a reference, 2D numpy array
        imToAlign: imgage to be aligned, 2D numpy array
        parameterGuess: initial guess for theta, xshift, yshift
        parameterLowerLimit: strict lower limit in parameter space
        parameterUpperLimit: strict upper limit in parameter space
        model: returns the value to minimize
    """
    parinfo = []
    for k in range(len(parameterGuess)):
        lowLimit=True
        upLimit=True
        parMinStep=parameterMinStep[k]
        if parameterLowerLimit[k]=='None': lowLimit=False
        if parameterUpperLimit[k]=='None': upLimit=False
        fixGuess = False
        if parameterLowerLimit[k]==parameterUpperLimit[k] and parameterLowerLimit!=None: fixGuess = True
        par={'n':k,'value':parameterGuess[k],'limits':[parameterLowerLimit[k],parameterUpperLimit[k]],'limited':[lowLimit,upLimit],'fixed':fixGuess,'step':parMinStep}
        parinfo.append(par)
    quiet=True
    fa={'refImage':refImage,'imToAlign':imToAlign}

    #with warnings.catch_warnings():
     #   warnings.simplefilter("ignore")
    #m = mpfit.mpfit(correlation, functkw=fa, parinfo=parinfo, maxiter=1000, xtol=10**(-20),ftol=10**(-15),gtol=10**(-30),quiet=quiet)
    m = mpfit.mpfit(correlation, functkw=fa, parinfo=parinfo, maxiter=1000, xtol=10**(-20),ftol=10**(-20),gtol=10**(-30),quiet=quiet)
    print('status', m.status, 'errmsg', m.errmsg)
    mpPar=m.params
    print('parameters:', mpPar, "\n")
    return mpPar

def calculateRotationAngle(refImage=None,imToAlign=None,intTime=None):
    """
    looks at the timestamps of the reference image or at the integration time and the image to align and calculates the rotation angle
    takes  either two obs files or a time interval and calculates the rotation angle either between the beginning of the 2 obs files
    or between the beginning and the end of the the integration time
    """
    if refImage==None and imToAlign==None and intTime != None:
         timeInterval=intTime
    elif refImage!=None and imToAlign!=None:
        refTimeStamp=refImage.getFromHeader('unixtime')
        imToAlignTimeStamp=imToAlign.getFromHeader('unixtime')
        timeInterval = imToAlignTimeStamp-refTimeStamp
    else: 
        print("Specify the integration interval or two obs files")
        return
    rotationAngle = 360.0*timeInterval/secondsPerDay
    return rotationAngle


if __name__ == "__main__":
    pass
