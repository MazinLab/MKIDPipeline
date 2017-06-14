import matplotlib                                                                                                                                
#matplotlib.use('tkAgg')
import ObsFile
import numpy as np
from util.FileName import FileName
from util.utils import interpolateImage
import matplotlib.pyplot as plt 
import pickle
from util.popup import plotArray,PopUp
import commands, warnings
from scipy import ndimage
#import mpfit
import tables
secondsPerDay = 24*60*60

#def makeArray(run='PAL2014',date='20141021',timeStamp='20141022-114831'):
#    """
#
#    """
#    obsName = FileName(run=run, date=date, tstamp=timeStamp).obs()
#    obs = ObsFile.ObsFile(obsName)
#    obs.loadAllCals()
#    obsValues = obs.getPixelCountImage(weighted=True, fluxWeighted=True, scaleByEffInt=True)
#    return obsValues

def makeArray(obs,firstSec=0, integrationTime=30):
    
    """
    outputs a 2D array
    """
    obsValues = obs.getPixelCountImage(weighted=True, fluxWeighted=True, scaleByEffInt=True, firstSec=firstSec, integrationTime=integrationTime)
    return obsValues

def divideImageinFrames(obs,firstSec=0,lastSecond=None, integrationTime=30):
    """
    gets an obs files and divides it in subframes to be aligned.
    If last second is None, it subdivides the entire obs file
    """
    exptime=obs.getFromHeader('exptime')
    if lastSecond!=None:
        endSecond=lastSecond
    else:
        endSecond=exptime
    if exptime<endSecond or endSecond<firstSec:
        print "Error in divedeImageinFrames: lastSecond is bigger than exptime or lastSecond is smaller than firstSec \n"
        return
    
    else:
        frames = []
        frameStart=range(firstSec,endSecond, integrationTime)
        for sec in frameStart:
            if sec+integrationTime<=endSecond:
                frames.append(makeArray(obs,sec,integrationTime))
            else:
                intTime=endSecond-sec
                print "last frame is only %d seconds long" % (intTime)
                frames.append(makeArray(obs,sec,intTime))
    return frames

def alignObsFileToBegin(obs, run, date, timeStamp, xTolerance=3, yTolerance=3, firstSec=0, lastSec=None, upsampling=10, frameSize=1):
    '''looks at the centroid list for a given obs file and uses the centroid as a starting point for alignment
    the offset is calculated with respect to the first frame in the obs file
    '''
    f=FileName(run,date,timeStamp)
    centList=f.centroidList()
    h=tables.openFile(centList,'r')
    xCent=h.root.centroidlist.xPositions[:]
    yCent=h.root.centroidlist.yPositions[:]
    times=h.root.centroidlist.times[:]
    ###finds the integration time used to find the centroids
    intTime=times[1]-times[0]
    frames=divideImageinFrames(obs,firstSec,lastSec,intTime)
    refFrame=frames[0]['image']
    upSampledRef = upSampleIm(refFrame,upsampling)
    refImage = embedInLargerArray(upSampledRef,frameSize)
    refImage=convertNegativeValuesToNAN(refImage)
    xRefCentroid=xCent[0]
    yRefCentroid=yCent[0]
    rotAngle=calculateRotationAngle(intTime=intTime)
    centroids={}
    centroids['x']=[]
    centroids['y']=[]
    centroids['theta']=[]
    centroids['times']=times
    centroids['fileName']=centList
    centroids['xOffset']=[]
    centroids['yOffset']=[]
    ###temp test
    #fout=open('testCent.txt','a')
    for iframe, frame  in enumerate(frames[1:]):
        xCentroid=xCent[iframe+1]
        yCentroid=yCent[iframe+1]
        xOffset=xRefCentroid-xCentroid
        yOffset=yRefCentroid-yCentroid
        xLowerLimit=(xOffset-xTolerance)*upsampling
        xUpperLimit=(xOffset+xTolerance)*upsampling
        yLowerLimit=(yOffset-yTolerance)*upsampling
        yUpperLimit=(yOffset+yTolerance)*upsampling
        thetaLowerLimit=rotAngle*(iframe+1)
        thetaUpperLimit=rotAngle*(iframe+1)

        upSampledArray = upSampleIm(frame['image'],upsampling)
        paddedIm = embedInLargerArray(upSampledArray,frameSize)
        imToAlign=convertNegativeValuesToNAN(paddedIm)
        
        pGuess=[rotAngle*(iframe+1),xOffset*upsampling,yOffset*upsampling]

        pLowLimit=[thetaLowerLimit,xLowerLimit,yLowerLimit]
        pUpLimit=[thetaUpperLimit,xUpperLimit,yUpperLimit]
        pars=alignImages(refImage,imToAlign,parameterGuess=pGuess,parameterLowerLimit=pLowLimit,parameterUpperLimit=pUpLimit)
        centroids['theta'].append(pars[0])
        centroids['x'].append(pars[1]/upsampling)
        centroids['y'].append(pars[2]/upsampling)
        centroids['xOffset'].append(xOffset)
        centroids['yOffset'].append(yOffset)
        st=str(xOffset)+" "+str(yOffset)+"  "+str(np.array(pars[0]))+"  "+str(np.array(pars[1])/upSampling)+"  "+str(np.array(pars[2])/upSampling)


        #fout.write(st)
        #fout.write("\n")
        #print xOffset*upsampling, yOffset*upsampling, pars
    #fout.write("\n\n")
    return centroids

def selectObsFiles(start=114800,stop=124500,path='/Scratch/ScienceData/PAL2014/20141021/',run="PAL2014",date="20141021",date_2= "20141022"):
    """
    returns the list of time stamps in the interval between start and stop for a given run and date. The format of the
    time stamp is '20141022-114831'
    """
    stat, out = commands.getstatusoutput("ls "+ path + "| grep h5")
    obsn = out.split("\n")
    obs_ts = [f.split("-")[-1].split(".")[0] for f in obsn]
    tsl = [date_2+"-"+f for f in obs_ts if (int(f)>=start and int(f)<=stop)]
    return tsl

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
    #print "pars", p
    ###rotates and shifts the image
    rot=rotateShiftImage(imToAlign,p[0],p[1],p[2])
    #the figure of merit is calculated by multiplying each pixel in the two images and dividing by the maximum of the two pixels squared.
    #This gives a number between 0 and 1 for each pixel
    corrIm=refImage*rot/np.maximum(refImage,rot)**2

    
    #plt.figure(0)
    #plt.imshow(corrIm,origin="lower",interpolation='none')
    #plt.figure(1)
    #plt.imshow(refImage,origin="lower",interpolation='none')
    #plt.show()
    ###calculates the sum of all the pixels that are not nan and are >0 (i.e. all the valid pixels)
    corr=np.sum(corrIm[corrIm>0])
    ###calculates the number of valid pixels so that the correlation can be normalized
    nValidPixel=np.sum(corrIm>0)
    ###deviates is the quantity that mpfit will minimize. The normalized correlation is nCorr=corr/nValidPixel
    ###so deviates is 1/nCorr
    #print "corr test", corr
    deviatesValue=nValidPixel/corr
    if returnCorrelation:
        return([corrIm,nValidPixel,deviatesValue])
    ###deviates needs to be an array or mpfit won't work
    deviates=np.empty(len(corrIm))
    deviates.fill(deviatesValue)
    status=0
    #print deviates[0]
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
    print 'status', m.status, 'errmsg', m.errmsg
    mpPar=m.params
    #print 'parameters:', mpPar, "\n"
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
        print "Specify the integration interval or two obs files"
        return
    rotationAngle = 360.0*timeInterval/secondsPerDay
    return rotationAngle

if __name__ == "__main__":
    test=False
    #test=True
    tsl = selectObsFiles(start=114800,stop=124500)
    #tsl = selectObsFiles(start=114800,stop=114900)
    run = "PAL2014"
    date = "20141021"
    frameSize=1 #size of the frame of -1 to put around the image in units of biggest size image size: if the image.shape is (42,46), 1 means that 46 pixels are added on each side of the image 
    upSampling=10 #side of the pixel box each pixel is converted into
    

    if test==True:
        testAr=np.array([[1,1,1,1,1],[1,2,2,2,1],[1,2,2,2,1],[1,2,2,2,1],[1,1,1,1,1]])
    
    else:
                
        #refTimeStamp='20141022-114831'
        #obsRefName=FileName(run=run, date=date, tstamp=refTimeStamp).obs()
        #obsRef=ObsFile.ObsFile(obsRefName)
        #obsRef.loadAllCals()
        
        #alignObsFileToBegin(obsRef, run, date, refTimeStamp, xTolerance=3, yTolerance=3, firstSec=0, lastSec=None, upsampling=10, frameSize=1)
        currentPath=commands.getoutput("pwd") 
        randLen=10
        randPars=np.random.rand(randLen,2)
        for j,obsFile in enumerate(tsl):
            fOut=currentPath+"/"+obsFile+'.p'
            #fout=open(fOut,'a')
            obsRefName=FileName(run=run, date=date, tstamp=obsFile).obs()
            obsRef=ObsFile.ObsFile(obsRefName)
            obsRef.loadAllCals()
            
            params=alignObsFileToBegin(obsRef, run, date, obsFile, xTolerance=3, yTolerance=3, firstSec=0, lastSec=None, upsampling=10, frameSize=1)
            #fout.write(st)
            #fout.write("\n")
            pickle.dump(params, open(fOut,'wb'))
            #test=pickle.load(open(fOut,'rb'))
            #print test['xOffset']
        #refIm=makeArray(obsRef)
        #upSampledRef = upSampleIm(refIm['image'],upSampling)
        #refImage = embedInLargerArray(upSampledRef,frameSize)
        #refImage=convertNegativeValuesToNAN(refImage)
        #
        ##xoffset=np.concatenate(([0,81.886468487460306/10-2.0],np.random.rand(10)*7))
        ##yoffset=np.concatenate(([0,190.94513468969444/10-15.0],np.random.rand(10)*7))
        #
        ##for j,obsFile in enumerate(tsl):
        #for  
        # obsToAlignName=FileName(run=run, date=date, tstamp=obsFile).obs()
        #    obsToAlign=ObsFile.ObsFile(obsToAlignName)
        #    obsToAlign.loadAllCals()
        #    obsValues = makeArray(obsToAlign)
        #    
        #    rotAngle=calculateRotationAngle(obsRef,obsToAlign)
        #    thetaLowerLimit=rotAngle
        #    thetaUpperLimit=rotAngle

        #    xLowerLimit=-35*upSampling
        #    xUpperLimit=35*upSampling
        #    yLowerLimit=-35*upSampling
        #    yUpperLimit=35*upSampling

        #    upSampledArray = upSampleIm(obsValues['image'],upSampling)
        #    paddedIm = embedInLargerArray(upSampledArray,frameSize)
        #    imToAlign=convertNegativeValuesToNAN(paddedIm)
        #    #pGuessWeird=[3.7875,-150.,488.58987924]
        #    #corrT= correlation(pGuessWeird,refImage=refImage,imToAlign=imToAlign,returnCorrelation=True)
        #    #plt.figure(10)
        #    #plt.imshow(refImage,origin="lower",interpolation='none')
        #    #rot10=rotateShiftImage(imToAlign,pGuessWeird[0],pGuessWeird[1],pGuessWeird[2])
        #    #plt.imshow(rot10,origin="lower",interpolation='none')
        #    #plt.show()
        #    for iOffset, offset in enumerate(xoffset):
        #        pGuess=[rotAngle,(2+xoffset[iOffset])*upSampling,(15+yoffset[iOffset])*upSampling]
        #        #pGuessF=[rotAngle,0*upSampling,18*upSampling]
        #        print "guess #", iOffset, "\n", pGuess 
        #        corrT=correlation(pGuess,refImage=refImage,imToAlign=imToAlign,returnCorrelation=True)
        #        print "initial correlation", corrT[2:3]

        #        pLowLimit=[thetaLowerLimit,xLowerLimit,yLowerLimit]
        #        pUpLimit=[thetaUpperLimit,xUpperLimit,yUpperLimit]
        #        pars=alignImages(refImage,imToAlign,parameterGuess=pGuess,parameterLowerLimit=pLowLimit,parameterUpperLimit=pUpLimit)
        #        print "result #", iOffset, "\n", pars, "\n\n"

        #        #plt.figure(10)
        #        #plt.imshow(refImage,origin="lower",interpolation='none')
        #        #rot10=rotateShiftImage(imToAlign,pGuess[0],pGuess[1],pGuess[2])
        #        #plt.imshow(rot10,origin="lower",interpolation='none')
