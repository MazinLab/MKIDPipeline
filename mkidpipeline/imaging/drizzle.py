'''
Author: Rupert Dodkins, Julian van Eyken            Date: Jan 2019

Reads in obsfiles for dither offsets (preferably fully reduced and calibrated) as well as a dither log file and creates
a stacked image.

This code is adapted from Julian's testImageStack from the ARCONS pipeline.
'''

import warnings
import pickle
import os
import os.path
import glob
import scipy.stats
import numpy as np
#from astropy import coordinates as coord
# import pyfits
import matplotlib.pylab as plt

plt.plot(range(5))
plt.show()
import time
import RADecImage as rdi
from utils import utils
from mkidpipeline.hdf.photontable import ObsFile
import CentroidCalc

from pprint import pprint

from inspect import getframeinfo, stack
def dprint(message):
    caller = getframeinfo(stack()[1][0])
    print("%s:%d - %s" % (caller.filename, caller.lineno, message))

def makeImageStack(fileNames='*.h5', dir=os.getenv('MKID_PROC_PATH', default="/Scratch") + 'photonLists/',
                   dithLogFilename = 'KAnd_1545626974_dither.log', detImage=False, saveFileName='stackedImage.pkl',
                   wvlMin=3500, wvlMax=12000, doWeighted=True, medCombine=False, vPlateScale=0.2,
                   nPixRA=250, nPixDec=250, maxBadPixTimeFrac=0.2, integrationTime=-1,
                   outputdir=''):

    '''
    Create an image stack
    INPUTS:
        filenames - string, list of photon-list .h5 files. Can either
                    use wildcards (e.g. 'mydirectory/*.h5') or if string
                    starts with an @, supply a text file which contains
                    a list of file names to stack. (e.g.,
                    'mydirectory/@myfilelist.txt', where myfilelist.txt
                    is a simple text file with one file name per line.)
        dir - to provide name of a directory in which to find the files
        detImage - if True, show the images in detector x,y coordinates instead
                    of transforming to RA/dec space.
        saveFileName - name of output pickle file for saving final resulting object.
        doWeighted - boolean, if True, do the image flatfield weighting.
        medCombine - experimental, if True, do a median combine of the image stack
                     instead of just adding them all.... Prob. should be implemented
                     properly at some point, just a fudge for now.
        vPlateScale - (arcsec/virtual pixel) - to set the plate scale of the virtual
                     pixels in the outputs image.
        nPixRA,nPixDec - size of virtual pixel grid in output image.
        maxBadPixTimeFrac - Maximum fraction of time which a pixel is allowed to be
                     flagged as bad (e.g., hot) for before it is written off as
                     permanently bad for the duration of a given image load (i.e., a
                     given obs file).
        integrationTime - the integration time to use from each input obs file (from
                     start of file).
    OUTPUTS:
        Returns a stacked image object, saves the same out to a pickle file, and
        (depending whether it's still set to or not) saves out the individual non-
        stacked images as it goes.
    '''

    # Get the list of filenames
    if fileNames[0] == '@':
        # (Note, actually untested, but should be more or less right...)
        files = []
        with open(fileNames[1:]) as f:
            for line in f:
                files.append(os.path.join(dir, line.strip()))
    else:
        files = glob.glob(os.path.join(dir, fileNames))


    ditherDict = loadDitherLog(dithLogFilename)
    con2pix = getCon2Pix(files[0], files[1], ditherDict, filename = dir+'con2pix.txt')
    ditherDict = getPixOff(ditherDict,con2pix)

    # Initialise empty image centered on Crab Pulsar
    virtualImage = rdi.RADecImage(nPixRA=nPixRA, nPixDec=nPixDec, vPlateScale=vPlateScale,
                                  cenRA=1.4596725441339724, cenDec=0.38422539085925933,
                                  ditherDict=ditherDict)
    imageStack = []
    # pprint(virtualImage.__dict__)

    for ix, eachFile in enumerate(files+files+files+files):
        if os.path.exists(eachFile):
            print 'Loading: ', os.path.basename(eachFile)
            # fullFileName=os.path.join(dir,eachFile)
            phList = ObsFile(eachFile)
            baseSaveName, ext = os.path.splitext(os.path.basename(eachFile))

            if detImage is True:
                imSaveName = os.path.join(outputdir, baseSaveName + 'det.tif')
                im = phList.getImageDet(wvlMin=wvlMin, wvlMax=wvlMax)
                dprint(im)
                utils.plotArray(im)
                plt.imsave(fname=imSaveName, arr=im, colormap=plt.cm.gnuplot2, origin='lower')
                if eachFile == files[0]:
                    virtualImage = im
                else:
                    virtualImage += im
            else:
                # imSaveName = os.path.join(outputdir, baseSaveName + '.tif')
                tic = time.clock()
                photons = virtualImage.loadObsFile(phList, ditherInd=ix,
                                       wvlMin=wvlMin, wvlMax=wvlMax, doWeighted=doWeighted,
                                       maxBadPixTimeFrac=maxBadPixTimeFrac, integrationTime=integrationTime)
                virtualImage.stackExposure(photons, ditherInd=ix, doStack=not medCombine, savePreStackImage=None)
                print 'Image load done. Time taken (s): ', time.clock() - tic
                imageStack.append(virtualImage.image * virtualImage.expTimeWeights)  # Only makes sense if medCombine==True, otherwise will be ignored
                # if medCombine == True:
                #     medComImage = scipy.stats.nanmedian(np.array(imageStack), axis=0)
                #     toDisplay = np.copy(medComImage)
                #     toDisplay[~np.isfinite(toDisplay)] = 0
                #     utils.plotArray(toDisplay, pclip=0.1, cbar=True, colormap=plt.cm.gray)
                # else:
                #     virtualImage.display(pclip=0.5, colormap=plt.cm.gray)
                #     medComImage = None

            plt.show()


        else:
            print 'File doesn''t exist: ', eachFile

    # Save the results.
    # Note, if median combining, 'vim' will only contain one frame. If not, medComImage will be None.
    results = {'vim': virtualImage, 'imstack': imageStack, 'medim': medComImage}

    try:
        output = open(os.path(outputdir, saveFileName), 'wb')
        pickle.dump(results, output, -1)
        output.close()

    except:
        warnings.warn('Unable to save results for some reason...')

    return results

def getCon2Pix(ObsFilename1, ObsFilename2, ditherDict, filename):
    '''Essentially a wrapper for calcCon2Pix'''
    if os.path.exists(filename):
        con2Pix = np.loadtxt(filename, delimiter=',')
        print con2Pix
        return con2Pix
    else:
        con2Pix = calcCon2Pix(ObsFilename1, ObsFilename2, ditherDict, filename)
        return con2Pix

def calcCon2Pix(ObsFilename1, ObsFilename2, ditherDict, filename):
    '''Quick and dirty implementation to get the conversation between connex values and pixel offsets'''

    ObsFile1 = ObsFile(ObsFilename1)
    ObsFile2 = ObsFile(ObsFilename2)

    # from matplotlib.colors import LogNorm
    # img1 = ObsFile1.getPixelCountImage(firstSec =0, integrationTime=1)
    # image1 = img1['image']
    # plt.imshow(image1, aspect='equal', norm= LogNorm())
    # plt.show()
    # img2 = ObsFile2.getPixelCountImage(firstSec =0, integrationTime=1)
    # image2 = img2['image']
    # plt.imshow(image2, aspect='equal', norm= LogNorm())
    # plt.show()

    # change these from hard coded
    centroid_RA = '09:26:38.7'
    centroid_DEC = '36:24:02.4'

    centroidDictList1 = CentroidCalc.centroidCalc(ObsFile1, centroid_RA, centroid_DEC, guessTime=1, integrationTime=1,
                                                 secondMaxCountsForDisplay=500)
    centroidDictList2 = CentroidCalc.centroidCalc(ObsFile2, centroid_RA, centroid_DEC, guessTime=1, integrationTime=1,
                                                 secondMaxCountsForDisplay=500)

    locPix1 = [centroidDictList1[0]['xPositionList'], centroidDictList1[0]['yPositionList']]
    locPix2 = [centroidDictList2[0]['xPositionList'], centroidDictList2[0]['yPositionList']]

    pixVec = locPix1 - locPix2

    locCon1 = [ditherDict['xPos'][0],ditherDict['yPos'][0]]
    locCon2 = [ditherDict['xPos'][1],ditherDict['yPos'][1]]

    conVec = locCon1 - locCon2

    con2pix = conVec/pixVec

    # save the conversion for next time
    np.savetxt(filename, con2pix)

    return con2pix

def loadDitherLog(fileName):

    logPath = os.getenv('MKID_PROC_PATH',default="/Scratch") + 'photonLists/'
    log = os.path.join(logPath,fileName)
    ditherDict = {}
    with open(log) as f:
        ditherDict['startTimes'] = np.float_(f.readline()[14:-3].split(','))
        ditherDict['endTimes'] = np.float_(f.readline()[12:-3].split(','))
        ditherDict['xPos'] = np.float_(f.readline()[8:-3].split(','))
        ditherDict['yPos'] = np.float_(f.readline()[8:-3].split(','))
        ditherDict['intTime'] = np.float(f.readline()[10:])
        ditherDict['nSteps'] = np.float(f.readline()[9:])

    firstSec = ditherDict['startTimes'][0]
    ditherDict['startTimes'] = ditherDict['startTimes'] - firstSec
    ditherDict['endTimes'] = ditherDict['endTimes'] - firstSec

    return ditherDict


def getPixOff(ditherDict, con2pix=None):
    ''' A function to convert the connex offset to pixel displacement'''#

    if con2pix is None:
        np.array([[-20, 1], [1,-20]])
    conPos = np.array([ditherDict['xPos'],ditherDict['yPos']])
    ditherDict['xPixOff'], ditherDict['yPixOff'] = np.int_(np.matmul(conPos.T, con2pix)).T

    # xFirst = np.min(ditherDict['xPos'])
    # yFirst = np.min(ditherDict['yPos'])
    # ditherDict['xPos'] = ditherDict['xPos'] - xFirst
    # ditherDict['yPos'] = ditherDict['yPos'] - yFirst

    return ditherDict

if __name__ == '__main__':
    makeImageStack(fileNames='*.h5', dir=os.getenv('MKID_PROC_PATH') + 'photonLists/',
                   detImage=False, saveFileName='stackedImage.pkl', wvlMin=-200,
                   wvlMax=-150, doWeighted=False, medCombine=False, vPlateScale=1,
                   nPixRA=250, nPixDec=250, maxBadPixTimeFrac=None, integrationTime=5,
                   outputdir='')