'''
Author: Alex Walter
Date: Dec 3, 2014
This code makes a light curve using the photometry modules

Updated by: Sarah Steiger   Date: April 7, 2020
'''

import os
import warnings
import numpy as np
import inspect
from mkidcore.corelog import getLogger
from mkidpipeline.hdf.photontable import ObsFile
from util.readDict import readDict
from util.popup import *
from util.ObsFileSeq import ObsFileSeq
from astrometry.CentroidCalc import quickCentroi
from headers.DisplayStackHeaders import writePhotometryFile, readPhotometryFile, PSFPhotometryDataDescription, \
    aperPhotometryDataDescription, readImageStack, writeCentroidFile, readCentroidFile


def isPSFString(str_var):
    return str_var in ['point spread function', 'PSF', 'psf']


def isAperString(str_var):
    return str_var in ['aperture', 'Aperture', 'aper', 'Aper', 'aperture photometry', 'Aperture Photometry']

def aperture(startpx, startpy, radius):
    r = radius
    length = 2 * r
    height = length
    allx = xrange(startpx - int(np.ceil(length / 2.0)), startpx + int(np.floor(length / 2.0)) + 1)
    ally = xrange(startpy - int(np.ceil(height / 2.0)), startpy + int(np.floor(height / 2.0)) + 1)
    mask = np.zeros((46, 44))

    for x in allx:
        for y in ally:
            if (np.abs(x - startpx)) ** 2 + (np.abs(y - startpy)) ** 2 <= (r) ** 2 and 0 <= y and y < 46 and 0 <= x and x < 44:
                    mask[y, x] = 1.
    return mask


class LightCurve:
    def __init__(self, file='', path='', targetName=None, run=None, verbose=False, showPlot=False):
        '''
        Constructs a list of obs FileName objects from the dictionary in the path.

        Inputs:
            fileID - identifier in filename. eg centroid_fileID.h5 or ImageStack_fileID.h5
            path - path to the display stack target info
            targetName - Name of target. Assumes there's a dictionary in path called targetName.dict
                         If None, just searches for any xxx.dict file in the path and assigns targetName=xxx
            run - name of run. eg PAL2014
                  if None, assumes run is the second to last directory in the path
            verbose
            showPlot
        '''

        self.path = path
        self.file = file
        self.targetName = targetName
        self.run = run
        self.verbose = verbose
        self.showPlot = showPlot

        if self.targetName is None or not os.path.isfile(self.path + os.sep + self.targetName + '.dict'):
            # Assumes there's only one '.dict' file in the directory
            for f in os.listdir(path):
                if f.endswith(".dict"):
                    if self.verbose: print
                    'Loading params from ', path + os.sep + f
                    self.targetName = f.split('.dict')[0]
        try:
            self.params = readDict(self.path + os.sep + self.targetName + '.dict')
            self.params.readFromFile(self.path + os.sep + self.targetName + '.dict')
        except:
            printco
            "Provide a target name that leads to a dictionary in the path!"
            print
            path
            print
            "target: ", self.targetName
            raise ValueError

        if self.run is None:
            # run is assumed to be the second to last directory in the path
            self.run = os.path.basename(os.path.dirname(self.path))

        # Get images
        try:
            self.loadImageStack()
        except:
            print
            "Need to make image stack for this object"
        # Get Centroids
        try:
            self.loadAllCentroidFiles()
        except:
            print
            "Need to make centroids for this object"

    def viewLightCurve(self, photometryFilename='', photometryType=None):
        if photometryFilename is '':
            if isPSFString(photometryType):
                photometryFilename = self.path + os.sep + 'FittedStacks' + os.sep + 'FittedStack_' + self.fileID + '.h5'
            elif isAperString(photometryType):
                photometryFilename = self.path + os.sep + 'ApertureStacks' + os.sep + 'ApertureStack_' + self.fileID + '.h5'
            else:
                print
                'You must provide a valid photometry type if you want to automatically generate the photometry filename'
                raise ValueError

        headerDict, dataDict = readPhotometryFile(photometryFilename)
        photometryType = headerDict['photometryType']
        # if isPSFString(photometryType):
        #    pop = PSF_Popup()
        print
        "implement this!"

    def loadLightCurve(self, photometryFilename='', photometryType=None):
        if photometryFilename is '':
            if isPSFString(photometryType):
                photometryFilename = self.path + os.sep + 'FittedStacks' + os.sep + 'FittedStack_' + self.fileID + '.h5'
            elif isAperString(photometryType):
                photometryFilename = self.path + os.sep + 'ApertureStacks' + os.sep + 'ApertureStack_' + self.fileID + '.h5'
            else:
                print
                'You must provide a valid photometry type if you want to automatically generate the photometry filename'
                raise ValueError

        self.photometry_params, self.photometry_dict = readPhotometryFile(photometryFilename)
        self.photometryType = self.photometry_params['photometryType']

    def makeLightCurve(self, photometryType, photometryFilename='', **photometryKwargs):
        '''
        Loops through each image and performs photometry on it.
        Saves the results in the file specified (photometryFilename='' saves in default location).

        Inputs:
            photometryType - Can be 'PSF' or 'aperture.' See isPSFString() and isAperString()
            photometryFilename - full path to save photometry file.
                                 If None then don't save.
                                 If empty string then save in self.path with self.fileID
            photometryKwargs - Provide parameters for the photometry operation.
                               You can give a single value for a parameter to be used on each image. ie., aper_radius=5
                               Or you can give a list of values (same length as number of images) to have a different value for that parameter for each image. ie., aper_radius=[5,4,...,5]

        Returns:
            photometryData - Dictionary where each key maps to a list of values corresponding to the value for each image
                           - valid keywords are the intersection of keywords from whatever the photometry module returns and the corresponding description in headers.DisplayStackHeaders
                                ie. for 'PSF', check PSFphotometry.PSFfit() in PSFphotometry.py and PSFPhotometryDataDescription() in DisplayStackHeaders.py

        Warning!
            If you need to pass a photometry parameter that is an array, and it happens that you have the
            same number of images as the length of the array, then the code won't know if it is supposed
            to pass the whole array with each image or one element of the array with each image. To avoid
            this, pass a list of arrays for the images.
            ie., param=np.tile(array, (NFrames,1))
        '''
        if not hasattr(self, 'im_dict'):
            print
            "Need to make image stack for this object"
            return
        if not hasattr(self, 'centroids'):
            print
            "Need to make centroids for this object"
            return

        # Parameters needed for photometry class __init__()
        images = self.im_dict['images']
        assert len(images) > 0
        expTimes = self.im_dict['pixIntTimes']
        # centroids = zip(self.centroid_dict['xPositions'], self.centroid_dict['yPositions'])
        # flags = self.centroid_dict['flag']
        flags = self.flags
        # Specific parameters for photometry operation
        if isPSFString(photometryType):
            photometryParamKeys = set(inspect.getargspec(PSFphotometry.PSFfit)[0])
            photometryDataKeys = PSFPhotometryDataDescription(1, 1).keys()
        elif isAperString(photometryType):
            photometryParamKeys = set(inspect.getargspec(AperPhotometry.AperPhotometry)[0])
            photometryDataKeys = aperPhotometryDataDescription(1).keys()
        photometryParamKeys = list(photometryParamKeys.intersection(
            photometryKwargs.keys()))  # unique keywords given in **photometryKwargs that can be passed to photometry module

        # First add all the keywords that stay the same with each image
        photometryParamDict = {}
        for keyword in np.copy(photometryParamKeys):
            val = photometryKwargs[keyword]
            try:
                assert type(val) is not str  # make sure it's not a single string
                val = list(val)  # make sure it can be cast to a list
                if len(val) != len(images):
                    warnings.warn(
                        "Be careful passing an array as a parameter. You may want to pass a list of arrays, one for each image.",
                        UserWarning)
                assert len(val) == len(images)  # make sure the list is the right length
            except:
                photometryParamDict[keyword] = val
                photometryParamKeys.remove(keyword)

        # Grab the rest of the keywords for the first image
        for keyword in photometryParamKeys:
            photometryParamDict[keyword] = photometryKwargs[keyword][0]
        # Perform photometry on the first image.
        fluxDict = self.performPhotometry(photometryType, images[0], self.centroids[0], expTimes[0],
                                          **photometryParamDict)
        if flags[0] > 0:
            # fluxDict['flux']=np.asarray(fluxDict['flux'])*0.       # Force flux --> if centroid flag is set
            fluxDict['flag'] = flags[0]  # Force photometry flag set if centroid flag was set
        # Check for unexpected dictionary keys
        if not set(fluxDict.keys()).issubset(photometryDataKeys):
            warnings.warn("The following keys returned by the photometry module dictionary " +
                          "don't match the keys expected by writePhotometryFile() and won't " +
                          "be saved: " + str(set(fluxDict.keys()).difference(photometryDataKeys)), UserWarning)

        # initialize dictionary of listed values for return data
        photometryData = {}
        for keyword in set(fluxDict.keys()).intersection(photometryDataKeys):
            if len(images) > 1:
                photometryData[keyword] = [fluxDict[keyword]]
            else:
                photometryData[keyword] = fluxDict[keyword]
        # Loop through the rest of the images
        for i in range(1, len(images)):
            for keyword in photometryParamKeys:
                photometryParamDict[keyword] = photometryKwargs[keyword][i]
            fluxDict = self.performPhotometry(photometryType, images[i], self.centroids[i], expTimes[i],
                                              **photometryParamDict)
            if flags[i] > 0:
                # fluxDict['flux']=np.asarray(fluxDict['flux'])*0.       # Force flux --> if centroid flag is set
                fluxDict['flag'] = flags[i]  # Force photometry flag set if centroid flag was set
            for keyword in set(fluxDict.keys()).intersection(photometryDataKeys):
                photometryData[keyword].append(fluxDict[keyword])

        if photometryFilename is not None:
            if photometryFilename is '':
                if isPSFString(photometryType):
                    photometryFilename = self.path + os.sep + 'FittedStacks' + os.sep + 'FittedStack_' + self.fileID + '.h5'
                elif isAperString(photometryType):
                    photometryFilename = self.path + os.sep + 'ApertureStacks' + os.sep + 'ApertureStack_' + self.fileID + '.h5'
                else:
                    print
                    'Choose a valid photometry type!'
                    raise ValueError
            writePhotometryFile(photometryFilename, photometryType, targetName=self.targetName, run=self.run,
                                maxExposureTime=self.im_params['maxExposureTime'],
                                imageStackFilename=self.imageStackFilename,
                                centroidFilenames=self.centroidFilenames, startTimes=self.im_dict['startTimes'],
                                endTimes=self.im_dict['endTimes'], intTimes=self.im_dict['intTimes'], **photometryData)

        if self.showPlot:
            self.viewLightCurve(photometryType, photometryFilename)
        return photometryData

    def performPhotometry(self, photometryType, image, centroid, expTime=None, **photometryKwargs):
        '''
        Perform the photometry on an image.

        Input:
            photometryType - Can be 'PSF' or 'aperture.' See isPSFString() and isAperString()
            image - 2D image of data (0 for dead pixel, shouldn't be any nan's or infs)
                        Should be fully calibrated, dead time corrected, and scaled up to the effective integration time
            centroid - list of (col,row) tuples. The first tuple is the target location. The next are reference stars in the field
            expTime - 2D array of pixel exposure times (0 for dead pixels)
                        optional. But can be used for distinguishing 0 count pixels from bad pixels
            photometryKwargs - For photometry module

        Output:
            fluxDict - dictionary output from photometry module. Should contain keyword 'flux'
                     - 'flux:' array of flux values. [target_flux, ref0_flux, ...]
        '''

        if isPSFString(photometryType):
            PSFphoto = PSFphotometry(image, centroid, expTime, verbose=self.verbose, showPlot=self.showPlot)
            fluxDict = PSFphoto.PSFfit(**photometryKwargs)
            del PSFphoto
        elif isAperString(photometryType):
            aperPhoto = AperPhotometry(image, centroid, expTime, verbose=self.verbose, showPlot=self.showPlot)
            fluxDict = aperPhoto.AperPhotometry(**photometryKwargs)
            del aperPhoto
        else:
            getLogger(__name__).info('Invalid Photometry Type Specified')
            raise ValueError

        return fluxDict

    def makeImageStack(self, imageStackFilename='', dt=30, wvlStart=None, wvlStop=None,
                       weighted=True, fluxWeighted=False, getRawCount=False,
                       scaleByEffInt=True, deadTime=100.e-6, filterName=None):
        '''
        This function makes an image stack using the ObsFileSeq class

        Inputs:
            imageStackFilename - full path of file.
                               - An empty string means use the default location using self.path and self.fileID
            dt - the maximum number of seconds for one frame
            keywords for image
        '''
        tsl = []
        for day_i in range(len(self.params['utcDates'])):
            for tstamp in self.params['obsTimes'][day_i]:
                tsl.append(self.params['utcDates'][day_i] + '-' + tstamp)
        # tsl=tsl[:1]
        ofs = ObsFileSeq(name=self.targetName, run=self.run, date=self.params['sunsetDates'][0], timeStamps=tsl, dt=dt)
        if imageStackFilename is None or imageStackFilename is '': imageStackFilename = self.path + os.sep + 'ImageStacks' + os.sep + 'ImageStack_' + self.fileID + '.h5'
        self.im_params, self.im_dict = ofs.loadImageStack(imageStackFilename, wvlStart=wvlStart, wvlStop=wvlStop,
                                                          weighted=weighted, fluxWeighted=fluxWeighted,
                                                          getRawCount=getRawCount,
                                                          scaleByEffInt=scaleByEffInt, deadTime=deadTime,
                                                          filterName=filterName)
        self.imageStackFilename = imageStackFilename
        # return self.im_params, self.im_dict

    def printImageStackInfo(self, dt=30):
        tsl = []
        for day_i in range(len(self.params['utcDates'])):
            for tstamp in self.params['obsTimes'][day_i]:
                tsl.append(self.params['utcDates'][day_i] + '-' + tstamp)
        # tsl=tsl[:2]
        ofs = ObsFileSeq(name=self.targetName, run=self.run, date=self.params['sunsetDates'][0], timeStamps=tsl, dt=dt)
        for iframe in range(len(ofs.frameObsInfos)):
            # if iframe < 300 or iframe > 310:
            #    continue
            frameNum = "%04.d" % (iframe,)
            if len(ofs.frameObsInfos[iframe]) < 1:
                print
                frameNum
            for obsDesc in ofs.frameObsInfos[iframe]:
                print
                frameNum, 'obs:', obsDesc['iObs'], 'firstSec:', obsDesc['firstSec'], 'intTime:', obsDesc[
                    'integrationTime'], 'fn:', obsDesc['obs'].fullFileName

        # ofs.getFrameList()

    def loadImageStack(self, imageStackFilename=''):
        '''
        This function will load in a new image stack from the file specified.

        Inputs:
            imageStackFilename - full path of file.
                               - An empty string means use the default location using self.path and self.fileID
            kwargs - keywords for makeImageStack()
        '''
        if imageStackFilename is None or imageStackFilename is '': imageStackFilename = self.path + os.sep + 'ImageStacks' + os.sep + 'ImageStack_' + self.fileID + '.h5'
        self.im_params, self.im_dict = readImageStack(imageStackFilename)
        self.imageStackFilename = imageStackFilename
        # return self.im_params, self.im_dict

    def removeMostlyHotPixels(self, maxTimeHot=0.80):
        try:
            num_images = len(self.im_dict['images'])
            assert num_images > 0
        except:
            print
            "Need to load image stack for this object"
            return

        for i in range(num_images):
            badPixels = np.where((self.im_dict['pixIntTimes'][i] < self.im_dict['intTimes'][i] * (1. - maxTimeHot)) * (
                        self.im_dict['pixIntTimes'][i] > 0.))
            nbad = np.sum((self.im_dict['pixIntTimes'][i] < self.im_dict['intTimes'][i] * (1. - maxTimeHot)) * (
                        self.im_dict['pixIntTimes'][i] > 0.))
            if nbad > 0:
                self.im_dict['pixIntTimes'][i][badPixels] = 0.
                self.im_dict['images'][i][badPixels] = 0.
                if self.verbose:
                    print
                    'Removed', nbad, 'bad pixels from image', i

    def makeAllCentroidFiles(self, centroidFilenames=[''], radiusOfSearch=[10], maxMove=[4], usePsfFit=[False]):
        centroidDir = 'CentroidLists'
        targetDir = 'Target'
        refDir = 'Reference'

        try:
            num_images = len(self.im_dict['images'])
            assert num_images > 0
            self.centroidFilenames = []
        except:
            print
            "Need to make image stack for this object"
            return

        if centroidFilenames is None or len(centroidFilenames) < 1 or centroidFilenames[0] is '':
            try:
                centroidFilenames[
                    0] = self.path + os.sep + centroidDir + os.sep + targetDir + os.sep + 'Centroid_' + self.fileID + '.h5'
            except NameError:
                centroidFilenames = [
                    self.path + os.sep + centroidDir + os.sep + targetDir + os.sep + 'Centroid_' + self.fileID + '.h5']
        for i in range(1, len(centroidFilenames)):
            ref_num_str = "%02.d" % (i - 1,)  # Format as 2 digit integer with leading zeros
            centroidFilenames[
                i] = self.path + os.sep + centroidDir + os.sep + refDir + ref_num_str + os.sep + 'Centroid_' + self.fileID + '.h5'

        centroid_list = []
        flags = np.zeros(num_images)
        for i in range(len(centroidFilenames)):
            if i == 0:
                print
                "\tSelect Target Star"
            else:
                print
                "\tSelect Reference Star #" + str(i - 1)
            try:
                radiusOfSearch_i = radiusOfSearch[i]
            except IndexError:
                radiusOfSearch_i = radiusOfSearch[0]
            except TypeError:
                radiusOfSearch_i = radiusOfSearch
            try:
                maxMove_i = maxMove[i]
            except IndexError:
                maxMove_i = maxMove[0]
            except TypeError:
                maxMove_i = maxMove
            try:
                usePsfFit_i = usePsfFit[i]
            except IndexError:
                usePsfFit_i = usePsfFit[0]
            except TypeError:
                usePsfFit_i = usePsfFit
            self.makeCentroidFile(centroidFilenames[i], radiusOfSearch_i, maxMove_i, usePsfFit_i)
            centroid_list.append(self.centroids)
            flags += self.flags

        if len(centroid_list) > 1: self.centroids = np.asarray(zip(*centroid_list))
        self.flags = 1.0 * (np.asarray(flags) > 0.)

    def loadAllCentroidFiles(self, centroidFilenames=[]):
        centroidDir = 'CentroidLists'
        targetDir = 'Target'
        refDir = 'Reference'
        try:
            num_images = len(self.im_dict['images'])
            assert num_images > 0
            self.centroidFilenames = []
        except:
            print
            "Need to load image stack before loading centroids"
            return

        if centroidFilenames is None or len(centroidFilenames) < 1:
            nStars = 0
            for file_i in os.listdir(self.path + os.sep + centroidDir):
                nStars += int(os.path.isdir(self.path + os.sep + centroidDir + os.sep + file_i))
            centroidFilenames = [''] * nStars
        if centroidFilenames[0] is '':
            try:
                centroidFilenames[
                    0] = self.path + os.sep + centroidDir + os.sep + targetDir + os.sep + 'Centroid_' + self.fileID + '.h5'
            except NameError:
                centroidFilenames = [
                    self.path + os.sep + centroidDir + os.sep + targetDir + os.sep + 'Centroid_' + self.fileID + '.h5']
        for i in range(1, len(centroidFilenames)):
            ref_num_str = "%02.d" % (i - 1,)  # Format as 2 digit integer with leading zeros
            centroidFilenames[
                i] = self.path + os.sep + centroidDir + os.sep + refDir + ref_num_str + os.sep + 'Centroid_' + self.fileID + '.h5'

        centroid_list = []
        flags = np.zeros(num_images)
        for i in range(len(centroidFilenames)):
            self.loadCentroidFile(centroidFilenames[i])
            centroid_list.append(self.centroids)
            flags += self.flags

        if len(centroid_list) > 1: self.centroids = np.asarray(zip(*centroid_list))
        self.flags = 1.0 * (np.asarray(flags) > 0.)
        self.centroidFilenames = centroidFilenames

    def makeCentroidFile(self, centroidFilename='', radiusOfSearch=10, maxMove=4, usePsfFit=False):
        '''
        This function makes a centroid file using CentroidCal.quickCentroid()

        Inputs:
            centroidFilename - full path of file.
                             - An empty string means use the default location using self.path and self.fileID
            kwargs -

        Returns:
            centroids
            flags
        '''
        centroidDir = 'CentroidLists'
        targetDir = 'Target'
        refDir = 'Reference'
        # Get images
        try:
            images = self.im_dict['images']
        except:
            print
            "Need to make image stack for this object"
            return

        # reducedImages = images/self.im_dict['intTimes']
        reducedImages = [images[i] / self.im_dict['intTimes'][i] for i in range(len(images))]
        xPositionList, yPositionList, flagList = quickCentroid(reducedImages, radiusOfSearch=radiusOfSearch,
                                                               maxMove=maxMove, usePsfFit=usePsfFit)

        if centroidFilename is None or centroidFilename is '':
            centroidFilename = self.path + os.sep + centroidDir + os.sep + targetDir + os.sep + 'Centroid_' + self.fileID + '.h5'

        centroid_params = {'targetName': self.targetName, 'run': self.run, 'nFrames': len(images),
                           'imageStackFilename': self.imageStackFilename}
        centroid_dict = {'startTimes': self.im_dict['startTimes'], 'endTimes': self.im_dict['endTimes'],
                         'intTimes': self.im_dict['intTimes'], 'xPositions': xPositionList,
                         'yPositions': yPositionList, 'flag': flagList}
        # writeCentroidFile(centroidFilename, **centroid_params,**centroid_dict)
        writeCentroidFile(centroidFilename, **dict(centroid_params.items() + centroid_dict.items()))
        try:
            self.centroidFilenames.append(centroidFilename)
        except NameError:
            self.centroidFilenames = [centroidFilename]

        self.centroids = zip(xPositionList, yPositionList)
        self.flags = flagList

    def loadCentroidFile(self, centroidFilename=''):
        centroidDir = 'CentroidLists'
        targetDir = 'Target'
        refDir = 'Reference'
        if centroidFilename is None or centroidFilename is '':
            centroidFilename = self.path + os.sep + centroidDir + os.sep + targetDir + os.sep + 'Centroid_' + self.fileID + '.h5'
        centroid_params, centroid_dict = readCentroidFile(centroidFilename)
        self.centroids = zip(centroid_dict['xPositions'], centroid_dict['yPositions'])
        self.flags = centroid_dict['flag']
        try:
            self.centroidFilenames.append(centroidFilename)
        except NameError:
            self.centroidFilenames = [centroidFilename]


class PSFphotometry:

    def __init__(self, image, centroid, expTime=None, verbose=False, showPlot=False):
        '''
        Inputs:
            image - 2D array of data (0 for dead pixel, shouldn't be any nan's or infs)
                  - Should be fully calibrated, dead time corrected, and scaled up to the effective integration time
            centroid - list of (col,row) tuples. The first tuple is the target location. The next are reference stars in the field
            expTime - 2d array of effective exposure times (same size as image)
            verbose - show error messages
            showPlot - show and pause after each PSF fit
        '''
        self.verbose = verbose
        self.showPlot = showPlot
        self.image = image
        # self.model=model

        super(PSFphotometry, self).__init__(image=np.copy(image), centroid=centroid, expTime=np.copy(expTime))
        self.image[np.invert(np.isfinite(self.image))] = 0.
        if expTime is None:
            self.expTime = np.ones(self.image.shape)
            self.expTime[np.where(self.image == 0.)] = 0.
        else:
            self.expTime[np.invert(np.isfinite(self.expTime))] = 0.

    def guess_parameters(self, model, aper_radius=9, tie_sigmas=True):
        '''
        Inputs:
            model - model used for fit. Options in util.fitFunctions
            aper_radius - double or list of doubles of the same length as self.centroid. Number of pixels around the
            star to be used in estimating parameters. -1 for the whole array.
            tie_sigmas - By default, tells mpfit to tie sigmas for multiple stars to the same value

        Outputs:
            parameter_guess - List of best guesses for fit parameters
            parameter_lowerlimit - Lower limit for each parameter to constrain fit
            parameter_upperlimit - Upper limit
            parameter_ties - Parameters to be tied should have matching numbers in their indices in this array.
                           - eg. [0,0,1,0,1] would tie parameters 3 and 5 together
                           - anything <=0 is ignored
                           - Should be same length as parameter_guess
                           - Or can return parameter_ties=None
        '''

        if model == 'multiple_2d_circ_gauss_func':
            # p[0] = background
            # p[1] = amplitude
            # p[2] = x_offset    column
            # p[3] = y_offset    row
            # p[4] = sigma
            # And so on for the 2nd and 3rd gaussians etc...
            # A+Be^-((x-xo)^2+(y-y0)^2)/2s^2 + Ce^-((x-x1)^2+(y-y1)^2)/2d^2 + ...

            bkgdPercentile = 30.0
            overallBkgd = np.percentile(self.image[np.where(np.isfinite(self.image) & (self.image > 0.))],
                                        bkgdPercentile)  # This doesn't seem to work for some reason...
            # print 'bkgd ',overallBkgd
            # overallBkgd=1.
            parameter_guess = [overallBkgd]
            parameter_lowerlimit = [0.0]
            parameter_upperlimit = [np.mean(self.image[np.where(np.isfinite(self.image) & (self.image > 0.))])]
            parameter_ties = [0.]

            for star_i in range(len(self.centroid)):
                # p_guess = [1.,self.centroid[star_i][0],self.centroid[star_i][1],0.01]

                try:
                    radius = aper_radius[star_i]
                except TypeError:
                    radius = aper_radius

                x_guess = self.centroid[star_i][0]
                x_ll = 0.
                x_ul = len(self.image[0])  # number of columns
                y_guess = self.centroid[star_i][1]
                y_ll = 0.
                y_ul = len(self.image)  # number of rows
                if radius > 0.:
                    x_ll = max(self.centroid[star_i][0] - radius, x_ll)
                    x_ul = min(self.centroid[star_i][0] + radius, x_ul)
                    y_ll = max(self.centroid[star_i][1] - radius, y_ll)
                    y_ul = min(self.centroid[star_i][1] + radius, y_ul)

                pixLoc = np.where(np.isfinite(self.image))
                if radius > 0.:
                    x_arr = np.tile(range(len(self.image[0])), (len(self.image), 1))
                    y_arr = np.tile(range(len(self.image)), (len(self.image[0]), 1)).transpose()
                    d_arr = np.sqrt((x_arr - x_guess) ** 2 + (y_arr - y_guess) ** 2)

                    pixLoc = np.where(np.isfinite(self.image) * d_arr <= radius)

                amp_guess = np.amax(self.image[pixLoc]) - overallBkgd
                # amp_guess = 10000.
                # amp_ul = amp_guess + 5.*np.sqrt(amp_guess)
                # amp_ul = 2.*amp_guess
                amp_ul = None
                amp_ll = max(amp_guess - 3. * np.sqrt(amp_guess), 0.)
                # amp_ll = amp_guess/2.
                amp_ll = 0.

                sig_guess = 1.8
                sig_ll = 0.3
                sig_ul = 2.5
                if radius > 0. and sig_ul > radius: sig_ul = radius

                p_guess = [amp_guess, x_guess, y_guess, sig_guess]
                p_ll = [amp_ll, x_ll, y_ll, sig_ll]
                p_ul = [amp_ul, x_ul, y_ul, sig_ul]
                parameter_guess += p_guess
                parameter_lowerlimit += (p_ll)
                parameter_upperlimit += (p_ul)
                if tie_sigmas == True:
                    parameter_ties += [0., 0., 0., 1]
                else:
                    parameter_ties += [0., 0., 0., 0.]

        # print_guesses(parameter_guess, parameter_lowerlimit, parameter_upperlimit, parameter_guess)
        return parameter_guess, parameter_lowerlimit, parameter_upperlimit, parameter_ties

    def PSFfit(self, model='multiple_2d_circ_gauss_func', aper_radius=-1, tie_sigmas=True):
        '''
        Inputs:
            model - model used for fit. Options in util.fitFunctions
            aper_radius - double or list of doubles of the same length as self.centroid. Number of pixels around the star to be used in estimating parameters. -1 for the whole array.
            tie_sigmas - By default, tells mpfit to tie sigmas for multiple stars to the same value

        Returns: Dictionary with keywords
            flux - array of flux values. [target_flux, ref0_flux, ...]
            parameters - parameters used for fit
            mpperr - error on parameters from mpfit
            redChi2 - reduced chi^2 of fit
            flag - flag indicating if fit failed. 0 means success
            model - model used to fit. Just returns the string you inputted
        '''

        intTime = np.max(self.expTime)
        if intTime > 0:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", 'invalid value encountered in divide', RuntimeWarning)
                warnings.filterwarnings("ignore", 'divide by zero encountered in divide', RuntimeWarning)
                errs = np.sqrt(self.image * intTime / self.expTime)
            errs[np.where(self.image == 0.)] = 1.5
            errs[np.where(self.expTime <= 0.)] = np.inf
        else:
            errs = np.sqrt(self.image)
            errs[np.where(self.image == 0.)] = np.inf

        parameter_guess, parameter_lowerlimit, parameter_upperlimit, parameter_ties = self.guess_parameters(model,
                                                                                                            aper_radius=aper_radius,
                                                                                                            tie_sigmas=tie_sigmas)

        models = model_list[model](parameter_guess)(p=np.ones(len(parameter_guess)), data=self.image,
                                                    return_models=True)
        guess = models[0]
        for m in models[1:]:
            guess += m
        if self.showPlot:
            pop(plotFunc=lambda popupOb: plot3DImage(popupOb.fig, popupOb.axes, self.image, errs=errs, fit=guess),
                title="Guess")

        # p_guess = np.ones(len(parameter_guess))
        # p_ll = np.asarray(parameter_lowerlimit,dtype=np.float)/np.asarray(parameter_guess,dtype=np.float)
        # p_ul = np.asarray(parameter_upperlimit,dtype=np.float)/np.asarray(parameter_guess,dtype=np.float)
        parameter_fit, redchi2gauss2, mpperr = fitData2D(np.copy(self.image), np.copy(errs), parameter_guess,
                                                         parameter_lowerlimit, parameter_upperlimit, model,
                                                         parameter_ties=parameter_ties, verbose=self.verbose)

        models2 = model_list[model](parameter_fit)(p=np.ones(len(parameter_fit)), data=self.image, return_models=True)
        fitModelImg = models2[0]
        for m in models2[1:]:
            fitModelImg += m
        if self.showPlot:
            pop(plotFunc=lambda popupOb: plot3DImage(popupOb.fig, popupOb.axes, self.image, errs=errs, fit=fitModelImg),
                title="Fitted")

        flag = 2.0 * self.fitHitLimit(parameter_fit, parameter_guess, parameter_lowerlimit, parameter_upperlimit)
        # return self.getFlux(parameter_fit)
        return {'flux': self.getFlux(model, parameter_fit), 'parameters': parameter_fit, 'perrors': mpperr,
                'redChi2': redchi2gauss2, 'flag': flag, 'model': model}

    def fitHitLimit(self, parameter_fit, parameter_guess, parameter_lowerlimit, parameter_upperlimit):
        """
            This function just checks if the fit is railed against one of its parameter limits

            Returns:
                True if fit has a parameter that hit its upper or lower limit --> Bad fit
                False otherwise
        """
        fixed_guess = (parameter_lowerlimit == parameter_upperlimit) * (
                    parameter_lowerlimit != np.asarray([None] * len(parameter_lowerlimit)))

        s1 = np.sum((parameter_fit == parameter_lowerlimit) * (np.logical_not(fixed_guess)))
        s2 = np.sum((parameter_fit == parameter_upperlimit) * (np.logical_not(fixed_guess)))
        s3 = np.sum((parameter_fit == parameter_guess) * (np.logical_not(fixed_guess)))
        if s1 > 0 or s2 > 0 or s3 > 0:
            return True
        else:
            return False

    def getFlux(self, model, parameter_fit):
        '''
        Inputs:
            model - model used for fit. Options in util.fitFunctions
            parameter_fit - parameters of fit to calculate flux from
        Returns:
            flux - array of flux values. [target_flux, ref0_flux, ...]
        '''
        flux = []
        if model == 'multiple_2d_circ_gauss_func':
            for i in range(len(parameter_fit[1:]) / 4):
                star_flux = 2. * np.pi * parameter_fit[1 + 4 * i] * (parameter_fit[4 + 4 * i]) ** 2
                flux.append(star_flux)

        return np.asarray(flux)


class AperPhotometry:

    def __init__(self, image, centroid, expTime=None, save_plot=False):
        '''
        Inputs:
            image - 2D array of data (0 for dead pixel, shouldn't be any nan's or infs)
                  - Should be fully calibrated, dead time corrected, and scaled up to the effective integration time
            centroid - list of (col,row) tuples. The first tuple is the target location. The next are reference stars in the field
            expTime - 2d array of effective exposure times (same size as image)
            verbose - show error messages
            showPlot - show and pause after each frame
        '''
        self.save_plot = save_plot
        self.image = image
        self.centroid = centroid
        super(AperPhotometry, self).__init__(image=image, centroid=centroid, expTime=expTime)

    def AperPhotometry(self, aper_radius=5, sky_sub="median", annulus_inner=10, annulus_outer=15,
                       interpolation="linear"):
        '''

        :param aper_radius: double or list of doubles of the same length as self.centroid. Number of pixels around the star to be used in aperture
        :param sky_sub: indicates type of sky subtraction to be performed.
                      "median" - to use median sky value in annulus, only parameter that is currently supported
                      "fit" - to mask objects and estimate sky in aperture with polynomial fit
        :param annulus_inner: double or list of doubles of same length as self.centroid. Gives radius of inner part of sky annulus for use in "median" sky sub. Ignored if sky_sub is not "median" [target_apertureRad, ref0_apertureRad, ...]
        :param annulus_outer: double or list of doubles of same length as self.centroid. Gives radius of outer part of sky annulus for use in "median" sky sub. Ignored if sky_sub is not "median"
        :param interpolation: not currently supported
        :return: Dictionary with keywords. These keywords should be the same as in aperPhotometryDataDescription in headers.DisplayStackHeaders
                flux - array of flux values. [target_flux, target_sky_flux, ref0_flux, ...]
                skyFlux - array of sky flux values, scaled to same n_pix as object flux. [target_sky, ref0_sky, ...]
                apertureRad - same as input
                annulusInnerRad - same as input
                annulusOuterRad - same as input
                interpolation - same as input
        '''
        flux = np.zeros(len(self.centroid))
        sky = np.zeros(len(self.centroid))
        # if sky fitting is selected for sky subtraction, do masking and produce polynomial sky image
        if sky_sub == "fit":
            getLogger(__name__).warning("Sky fitting not currently supported")
            use_image = self.image
        if interpolation != None:
            getLogger(__name__).warning('Interpolation not currently supported')
            use_image = self.image
        else:
            use_image = self.image

        # step through each star in centroid list: [target, ref0, ref1, ...]
        for star_i, centroid in enumerate(self.centroid):
            try:  # check if different aperture radii are set for each star
                radius = aper_radius[star_i]
            except TypeError:
                radius = aper_radius
            try:  # check if different annulus radii are set for each star
                ann_in = annulus_inner[star_i]
            except TypeError:
                ann_in = annulus_inner
            try:  # check if different annulus radii are set for each star
                ann_out = annulus_outer[star_i]
            except TypeError:
                ann_out = annulus_outer

            objectFlux, nObjPix = self.get_aperture_counts(use_image, radius, self.centroid[star_i])
            skyFlux, nSkyPix = self.get_annulus_counts(use_image, ann_in, ann_out, self.centroid[star_i])
            flux[star_i] = objectFlux
            sky[star_i] = skyFlux * (float(nObjPix) / float(nSkyPix))
        return {'flux': flux, 'skyFlux': sky, 'apertureRad': np.array(aper_radius),
                'annulusInnerRad': np.array(annulus_inner), 'annulusOuterRad': np.array(annulus_outer),
                'interpolation': interpolation}

    def get_aperture_counts(self, im, radius, center):
        startpx = int(np.round(center[0]))
        startpy = int(np.round(center[1]))
        apertureMask = aperture(startpx, startpy, radius)
        nanMask = np.isnan(im)
        im[nanMask] = 0.0  # set to finite value that will be ignored
        aperturePixels = np.array(np.where(np.logical_and(apertureMask == 1, nanMask == False)))
        nApPix = aperturePixels.shape[1]
        apertureCounts = np.sum(im[aperturePixels[0], aperturePixels[1]])
        getLogger(__name__).debug('Aperture Counts = {}').format(apertureCounts)
        getLogger(__name__).debug('Number of Aperture Pixels = {}').format(nApPix)
        return [apertureCounts, nApPix]

    def get_annulus_counts(self, im, annulusInner, annulusOuter, center):
        startpx = int(np.round(center[0]))
        startpy = int(np.round(center[1]))
        innerMask = aperture(startpx, startpy, annulusInner)
        outerMask = aperture(startpx, startpy, annulusOuter)
        annulusMask = outerMask - innerMask
        nanMask = np.isnan(im)
        annulusPixels = np.array(np.where(np.logical_and(annulusMask == 1, nanMask == False)))
        nAnnPix = annulusPixels.shape[1]
        annulusCounts = np.nanmedian(im[annulusPixels[0], annulusPixels[1]]) * nAnnPix
        getLogger(__name__).debug('Annulus Counts = {}').format(annulusCounts)
        getLogger(__name__).debug('Number of Annulus Pixels = {}').format(nAnnPix)
        return [annulusCounts, nAnnPix]

    # TODO add functionality to fit the sky background behind the aperture