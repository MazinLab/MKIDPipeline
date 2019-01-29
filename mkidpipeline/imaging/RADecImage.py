'''
Author: Julian van Eyken                    Date: May 15 2013

Package/class for handling of images created from photon lists that are derotated and
mapped to sky coordinates, and stacked.

(NB - NEED TO COMPLETE DOCUMENTATION OF OBJECT ATTRIBUTES)

'''


import numpy as np
# np.set_printoptions(threshold=np.inf)
import tables
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import astropy.io.fits as pyfits
import utils
import boxer
from mkidcore import pixelflags
import ephem
import astropy.time
from inspect import getframeinfo, stack


class RADecImage(object):
    '''
    Class to hold derotated, integrated and possibly stacked images, in the sky coordinate
    frame.

    NOTE - to be able to import boxer for this routine,
    boxer_old.so (exact name may vary by platform) needs to be created
    from fortran boxer.f (in the same directory) using command:

    f2py -c -m boxer boxer.f

    This command should work on any platform, I think, though the
    output file may have a different file extension. In any case,
    should all then be importable in python using 'import boxer',
    and this module should all then run fine (see also readme.txt
    in this directory).

    OBJECT ATTRIBUTES (assume read-only unless otherwise stated!):

        .nPixRA, .nPixDec   - number of virtual pixels in RA and Dec directions
        .cenRA, .cenDec     - RA and dec of center of virtual image grid (radians)
        .vPlateScale        - Arcseconds per pixel for virtual image grid
        .imageIsLoaded      - True if image data has been loaded into the object

        .image              - nPixRA x nPixDec array representing the virtual image stored
                                (NOT weighted by exposure time!)
        .effIntTimes        - nPixRA x nPixDec array of total effective exposure times for each
                                virtual image pixel (seconds)
        .expTimeWeights     - weights to apply to .image to account for effective exposure times
                                of each pixel. i.e., to get a fully exposure time corrected image,
                                use .image x .expTimeWeights. Weights pixels to correspond to the
                                total exposure time for the full coadded image.
        .gridRA, .gridDec   - 1D arrays containing the virtual pixel boundaries in the RA and dec
                                directions.
        .totExpTime         - Scalar, total exposure time for the current image (seconds)
    '''

    def __init__(self, ditherDict, nPixRA=None, nPixDec=None, cenRA=None, cenDec=None,
                 # vPlateScale=0.1, detPlateScale=0.44, firstSec=0, integrationTime=-1,
                 vPlateScale=1, detPlateScale=1, firstDitherSec=0,
                 dithIntTime=-1, doWeighted=True, wvlMin=None, wvlMax=None,
                 maxBadPixTimeFrac=0.5, savePreStackImage=None):
        # expWeightTimeStep=1.0):
        '''
        Initialise a (possibly empty) RA-dec coordinate frame image.

        INPUTS:
            nPixRA, nPixDec: integers, Number of pixels in RA and Dec directions
                        for the virtual image.
            cenRA, cenDec: floats, location of center of virtual image in RA and
                        dec (both in radians)
            vPlateScale: float, plate scale for virtual image (arcseconds per
                        virtual image pixel). Note that the attribute, self.vPlateScale
                        is stored in *radians* per pixel, as is self.detPlateScale (plate
                        scale for the detector pixels).
            detPlateScale: override the assumed detector plate scale (arcseconds
                        per detector pixel)
            firstSec: float, time from beginning of photon-list file at which to
                        begin integration
            integrationTime: float, length of time to integrate for in seconds. If
                        -1, integrate to end of photon list.
            doWeighted,wvlMin,wvlMax,maxBadPixTimeFrac,savePreStackImage - passed on to loadImage() if
                        'ObsFile' keyword is provided. Otherwise ignored.
            #### REMOVED #####
            expWeightTimeStep: float, time step to use when calculating exposure
                        time weights for the virtual pixels (seconds).
            #####################
        '''

        self.nPixRA = nPixRA  # No. of virtual pixels in RA direction
        self.nPixDec = nPixDec  # No. of virtual pixels in dec. direction
        self.cenRA = cenRA  # RA location of center of field (radians)
        self.cenDec = cenDec  # Dec location of center of field (rad)
        self.vPlateScale = vPlateScale * 2 * np.pi / 1296000  # No. of radians on sky per virtual pixel.
        self.imageIsLoaded = False  # Flag to indicate whether an actual image has been loaded yet.
        self.detPlateScale = detPlateScale * 2 * np.pi / 1296000
        self.plateScale = 0.44
        if nPixRA is not None and nPixDec is not None:
            self.image = np.empty((self.nPixDec, self.nPixRA),
                                  dtype=float)  # To take a (possibly stacked) image in virtual
            self.image.fill(np.nan)
            self.effIntTimes = np.empty_like(self.image)  # Effective integration times for each pixel, in seconds.
            self.effIntTimes.fill(np.nan)
            self.expTimeWeights = np.empty((self.nPixDec, self.nPixRA),
                                           dtype=float)  # Weights for each pixel in the virtual image to account for effective integration time on each pixel.
            self.expTimeWeights.fill(np.nan)
            self.gridRA = np.empty((self.nPixRA), dtype=float)  # Virtual pixel boundaries in the RA direction
            self.gridRA.fill(np.nan)
            self.gridDec = np.empty((self.nPixDec), dtype=float)  # Virtual pixel boundaries in the dec. direction.
            self.gridDec.fill(np.nan)
            self.totExpTime = np.nan  # Total exposure time included in current image
            # self.expWeightTimeStep = expWeightTimeStep
        else:
            self.image = None
            self.effIntTimes = None
            self.expTimeWeights = None
            self.gridRA = None
            self.gridDec = None
            self.totExpTime = None
            # self.expWeightTimeStep = expWeightTimeStep
        if (cenRA is not None and cenDec is not None and vPlateScale is not None
                and nPixRA is not None and nPixDec is not None):
            self.setCoordGrid()
        # if ObsFile is not None:
        #     self.loadImage(ObsFile, firstSec=firstSec, integrationTime=integrationTime,
        #                    doWeighted=doWeighted, wvlMin=wvlMin, wvlMax=wvlMax,
        #                    maxBadPixTimeFrac=maxBadPixTimeFrac, savePreStackImage=savePreStackImage)

        # This was previously read from the dither log file. Lets try using cenRA and Dec instead
        self.centroidRightAscension = cenRA
        self.centroidDeclination = cenDec

        self.times = ditherDict['startTimes']
        self.xCentroids = ditherDict['xPixOff'] + 73
        self.yCentroids = ditherDict['yPixOff'] + 73
        self.centroidFlags = np.zeros_like(self.times)

        # self.hourAngles = np.ones_like(self.times)*np.pi/3#np.array([0,0])
        # print ephem.hours(self.centroidRightAscension).real
        self.hourAngles = self.calcHAs(self.times, ephem.hours(self.centroidRightAscension).real)

        # print self.hourAngles, 'hourangles'

        # # Account for 0.5 offset in centroiding algorithm
        # self.xCentroids = self.xCentroids - 0.5
        # self.yCentroids = self.yCentroids - 0.5

        # Note - changed rotation direction here - JvE 6/5/2013
        self.rotationMatrix = np.array([[np.cos(self.hourAngles), -np.sin(self.hourAngles)],
                                        [np.sin(self.hourAngles), np.cos(self.hourAngles)]]).T
        self.centroidRotated = np.dot(self.rotationMatrix, np.array([self.xCentroids, self.yCentroids])).diagonal(
            axis1=0, axis2=2)

        # plt.imshow(self.centroidRotated)
        # plt.show()

        # And start figuring out the exposure time weights....
        print('Calculating effective exposure times')
        # First find start/end times of each timestep ('frame') for calculating effective exp. times
        # Use the same timesteps as used in calculating the astrometry.
        self.firstDitherSec = firstDitherSec

        if dithIntTime != -1:
            self.lastDithSec = ditherDict['relStartTimes'][0]+dithIntTime
        else:
            self.lastDithSec = ditherDict['endTimes'][-1]

        withinIntegration = ((ditherDict['relStartTimes'] < self.lastDithSec) & (ditherDict['relEndTimes'] > self.firstDitherSec))
        self.tStartFrames = ditherDict['relStartTimes'][withinIntegration].clip(
            min=self.firstDitherSec)  # Now clip so that everything is within the requested integration time.
        self.tEndFrames = ditherDict['relEndTimes'][withinIntegration].clip(max=self.lastDithSec)
        self.nFrames = len(self.tStartFrames)
        assert self.nFrames > 0  # Otherwise we have a problem....
        assert np.all(self.tStartFrames <= self.lastDithSec) and np.all(self.tEndFrames >= self.firstDitherSec)

        # Get x,y locations of detector pixel corners (2D array of each x,y value, in detector space)
        # Assume definition where integer values represent location of pixel center.

    def loadObsFile(self, ObsFile, firstObsTime=0, integrationTime=-1, ditherInd=0, wvlMin=None, wvlMax=None,
                 doWeighted=True,
                  maxBadPixTimeFrac=0.5):  # savePreStackImage is sort of temporary for test purposes
        '''
        load photons for a given effective exposure - typically a dither position or orientation on the sky

        Old intro - needs looking over
        Build a de-rotated stacked image from a photon list (ObsFile) object.
        If the RADecImage instance already contains an image, the new image is added to it.

        INPUTS:
            ObsFile - a ObsFile object from which to construct the image.

            firstSec - time from start of exposure to start the 'integration' for the image (seconds)

            integrationTime - duration of integration time to include in the image (in seconds; -1 or NaN => to end of exposure)

            wvlMin, wvlMax - min and max wavelengths of photons to include in the image (Angstroms).

            doStack - boolean; if True, then stack the image to be loaded on top of any image data already present.

            wvlMin, wvlMax - set min and max wavelength cutoffs for photons to be loaded in.

            savePreStackImage - temporary fudge, set to a file-name to save the image out to a file prior to stacking.

            doWeighted - if True, includes flat and flux weighting (i.e. flatfielding and spectral response) factors from photons,
                                and rejects photons from pixels where the flatfield is bad at any wavelength within the requested
                                wavelength range (all if wvlMin/wvl Max not specified).
                                ****NOTE - FLUX WEIGHTING NOT FULLY TESTED -- but looks probably okay.****

            maxBadPixTimeFrac - Maximum fraction of time for which a pixel is allowed to be flagged as hot (or otherwise bad)
                                before it is written off as bad for the entire duration of the requested integration time.
        '''



        photTable = ObsFile.file.root.Photons.PhotonTable  # Shortcut to table

        # img = ObsFile.getPixelCountImage(firstSec =0, integrationTime=1)
        # image = img['image']
        # plt.imshow(image, aspect='equal')
        # plt.show()

        # if expWeightTimeStep is not None:
        #    self.expWeightTimeStep=expWeightTimeStep

        if ObsFile.getFromHeader('timeMaskExists'):
            # If hot pixels time-mask data not already parsed in (presumably not), then parse it.
            if ObsFile.hotPixTimeMask is None:
                ObsFile.parseHotPixTimeMask()  # Loads time mask dictionary into ObsFile.hotPixTimeMask

        if wvlMin is not None and wvlMax is None: wvlMax = np.inf
        if wvlMin is None and wvlMax is not None: wvlMin = 0.0

        # Figure out last second of integration
        obsFileExpTime = ObsFile.header.cols.expTime[0]
        if integrationTime == -1 or firstObsTime + integrationTime > obsFileExpTime:
            lastObsTime = obsFileExpTime
        else:
            lastObsTime = firstObsTime + integrationTime

        # print lastObsTime, integrationTime, obsFileExpTime,
        lastObsTime*=1e6 # convert to microseconds. Change the variable name maybe?
        # print lastObsTime

        # If virtual coordinate grid is not yet defined, figure it out.
        if self.gridRA is None or self.gridDec is None:
            # Find RA/dec range needed, taking advantage of the fact that the ra/dec columns are (or should be) indexed....
            print('Finding RA/dec ranges')
            self.raMin = photTable.cols.ra[photTable.colindexes['ra'][0]]
            self.raMax = photTable.cols.ra[photTable.colindexes['ra'][-1]]
            self.decMin = photTable.cols.dec[photTable.colindexes['dec'][0]]
            self.decMax = photTable.cols.dec[photTable.colindexes['dec'][-1]]
            self.cenRA = (self.raMin + self.raMax) / 2.0
            self.cenDec = (self.decMin + self.decMax) / 2.0
            # Set size of virtual grid to accommodate.
            if self.nPixRA is None:
                # +1 for round up; +1 because coordinates are the boundaries of the virtual pixels, not the centers.
                self.nPixRA = int((self.raMax - self.raMin) // self.vPlateScale + 2)
            if self.nPixDec is None:
                self.nPixDec = int((self.decMax - self.decMin) // self.vPlateScale + 2)
            self.setCoordGrid()

        beamFlagImage = np.transpose(ObsFile.beamFlagImage.read())
        self.nDPixRow, self.nDPixCol = beamFlagImage.shape

        # nVPixRA, nVPixDec = self.nPixRA, self.nPixDec

        # # Calculate ratio of virtual pixel area to detector pixel area
        # vdPixAreaRatio = (self.vPlateScale / self.detPlateScale) ** 2

        # Make a boolean mask of dead (non functioning for whatever reason) pixels
        # True (1) = good; False (0) = dead
        # First on the basis of the wavelength cals:

        if ObsFile.getFromHeader('isWvlCalibrated'):
            # wvlCalFlagImage = ObsFile.getBadWvlCalFlags()
            print('This needs to be updated. No flags loaded')
            wvlCalFlagImage = np.zeros_like(beamFlagImage)
        else:
            wvlCalFlagImage = np.zeros_like(beamFlagImage)

        deadPixMask = np.where(wvlCalFlagImage == pixelflags.speccal, 1, 0)  # 1.0 where flag is good; 0.0 otherwise. (Straight boolean mask would work, but not guaranteed for Python 4....)
        print('# Dead detector pixels to reject on basis of wavelength cal: ', np.sum(deadPixMask == 0))

        # Next a mask on the basis of the flat cals (or all ones if weighting not requested)
        if doWeighted:
            flatCalFlagArray = ObsFile.file.root.flatcal.flags.read()  # 3D array - nRow * nCol * nWavelength Bins.
            flatWvlBinEdges = ObsFile.file.root.flatcal.wavelengthBins.read()  # 1D array of wavelength bin edges for the flat cal.
            lowerEdges = flatWvlBinEdges[0:-1]
            upperEdges = flatWvlBinEdges[1:]
            if wvlMin is None and wvlMax is None:
                inRange = np.ones(len(lowerEdges), dtype=bool)  # (all bins in range implies all True)
            else:
                inRange = ((lowerEdges >= wvlMin) & (lowerEdges < wvlMax) |
                           (upperEdges >= wvlMin) & (
                                       lowerEdges < wvlMax))  ####SOMETHING NOT RIGHT HERE? DELETE IF NO ASSERTION ERROR THROWN BELOW!##########
                # Bug fix - I think this is totally equivalent - first term above is redundant, included in second term:
                inRangeOld = np.copy(inRange)  # Can delete if no assertion error thrown below
                inRange = (upperEdges >= wvlMin) & (lowerEdges < wvlMax)
                assert np.all(inRange == inRangeOld)  # Can delete once satisfied this works.
                # If this never complains, then can switch to the second form.

            flatCalMask = np.where(np.all(flatCalFlagArray[:, :, inRange] == False, axis=2), 1,
                                   0)  # Should be zero where any pixel has a bad flag at any wavelength within the requested range; one otherwise. Spot checked, seems to work.
            print('# Detector pixels to reject on basis of flatcals: ', np.sum(flatCalMask == 0))
        else:
            flatCalMask = np.ones((self.nDPixRow, self.nDPixCol))

        # And now a mask based on how much hot pixel behaviour each pixel exhibits:
        # if a given pixel is bad more than a fraction maxBadTimeFrac of the time,
        # then write it off as permanently bad for the duration of the requested
        # integration.
        if maxBadPixTimeFrac is not None:
            print('Rejecting pixels with more than ', 100 * maxBadPixTimeFrac, '% bad-flagged time')
            detGoodIntTimes = ObsFile.hotPixTimeMask.getEffIntTimeImage(firstSec=firstObsTime,
                                                                         integrationTime=lastObsTime - firstObsTime)
            badPixMask = np.where(detGoodIntTimes / (lastObsTime - firstObsTime) > (1. - maxBadPixTimeFrac), 1,
                                  0)  # Again, 1 if okay, 0 bad. Use lastObsTime-firstObsTime instead of integrationTime in case integrationTime is -1.
            print('# pixels to reject: ', np.sum(badPixMask == 0))
            print('# pixels to reject with eff. int. time > 0: ', np.sum((badPixMask == 0) & (detGoodIntTimes > 0)))
        else:
            badPixMask = np.ones((self.nDPixRow, self.nDPixCol))

        # Finally combine all the masks together into one detector pixel mask:
        detPixMask = deadPixMask * flatCalMask * badPixMask  # Combine masks
        print('Total detector pixels to reject: ', np.sum(
            detPixMask), "(may not equal sum of the above since theres overlap!)")

        # Get array of effective exposure times for each detector pixel based on the hot pixel time mask
        # Multiply by the bad pixel mask and the flatcal mask so that non-functioning pixels have zero exposure time.
        # Flatten the array in the same way as the previous arrays (1D array, nRow*nCol elements).
        # detExpTimes = (hp.getEffIntTimeImage(ObsFile.hotPixTimeMask, integrationTime=tEndFrames[iFrame]-tStartFrames[iFrame],
        #                                     firstObsTime=tStartFrames[iFrame]) * detPixMask).flatten()
        if ObsFile.getFromHeader('timeMaskExists'):
            self.detExpTimes = (ObsFile.hotPixTimeMask.getEffIntTimeImage(firstSec=self.tStartFrames[ditherInd],
                                                                      integrationTime=self.tEndFrames[ditherInd] - self.tStartFrames[
                                                                          ditherInd]) * detPixMask).flatten()
        else:
            self.detExpTimes =None

        # Now get the photons
        print('Getting photon coords')
        print('wvlMin, wvlMax: ', wvlMin, wvlMax)
        if wvlMin is None:
            assert wvlMin is None and wvlMax is None
            print('getting all wavelengths')
            # tic = time.clock()
            photons = photTable.read_where('(Time>=firstObsTime) & (Time<=lastObsTime)')
            # print 'v1 time taken (s): ', time.clock()-tic
            # tic = time.clock()
            # photons = np.array([row.fetch_all_fields() for row in photTable.where('(arrivalTime>=firstObsTime) & (arrivalTime<=lastObsTime)')])
            # photIndices = photTable.getWhereList('(arrivalTime>=firstObsTime) & (arrivalTime<=lastObsTime)')
            # print 'v2 time taken (s): ', time.clock()-tic
            # print 'Doing by second method'
            # tic = time.clock()
            # photons2 = [x for x in photons.iterrows() if (x['arrivalTime']>=firstObsTime) and (x['arrivalTime']<=lastObsTime)]
            # print 'Time taken (s): ',time.clock()-tic
        else:
            assert wvlMin is not None and wvlMax is not None
            print('(trimming wavelength range) ')
            photons = photTable.read_where(
                '(Time>=firstObsTime) & (Time<=lastObsTime) & (Wavelength>=wvlMin) & (Wavelength<=wvlMax)')

        # And filter out photons to be masked out on the basis of the detector pixel mask
        print('Finding photons in masked detector pixels...')
        whereBad = np.where(detPixMask == 0)

        # badXY = pl.xyPack(whereBad[0],
        #                   whereBad[1])  # Array of packed x-y values for bad pixels (CHECK X,Y THE RIGHT WAY ROUND!)
        xyPackMult = 100
        badXY = xyPackMult * whereBad[0] + whereBad[1]

        # allPhotXY = photons['xyPix']  # Array of packed x-y values for all photons
        allPhotXY = []
        for row in np.arange(self.nDPixRow):
            for col in range(self.nDPixCol):
                allPhotXY.append(xyPackMult*row + col)
        # Get a boolean array indicating photons whose packed x-y coordinate value is in the 'bad' list.
        toReject = np.where(np.in1d(allPhotXY, badXY))[
            0]  # [0] to take index array out of the returned 1-element tuple.
        # Chuck out the bad photons
        print('Rejecting photons from bad pixels...')
        # photons = np.delete(photons, toReject)
        #########################################################################

        self.photWeights = None
        if ObsFile.getFromHeader('isFlatCalibrated') and ObsFile.getFromHeader('isSpecCalibrated'):
            print('INCLUDING FLUX WEIGHTS!')
            self.photWeights = photons['flatWeight'] * photons['fluxWeight']  # ********EXPERIMENTING WITH ADDING FLUX WEIGHT - NOT FULLY TESTED, BUT SEEMS OKAY....********

        n_photons = len(photons)
        times = np.zeros(n_photons)
        icols = np.zeros(n_photons)
        irows = np.zeros(n_photons)
        photWavelengths = np.zeros(n_photons)
        # Use for loop for now. Photons is a np.array of np.void values so hard to index across that axis...
        for p, photon in enumerate(photons):
            # print p, photon
            times[p] = photon[1]
            icols[p], irows[p] = np.where(ObsFile.beamImage==photon[0])
            # resIDInd = np.where(resIDList == ObsFile.beamImage[xCoord, yCoord])[0]
            # icols[p], irows[p] = pl.xyUnpack(photon[0])
            photWavelengths[p] = photon[2]

        if wvlMin is not None or wvlMax is not None:
            assert all(photWavelengths >= wvlMin) and all(photWavelengths <= wvlMax)
        print('Min, max photon wavelengths found: ', np.min(photWavelengths), np.max(photWavelengths))

        reformat_photons = [times, icols, irows, photWavelengths]

        return reformat_photons

    def calcHAs(self, times, RAs, observatory='greenwich'):
        # RA is a function of xPos and yPos too right?
        # **** This needs to be verified!!! ****

        if observatory == 'Mauna Kea':
            longitude = 155.0903 #
        else:
            longitude = 'greenwich'

        LSTs = astropy.time.Time(val=times, format='unix').sidereal_time('mean', longitude).radian
        # print times, RAs, longitude, LSTs
        LHAs = LSTs - RAs

        return LHAs

    def getRaDec(self, timestamp, xPhotonPixel, yPhotonPixel, ditherInd, show=False):
        self.pixelCount = self.nDPixCol * self.nDPixRow
        self.values = np.zeros((2, self.pixelCount))
        for i in range(self.pixelCount):
            self.values[0, i] = i / self.nDPixRow
            self.values[1, i] = i % self.nDPixRow
        self.rotatedValues = np.dot(self.rotationMatrix, self.values)

        self.timestamp = np.array(timestamp)
        self.xPhotonPixel = np.array(xPhotonPixel) #Don't require integer values for inputs - JvE 7/19/2013
        self.yPhotonPixel = np.array(yPhotonPixel)
        self.xyPhotonPixel = np.array([self.xPhotonPixel, self.yPhotonPixel])  # 2 x nPhotons array of x,y pairs

        self.inputLength = len(self.timestamp)

        # self.deltaTime = (self.times[1] - self.times[0])*1e6
        # self.binNumber = np.array(self.timestamp / self.deltaTime).astype('int')
        self.binNumber = np.ones_like((self.timestamp), dtype=np.int8)*ditherInd
        self.photonHourAngle = self.hourAngles[self.binNumber]

        self.indexArray = np.array(self.xPhotonPixel * self.nDPixRow + self.yPhotonPixel)

        self.xPhotonRotated = np.zeros(len(self.timestamp))
        self.yPhotonRotated = np.zeros(len(self.timestamp))

        # NEW VERSION (calculate on the fly, no look-up table, can handle non-integers)
        for iBin in np.arange(np.min(self.binNumber), np.max(self.binNumber) + 1):
            # print iBin
            inThisBin = np.where(self.binNumber == iBin)[0]  # [0] just to move array result outside tuple
            if len(inThisBin) == 0: continue
            rotatedValues = np.dot(self.rotationMatrix[iBin, :, :], self.xyPhotonPixel[:, inThisBin])
            self.xPhotonRotated[inThisBin] = rotatedValues[0, :]
            self.yPhotonRotated[inThisBin] = rotatedValues[1, :]
            self.rotatedValues = np.dot(self.rotationMatrix, self.values)

        # Use the centroid as the zero point for ra and dec offsets
        self.rightAscensionOffset = -1.0 * self.plateScale * (
                    self.xPhotonRotated - self.centroidRotated[0][self.binNumber])
        self.declinationOffset = self.plateScale * (
                    self.yPhotonRotated - self.centroidRotated[1][self.binNumber])

        # if show:
        #     thisImage, xedges, yedges = np.histogram2d(self.rightAscensionOffset, self.declinationOffset, bins=146)
        #     plt.imshow(thisImage)
        #     plt.show()
        #     plt.plot(self.rightAscensionOffset, marker='o'); plt.show()
        #     plt.plot(self.declinationOffset, marker='o'); plt.show()
        #     plt.plot(xedges, marker='o'); plt.show()
        #     plt.plot(yedges, marker='o'); plt.show()
        #     plt.plot(self.centroidRotated[0], marker='o'); plt.show()
        #     plt.plot(self.centroidRotated[1], marker='o'); plt.show()
        #     plt.plot(self.xPhotonRotated, marker='o'); plt.show()
        #     plt.plot(self.yPhotonRotated, marker='o'); plt.show()

        # Convert centroid positions in DD:MM:SS.S and HH:MM:SS.S format to radians.
        self.centroidRightAscensionRadians = ephem.hours(self.centroidRightAscension).real
        self.centroidDeclinationRadians = ephem.degrees(self.centroidDeclination).real

        # Convert centroid position radians to arcseconds.

        degreesToRadians = np.pi / 180.0
        radiansToDegrees = 180.0 / np.pi

        self.centroidDeclinationArcseconds = self.centroidDeclinationRadians * radiansToDegrees * 3600.0
        self.centroidRightAscensionArcseconds = self.centroidRightAscensionRadians * radiansToDegrees * 3600.0

        # Add the photon arcsecond offset to the centroid offset.
        self.photonDeclinationArcseconds = self.centroidDeclinationArcseconds + self.declinationOffset
        self.photonRightAscensionArcseconds = self.centroidRightAscensionArcseconds + self.rightAscensionOffset

        # Convert the photon positions from arcseconds to radians
        self.photonDeclinationRadians = (self.photonDeclinationArcseconds / 3600.0) * degreesToRadians
        self.photonRightAscensionRadians = (self.photonRightAscensionArcseconds / 3600.0) * degreesToRadians

        return self.photonRightAscensionRadians, self.photonDeclinationRadians, self.photonHourAngle

    def setCoordGrid(self):
        '''
        Establish RA and dec coordinates for pixel boundaries in the virtual pixel grid,
        given the number of pixels in each direction (self.nPixRA and self.nPixDec), the
        location of the centre of the array (self.cenRA, self.cenDec), and the plate scale
        (self.vPlateScale).
        '''
        # self.gridRA = np.empty((self.nPixDec,self.nPixRA),dtype=float)
        # self.gridRA.fill(np.nan)
        # self.gridDec = np.empty((self.nPixDec,self.nPixRA),dtype=float)
        # self.gridDec.fill(np.nan)

        # Note - +1's are because these are pixel *boundaries*, not pixel centers:
        # print self.cenRA, (np.arange(self.nPixRA + 1) - ((self.nPixRA + 1) // 2)), (np.arange(self.nPixRA + 1) - ((self.nPixRA + 1) // 2))*self.vPlateScale, 'cenRA'

        self.gridRA = self.cenRA + (self.vPlateScale * (np.arange(self.nPixRA + 1) - ((self.nPixRA + 1) // 2)))
        self.gridDec = self.cenDec + (self.vPlateScale * (np.arange(self.nPixDec + 1) - ((self.nPixDec + 1) // 2)))



    def stackExposure(self, photons, ditherInd=0, doStack=False, savePreStackImage=None):

        # photons = np.asarray(photons)

        # photons = photons.view(dtype=np.ndarray)
        print('Calculating RA/Decs...')
        photRAs, photDecs, photHAs = self.getRaDec(photons[0], photons[1], photons[2], ditherInd, show=True)
        thisImage, thisGridDec, thisGridRA = np.histogram2d(photDecs, photRAs, bins=146)
        plt.imshow(thisImage)
        plt.show()

        nPhot = len(photRAs)

        # Add uniform random dither to each photon, distributed over a square
        # area of the same size and orientation as the originating pixel at
        # the time of observation (assume RA and dec are defined at center of pixel).
        xRand = np.random.rand(nPhot) * self.detPlateScale - self.detPlateScale / 2.0
        yRand = np.random.rand(nPhot) * self.detPlateScale - self.detPlateScale / 2.0  # Not the same array!
        ditherRAs = xRand * np.cos(photHAs) - yRand * np.sin(photHAs)
        ditherDecs = yRand * np.cos(photHAs) + xRand * np.sin(photHAs)

        photRAs = photRAs + ditherRAs
        photDecs = photDecs + ditherDecs

        thisImage, thisGridDec, thisGridRA = np.histogram2d(photDecs, photRAs, bins=146)
        plt.imshow(thisImage)
        plt.show()

        # print self.gridDec, photDecs, photDecs.shape

        # plt.plot(self.gridDec, alpha=0.5)
        # plt.plot(photDecs, alpha=0.5)
        # plt.show(block=True)
        #
        # plt.plot(self.gridRA, alpha=0.5)
        # plt.plot(photRAs, alpha=0.5)
        # plt.show()

        # plt.hist(photDecs, alpha=0.5)
        # plt.show()
        # plt.hist(photRAs, alpha =0.5)
        # plt.show()

        # Make the image for this integration
        if self.photWeights:
            print('Making weighted image')
            thisImage, thisGridDec, thisGridRA = np.histogram2d(photDecs, photRAs, [self.gridDec, self.gridRA],
                                                                weights=self.photWeights)
        else:
            print('Making unweighted image')
            thisImage, thisGridDec, thisGridRA = np.histogram2d(photDecs, photRAs, [self.gridDec, self.gridRA])

        # plt.imshow(thisImage, norm= LogNorm())
        # plt.show()

        # Save the time slice images in detector coordinates if image saving is requested.
        if savePreStackImage is not None:
            saveName = 'det-' + savePreStackImage
            print('Making detector-frame image slice for diagnostics: ' + saveName)
            detImSlice = np.histogram2d(photons[1], photons[2], bins=[self.nDPixRow, self.nDPixCol])[0]
            plt.imshow(detImSlice)
            plt.show()
            plt.imsave(fname=saveName, arr=detImSlice, origin='lower',
                       cmap=plt.cm.gray, vmin=np.percentile(detImSlice, 0.5), vmax=np.percentile(detImSlice, 99.5))

        # ------------
        # Time masking
        # ------------

        dPixXmin = np.indices((self.nDPixRow, self.nDPixCol))[1] - 0.5
        dPixXmax = np.indices((self.nDPixRow, self.nDPixCol))[1] + 0.5
        dPixYmin = np.indices((self.nDPixRow, self.nDPixCol))[0] - 0.5
        dPixYmax = np.indices((self.nDPixRow, self.nDPixCol))[0] + 0.5
        dPixXminFlat = dPixXmin.flatten()  # Flattened versions of the same since getRaDec() only works on flat arrays.
        dPixXmaxFlat = dPixXmax.flatten()
        dPixYminFlat = dPixYmin.flatten()
        dPixYmaxFlat = dPixYmax.flatten()

        # # Create (1D) arrays for normalised center locations of virtual pixel grid (=index numbers, representing location of unit squares)
        # vPixRANormCen = np.arange(nVPixRA)  # np.indices(self.nPixDec,nVPixRA)[1]
        # vPixDecNormCen = np.arange(self.nPixDec)  # np.indices(self.nPixDec,nVPixRA)[0]

        # Create 1D arrays marking edges of virtual pixels (in 'normalised' space...)
        vPixRANormMin = np.arange(self.nPixRA) - 0.5
        vPixRANormMax = np.arange(self.nPixRA) + 0.5
        vPixDecNormMin = np.arange(self.nPixDec) - 0.5
        vPixDecNormMax = np.arange(self.nPixDec) + 0.5

        # Find origin of virtual array (center of virtual pixel 0,0) in RA/dec space.
        vPixOriginRA = np.mean(self.gridRA[0:2])
        vPixOriginDec = np.mean(self.gridDec[0:2])
        vPixSize = self.vPlateScale  # Short hand, Length of side of virtual pixel in radians (assume square pixels)

        # Make array to take the total exposure times for each virtual pixel at each time step
        vExpTimesStack = np.zeros((self.nPixDec, self.nPixRA, self.nFrames))
        # vExpTimesStack2 = np.zeros((self.nPixDec,self.nPixRA,nFrames))  #FOR TEST PURPOSES

        # # And one for the total exposure time at each pixel summed over all time steps
        # vExpTimes = np.zeros((self.nPixDec, self.nPixRA))

        # Array to hold list of (equal) timestamps for each pixel at each timestep
        # (just for calculating the RA/dec coordinates of the pixel corners)
        frameTimeFlat = np.zeros((self.nDPixRow * self.nDPixCol))  # Also flat array for the purposes of getRaDec()
        frameTimeFlat.fill(np.nan)

        # ------------ Loop through the time steps ----------
        # for iFrame in range(self.nFrames):
        iFrame = ditherInd
        print('Time slice: ', iFrame + 1, '/', self.nFrames)

        # Calculate detector pixel corner locations in RA/dec space (needs to be clockwise in RA/dec space! (checked, gives +ve answers).
        frameTimeFlat.fill(self.tStartFrames[iFrame])
        dPixRA1, dPixDec1, dummy = self.getRaDec(frameTimeFlat, dPixXminFlat, dPixYminFlat, ditherInd)  # dPix* should all be flat
        dPixRA2, dPixDec2, dummy = self.getRaDec(frameTimeFlat, dPixXminFlat, dPixYmaxFlat, ditherInd)
        dPixRA3, dPixDec3, dummy = self.getRaDec(frameTimeFlat, dPixXmaxFlat, dPixYmaxFlat, ditherInd)
        dPixRA4, dPixDec4, dummy = self.getRaDec(frameTimeFlat, dPixXmaxFlat, dPixYminFlat, ditherInd)

        # Reshape the flat-array results into arrays matching the detector shape.
        # Default ordering for reshape should just be the reverse of flatten().
        # (Note all this can probably be avoided by just using flat arrays throughout
        # - this is just a bit more intuitive this way at the moment).
        # dPixRA1,dPixDec1 = dPixRA1Flat.reshape(detShape),dPixDec1Flat.reshape(detShape)
        # dPixRA2,dPixDec2 = dPixRA2Flat.reshape(detShape),dPixDec2Flat.reshape(detShape)
        # dPixRA3,dPixDec3 = dPixRA3Flat.reshape(detShape),dPixDec3Flat.reshape(detShape)
        # dPixRA4,dPixDec4 = dPixRA4Flat.reshape(detShape),dPixDec4Flat.reshape(detShape)

        # Normalise to scale where virtual pixel size=1 and origin is the origin of the virtual pixel grid
        dPixNormRA1 = (dPixRA1 - vPixOriginRA) / vPixSize  # dPixNorm* should all be flat.
        dPixNormRA2 = (dPixRA2 - vPixOriginRA) / vPixSize
        dPixNormRA3 = (dPixRA3 - vPixOriginRA) / vPixSize
        dPixNormRA4 = (dPixRA4 - vPixOriginRA) / vPixSize
        dPixNormDec1 = (dPixDec1 - vPixOriginDec) / vPixSize
        dPixNormDec2 = (dPixDec2 - vPixOriginDec) / vPixSize
        dPixNormDec3 = (dPixDec3 - vPixOriginDec) / vPixSize
        dPixNormDec4 = (dPixDec4 - vPixOriginDec) / vPixSize

        # Get min and max RA/decs for each of the detector pixels
        dPixCornersRA = np.array([dPixNormRA1, dPixNormRA2, dPixNormRA3,
                                  dPixNormRA4])  # 2D array, 4 by nRow*nCol - should be clockwise, I think!
        dPixCornersDec = np.array([dPixNormDec1, dPixNormDec2, dPixNormDec3, dPixNormDec4])
        # dPixCornersRA = np.array([dPixNormRA4,dPixNormRA3,dPixNormRA2,dPixNormRA1])      #2D array, 4 by nRow*nCol - reversed, but gives -ve results, so prob. anti-clockwise....
        # dPixCornersDec = np.array([dPixNormDec4,dPixNormDec3,dPixNormDec2,dPixNormDec1])
        dPixRANormMin = dPixCornersRA.min(axis=0)  # Flat 1D array, nRow * nCol
        dPixRANormMax = dPixCornersRA.max(axis=0)
        dPixDecNormMin = dPixCornersDec.min(axis=0)
        dPixDecNormMax = dPixCornersDec.max(axis=0)



        # Loop over the detector pixels.... (should be faster than looping over virtual pixels)
        for iDPix in np.arange(self.nDPixRow * self.nDPixCol):
            # Find the pixels which are likely to be overlapping (note - could do this as a sorted search to make things faster)
            maybeOverlappingRA = \
            np.where((dPixRANormMax[iDPix] > vPixRANormMin) & (dPixRANormMin[iDPix] < vPixRANormMax))[0]
            maybeOverlappingDec = \
            np.where((dPixDecNormMax[iDPix] > vPixDecNormMin) & (dPixDecNormMin[iDPix] < vPixDecNormMax))[0]

            for overlapLocRA in maybeOverlappingRA:
                for overlapLocDec in maybeOverlappingDec:
                    # NB Boxer needs its input coordinates in *clockwise* direction; otherwise output behaviour is unspecified
                    # (though looks like it just gives -ve results. Could put an 'abs' in front of it to save bother, but
                    # not sure I'd want to guarantee that's safe)
                    overlapFrac = boxer.boxer(overlapLocDec, overlapLocRA, dPixCornersDec[:, iDPix],
                                              dPixCornersRA[:, iDPix])
                    # if ObsFile.getFromHeader('timeMaskExists'):
                    if self.detExpTimes:
                        expTimeToAdd = overlapFrac * self.detExpTimes[iDPix]
                    else:
                        expTimeToAdd = overlapFrac
                    vExpTimesStack[overlapLocDec, overlapLocRA, iFrame] += expTimeToAdd

        # ------------ End loop through time steps ----------

        # Sum up the exposure times from each frame:
        vExpTimes = np.sum(vExpTimesStack, axis=2)

        # # Check that wherever the exposure time is zero, there are no photons that have not been rejected
        # assert np.all(thisImage[vExpTimes == 0] == 0)
        # print 'Dunno why, but it passed the assertion...'

        if savePreStackImage is not None:
            print('Saving exp.time weighted pre-stacked image to ' + savePreStackImage)
            print('cmap: ', plt.cm.gray)
            imToSave = thisImage / vExpTimes
            plt.imsave(fname=savePreStackImage, arr=imToSave, origin='lower', cmap=plt.cm.gray,
                       vmin=np.percentile(imToSave, 1.0), vmax=np.percentile(imToSave, 99.0))

        plt.imshow(thisImage, norm=LogNorm())
        plt.show(block=True)

        if self.imageIsLoaded is False or doStack is False:
            self.image = thisImage  # For now, let's keep it this way.... Since weighting does odd things.
            self.effIntTimes = vExpTimes
            self.totExpTime = self.lastDithSec - self.firstDitherSec
            self.expTimeWeights = self.totExpTime / self.effIntTimes
            self.vExpTimesStack = vExpTimesStack  # TEMPORARY FOR DEBUGGING PURPOSES
            self.imageIsLoaded = True
        else:
            assert self.imageIsLoaded == True
            print('Stacking')
            self.image += thisImage
            self.effIntTimes += vExpTimes
            self.totExpTime += self.lastDithSec - self.firstDitherSec
            self.expTimeWeights = self.totExpTime / self.effIntTimes

        plt.imshow(self.image, norm= LogNorm())
        plt.show(block=True)



    def display(self, normMin=None, normMax=None, expWeight=True, pclip=None, colormap=plt.cm.hot,
                image=None, logScale=False, fileName=None, ds9=False, cbar=False, noAxis=False):
        '''
        Display the current image. Currently just a short-cut to utils.plotArray,
        but needs updating to mark RA and Dec on the axes.

        NOTE: FOR NOW, PLOTTING AGAINST RA/DEC IS SWITCHED OFF, SINCE IT LOOKS LIKE THE IMAGE FLIP/ROTATION
        ASSUMED BY CALCULATERADEC.PY MAY BE WRONG. NEEDS SORTING OUT, BUT IN THE MEANTIME, JUST SHOW THE IMAGE
        THE RIGHT WAY ROUND, AND DON'T LABEL THE AXES.

        INPUTS:
            normMin, normMax: Minimum and maximum values for the color-scale stretch
            expWeight: if True, scale each virtual pixel by its exposure time
            pclip: Apply a percentile clip to the image at the lower [pclip] and upper [100-pclip]'th
                    percentiles (where pclip is in percent). Overrides normMin and normMax.
            colorMap: as for plotArray, can specify the matplotlib color map to use.
            image: if set to a 2D array, displays that array instead of the default image.
            logScale: if True, display the intensities on a log scale.
            fileName: if a string, save the plot to this filename. If anything else (inc. None),
                      display to screen.
            ds9: if True, send image to DS9 (in which case pretty much all the other parameters
                 are ignored).
            cbar: if True, add a colour bar (note you can end up with multiple color bars if you
                    call this repeatedly though.)
            noAxis: if True, don't mark or label the axes.

        '''

        showCoordsOnAxes = False  # For now, don't try, since it looks like image flip/rotation assumed by CalculateRaDec may be wrong.

        assert np.all(self.image[self.effIntTimes == 0] == 0)

        if expWeight:
            toDisplay = np.copy(self.image * self.expTimeWeights)
        else:
            toDisplay = np.copy(self.image)

        if ds9 is True:
            utils.ds9Array(toDisplay)
            return  # Don't need to do anything else in this case.

        if logScale is True: toDisplay = np.log10(toDisplay)

        if image is not None: toDisplay = np.copy(image)

        #####################
        # Rotate by 180deg so north is up and east is left - just a fudge for now, need to
        # sort this out properly.
        if showCoordsOnAxes is False:
            toDisplay = np.rot90(np.rot90(toDisplay))
        #####################

        if pclip:
            normMin = np.percentile(toDisplay[np.isfinite(toDisplay)], q=pclip)
            normMax = np.percentile(toDisplay[np.isfinite(toDisplay)], q=100.0 - pclip)

        # Display NaNs as zeros so it looks better
        toDisplay[np.isnan(toDisplay)] = 0

        # Find the coordinates of the centers of the virtual pixels in degrees
        # raMin = (self.gridRA[0:-1] + self.gridRA[1:])/2.0 / np.pi * 180.
        # dec = (self.gridDec[0:-1] + self.gridDec[1:])/2.0 / np.pi * 180.

        # utils.plotArray(toDisplay,cbar=True,normMin=normMin,normMax=normMax,
        #                colormap=colormap,plotFileName=fileName,
        #                showMe=(type(fileName) is not str))
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
        plt.ticklabel_format(style='plain', useOffset=False)
        plt.tick_params(direction='out')

        ### Use until we get image rotation/flip sorted out properly
        if showCoordsOnAxes is True:
            origin = 'lower'
            extent = ((180. / np.pi) *
                      np.array([self.gridRA[0], self.gridRA[-1], self.gridDec[0], self.gridDec[-1]]))
        else:
            origin = 'lower'
            extent = None
        ###

        if noAxis is True: plt.axis(
            'off')  # Hopefully overrides any axis labeling that comes later. But haven't checked...

        plt.imshow(toDisplay, vmin=normMin, vmax=normMax, extent=extent,
                   origin=origin, cmap=colormap)
        # plt.ticklabel_format(style='plain',useOffset=False)

        if showCoordsOnAxes is True:
            ax = plt.gca()
            xtickBase = 10 ** (np.round(np.log10((self.gridRA[-1] - self.gridRA[0]) * 180. / np.pi / 10.)))
            ytickBase = 10 ** (np.round(np.log10((self.gridDec[-1] - self.gridDec[0]) * 180. / np.pi / 10.)))
            ax.xaxis.set_major_locator(plt.MultipleLocator(base=xtickBase))
            ax.yaxis.set_major_locator(plt.MultipleLocator(base=ytickBase))
            plt.xticks(rotation=90)
            plt.xlabel('R.A. (deg)')
            plt.ylabel('Dec. (deg)')
        if cbar is True: plt.colorbar()

        utils.showzcoord()

        # ax.xaxis.set_major_locator(plt.MultipleLocator(base=0.001))
        # ax.yaxis.set_major_locator(plt.MultipleLocator(base=0.001))
        # plt.show()

    def writeFits(self, fileName='RADecImage.fits', expWeight=True):
        '''
        Write the current image (if present) to a fits file.

        INPUTS:
            fileName - name of output file to write
            expWeight - if True, scale by the per-pixel exposure time weights.
        '''

        if self.image is None:
            raise RuntimeError('No current image to write to FITS file')

        if expWeight is True:
            hdu = pyfits.PrimaryHDU(self.image * self.expTimeWeights)
        else:
            hdu = pyfits.PrimaryHDU(self.image)

        hdu.writeto(fileName)


def test(ObsFileFileName='/Users/vaneyken/Data/UCSB/ARCONS/Palomar2012/corot18/testPhotonList-blosc.h5',
         vPlateScale=0.1, integrationTime=-1, firstSec=0):
    ObsFile = tables.openFile(ObsFileFileName, mode='r')
    try:
        im = RADecImage(ObsFile, vPlateScale=vPlateScale, firstSec=firstSec,
                        integrationTime=integrationTime)
    finally:
        print('Closing phot. list file.')
        ObsFile.close()
    im.display()
    return im
