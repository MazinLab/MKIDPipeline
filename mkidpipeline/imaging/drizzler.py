"""
TODO
Spatial and Temporal drizzling

Add ephem, pyguide, astropy_helpers, drizzle to yml. these were all pip installed except PyGuide is just downloaded
from github (no installation)

Get con2pix calibration from Isabel's code and remove from here

Build form() wrapper

Better handling of savestate of photonlists

Move photonlist class functions to photontable.py

Usage
-----

python drizzle.py

Author: Rupert Dodkins, Julian van Eyken            Date: Jan 2019


This code is adapted from Julian's testImageStack from the ARCONS pipeline.
"""

import os
import numpy as np
import time
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
import ephem
import glob
from astropy import wcs
from astropy.io import fits
from astropy.coordinates import EarthLocation, Angle
import astropy
from drizzle import drizzle as stdrizzle
from mkidcore import pixelflags
from mkidpipeline.hdf.photontable import ObsFile
from mkidcore.corelog import getLogger
from mkidpipeline.config import MKIDObservingDataDescription, MKIDObservingDither

import sys
sys.path.append('/Users/dodkins/PythonProjects/MEDIS/')
from medis.Utils.plot_tools import loop_frames, quicklook_im, view_datacube, compare_images, indep_images, grid

def con2pix(xCon, yCon):
    '''
    This is temporary. Should grab this conversion from elsewhere

    :param xCon:
    :param yCon:
    :return:
    '''
    from scipy.optimize import curve_fit

    def func(x, slope, intercept):
        return x * slope + intercept

    xCon0 = -0.035
    xCon1 = 0.23
    xCon2 = 0.495

    yCon0 = -0.76
    yCon1 = -0.38
    yCon2 = 0.0
    yCon3 = 0.38

    xPos0array = np.array([125.46351537124369, 124.79156638384541])
    xPos1array = np.array([107.98640545380867, 106.53992257621843, 106.04177093203712])
    xPos2array = np.array([93.809781273378277, 93.586178673966316, 91.514837557492427, 89.872003744327927])

    yPos0array = np.array([36.537397207881689])
    yPos1array = np.array([61.297923464154792, 61.535802615842933, 61.223871938056725])
    yPos2array = np.array([88.127237564834743, 90.773675516601259, 90.851982786156569])
    yPos3array = np.array([114.66071882865981, 115.42948957872515])

    xPos0 = np.median(xPos0array)
    xPos1 = np.median(xPos1array)
    xPos2 = np.median(xPos2array)

    yPos0 = np.median(yPos0array)
    yPos1 = np.median(yPos1array)
    yPos2 = np.median(yPos2array)
    yPos3 = np.median(yPos3array)

    xPos0err = np.std(xPos0array)
    xPos1err = np.std(xPos1array)
    xPos2err = np.std(xPos2array)

    yPos0err = np.std(yPos0array)
    yPos1err = np.std(yPos1array)
    yPos2err = np.std(yPos2array)
    yPos3err = np.std(yPos3array)

    xConFit = np.array([xCon0, xCon1, xCon2])
    xPosFit = np.array([xPos0, xPos1, xPos2])
    xPoserrFit = np.array([xPos0err, xPos1err, xPos2err])

    yConFit = np.array([yCon0, yCon1, yCon2, yCon3])
    yPosFit = np.array([yPos0, yPos1, yPos2, yPos3])
    yPoserrFit = np.array([np.sqrt(yPos0array[0]), yPos1err, yPos2err, yPos3err])

    xopt, xcov = curve_fit(func, xConFit, xPosFit, sigma=xPoserrFit)
    yopt, ycov = curve_fit(func, yConFit, yPosFit, sigma=yPoserrFit)

    def con2Pix(xCon, yCon, func):
        return [func(xCon, *xopt),func(yCon, *yopt)]

    xPos, yPos = con2Pix(xCon, yCon, func)

    return [xPos, yPos]

def ditherp_2_pixel(positions):
    """ A function to convert the connex offset to pixel displacement"""
    positions = np.array(positions)
    pix = con2pix(positions[:,0], positions[:,1])
    return pix

def form(observations, *args, cfg=None, **kwargs):
    """
    Form a (possibly drizzled) image from a set of observations. Forms a simple image for single observations.

    TODO Sort out what arguments and kw arguments are needed
    """
    try:
        iter(observations)
    except:
        observations = tuple(observations)

    #Determine what we are drizzling onto
    #e.g. 4d equivalent (sky_x, sky_y, wave, time) of an rdi.RADecImage

    for obs in observations:
        if isinstance(obs, MKIDObservingDataDescription):
            #TODO formImage on a single obs per
            raise NotImplementedError
        elif isinstance(obs, MKIDObservingDither):
            #TODO form from a dither
            drizzleDither(obs, *args, **kwargs)
        else:
            #TODO do we need to enable drizzling lists of arbitrary observations?
            raise ValueError('Unknown input')


def drizzleDither(dither, *args, **kwargs):
    """Form a drizzled image from a dither"""
    drizzleddata = drizzle([ObsFile(get_h5_filename(ob)) for ob in dither], **kwargs)
    drizzleddata.save(dither.name)

def getmetafromh5():
    '''
    Helper function not properly implemented yet. Hard coded values from Trap

    :param ditherdesc:
    :return:
    '''

    # raise NotImplementedError

    cenRA = 1.463
    cenDec = 0.0951
    xpix = 140
    ypix = 146
    platescale = 0.44

    return (cenRA, cenDec, xpix, ypix, platescale)

class DitherDescription(object):
    '''
    Info on the dither

    rotate determines if the effective integrations are pupil stablised or not

    TODO implement center of rotation
    '''
    def __init__(self, mkid_observing_dither, h5dithdata, drizzleconfig, observatory='greenwich', rotate=False, plotdithlocs=False):
        self.description = mkid_observing_dither
        self.pos = mkid_observing_dither.pos

        # TODO these definitions need to be redone
        self.cenRA = h5dithdata[0]
        self.cenDec = h5dithdata[1]
        self.xpix = h5dithdata[2]
        self.ypix = h5dithdata[3]
        self.platescale = h5dithdata[4]

        # TODO these definitions need to be redone
        self.wvlMin = drizzleconfig[0]
        self.wvlMax = drizzleconfig[1]
        self.firstObsTime = drizzleconfig[2]
        self.integrationTime = drizzleconfig[3]

        xCentroids, yCentroids = ditherp_2_pixel(self.pos)
        self.virxPixCen, self.viryPixCen = ditherp_2_pixel([(0,0)])
        self.xCenRes, self.yCenRes = xCentroids - self.virxPixCen, yCentroids - self.viryPixCen

        if rotate:
            times = [o.start for o in mkid_observing_dither.obs]
            site = EarthLocation.of_site(observatory)
            LSTs = astropy.time.Time(val=times, format='unix').sidereal_time('mean', site.lon).radian
            # self.hourAngles = LSTs - ephem.hours(self.cenRA).real
            self.dithHAs = LSTs - LSTs[0] # in radians
        else:
            self.dithHAs = np.zeros_like((mkid_observing_dither.obs), dtype=np.float)
        getLogger(__name__).debug("HAs: %s", self.dithHAs)

        # self.rotationMatrix = np.array([[np.cos(self.dithHAs), -np.sin(self.dithHAs)],
        #                                 [np.sin(self.dithHAs), np.cos(self.dithHAs)]]).T
        #
        # # self.centroidRotated = np.dot(self.rotationMatrix,
        # #                               np.array([xCentroids, yCentroids])).diagonal(axis1=0,axis2=2)
        # self.centroidRotated = np.dot(self.rotationMatrix,
        #                               np.array([self.xCenRes, self.yCenRes])).diagonal(axis1=0,axis2=2) + [self.virxPixCen, self.viryPixCen]

        if plotdithlocs:
            # plt.plot(np.array(self.pos)[:, 0], np.array(self.pos)[:, 1])
            # plt.figure()
            plt.plot(xCentroids, yCentroids)
            plt.plot(self.virxPixCen, self.viryPixCen, marker='x')
            plt.plot(self.centroidRotated[0], self.centroidRotated[1])
            plt.show()

class Drizzler(object):
    def __init__(self, photonlists, metadata):
        #TODO Implement
        # Assume obsfiles either have their metadata or needed metadata is passed, e.g. WCS information, target info,
        # etc

        #TODO determine appropirate value from area coverage of dataset and oversampling, even longerterm there
        # the oversampling should be selected to optimize total phase coverage to extract the most resolution at a
        # desired minimum S/N

        # self.randoffset = False # apply random spatial offset to each photon
        self.nPixRA = 250
        self.nPixDec = 250
        # self.photWeights = None

        self.config = None
        self.files = photonlists

        # self.meta = metadata
        self.cenRA = metadata.cenRA
        self.cenDec = metadata.cenDec
        self.vPlateScale = metadata.platescale
        self.detPlateScale = metadata.platescale

        self.vPlateScale = self.vPlateScale * 2 * np.pi / 1296000  # No. of radians on sky per virtual pixel.
        self.detPlateScale = self.detPlateScale * 2 * np.pi / 1296000

        print('Finding RA/dec ranges')

        raMin, raMax, decMin, decMax = [], [], [], []
        for photonlist in photonlists:
            raMin.append(min(photonlist.photRARad))
            raMax.append(max(photonlist.photRARad))
            decMin.append(min(photonlist.photDecRad))
            decMax.append(max(photonlist.photDecRad))
        raMin = min(raMin)
        raMax = max(raMax)
        decMin = min(decMin)
        decMax = max(decMax)

        self.cenRA = (raMin + raMax) / 2.0
        self.cenDec = (decMin + decMax) / 2.0

        # Set size of virtual grid to accommodate.

        if self.nPixRA is None:
            # +1 for round up; +1 because coordinates are the boundaries of the virtual pixels, not the centers.
            self.nPixRA = int((raMax - raMin) // self.vPlateScale + 2)
        if self.nPixDec is None:
            self.nPixDec = int((decMax - decMin) // self.vPlateScale + 2)

        self.generate_coordinate_grid()

        self.get_header()

    def generate_coordinate_grid(self):
        """
        Establish RA and dec coordinates for pixel boundaries in the virtual pixel grid,
        given the number of pixels in each direction (self.nPixRA and self.nPixDec), the
        location of the centre of the array (self.cenRA, self.cenDec), and the plate scale
        (self.vPlateScale).
        """
        # Note - +1's are because these are pixel *boundaries*, not pixel centers:
        self.gridRA = self.cenRA + (self.vPlateScale * (np.arange(self.nPixRA + 1) - ((self.nPixRA + 1) // 2)))
        self.gridDec = self.cenDec + (self.vPlateScale * (np.arange(self.nPixDec + 1) - ((self.nPixDec + 1) // 2)))

    def get_header(self):
        #TODO implement something like this
        # w = mkidcore.buildwcs(self.nPixRA, self.nPixDec, self.vPlateScale, self.cenRA, self.cenDec)
        # TODO implement the PV distortion?
        # eg w.wcs.set_pv([(2, 1, 45.0)])

        self.w = wcs.WCS(naxis=2)
        self.w.wcs.crpix = [self.nPixRA/2., self.nPixDec/2.]
        self.w.wcs.cdelt = np.array([self.vPlateScale, self.vPlateScale])
        self.w.wcs.crval = [self.cenRA, self.cenDec]
        self.w.wcs.ctype = ["RA---AIR", "DEC--AIR"]
        self.w._naxis1 = self.nPixRA
        self.w._naxis2 = self.nPixDec

class SpectralDrizzler(Drizzler):
    """ Generate a spatially dithered fits dataacube from a set dithered dataset """
    def __init__(self, photonlists, metadata, pixfrac=1):
        self.nwvlbins = 3
        self.wvlbins = np.linspace(metadata.wvlMin, metadata.wvlMax, self.nwvlbins+1)
        print(self.wvlbins)
        Drizzler.__init__(self, photonlists, metadata)
        self.drizcube = [stdrizzle.Drizzle(outwcs=self.w, pixfrac=pixfrac)] * self.nwvlbins

    def run(self, save_file=None):
        for ix, file in enumerate(self.files):

            getLogger(__name__).debug('Processing %s', file)
            tic = time.clock()
            insci, inwcs = self.makeCube(file)
            # indep_images(insci, logAmp=True)
            getLogger(__name__).debug('Image load done. Time taken (s): %s', time.clock() - tic)
            for iw in range(self.nwvlbins):
                print(ix, iw)
                self.drizcube[iw].add_image(insci[iw], inwcs, inwht=np.int_(np.logical_not(insci[iw]==0)))

        self.cube = [d.outsci for d in self.drizcube]

        # TODO add the wavelength WCS
        # if save_file:
        #     self.driz.write(save_file)

    def makeCube(self, file):
        sample = np.vstack(( file.wavelengths, file.photDecRad, file.photRARad))
        bins = np.array([self.wvlbins, self.gridDec, self.gridRA])
        print(sample.shape)
        datacube, bins= np.histogramdd(sample.T, bins)
        print(datacube.shape)

        wavelengths, thisGridDec, thisGridRA = bins

        # quicklook_im(datacube[0], logAmp=True, vmax=10, vmin=5, show=True)

        # w = wcs.WCS(naxis=3)
        # w.wcs.crpix = [0., 0., 0.]
        # w.wcs.cdelt = np.array([wavelengths[1] - wavelengths[0],
        #                         thisGridRA[1]  - thisGridRA[0],
        #                         thisGridDec[1] - thisGridDec[0]])
        # w.wcs.crval = [wavelengths[0], thisGridRA[0], thisGridDec[0]]
        # w.wcs.ctype = ["WAVE", "RA---", "DEC-"]
        # w._naxis1 = 0
        # w._naxis2 = 250
        # w._naxis3 = 250

        w = wcs.WCS(naxis=2)
        w.wcs.crpix = [0., 0.]
        w.wcs.cdelt = np.array([thisGridRA[1]-thisGridRA[0], thisGridDec[1]-thisGridDec[0]])
        w.wcs.crval = [thisGridRA[0], thisGridDec[0]]
        w.wcs.ctype = ["RA---AIR", "DEC--AIR"]
        w._naxis1 = 250
        w._naxis2 = 250


        return datacube, w


class TemporalDrizzler(Drizzler):
    """ Generate a spatially dithered fits hypercube from a set dithered dataset """
    def __init__(self, photonlists, metadata, pixfrac=1):
        self.nwvlbins = 8
        self.timestep = 0.01 # seconds

        self.wvlbins = np.linspace(metadata.wvlMin, metadata.wvlMax, self.nwvlbins+1)
        self.ntimebins = int(metadata.integrationTime / self.timestep)
        self.timebins = np.linspace(metadata.firstObsTime,
                                    metadata.firstObsTime + metadata.integrationTime,
                                    self.ntimebins+1) * 1e6  # timestamps are in microseconds

        print(self.timebins, self.ntimebins)
        Drizzler.__init__(self, photonlists, metadata)
        self.drizhyper = [[stdrizzle.Drizzle(outwcs=self.w, pixfrac=pixfrac)] * self.nwvlbins] * self.ntimebins
        # self.drizhyper = np.empty((self.ntimebins, self.nwvlbins), dtype=object)
        # for it in range(self.ntimebins):
        #     for iw in range(self.nwvlbins):
        #         self.drizhyper[it, iw] = stdrizzle.Drizzle(outwcs=self.w, pixfrac=pixfrac)

    def run(self, save_file=None):
        tic = time.clock()
        for ix, file in enumerate(self.files):

            getLogger(__name__).debug('Processing %s', file)

            insci, inwcs = self.makeHyper(file)
            # for datacube in insci:
            #     indep_images(datacube, logAmp=True)

            for it in range(self.ntimebins):
                for iw in range(self.nwvlbins):
                    self.drizhyper[it][iw].add_image(insci[it,iw], inwcs, inwht=np.int_(np.logical_not(insci[it][iw]==0)))

        self.hypercube = [self.drizhyper[it][iw].outsci for it in range(self.ntimebins) for iw in range(self.nwvlbins)]
        getLogger(__name__).debug('Image load done. Time taken (s): %s', time.clock() - tic)
        print('Image load done. Time taken (s): %s', time.clock() - tic)
        # TODO add the wavelength WCS
        # if save_file:
        #     self.driz.write(save_file)

    def makeHyper(self, file):
        sample = np.vstack((file.timestamps, file.wavelengths, file.photDecRad, file.photRARad))
        bins = np.array([self.timebins, self.wvlbins, self.gridDec, self.gridRA])
        print(sample.shape)
        hypercube, bins = np.histogramdd(sample.T, bins)
        print(hypercube.shape)

        times, wavelengths, thisGridDec, thisGridRA = bins

        # for datacube in hypercube:
        #     print(self.timebins, times)
        #     indep_images(datacube, logAmp=True)

        w = wcs.WCS(naxis=2)
        w.wcs.crpix = [0., 0.]
        w.wcs.cdelt = np.array([thisGridRA[1]-thisGridRA[0], thisGridDec[1]-thisGridDec[0]])
        w.wcs.crval = [thisGridRA[0], thisGridDec[0]]
        w.wcs.ctype = ["RA---AIR", "DEC--AIR"]
        w._naxis1 = 250
        w._naxis2 = 250


        return hypercube, w


class SpatialDrizzler(Drizzler):
    """ Generate a spatially dithered fits image from a set dithered dataset """
    def __init__(self, photonlists, metadata, pixfrac=1):
        Drizzler.__init__(self, photonlists, metadata)
        self.driz = stdrizzle.Drizzle(outwcs=self.w, pixfrac=pixfrac)

    def run(self, save_file=None):
        # imageStack, detStack, expWeightStack, offset = [], [], [], []
        for ix, file in enumerate(self.files):

            getLogger(__name__).debug('Processing %s', file)
            tic = time.clock()
            insci, inwcs = self.makeImage(file)
            getLogger(__name__).debug('Image load done. Time taken (s): %s', time.clock() - tic)
            # imageStack.append(self.image)

            # self.driz.add_image(insci, inwcs, wt_scl = file.integrationTime, inwht=file.pixelMask)
            # self.driz.add_image(insci, inwcs, wt_scl = file.integrationTime, inwht=np.int_(np.logical_not(insci==0)))
            self.driz.add_image(insci, inwcs, inwht=np.int_(np.logical_not(insci==0)))
            # self.driz.add_fits_file(self.tempfile)

            # ret = astropy.io.fits.ImageHDU(data=self.image)
            # ret.header['imgname'] = save_file
            # # ret.header['utc'] = datetime.utcfromtimestamp(self.meta[ix].start).strftime('%Y-%m-%d %H:%M:%S')
            # ret.header['exptime'] = self.meta.total_exp_time
            # hdul = astropy.io.fits.HDUList([astropy.io.fits.PrimaryHDU(), ret])
            # offset[ix] = 'dith%i.fits' % ix
            # hdul.writeto(os.path.join(self.config.paths.out, offset[ix]))
        # plt.show()

        if save_file:
            self.driz.write(save_file)

        # # Save the results.
        # results = {'vim': self, 'imstack': imageStack}

        # TODO introduce total_exp_time variable and complete these steps
        # effective_timestamp = self.meta.inttime+self.meta.total_exp_time/2  #TODO this is only approximate
        # ret = astropy.io.fits.ImageHDU(data=self.image)
        # ret.header['imgname'] = save_file
        # ret.header['utc'] = datetime.utcfromtimestamp(effective_timestamp).strftime('%Y-%m-%d %H:%M:%S')
        # ret.header['exptime'] = self.meta.total_exp_time
        # hdul = astropy.io.fits.HDUList([astropy.io.fits.PrimaryHDU(), ret])
        # hdul.writeto(os.path.join(self.config.paths.out, save_file))

        # return results


    def makeImage(self, file):
        # nPhot = len(file.photRARad)
        #
        # if self.randoffset:
        #     # Add uniform random dither to each photon, distributed over a square
        #     # area of the same size and orientation as the originating pixel at
        #     # the time of observation (assume RA and dec are defined at center of pixel).
        #     np.random.seed(42) # so random values always same
        #     xRand = np.random.rand(nPhot) * self.detPlateScale - self.detPlateScale / 2.0
        #     yRand = np.random.rand(nPhot) * self.detPlateScale - self.detPlateScale / 2.0  # Not the same array!
        #     ditherRAs = xRand * np.cos(file.photHAs) - yRand * np.sin(file.photHAs)
        #     ditherDecs = yRand * np.cos(file.photHAs) + xRand * np.sin(file.photHAs)
        # else:
        #     ditherRAs = 0
        #     ditherDecs = 0
        #
        # photRAs = file.photRARad + ditherRAs
        # photDecs = file.photDecRad + ditherDecs

        # self.detImage, thisGridDec, thisGridRA = np.histogram2d(photDecs, photRAs, bins=[146,140])

        thisImage, thisGridDec, thisGridRA = np.histogram2d(file.photDecRad, file.photRARad, [self.gridDec, self.gridRA])
        # quicklook_im(thisImage, logAmp=True, vmax=10, vmin=5, show=False)

        w = wcs.WCS(naxis=2)
        w.wcs.crpix = [0., 0.]
        w.wcs.cdelt = np.array([thisGridRA[1]-thisGridRA[0], thisGridDec[1]-thisGridDec[0]])
        w.wcs.crval = [thisGridRA[0], thisGridDec[0]]
        w.wcs.ctype = ["RA---AIR", "DEC--AIR"]
        w._naxis1 = 250
        w._naxis2 = 250
        # header = w.to_header()
        # hdu = fits.PrimaryHDU(data = thisImage, header=header)
        # hdu.writeto(self.tempfile, clobber=True)

        return thisImage, w


class photonlist(object):
    """
    Class for the reduced photon lists and the defining parameters

    """
    def __init__(self, obsfile, ditherdesc, ditherind):
        self.wvlMin= ditherdesc.wvlMin
        self.wvlMax= ditherdesc.wvlMax
        self.doWeighted=False
        self.medCombine=False
        self.maxBadPixTimeFrac=None
        self.firstObsTime =ditherdesc.firstObsTime
        self.integrationTime=ditherdesc.integrationTime
        self.randoffset = False # apply random spatial offset to each photon

        self.timestamps, xPhotonPixels, yPhotonPixels, self.wavelengths = self.reduce_obs(obsfile, ditherdesc, ditherind)

        self.photRARad, self.photDecRad = self.get_wcs(self.timestamps, xPhotonPixels,
                                                        yPhotonPixels, ditherind,
                                                        ditherdesc)

    def reduce_obs(self, obsfile, ditherdesc, ditherind):
        '''
        This function calibrates and queries an obsfile. A lot of copy pasta

        :returns
        list of photon times, positions, wavelengths
        '''

        # photTable = obsfile.file.root.Photons.PhotonTable  # Shortcut to table
        # img = obsfile.getPixelCountImage(firstSec =0, integrationTime=1, wvlStart=self.wvlMin, wvlStop=self.wvlMax)
        # self.testimage = img['image']
        # plt.imshow(self.testimage, aspect='equal', norm=LogNorm())
        # plt.show()

        # if expWeightTimeStep is not None:
        #    self.expWeightTimeStep=expWeightTimeStep

        if obsfile.getFromHeader('timeMaskExists'):
            # If hot pixels time-mask data not already parsed in (presumably not), then parse it.
            if obsfile.hotPixTimeMask is None:
                obsfile.parseHotPixTimeMask()  # Loads time mask dictionary into ObsFile.hotPixTimeMask

        if self.wvlMin is not None and self.wvlMax is None: self.wvlMax = np.inf
        if self.wvlMin is None and self.wvlMax is not None: self.wvlMin = 0.0

        # Figure out last second of integration
        obsFileExpTime = obsfile.header.cols.expTime[0]
        if self.integrationTime == -1 or self.firstObsTime + self.integrationTime > obsFileExpTime:
            lastObsTime = obsFileExpTime
        else:
            lastObsTime = self.firstObsTime + self.integrationTime

        lastObsTime *= 1e6  # convert to microseconds

        beamFlagImage = np.transpose(obsfile.beamFlagImage.read())
        self.nDPixRow, self.nDPixCol = beamFlagImage.shape

        # Make a boolean mask of dead (non functioning for whatever reason) pixels
        # True (1) = good; False (0) = dead
        # First on the basis of the wavelength cals:

        if obsfile.getFromHeader('isWvlCalibrated'):
            # wvlCalFlagImage = ObsFile.getBadWvlCalFlags()
            print('This needs to be updated. No flags loaded')
            wvlCalFlagImage = np.zeros_like(beamFlagImage)
        else:
            wvlCalFlagImage = np.zeros_like(beamFlagImage)

        deadPixMask = np.where(wvlCalFlagImage == pixelflags.speccal, 1,
                               0)  # 1.0 where flag is good; 0.0 otherwise. (Straight boolean mask would work, but not guaranteed for Python 4....)
        print('# Dead detector pixels to reject on basis of wavelength cal: ', np.sum(deadPixMask == 0))

        # Next a mask on the basis of the flat cals (or all ones if weighting not requested)
        if self.doWeighted:
            flatCalFlagArray = obsfile.file.root.flatcal.flags.read()  # 3D array - nRow * nCol * nWavelength Bins.
            flatWvlBinEdges = obsfile.file.root.flatcal.wavelengthBins.read()  # 1D array of wavelength bin edges for the flat cal.
            lowerEdges = flatWvlBinEdges[0:-1]
            upperEdges = flatWvlBinEdges[1:]
            if self.wvlMin is None and self.wvlMax is None:
                inRange = np.ones(len(lowerEdges), dtype=bool)  # (all bins in range implies all True)
            else:
                inRange = ((lowerEdges >= self.wvlMin) & (lowerEdges < self.wvlMax) |
                           (upperEdges >= self.wvlMin) & (
                                   lowerEdges < self.wvlMax))  ####SOMETHING NOT RIGHT HERE? DELETE IF NO ASSERTION ERROR THROWN BELOW!##########
                # Bug fix - I think this is totally equivalent - first term above is redundant, included in second term:
                inRangeOld = np.copy(inRange)  # Can delete if no assertion error thrown below
                inRange = (upperEdges >= self.wvlMin) & (lowerEdges < self.wvlMax)
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
        if self.maxBadPixTimeFrac is not None:
            print('Rejecting pixels with more than ', 100 * self.maxBadPixTimeFrac, '% bad-flagged time')
            detGoodIntTimes = obsfile.hotPixTimeMask.getEffIntTimeImage(firstSec=self.firstObsTime,
                                                                        integrationTime=lastObsTime - self.firstObsTime)
            badPixMask = np.where(detGoodIntTimes / (lastObsTime - self.firstObsTime) > (1. - self.maxBadPixTimeFrac), 1,
                                  0)  # Again, 1 if okay, 0 bad. Use lastObsTime-self.firstObsTime instead of integrationTime in case integrationTime is -1.
            print('# pixels to reject: ', np.sum(badPixMask == 0))
            print('# pixels to reject with eff. int. time > 0: ', np.sum((badPixMask == 0) & (detGoodIntTimes > 0)))
        else:
            badPixMask = np.ones((self.nDPixRow, self.nDPixCol))

        # Finally combine all the masks together into one detector pixel mask:
        # detPixMask = deadPixMask * flatCalMask * badPixMask  # Combine masks
        detPixMask = np.zeros_like(deadPixMask)
        print('Total detector pixels to reject: ', np.sum(
            detPixMask), "(may not equal sum of the above since theres overlap!)")

        # Get array of effective exposure times for each detector pixel based on the hot pixel time mask
        # Multiply by the bad pixel mask and the flatcal mask so that non-functioning pixels have zero exposure time.
        # Flatten the array in the same way as the previous arrays (1D array, nRow*nCol elements).
        # detExpTimes = (hp.getEffIntTimeImage(ObsFile.hotPixTimeMask, integrationTime=tEndFrames[iFrame]-tStartFrames[iFrame],
        #                                     self.firstObsTime=tStartFrames[iFrame]) * detPixMask).flatten()
        if obsfile.getFromHeader('timeMaskExists'):
            self.detExpTimes = (obsfile.hotPixTimeMask.getEffIntTimeImage(firstSec=ditherdesc.obs[ditherind].start,
                                                                          integrationTime=ditherdesc.obs[ditherind].end -
                                                                                          ditherdesc.obs[ditherind].start) * detPixMask).flatten()
        else:
            self.detExpTimes = None

        # Now get the photons
        print('Getting photon coords')

        photons = obsfile.query(startw=self.wvlMin, stopw=self.wvlMax,
                                startt=self.firstObsTime, stopt=self.firstObsTime+self.integrationTime)

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
                allPhotXY.append(xyPackMult * row + col)
        # Get a boolean array indicating photons whose packed x-y coordinate value is in the 'bad' list.
        toReject = np.where(np.in1d(allPhotXY, badXY))[
            0]  # [0] to take index array out of the returned 1-element tuple.
        # Chuck out the bad photons
        print('Rejecting photons from bad pixels...')
        # photons = np.delete(photons, toReject)
        #########################################################################

        self.photWeights = None
        if obsfile.getFromHeader('isFlatCalibrated') and ObsFile.getFromHeader('isSpecCalibrated'):
            print('INCLUDING FLUX WEIGHTS!')
            self.photWeights = photons['flatWeight'] * photons[
                'fluxWeight']  # ********EXPERIMENTING WITH ADDING FLUX WEIGHT - NOT FULLY TESTED, BUT SEEMS OKAY....********

        n_photons = len(photons)
        timestamps = np.zeros(n_photons)
        xPhotonPixels = np.zeros(n_photons)
        yPhotonPixels = np.zeros(n_photons)
        photWavelengths = np.zeros(n_photons)

        # Use for loop for now. Photons is a np.array of np.void values so hard to index across that axis...
        for p, photon in enumerate(photons):
            # print p, photon
            timestamps[p] = photon[1]
            xPhotonPixels[p], yPhotonPixels[p] = np.where(obsfile.beamImage == photon[0])
            photWavelengths[p] = photon[2]

        if self.wvlMin is not None or self.wvlMax is not None:
            assert all(photWavelengths >= self.wvlMin) and all(photWavelengths <= self.wvlMax)
        print('Min, max photon wavelengths found: ', np.min(photWavelengths), np.max(photWavelengths))

        # plt.figure()
        # thisImage, thisGridDec, thisGridRA = np.histogram2d(xPhotonPixels, yPhotonPixels, bins=[140,146])
        # plt.imshow(thisImage, norm=LogNorm())
        # plt.show()

        # self.pixMask = obsfile.pixelMask # load pixel mask here direct from photontable

        return [timestamps, xPhotonPixels, yPhotonPixels, photWavelengths]

    def get_wcs(self, timestamps, xPhotonPixels, yPhotonPixels, ditherind, ditherdesc, toa_rotation=False):
        '''
        :param timestamps:
        :param xPhotonPixels:
        :param yPhotonPixels:
        :param ditherind:
        :param ditherdesc:
        :param toa_rotation:
        If False each dither position is a fixed orientation. If True the HA of each photon receives an additional
        contribution based on the TOA allowing for rotation effects during each dither integration.

        :return:
        '''

        print('Calculating RA/Decs for dither %i' % ditherind)

        if toa_rotation:
            photHAs = timestamps * 1e-6 * 1./86164.1 * 2*np.pi #* 500

            hourangles = ditherdesc.dithHAs[ditherind] + photHAs

            rotationMatrix = np.array([[np.cos(hourangles), -np.sin(hourangles)],
                                       [np.sin(hourangles), np.cos(hourangles)]]).T

            centroids = np.array([-1*ditherdesc.xCenRes[ditherind] + xPhotonPixels - ditherdesc.xpix/2,
                                  -1*ditherdesc.yCenRes[ditherind] + yPhotonPixels - ditherdesc.ypix/2])

            centroidRotated = np.dot(rotationMatrix, centroids).diagonal(axis1=0,axis2=2)

        else:
            hourangles = ditherdesc.dithHAs[ditherind]

            rotationMatrix = np.array([[np.cos(hourangles), -np.sin(hourangles)],
                                       [np.sin(hourangles), np.cos(hourangles)]]).T

            centroids = np.array([-1*ditherdesc.xCenRes[ditherind] + xPhotonPixels - ditherdesc.xpix/2,
                                  -1*ditherdesc.yCenRes[ditherind] + yPhotonPixels - ditherdesc.ypix/2])

            centroidRotated = np.dot(rotationMatrix, centroids)

        rightAscensionOffset = -1*ditherdesc.platescale * (centroidRotated[0]) # -1 here just orientates the image
        declinationOffset = ditherdesc.platescale * (centroidRotated[1])

        # Convert centroid positions in DD:MM:SS.S and HH:MM:SS.S format to radians.
        centroidRightAscensionRadians = ephem.hours(ditherdesc.cenRA).real
        centroidDeclinationRadians = ephem.degrees(ditherdesc.cenDec).real

        # Convert centroid position radians to arcseconds.
        degreesToRadians = np.pi / 180.0
        radiansToDegrees = 180.0 / np.pi

        centroidDeclinationArcseconds = centroidDeclinationRadians * radiansToDegrees * 3600.0
        centroidRightAscensionArcseconds = centroidRightAscensionRadians * radiansToDegrees * 3600.0

        # Add the photon arcsecond offset to the centroid offset.
        photonDeclinationArcseconds = centroidDeclinationArcseconds + declinationOffset
        photonRightAscensionArcseconds = centroidRightAscensionArcseconds + rightAscensionOffset

        # Convert the photon positions from arcseconds to radians
        photDecRad = (photonDeclinationArcseconds / 3600.0) * degreesToRadians
        photRARad = (photonRightAscensionArcseconds / 3600.0) * degreesToRadians

        nPhot = 1#len(hourangles)

        if self.randoffset:
            # Add uniform random dither to each photon, distributed over a square
            # area of the same size and orientation as the originating pixel at
            # the time of observation (assume RA and dec are defined at center of pixel).
            np.random.seed(42) # so random values always same
            xRand = np.random.rand(nPhot) * ditherdesc.plateScale - ditherdesc.plateScale / 2.0
            yRand = np.random.rand(nPhot) * ditherdesc.plateScale - ditherdesc.plateScale / 2.0  # Not the same array!
            ditherRAs = xRand * np.cos(hourangles) - yRand * np.sin(hourangles)
            ditherDecs = yRand * np.cos(hourangles) + xRand * np.sin(hourangles)
        else:
            ditherRAs = 0
            ditherDecs = 0

        photRARad = photRARad + ditherRAs
        photDecRad = photDecRad + ditherDecs

        return photRARad, photDecRad

if __name__ == '__main__':
    # Get dither offsets
    name = 'Trapezium'
    file = 'Trapezium.log'

    wvlMin = 850
    wvlMax = 1100
    firstObsTime = 0
    integrationTime = 4
    drizzleconfig = [wvlMin, wvlMax, firstObsTime, integrationTime]

    loc = os.path.join(os.getenv('MKID_DATA_DIR'), name, 'wavecal', file)
    logdithdata = MKIDObservingDither(name, loc, None, None)
    h5dithdata = getmetafromh5()
    ditherdesc = DitherDescription(logdithdata, h5dithdata, drizzleconfig, rotate=True)

    # Quick save method for the reduced photon packets
    import pickle
    pkl_save = './phlist.pkl'

    if os.path.exists(pkl_save):
        with open(pkl_save, 'rb') as handle:
            photonlists = pickle.load(handle)
    else:
        dir = os.getenv('MKID_PROC_PATH')
        filenames = sorted(glob.glob(os.path.join(dir, name, 'wavecal', '*.h5')), key=os.path.getmtime)
        obsfiles = [ObsFile(file) for file in filenames]

        photonlists = []
        for ditherind, obsfile in enumerate(obsfiles[:10]):
            photonlists.append(photonlist(obsfile, ditherdesc, ditherind))
        plt.show()
        with open(pkl_save, 'wb') as handle:
            pickle.dump(photonlists, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Do the dither
    # scimaps = []
    # for pixfrac in [1]:
    #     driz = SpatialDrizzler(photonlists, ditherdesc, pixfrac=pixfrac)
    #     driz.run()
    #     scimaps.append(driz.driz.outsci)
    # view_datacube(scimaps, logAmp=True, vmin=2, vmax=150)
    # view_datacube(scimaps,logAmp=False, vmin=0, vmax=100)
    # # view_datacube(np.roll(scimaps, axis=0, shift=1)-scimaps)
    # # view_datacube(np.roll(scimaps, axis=0, shift=1) - scimaps, logAmp=True)
    # loop_frames(scimaps, logAmp=True, vmin=2, vmax=150)
    # loop_frames(scimaps, logAmp=False, vmin=0, vmax=100)
    # # loop_frames(np.roll(scimaps, axis=0, shift=1)-scimaps, logAmp=True)

    # driz = SpectralDrizzler(photonlists, ditherdesc)
    # driz.run()
    # indep_images(driz.cube)

    driz = TemporalDrizzler(photonlists, ditherdesc)
    driz.run()
    # grid(driz.hypercube, width=5, nrows=3, vmaxs=[20]*15, vmins=[1]*15)
