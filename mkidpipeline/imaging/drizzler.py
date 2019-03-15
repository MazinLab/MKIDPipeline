"""
TODO
Add astroplan, ephem, drizzle, SharedArray to setup.py/yml. ephem can be conda installed, drizzle and SharedArray need to be pip
installed. I found that  astroplan needed to be pip installed otherwise some astropy import fails

Get con2pix calibration from Isabel's code and remove from here

Convert print to logs

Build form() wrapper

Better handling of savestate of photonlists

Move photonlist class functions to photontable.py

Usage
-----

python drizzle.py

Author: Rupert Dodkins, Julian van Eyken            Date: Jan 2019


This code is adapted from Julian's testImageStack from the ARCONS pipeline.
"""
import matplotlib
print(matplotlib.matplotlib_fname())
matplotlib.use('QT5Agg', force=True)
matplotlib.rcParams['backend'] = 'Qt5Agg'
import matplotlib.pylab as plt

# plt.plot(range(5))
# plt.show()

# # plt.switch_backend('Qt5Agg')
# print(matplotlib.get_backend())

import os
import numpy as np
import time
import multiprocessing as mp
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
import glob
import ephem
from astropy import wcs
from astropy.coordinates import EarthLocation, Angle, SkyCoord
import astropy.units as u
from astroplan import Observer
import astropy
from drizzle import drizzle as stdrizzle
from mkidcore import pixelflags
from mkidpipeline.hdf.photontable import ObsFile
from mkidcore.corelog import getLogger
from mkidpipeline.config import MKIDObservingDataDescription, MKIDObservingDither
# import _pickle as pickle
import pickle



def con2pix(xCon, yCon):
    """
    This is temporary. Should grab this conversion from elsewhere

    :param xCon:
    :param yCon:
    :return:
    """
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
        return [func(xCon, *xopt), func(yCon, *yopt)]

    xPos, yPos = con2Pix(xCon, yCon, func)

    # return np.array([xPos, yPos]).T
    return [-xPos, -yPos]


def ditherp_2_pixel(positions):
    """ A function to convert the connex offset to pixel displacement"""
    positions = np.asarray(positions)
    pix = np.array(con2pix(positions[:, 0], positions[:, 1])) - np.array(con2pix(0, 0)).reshape(2,1)
    print(pix.shape)
    # pix = (positions*150).T
    # print(np.shape(pix))
    # exit()
    return pix


def getmetafromh5():
    """
    Helper function not properly implemented yet. Hard coded values from Trap

    :param ditherdesc:
    :return:
    """
    # raise NotImplementedError
    getLogger(__name__).warning('Using hard coded RA/Dec metadata in place of real h5 data.')

    xpix = 140
    ypix = 146
    platescale = 10 *u.mas #mas #0.44

    return {'xpix': xpix, 'ypix': ypix, 'platescale': platescale}


class DitherDescription(object):
    """
    Info on the dither

    rotate determines if the effective integrations are pupil stablised or not

    TODO implement center of rotation
    """
    def __init__(self, mkid_observing_dither, wvlMin=0, wvlMax=np.inf, startt=0, intt=1,
                 observatory='Subaru', target='* kap And', multiplier=1, plotdithlocs=False):
        self.description = mkid_observing_dither
        self.pos = mkid_observing_dither.pos

        # TODO these definitions need to be redone
        self.coords = SkyCoord.from_name(target)
        self.cenRA = self.coords.ra.deg
        self.cenDec = self.coords.dec.deg
        print(self.cenRA, 'cenRA')
        h5dithdata = getmetafromh5()
        self.xpix = h5dithdata['xpix']
        self.ypix = h5dithdata['ypix']
        self.platescale = h5dithdata['platescale'].to(u.deg).value

        # TODO these definitions need to be redone
        self.wvlMin = wvlMin
        self.wvlMax = wvlMax
        self.firstObsTime = startt
        self.integrationTime = intt

        self.centroids = ditherp_2_pixel(self.pos)
        # self.virPixCen = np.array([[100,0]]).T#ditherp_2_pixel([(0,0)]) #+ np.array([(-30,-30)]).T
        self.virPixCen = np.array([[25,0]]).T#ditherp_2_pixel([(0,0)]) #+ np.array([(-30,-30)]).T

        # self.virPixCen = np.array([(0,0)]).T
        print('virPixCen', self.virPixCen)

        if multiplier != 0:
            times = np.array([o.start for o in mkid_observing_dither.obs])
            print(times)
            site = EarthLocation.of_site(observatory)
            unixtimes = astropy.time.Time(val=times, format='unix')
            LSTs = unixtimes.sidereal_time('mean', site.lon)
            # print(LSTs)
            radLSTs = LSTs.radian
            # self.hourAngles = radLSTs - ephem.hours(self.cenRA).real

            # target = SkyCoord(ra=355.1 * u.deg, dec=44.3 * u.deg)


            # print(target)

            apo = Observer.at_site(observatory)
            altaz = apo.altaz(unixtimes, self.coords)

            Earthrate = 2 * np.pi / u.sday.to(u.second) # * 500

            # obs_const = Earthrate * np.cos(19.7 * np.pi / 180)
            obs_const = Earthrate * np.cos(site.geodetic.lat.rad)
            rot_rate = obs_const * np.cos(altaz.az.radian) / np.cos(altaz.alt.radian)
            rot_rate = np.array(rot_rate)#*2

            # self.dithHAs = direction*(radLSTs - radLSTs[0]) # in radians
            if multiplier:
                rot_rate = rot_rate*multiplier
            # plt.plot(angles)
            # plt.show()
            # self.dithHAs = np.cumsum(rot_rate*(times - times[0]))# * direction # in radians
            self.dithHAs = [np.trapz(rot_rate[:ix], x= times[:ix] - times[0]) for ix in range(1, len(times)+1)]
            self.dithHAs -= self.dithHAs[0]
            print(self.dithHAs)

            # self.dithHAs = np.trapz(rot_rate)#*(times - times[0])) * direction # in radians
            # plt.plot(self.dithHAs, 'o')
            # plt.show()

        else:
            self.dithHAs = np.zeros(len(mkid_observing_dither.obs), dtype=np.float)
        getLogger(__name__).debug("HAs: %s", self.dithHAs)

    def plot(self):
        self.xCenRes, self.yCenRes = self.centroids[0] - self.virPixCen[0], self.centroids[1] - self.virPixCen[1]
        rotationMatrix = np.array([[np.cos(self.dithHAs), -np.sin(self.dithHAs)],
                                        [np.sin(self.dithHAs), np.cos(self.dithHAs)]]).T

        centroidRotated = np.dot(rotationMatrix,
                            np.array([self.xCenRes, self.yCenRes])).diagonal(axis1=0,axis2=2) + [
                            self.virPixCen[0], self.virPixCen[1]]


        plt.plot(-self.centroids[0], -self.centroids[1], '-o')
        plt.plot(-self.virPixCen[0], -self.virPixCen[1], marker='x')
        plt.plot(-centroidRotated[0], -centroidRotated[1], '-o')
        plt.show()

class Drizzler(object):
    def __init__(self, photonlists, metadata):
        # TODO Implement
        # Assume obsfiles either have their metadata or needed metadata is passed, e.g. WCS information, target info,
        # etc

        # TODO determine appropirate value from area coverage of dataset and oversampling, even longerterm there
        # the oversampling should be selected to optimize total phase coverage to extract the most resolution at a
        # desired minimum S/N

        # self.randoffset = False # apply random spatial offset to each photon
        self.nPixRA = None
        self.nPixDec = None

        self.config = None
        self.files = photonlists

        self.xpix = metadata.xpix
        self.ypix = metadata.ypix
        self.cenRA = metadata.coords.ra.deg
        self.cenDec = metadata.coords.dec.deg
        print(self.cenRA, self.cenDec, 'cenRA cenDec')
        self.vPlateScale = metadata.platescale
        self.virPixCen = metadata.virPixCen

        # self.vPlateScale = self.vPlateScale * 2 * np.pi / 1296000  # No. of radians on sky per virtual pixel.
        # self.detPlateScale = self.detPlateScale * 2 * np.pi / 1296000

        raMin, raMax, decMin, decMax = [], [], [], []
        for photonlist in photonlists:
            raMin.append(min(photonlist['photRARad']))
            raMax.append(max(photonlist['photRARad']))
            decMin.append(min(photonlist['photDecRad']))
            decMax.append(max(photonlist['photDecRad']))
        raMin = min(raMin)
        raMax = max(raMax)
        decMin = min(decMin)
        decMax = max(decMax)

        # self.cenRA = (raMin + raMax) / 2.0
        # self.cenDec = (decMin + decMax) / 2.0
        #
        # # Set size of virtual grid to accommodate.
        #
        # if self.nPixRA is None:
        #     # +1 for round up; +1 because coordinates are the boundaries of the virtual pixels, not the centers.
        #     self.nPixRA = int((raMax - raMin) // self.vPlateScale + 2)
        # if self.nPixDec is None:
        #     self.nPixDec = int((decMax - decMin) // self.vPlateScale + 2)
        self.nPixRA, self.nPixDec = 350, 350#250, 250

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
        # TODO implement something like this
        # w = mkidcore.buildwcs(self.nPixRA, self.nPixDec, self.vPlateScale, self.cenRA, self.cenDec)
        # TODO implement the PV distortion?
        # eg w.wcs.set_pv([(2, 1, 45.0)])

        self.w = wcs.WCS(naxis=2)
        self.w.wcs.crpix = np.array([self.nPixRA / 2., self.nPixDec / 2.]) + np.array([self.virPixCen[0][0],self.virPixCen[1][0]])
        self.w.wcs.cdelt = np.array([self.vPlateScale, self.vPlateScale])
        self.w.wcs.crval = [self.cenRA, self.cenDec]
        self.w.wcs.ctype = ["RA-----", "DEC----"]
        self.w._naxis1 = self.nPixRA
        self.w._naxis2 = self.nPixDec
        print(self.w)


class SpectralDrizzler(Drizzler):
    """ Generate a spatially dithered fits dataacube from a set dithered dataset """

    def __init__(self, photonlists, metadata, pixfrac=1):
        self.nwvlbins = 3
        self.wvlbins = np.linspace(metadata.wvlMin, metadata.wvlMax, self.nwvlbins + 1)
        super().__init__(photonlists, metadata)
        self.drizcube = [stdrizzle.Drizzle(outwcs=self.w, pixfrac=pixfrac)] * self.nwvlbins

    def run(self, save_file=None):
        for ix, file in enumerate(self.files):

            getLogger(__name__).debug('Processing %s', file)
            tic = time.clock()
            insci, inwcs = self.makeCube(file)
            getLogger(__name__).debug('Image load done. Time taken (s): %s', time.clock() - tic)
            for iw in range(self.nwvlbins):
                print(ix, iw)
                self.drizcube[iw].add_image(insci[iw], inwcs, inwht=np.int_(np.logical_not(insci[iw] == 0)))

        self.cube = [d.outsci for d in self.drizcube]

        # TODO add the wavelength WCS

    def makeCube(self, file):
        sample = np.vstack((file['wavelengths'], file['photDecRad'], file['photRARad']))
        bins = np.array([self.wvlbins, self.ypix, self.xpix])

        datacube, (wavelengths, thisGridDec, thisGridRA) = np.histogramdd(sample.T, bins)

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
        w.wcs.cdelt = np.array([thisGridRA[1] - thisGridRA[0], thisGridDec[1] - thisGridDec[0]])
        w.wcs.crval = [thisGridRA[0], thisGridDec[0]]
        w.wcs.ctype = ["RA---AIR", "DEC--AIR"]
        w._naxis1 = len(thisGridRA) - 1
        w._naxis2 = len(thisGridDec) - 1

        return datacube, w


class TemporalDrizzler(Drizzler):
    """
    Generate a spatially dithered fits 4D hypercube from a set dithered dataset. The cube size is
    ntimebins * ndithers X nwvlbins X nPixRA X nPixDec.
    """

    def __init__(self, photonlists, metadata, pixfrac=0):
        self.nwvlbins = 2
        self.timestep = 0.1  # seconds

        Drizzler.__init__(self, photonlists, metadata)
        self.ndithers = len(self.files)
        self.pixfrac = pixfrac
        self.wvlbins = np.linspace(metadata.wvlMin, metadata.wvlMax, self.nwvlbins + 1)
        self.ntimebins = int(metadata.integrationTime / self.timestep)
        self.timebins = np.linspace(metadata.firstObsTime,
                                    metadata.firstObsTime + metadata.integrationTime,
                                    self.ntimebins + 1) * 1e6  # timestamps are in microseconds
        self.totHypCube = None
        self.totWeightCube = None

    def run(self, save_file=None):
        tic = time.clock()

        self.totHypCube = np.zeros((self.ntimebins * self.ndithers, self.nwvlbins, self.nPixDec, self.nPixRA))
        self.totWeightCube = np.zeros((self.ntimebins, self.nwvlbins, self.nPixDec, self.nPixRA))
        for ix, file in enumerate(self.files):

            getLogger(__name__).debug('Processing %s', file)

            insci, inwcs = self.makeHyper(file)

            thishyper = np.zeros((self.ntimebins, self.nwvlbins, self.nPixDec, self.nPixRA), dtype=np.float32)

            for it, iw in np.ndindex(self.ntimebins, self.nwvlbins):
                drizhyper = stdrizzle.Drizzle(outwcs=self.w, pixfrac=self.pixfrac)
                drizhyper.add_image(insci[it, iw], inwcs, inwht=np.int_(np.logical_not(insci[it, iw] == 0)))
                thishyper[it, iw] = drizhyper.outsci
                self.totWeightCube[it, iw] += thishyper[it, iw] != 0

            self.totHypCube[ix * self.ntimebins: (ix + 1) * self.ntimebins] = thishyper

        getLogger(__name__).debug('Image load done. Time taken (s): %s', time.clock() - tic)
        # TODO add the wavelength WCS

    def makeHyper(self, file):
        sample = np.vstack((file['timestamps'], file['wavelengths'], file['photDecRad'], file['photRARad']))
        bins = np.array([self.timebins, self.wvlbins, self.ypix, self.xpix])
        hypercube, bins = np.histogramdd(sample.T, bins)

        times, wavelengths, thisGridDec, thisGridRA = bins

        w = wcs.WCS(naxis=2)
        w.wcs.crpix = [0., 0.]
        w.wcs.cdelt = np.array([thisGridRA[1] - thisGridRA[0], thisGridDec[1] - thisGridDec[0]])
        w.wcs.crval = [thisGridRA[0], thisGridDec[0]]
        w.wcs.ctype = ["RA---AIR", "DEC--AIR"]
        w._naxis1 = len(thisGridRA) - 1
        w._naxis2 = len(thisGridDec) - 1

        return hypercube, w


class SpatialDrizzler(Drizzler):
    """ Generate a spatially dithered fits image from a set dithered dataset """

    def __init__(self, photonlists, metadata, pixfrac=1):
        Drizzler.__init__(self, photonlists, metadata)
        self.driz = stdrizzle.Drizzle(outwcs=self.w, pixfrac=pixfrac)
        self.rotmat = np.array([[np.cos(metadata.dithHAs), np.sin(metadata.dithHAs)],
                           [-np.sin(metadata.dithHAs), np.cos(metadata.dithHAs)]])
        print(self.rotmat.shape, 'rotmat')

    def run(self, save_file=None):
        for ix, file in enumerate(self.files):
            getLogger(__name__).debug('Processing %s', file)
            tic = time.clock()
            insci, inwcs = self.makeImage(file)
            getLogger(__name__).debug('Image load done. Time taken (s): %s', time.clock() - tic)
            self.driz.add_image(insci, inwcs, inwht=(insci != 0).astype(int))
            # plt.imshow(self.driz.outsci, origin='lower', vmax=20)
            # plt.show()
        if save_file:
            self.driz.write(save_file)

        # TODO introduce total_exp_time variable and complete these steps

    def makeImage(self, file):
        thisImage, thisGridDec, thisGridRA = np.histogram2d(file['photDecRad'], file['photRARad'],
                                                            bins=[self.ypix, self.xpix])
        # virImage, virGridDec, virGridRA = np.histogram2d(file['photDecRad'], file['photRARad'],
        #                                                     [self.gridDec, self.gridRA])

        w = wcs.WCS(naxis=2)
        w.wcs.crpix = [self.xpix/2., self.ypix/2.]
        # w.wcs.crpix = [0, 0]
        w.wcs.cdelt = np.array([thisGridRA[1]-thisGridRA[0], thisGridDec[1]-thisGridDec[0]])
        # w.wcs.cdelt = np.array([self.vPlateScale, self.vPlateScale])
        w.wcs.crval = [thisGridRA[self.xpix//2], thisGridDec[self.ypix//2]]
        # w.wcs.crval = [thisGridRA[self.xpix//2], thisGridDec[self.ypix//2]]
        w.wcs.ctype = ["RA-----", "DEC----"]

        w._naxis1 = self.xpix
        w._naxis2 = self.ypix
        # w.wcs.cd = self.rotmat[:,:,ix] * self.vPlateScale
        print(w)

        # plt.imshow(thisImage, origin='lower', extent=[min(thisGridRA),max(thisGridRA),min(thisGridDec),max(thisGridDec)], norm=LogNorm())#)
        # plt.show()

        return thisImage, w


def reduce_obs(obsfile, ditherdesc, ditherind):
    """
    This function calibrates and queries an obsfile. A lot of copy pasta

    :returns
    list of photon times, positions, wavelengths
    """

    # photTable = obsfile.file.root.Photons.PhotonTable  # Shortcut to table
    # # print(photTable[::5000])
    # img = obsfile.getPixelCountImage(firstSec =0, integrationTime=1, wvlStart=self.wvlMin, wvlStop=self.wvlMax)
    # self.testimage = img['image']
    # plt.imshow(self.testimage, aspect='equal')
    # plt.show()

    # if expWeightTimeStep is not None:
    #    self.expWeightTimeStep=expWeightTimeStep

    doWeighted = False
    medCombine = False
    maxBadPixTimeFrac = None
    randoffset = False

    if obsfile.info['timeMaskExists']:
        # If hot pixels time-mask data not already parsed in (presumably not), then parse it.
        if obsfile.hotPixTimeMask is None:
            obsfile.parseHotPixTimeMask()  # Loads time mask dictionary into ObsFile.hotPixTimeMask

    if ditherdesc.wvlMin is not None and ditherdesc.wvlMax is None:
        ditherdesc.wvlMax = np.inf
    if ditherdesc.wvlMin is None and ditherdesc.wvlMax is not None:
        ditherdesc.wvlMin = 0.0

    # Figure out last second of integration
    obsFileExpTime = obsfile.header.cols.expTime[0]
    if ditherdesc.integrationTime == -1 or ditherdesc.firstObsTime + ditherdesc.integrationTime > obsFileExpTime:
        lastObsTime = obsFileExpTime
    else:
        lastObsTime = ditherdesc.firstObsTime + ditherdesc.integrationTime

    lastObsTime *= 1e6  # convert to microseconds

    beamFlagImage = np.transpose(obsfile.beamFlagImage.read())
    nDPixRow, nDPixCol = beamFlagImage.shape

    # Make a boolean mask of dead (non functioning for whatever reason) pixels
    # True (1) = good; False (0) = dead
    # First on the basis of the wavelength cals:

    if obsfile.info['isWvlCalibrated']:
        # wvlCalFlagImage = ObsFile.getBadWvlCalFlags()
        # print('This needs to be updated. No flags loaded')
        wvlCalFlagImage = np.zeros_like(beamFlagImage)
    else:
        wvlCalFlagImage = np.zeros_like(beamFlagImage)

    deadPixMask = np.where(wvlCalFlagImage == pixelflags.speccal, 1,
                           0)  # 1.0 where flag is good; 0.0 otherwise. (Straight boolean mask would work, but not guaranteed for Python 4....)
    # print('# Dead detector pixels to reject on basis of wavelength cal: ', np.sum(deadPixMask == 0))

    # Next a mask on the basis of the flat cals (or all ones if weighting not requested)
    if doWeighted:
        flatCalFlagArray = obsfile.file.root.flatcal.flags.read()  # 3D array - nRow * nCol * nWavelength Bins.
        flatWvlBinEdges = obsfile.file.root.flatcal.wavelengthBins.read()  # 1D array of wavelength bin edges for the flat cal.
        lowerEdges = flatWvlBinEdges[0:-1]
        upperEdges = flatWvlBinEdges[1:]
        if ditherdesc.wvlMin is None and ditherdesc.wvlMax is None:
            inRange = np.ones(len(lowerEdges), dtype=bool)  # (all bins in range implies all True)
        else:
            inRange = ((lowerEdges >= ditherdesc.wvlMin) & (lowerEdges < ditherdesc.wvlMax) |
                       (upperEdges >= ditherdesc.wvlMin) & (
                               lowerEdges < ditherdesc.wvlMax))  ####SOMETHING NOT RIGHT HERE? DELETE IF NO ASSERTION ERROR THROWN BELOW!##########
            # Bug fix - I think this is totally equivalent - first term above is redundant, included in second term:
            inRangeOld = np.copy(inRange)  # Can delete if no assertion error thrown below
            inRange = (upperEdges >= ditherdesc.wvlMin) & (lowerEdges < ditherdesc.wvlMax)
            assert np.all(inRange == inRangeOld)  # Can delete once satisfied this works.
            # If this never complains, then can switch to the second form.

        flatCalMask = np.where(np.all(flatCalFlagArray[:, :, inRange] == False, axis=2), 1,
                               0)  # Should be zero where any pixel has a bad flag at any wavelength within the requested range; one otherwise. Spot checked, seems to work.
        print('# Detector pixels to reject on basis of flatcals: ', np.sum(flatCalMask == 0))
    else:
        flatCalMask = np.ones((nDPixRow, nDPixCol))

    # And now a mask based on how much hot pixel behaviour each pixel exhibits:
    # if a given pixel is bad more than a fraction maxBadTimeFrac of the time,
    # then write it off as permanently bad for the duration of the requested
    # integration.
    if maxBadPixTimeFrac is not None:
        print('Rejecting pixels with more than ', 100 * maxBadPixTimeFrac, '% bad-flagged time')
        detGoodIntTimes = obsfile.hotPixTimeMask.getEffIntTimeImage(firstSec=ditherdesc.firstObsTime,
                                                                    integrationTime=lastObsTime - ditherdesc.firstObsTime)
        badPixMask = np.where(detGoodIntTimes / (lastObsTime - ditherdesc.firstObsTime) > (1. - maxBadPixTimeFrac), 1,
                              0)  # Again, 1 if okay, 0 bad. Use lastObsTime-self.firstObsTime instead of integrationTime in case integrationTime is -1.
        print('# pixels to reject: ', np.sum(badPixMask == 0))
        print('# pixels to reject with eff. int. time > 0: ', np.sum((badPixMask == 0) & (detGoodIntTimes > 0)))
    else:
        badPixMask = np.ones((nDPixRow, nDPixCol))

    # Finally combine all the masks together into one detector pixel mask:
    # detPixMask = deadPixMask * flatCalMask * badPixMask  # Combine masks
    detPixMask = np.zeros_like(deadPixMask)
    # print('Total detector pixels to reject: ', np.sum(
    #     detPixMask), "(may not equal sum of the above since theres overlap!)")

    # Get array of effective exposure times for each detector pixel based on the hot pixel time mask
    # Multiply by the bad pixel mask and the flatcal mask so that non-functioning pixels have zero exposure time.
    # Flatten the array in the same way as the previous arrays (1D array, nRow*nCol elements).
    # detExpTimes = (hp.getEffIntTimeImage(ObsFile.hotPixTimeMask, integrationTime=tEndFrames[iFrame]-tStartFrames[iFrame],
    #                                     self.firstObsTime=tStartFrames[iFrame]) * detPixMask).flatten()

    # Now get the photons
    # print('Getting photon coords')
    photons = obsfile.query(startw=ditherdesc.wvlMin, stopw=ditherdesc.wvlMax,
                            startt=ditherdesc.firstObsTime, stopt=ditherdesc.firstObsTime + ditherdesc.integrationTime)

    # And filter out photons to be masked out on the basis of the detector pixel mask
    # print('Finding photons in masked detector pixels...')
    # Get a boolean array indicating photons whose packed x-y coordinate value is in the 'bad' list.

    getLogger(__name__).debug("Number of photons read from obsfile: %i", len(photons))
    print("Number of photons read from obsfile: %i" % len(photons))
    timestamps = photons["Time"]
    flatbeam = obsfile.beamImage.flatten()
    beamsorted = np.argsort(flatbeam)
    ind = np.searchsorted(flatbeam[beamsorted], photons["ResID"])
    xPhotonPixels, yPhotonPixels = np.unravel_index(beamsorted[ind], obsfile.beamImage.shape)
    photWavelengths = photons["Wavelength"]

    if ditherdesc.wvlMin is not None or ditherdesc.wvlMax is not None:
        assert all(photWavelengths >= ditherdesc.wvlMin) and all(photWavelengths <= ditherdesc.wvlMax)
    # print('Min, max photon wavelengths found: ', np.min(photWavelengths), np.max(photWavelengths))

    # plt.figure()
    # thisImage, thisGridDec, thisGridRA = np.histogram2d(xPhotonPixels, yPhotonPixels, bins=[140,146])
    # plt.imshow(thisImage, norm=LogNorm())
    # plt.show()

    return [timestamps, xPhotonPixels, yPhotonPixels, photWavelengths]


def get_wcs(timestamps, xPhotonPixels, yPhotonPixels, ditherind, ditherdesc, toa_rotation=False, randoffset=False):
    """
    :param timestamps:
    :param xPhotonPixels:
    :param yPhotonPixels:
    :param ditherind:
    :param ditherdesc:
    :param toa_rotation:
    If False each dither position is a fixed orientation. If True the HA of each photon receives an additional
    contribution based on the TOA allowing for rotation effects during each dither integration.

    Add uniform random dither to each photon, distributed over a square
    area of the same size and orientation as the originating pixel at
    the time of observation (assume RA and dec are defined at center of pixel).

    :return:
    """

    print('Calculating RA/Decs for dither %i' % ditherind)

    if toa_rotation:
        # TODO update this rot_rate to altaz model
        Earthrate = 1./86164.1 * 2*np.pi #* 500
        obs_const = Earthrate * np.cos(19.7*np.pi/180)
        rot_rate = obs_const * np.cos(az) / np.cos(alt)
        photHAs = timestamps * 1e-6 * rot_rate


        hourangles = ditherdesc.dithHAs[ditherind] + photHAs

        rotationMatrix = np.array([[np.cos(hourangles), -np.sin(hourangles)],
                                   [np.sin(hourangles), np.cos(hourangles)]]).T

        centroids = np.array([-ditherdesc.xCenRes[ditherind] + xPhotonPixels - ditherdesc.xpix / 2,
                              -ditherdesc.yCenRes[ditherind] + yPhotonPixels - ditherdesc.ypix / 2])

        centroidRotated = np.dot(rotationMatrix, centroids).diagonal(axis1=0, axis2=2)

    else:
        hourangle = ditherdesc.dithHAs[ditherind]
        print('HAs', hourangle)

        rotationMatrix = np.array([[np.cos(hourangle), -np.sin(hourangle)],
                                   [np.sin(hourangle), np.cos(hourangle)]])

        # plt.plot(ditherdesc.centroids[0],ditherdesc.centroids[1])
        # plt.show(block=True)
        # print('lol', ditherdesc.virPixCen, rotationMatrix)

        # put each photon from the dither into its raster location on the virtual grid
        vgrid_photons = np.array([ditherdesc.centroids[0][ditherind] + xPhotonPixels - ditherdesc.xpix/2,
                                     ditherdesc.centroids[1][ditherind] + yPhotonPixels - ditherdesc.ypix/2])

        # offset these to the center of rotation
        cor_photons = vgrid_photons - ditherdesc.virPixCen

        # rotate that virtual grid so that that those photons now occupy the part of the sky that was sampled
        rotated_vgrid = np.dot(rotationMatrix, cor_photons)

        # undo the COR offset
        skyframe_photons = rotated_vgrid + ditherdesc.virPixCen

        # thisImage, thisGridDec, thisGridRA = np.histogram2d(skyframe_photons[0],skyframe_photons[1],
        #                                                     bins=[range(-250,250),range(-250,250)])
        #
        # plt.imshow(thisImage, origin='lower', extent=[min(thisGridRA),max(thisGridRA),min(thisGridDec),max(thisGridDec)], norm=LogNorm())#)
        # plt.show(block=True)
    rightAscensionOffset = ditherdesc.platescale * (skyframe_photons[0]) # -1 here just orientates the final image
    declinationOffset = ditherdesc.platescale * (skyframe_photons[1])

    # thisImage, thisGridDec, thisGridRA = np.histogram2d(rightAscensionOffset, declinationOffset, bins=[146,146])
    # plt.imshow(thisImage, origin='lower', extent=[min(thisGridRA),max(thisGridRA),min(thisGridDec),max(thisGridDec)], norm=LogNorm())#)
    # plt.show(block=True)

    # Convert centroid positions in DD:MM:SS.S and HH:MM:SS.S format to radians.
    centroidRADeg = ephem.hours(ditherdesc.cenRA).real
    centroidDecDeg = ephem.degrees(ditherdesc.cenDec).real

    # Add the photon arcsecond offset to the centroid offset.
    photDecDeg = centroidDecDeg + declinationOffset
    photRADeg = centroidRADeg + rightAscensionOffset

    nPhot = 1

    if randoffset:
        np.random.seed(42)  # so random values always same
        xRand = np.random.rand(nPhot) * ditherdesc.plateScale - ditherdesc.plateScale / 2.0
        yRand = np.random.rand(nPhot) * ditherdesc.plateScale - ditherdesc.plateScale / 2.0  # Not the same array!
        ditherRAs = xRand * np.cos(hourangles) - yRand * np.sin(hourangles)
        ditherDecs = yRand * np.cos(hourangles) + xRand * np.sin(hourangles)
    else:
        ditherRAs = 0
        ditherDecs = 0

    photRADeg = photRADeg + ditherRAs
    photDecDeg = photDecDeg + ditherDecs

    return photRADeg, photDecDeg


def drizzle_dither(dither, *args, **kwargs):
    """Form a drizzled image from a dither"""
    print('**This needs to be updated**')
    raise NotImplementedError

    obsfiles = [ObsFile(o.h5) for o in dither.obs]

    photonlists = []
    for ditherind, obsfile in enumerate(obsfiles[:25]):
        photonlists.append(photonlist(obsfile, ditherdesc, ditherind))
    plt.show()
    with open(pkl_save, 'wb') as f:
        pickle.dump(photonlists, f, protocol=pickle.HIGHEST_PROTOCOL)

    # The WCS can be reassigned here rather than loading from obs each time
    for ip, photonlist in enumerate(photonlists):
        photonlist.photRARad, photonlist.photDecRad = photonlist.get_wcs(photonlist.timestamps,
                                                                         photonlist.xPhotonPixels,
                                                                         photonlist.yPhotonPixels, ip, ditherdesc)

    # # Do the dither
    scimaps = []
    for pixfrac in [1]:
        driz = SpatialDrizzler(photonlists, ditherdesc, pixfrac=pixfrac)
        driz.run()
        scimaps.append(driz.driz.outsci)

    driz = TemporalDrizzler(photonlists, ditherdesc, pixfrac=0.5)
    driz.run()
    weights = driz.totWeightCube.sum(axis=0)[0]

    return scimaps, weights

# def form():
#     # TODO this needs to be implemented as a high level function
#     '''
#     This function has the sleeker multiprocessing suggestions from Jeb. It needs to be verified
#     :return:
#     '''
#     # Get dither offsets
#     # name = 'Trapezium'
#     # file = 'Trapezium_1547374552_dither.log'
#     name = 'KappaAnd_dither+lasercal'
#     file = 'KAnd_1545626974_dither.log'
#
#     wvlMin = 850
#     wvlMax = 1100
#     firstObsTime = 0
#     integrationTime = 10
#     drizzleconfig = [wvlMin, wvlMax, ]
#
#     # loc = os.path.join(os.getenv('MKID_DATA_DIR'), name, 'wavecal', file)
#     datadir = '/mnt/data0/isabel/mec'
#     loc = os.path.join(datadir, 'dithers', file)
#
#     ditherdesc = DitherDescription(MKIDObservingDither(name, loc, None, None), wvlMin=wvlMin, wvlMax=wvlMax,
#                                    startt=firstObsTime, intt=integrationTime, rotate=True)
#
#     # Quick save method for the reduced photon packets
#     # pkl_save = 'Trap_%i.pkl' % integrationTime
#     pkl_save = 'KAnd_%i.pkl' % integrationTime
#
#     # if mkidpipeline.pipe.tempfile(some_identifying_string, exists=True):
#     #     data = mkidpipeline.pipe.tempfile(some_identifying_string)
#     if os.path.exists(pkl_save):
#         with open(pkl_save, 'rb') as handle:
#             reduced_obslist = pickle.load(handle)
#     else:
#         begin = time.time()
#         print(os.path.join(datadir, 'out/Singles', name, 'wavecal_files', '*.h5'))
#         filenames = sorted(glob.glob(os.path.join(datadir, 'out/Singles', name, 'wavecal_files', '*.h5')))
#         if filenames ==[]:
#             print('No obsfiles found')
#         obsfiles = [ObsFile(file) for file in filenames]
#
#
#         def worker(mkidobdd, q, startw, stopw, foobar=5):
#             ob=mkidobdd
#             obfile=photontable.Obsfile(ob.h5)
#             q.put((ob, obfile.query(startw=startw, stopw=stopw, flags=some_flags),
#                    obfile.get_wcs()))
#             obfile.close()
#
#         arglist = [(ob, q, wvlMin, wvlMax) for ob in ditherdesc.description.obs]
#         pool = mp.Pool(ncpu)
#         photlists = pool.starmap(worker, arglist)
#         pool.close()
#
#         end = time.time()
#         print('Time spent: %f' % (end - begin))
#         # mkidpipeline.pipe.tempfile(some_identifying_string, data=data, overwrite=True)
#         with open(pkl_save, 'wb') as handle:
#             # pickle.dump(reduced_obslist, handle, protocol=pickle.HIGHEST_PROTOCOL)
#             pickle.dump(photlists, handle, protocol=-1)
#
#         # The WCS can be reassigned here rather than loading from obs each time
#         ditherdesc = DitherDescription(logdithdata, h5dithdata, drizzleconfig, multiplier=scale)
#         for ip, reduced_obs in enumerate(photlists):
#             reduced_obs['photRARad'], reduced_obs['photDecRad'] = get_wcs(reduced_obs['timestamps'],
#                                                                           reduced_obs['xPhotonPixels'],
#                                                                           reduced_obs['yPhotonPixels'],
#                                                                           ip, ditherdesc)
#
#     # # Do the dither
#     scimaps = []
#     for i, pixfrac in enumerate([0.85]):
#         driz = SpatialDrizzler(photlists[:25], ditherdesc, pixfrac=pixfrac)
#         driz.run()
#         scimaps.append(driz.driz.outsci)
#         plt.imshow(scimaps[0], origin='lower', norm=LogNorm())
#     plt.show(block=True)
#
#     print(scimaps[0].shape)  # This is the shape of the drizzled image
#     # >>> (268, 257)
#
#     exit()
#
#     # print(scimaps[0].shape)  # This is the shape of the drizzled image
#     # # >>> (268, 257)
#     #
#     # driz = TemporalDrizzler(reduced_obslist, ditherdesc, pixfrac=0.5)
#     # driz.run()
#     #
#     # print(np.shape(driz.totHypCube))  # This is the shape of the drizzled hypercube
#     # # >>> (6250, 2, 268, 257)  # 25 dither positions and integrationtime/frametime frames at each of them, nwvlbins
#     #                                                            # spectral frame
#     # for datacube in np.transpose(driz.totHypCube, (1,0,2,3))[::250]:  # iterate through the time axis to produce a series of
#     #                                                            # spectral cubes
#     #     for image in datacube:  #iterate through the wavelength axis
#     #         plt.imshow(image, origin='lower', norm=LogNorm())
#     #         plt.show()
#     #
#     # weights = np.sum(driz.totWeightCube, axis=0)[0]
#     # plt.imshow(np.sum(driz.totHypCube, axis=0)[0]/weights, origin='lower', norm=LogNorm())
#     # plt.show(block=True)  # This should plot the same 2d image as SpatialDrizzler(args).run().driz.outsci


if __name__ == '__main__':
    # plt.plot(range(5))
    # plt.show()
    # TODO cut this down so it uses form() once that works

    # Get dither offsets
    # name = 'Trapezium'
    # file = 'Trapezium_1547374552_dither.log'
    # name = 'KappaAnd_dither+lasercal'
    # file = 'KAnd_1545626974_dither.log'
    name = 'out/fordrizz/'
    file = 'HD34700_1547278116_dither.log'

    wvlMin = 850
    wvlMax = 1100
    firstObsTime = 0
    integrationTime = 1#10
    print(integrationTime)
    drizzleconfig = [wvlMin, wvlMax, firstObsTime, integrationTime]

    # loc = os.path.join(os.getenv('MKID_DATA_DIR'), name, 'wavecal', file)
    # datadir = '/mnt/data0/isabel/mec'
    # loc = os.path.join(datadir, 'dithers', file)
    datadir = '/mnt/data0/isabel/mec/'
    loc = os.path.join(datadir, 'dithers', file)

    logdithdata = MKIDObservingDither(name, loc, None, None)
    h5dithdata = getmetafromh5()
    # ditherdesc = DitherDescription(logdithdata, wvlMin, wvlMax, firstObsTime, integrationTime, multiplier=1)
    ditherdesc = DitherDescription(logdithdata, wvlMin, wvlMax, firstObsTime, integrationTime, multiplier=1, target='HD 34700')

    # Quick save method for the reduced photon packets
    # pkl_save = 'Trap_%i.pkl' % integrationTime
    pkl_save = 'KAnd_%i.pkl' % integrationTime

    if os.path.exists(pkl_save):
        with open(pkl_save, 'rb') as handle:
            reduced_obslist = pickle.load(handle)
    else:

        begin = time.time()
        # print(os.path.join(datadir, 'out/Singles', name, 'wavecal_files', '*.h5'))
        # filenames = sorted(glob.glob(os.path.join(datadir, 'out/Singles', name, 'wavecal_files', '*.h5')))
        filenames = sorted(glob.glob(os.path.join(datadir, name, '*.h5')))
        print(filenames)
        if filenames == []:
            print('No obsfiles found')
        obsfiles = [ObsFile(file) for file in filenames]


        def mp_worker(arg, reduced_obs_queue):

            obsfile, ditherdesc, ditherind = arg
            timestamps, xPhotonPixels, yPhotonPixels, wavelengths = reduce_obs(obsfile, ditherdesc, ditherind)
            photRARad, photDecRad = get_wcs(timestamps, xPhotonPixels, yPhotonPixels, ditherind, ditherdesc)
            reduced_obs = {'ditherind': ditherind,
                           'timestamps': timestamps,
                           'xPhotonPixels': xPhotonPixels,
                           'yPhotonPixels': yPhotonPixels,
                           'wavelengths': wavelengths,
                           'photRARad': photRARad,
                           'photDecRad': photDecRad}
            reduced_obs_queue.put(reduced_obs)


        ndither = len(ditherdesc.pos)
        print('stacking number of dithers: %i' % ndither)

        jobs = []
        reduced_obs_queue = mp.Queue()
        for ditherind, obsfile in enumerate(obsfiles[:ndither]):
            arg = ((obsfile, ditherdesc, ditherind))
            p = mp.Process(target=mp_worker, args=(arg, reduced_obs_queue))
            jobs.append(p)
            p.daemon = True
            p.start()
            # p.join()

        reduced_obslist = []
        order = np.zeros(ndither)
        for t in range(len(obsfiles[:ndither])):
            reduced_obslist.append(reduced_obs_queue.get())
            order[t] = reduced_obslist[t]['ditherind']
        reduced_obslist = np.array(reduced_obslist)
        sorted = np.argsort(order)
        reduced_obslist = reduced_obslist[sorted]

        end = time.time()
        print('Time spent: %f' % (end - begin))

        # with open(pkl_save, 'wb') as handle:
        #     pickle.dump(reduced_obslist, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #     # pickle.dump(reduced_obslist, handle, protocol=-1)

    for scale in [1]:  # np.linspace(-1, 1, 5):

        # The WCS can be reassigned here rather than loading from obs each time
        ditherdesc = DitherDescription(logdithdata, h5dithdata, drizzleconfig, multiplier=scale, target='HD 34700')
        for ip, reduced_obs in enumerate(reduced_obslist):
            reduced_obs['photRARad'], reduced_obs['photDecRad'] = get_wcs(reduced_obs['timestamps'],
                                                                          reduced_obs['xPhotonPixels'],
                                                                          reduced_obs['yPhotonPixels'],
                                                                          ip, ditherdesc)

        # # Do the dither
        scimaps = []
        for i, pixfrac in enumerate([0.25]):
            driz = SpatialDrizzler(reduced_obslist[:25], ditherdesc, pixfrac=pixfrac)
            driz.run()
            scimaps.append(driz.driz.outsci)
            plt.imshow(scimaps[i], origin='lower', vmax=300)  # , norm=LogNorm())
            # plt.xlim([0,400])
            # plt.ylim([0,400])
        plt.show(block=True)

    # exit()