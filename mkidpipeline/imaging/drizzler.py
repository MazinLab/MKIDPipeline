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
    def __init__(self, dither, observatory='Subaru', target='* kap And', multiplier=1):
        self.description = dither
        self.pos = dither.pos

        self.coords = dither.obs[0].lookup_coodinates(queryname=target)

        self.centroids = ditherp_2_pixel(dither.pos)
        self.virPixCen = np.array([[25, 0]]).T#ditherp_2_pixel([(0,0)]) #+ np.array([(-30,-30)]).T

        times = np.array([o.start for o in dither.obs])
        site = EarthLocation.of_site(observatory)

        apo = Observer.at_site(observatory)
        altaz = apo.altaz(astropy.time.Time(val=times, format='unix'), self.coords)
        Earthrate = 2 * np.pi / u.sday.to(u.second) # * 500
        obs_const = Earthrate * np.cos(site.geodetic.lat.rad)
        rot_rate = obs_const * np.cos(altaz.az.radian) / np.cos(altaz.alt.radian) * multiplier
        self.dithHAs = [np.trapz(rot_rate[:ix], x=times[:ix] - times[0]) for ix in range(1, len(times)+1)]

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


def get_wcs(time, x, y, coordinate_frame, toa_rotation=False, randoffset=False, nPhot=1):
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

    getLogger(__name__).info('Calculating RA/Decs for dither {}'.format(ditherind))

    if toa_rotation:
        # TODO update this rot_rate to altaz model
        earthrate = 1./86164.1 * 2*np.pi #* 500
        obs_const = earthrate * np.cos(np.deg2rad(19.7))
        rot_rate = obs_const * np.cos(az) / np.cos(alt)
        photHAs = timestamps * 1e-6 * rot_rate


        hourangles = ditherdesc.dithHAs[ditherind] + photHAs

        rotationMatrix = np.array([[np.cos(hourangles), -np.sin(hourangles)],
                                   [np.sin(hourangles), np.cos(hourangles)]]).T

        centroids = np.array([-ditherdesc.xCenRes[ditherind] + xPhotonPixels - ditherdesc.xpix / 2,
                              -ditherdesc.yCenRes[ditherind] + yPhotonPixels - ditherdesc.ypix / 2])

    else:
        hourangle = ditherdesc.dithHAs[ditherind]
        print('HAs', hourangle)

        rotationMatrix = np.array([[np.cos(hourangle), -np.sin(hourangle)],
                                   [np.sin(hourangle), np.cos(hourangle)]])

        # put each photon from the dither into its raster location on the virtual grid
        vgrid_photons = np.array([ditherdesc.centroids[0][ditherind] + xPhotonPixels - ditherdesc.xpix/2,
                                     ditherdesc.centroids[1][ditherind] + yPhotonPixels - ditherdesc.ypix/2])

        # offset these to the center of rotation
        cor_photons = vgrid_photons - ditherdesc.virPixCen

        # rotate that virtual grid so that that those photons now occupy the part of the sky that was sampled
        rotated_vgrid = np.dot(rotationMatrix, cor_photons)

        # undo the COR offset
        skyframe_photons = rotated_vgrid + ditherdesc.virPixCen

    rightAscensionOffset = ditherdesc.platescale * (skyframe_photons[0]) # -1 here just orientates the final image
    declinationOffset = ditherdesc.platescale * (skyframe_photons[1])

    # Convert centroid positions in DD:MM:SS.S and HH:MM:SS.S format to radians.
    centroidRADeg = ephem.hours(ditherdesc.cenRA).real
    centroidDecDeg = ephem.degrees(ditherdesc.cenDec).real

    # Add the photon arcsecond offset to the centroid offset.
    photDecDeg = centroidDecDeg + declinationOffset
    photRADeg = centroidRADeg + rightAscensionOffset

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
    drizzleconfig = [wvlMin, wvlMax, firstObsTime, integrationTime]

    # loc = os.path.join(os.getenv('MKID_DATA_DIR'), name, 'wavecal', file)
    # datadir = '/mnt/data0/isabel/mec'
    # loc = os.path.join(datadir, 'dithers', file)
    datadir = '/mnt/data0/isabel/mec/'
    loc = os.path.join(datadir, 'dithers', file)

    logdithdata = MKIDObservingDither('HD 34700', os.path.join(datadir, 'dithers', file), None, None)

    ditherdesc = DitherDescription(logdithdata, multiplier=1)

    pkl_save = 'drizzler_tmp_{}.pk'.format(logdithdata.name)
    try:
        with open(pkl_save, 'rb') as f:
            reduced_obslist = pickle.load(f)
    except IOError:

        begin = time.time()
        filenames = sorted(glob.glob(os.path.join(datadir, name, '*.h5')))

        if not filenames:
            print('No obsfiles found')

        def mp_worker(file, q, startt=None, intt=1):
            obsfile = ObsFile(file)
            photons = obsfile.query(startw=wvlMin, stopw=wvlMax, startt=startt, intt=intt,
                                    flagToUse=pixelflags.GOODPIXEL)  # hot, flat and wavelength masks?

            getLogger(__name__).info("Fetched {} photons from {}".format(len(photons), file))

            x, y = obsfile.xy(photons)
            obsfile.close()

            q.put({'file': file, 'timestamps': photons["Time"], 'xPhotonPixels': x, 'yPhotonPixels': y,
                   'wavelengths':  photons["Wavelength"], 'weight':0})

        ndither = len(ditherdesc.pos)
        getLogger(__name__).info('stacking number of dithers: %i'.format(ndither))

        jobs = []
        reduced_obs_queue = mp.Queue()
        for f in filenames[:ndither]:
            p = mp.Process(target=mp_worker, args=(f, reduced_obs_queue))
            jobs.append(p)
            p.daemon = True
            p.start()

        for j in jobs:
            j.join()

        data = []
        order = np.zeros(ndither)
        for t in range(ndither):
            data.append(reduced_obs_queue.get())
        data.sort(key=lambda k: filenames.index(k['file']))

        getLogger(__name__).debug('Time spent: %f' % (time.time() - begin))

        with open(pkl_save, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

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