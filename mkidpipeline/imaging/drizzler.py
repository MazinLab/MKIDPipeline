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
from mkidpipeline.config import MKIDObservingDataDescription, MKIDObservingDither, load_task_config
# import cPickle as pickle
import pickle

def con2pix(xCon, yCon):
    """
    This is temporary. Should grab this conversion from elsewhere

    :param xCon:
    :param yCon:
    :return:
    """
    from scipy.optimize import curve_fit


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

    def func(x, slope, intercept):
        return x * slope + intercept

    xopt, xcov = curve_fit(func, xConFit, xPosFit, sigma=xPoserrFit)
    yopt, ycov = curve_fit(func, yConFit, yPosFit, sigma=yPoserrFit)

    def con2Pix(xCon, yCon, func):
        return [func(xCon, *xopt), func(yCon, *yopt)]

    xPos, yPos = con2Pix(xCon, yCon, func)

    return [-xPos, -yPos]


def ditherp_2_pixel(positions):
    """ A function to convert the connex offset to pixel displacement"""
    positions = np.asarray(positions)
    pix = np.asarray(con2pix(positions[:, 0], positions[:, 1])) - np.array(con2pix(0, 0)).reshape(2, 1)
    return pix


class DitherDescription(object):
    """
    Info on the dither

    rotate determines if the effective integrations are pupil stablised or not

    TODO implement center of rotation
    """
    def __init__(self, dither, virPixCen=(25, 0), observatory='Subaru', target='* kap And', multiplier=1):

        self.description = dither
        self.coords = dither.obs[0].lookup_coodinates(queryname=target)
        self.cenRA, self.cenDec = self.coords.ra.deg, self.coords.dec.deg
        self.virPixCen = np.array([list(virPixCen)]).T

        self.centroids = ditherp_2_pixel(dither.pos)
        self.cenRes = self.centroids - self.virPixCen
        self.xCenRes, self.yCenRes = self.cenRes
        inst_info = dither.obs[0].instrument_info
        self.xpix = inst_info.beammap.ncols
        self.ypix = inst_info.beammap.nrows
        self.platescale = inst_info.platescale.to(u.deg).value  # 10 mas

        times = np.array([o.start for o in dither.obs])
        site = EarthLocation.of_site(observatory)
        apo = Observer.at_site(observatory)
        altaz = apo.altaz(astropy.time.Time(val=times, format='unix'), self.coords)
        earthrate = 2 * np.pi / u.sday.to(u.second)
        rot_rate = earthrate * np.cos(site.geodetic.lat.rad) * np.cos(altaz.az.radian) / np.cos(altaz.alt.radian)
        rot_rate *= multiplier

        self.dithHAs = [np.trapz(rot_rate[:ix], x=times[:ix] - times[0]) for ix in range(1, len(times)+1)]

        getLogger(__name__).debug("HAs: %s", self.dithHAs)

    def plot(self):
        rotationMatrix = np.array([[np.cos(self.dithHAs), -np.sin(self.dithHAs)],
                                   [np.sin(self.dithHAs), np.cos(self.dithHAs)]]).T

        centroidRotated = (np.dot(rotationMatrix, np.array([self.xCenRes, self.yCenRes])).diagonal(axis1=0, axis2=2) +
                           [self.virPixCen[0], self.virPixCen[1]])

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

    def __init__(self, photonlists, metadata, pixfrac=0, nwvlbins=2, timestep=0.1):

        super().__init__(photonlists, metadata)

        self.nwvlbins = nwvlbins
        self.timestep = timestep  # seconds

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

    def run(self, save_file=None):
        for ix, file in enumerate(self.files):
            getLogger(__name__).debug('Processing %s', file)
            tic = time.clock()
            insci, inwcs = self.makeImage(file)
            getLogger(__name__).debug('Image load done. Time taken (s): %s', time.clock() - tic)
            inwht = (insci != 0).astype(int)
            self.driz.add_image(insci, inwcs, inwht=inwht)
        if save_file:
            self.driz.write(save_file)

        # TODO introduce total_exp_time variable and complete these steps

    def makeImage(self, file):
        thisImage, thisGridDec, thisGridRA = np.histogram2d(file['photDecRad'], file['photRARad'],
                                                            weights=file['weight'],
                                                            bins=[self.ypix, self.xpix],
                                                            normed=False)
        # plt.hist(file['weight'])
        # plt.show()
        w = wcs.WCS(naxis=2)
        w.wcs.crpix = [self.xpix/2., self.ypix/2.]
        w.wcs.cdelt = np.array([thisGridRA[1]-thisGridRA[0], thisGridDec[1]-thisGridDec[0]])
        w.wcs.crval = [thisGridRA[self.xpix//2], thisGridDec[self.ypix//2]]
        w.wcs.ctype = ["RA-----", "DEC----"]

        w._naxis1 = self.xpix
        w._naxis2 = self.ypix
        return thisImage, w


def get_wcs(x, y, ditherdesc, ditherind, nxpix=146, nypix=140, toa_rotation=False, randoffset=False, nPhot=1,
            platescale=(10 * u.mas).to(u.deg).value):
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

    if toa_rotation:
        raise NotImplementedError
        # TODO update this rot_rate to altaz model
        earthrate = 1./86164.1 * 2*np.pi #* 500
        obs_const = earthrate * np.cos(np.deg2rad(19.7))
        rot_rate = obs_const * np.cos(az) / np.cos(alt)
        photHAs = time * 1e-6 * rot_rate

        hourangles = ha_ref + photHAs

        rotationMatrix = np.array([[np.cos(hourangles), -np.sin(hourangles)],
                                   [np.sin(hourangles), np.cos(hourangles)]]).T

        centroids = np.array([-ditherdesc.xCenRes[ditherind] + x - nxpix / 2,
                              -ditherdesc.yCenRes[ditherind] + y - nypix / 2])

    else:
        hourangles = ditherdesc.dithHAs[ditherind]

        rotationMatrix = np.array([[np.cos(hourangles), -np.sin(hourangles)],
                                   [np.sin(hourangles), np.cos(hourangles)]])

        # put each photon from the dither into its raster location on the virtual grid
        vgrid_photons = np.array([ditherdesc.centroids[0][ditherind] + x - nxpix/2,
                                  ditherdesc.centroids[1][ditherind] + y - nypix/2])

        # offset these to the center of rotation
        cor_photons = vgrid_photons - ditherdesc.virPixCen

        # rotate that virtual grid so that that those photons now occupy the part of the sky that was sampled
        rotated_vgrid = np.dot(rotationMatrix, cor_photons)

        # undo the COR offset
        skyframe_photons = rotated_vgrid + ditherdesc.virPixCen

    # Convert centroid positions in DD:MM:SS.S and HH:MM:SS.S format to radians.
    centroidRADeg = ephem.hours(ditherdesc.cenRA).real
    centroidDecDeg = ephem.degrees(ditherdesc.cenDec).real

    # Add the photon arcsecond offset to the centroid offset.
    photDecDeg = centroidDecDeg + platescale * skyframe_photons[1]
    photRADeg = centroidRADeg + platescale * skyframe_photons[0]

    if randoffset:
        np.random.seed(42)  # so random values always same
        xRand = np.random.rand(nPhot) * plateScale - plateScale / 2.0
        yRand = np.random.rand(nPhot) * plateScale - plateScale / 2.0  # Not the same array!
        ditherRAs = xRand * np.cos(hourangles) - yRand * np.sin(hourangles)
        ditherDecs = yRand * np.cos(hourangles) + xRand * np.sin(hourangles)
    else:
        ditherRAs = 0
        ditherDecs = 0

    photRADeg = photRADeg + ditherRAs
    photDecDeg = photDecDeg + ditherDecs

    return photRADeg, photDecDeg


if __name__ == '__main__':

    # Get dither offsets
    # name = 'Trapezium'
    # file = 'Trapezium_1547374552_dither.log'
    # name = 'KappaAnd_dither+lasercal'
    # file = 'KAnd_1545626974_dither.log'
    name = 'out/fordrizz/'
    file = 'HD34700_1547278116_dither.log'
    datadir = '/mnt/data0/isabel/mec/'
    wvlMin = 850
    wvlMax = 1100
    firstObsTime = 0
    integrationTime = 1
    pixfrac = .5

    load_task_config(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'pipe.yml'))

    dither = MKIDObservingDither('HD 34700', os.path.join(datadir, 'dithers', file), None, None)
    ndither = len(dither.obs)

    pkl_save = 'drizzler_tmp_{}.pkl'.format(dither.name)
    # os.remove(pkl_save)
    try:
        with open(pkl_save, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:

        begin = time.time()
        filenames = sorted(glob.glob(os.path.join(datadir, name, '*.h5')))
        # print(filenames)
        if not filenames:
            print('No obsfiles found')

        def mp_worker(file, q, startt=None, intt=1):
            obsfile = ObsFile(file)
            usableResIDs = obsfile.beamImage[obsfile.beamFlagImage == pixelflags.GOODPIXEL]
            photons = obsfile.query(startw=wvlMin, stopw=wvlMax, startt=startt, intt=intt, resid=usableResIDs)
            weights = photons['SpecWeight'] * photons['NoiseWeight']

            getLogger(__name__).info("Fetched {} photons from {}".format(len(photons), file))

            x, y = obsfile.xy(photons)
            del obsfile

            q.put({'file': file, 'timestamps': photons["Time"], 'xPhotonPixels': x, 'yPhotonPixels': y,
                   'wavelengths':  photons["Wavelength"], 'weight': weights})


        getLogger(__name__).info('stacking number of dithers: %i'.format(ndither))

        jobs = []
        data_q = mp.Queue()
        if ndither > 25:
            raise RuntimeError('Needs rewrite, will use too many cores')

        for f in filenames[:ndither]:
            p = mp.Process(target=mp_worker, args=(f, data_q))
            jobs.append(p)
            p.daemon = True
            p.start()

        # for j in jobs:
        #     j.join()

        data = []
        order = np.zeros(ndither)
        for t in range(ndither):
            data.append(data_q.get())
        data.sort(key=lambda k: filenames.index(k['file']))

        getLogger(__name__).debug('Time spent: %f' % (time.time() - begin))

        with open(pkl_save, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Do the dither
    ditherdesc = DitherDescription(dither, multiplier=-1)
    # inst_info = dither.obs[0].instrument_info

    for i, d in enumerate(data):
        radec = get_wcs(d['xPhotonPixels'], d['yPhotonPixels'], ditherdesc, i,
                        nxpix=140, nypix=146, toa_rotation=False,
                        randoffset=False, nPhot=1, platescale=(10 * u.mas).to(u.deg).value)
        d['photRARad'], d['photDecRad'] = radec

    driz = SpatialDrizzler(data, ditherdesc, pixfrac=pixfrac)
    driz.run()

    plt.imshow(driz.driz.outsci, origin='lower', vmax=10)
    plt.show(block=True)

    # tdriz = TemporalDrizzler(data, ditherdesc, pixfrac=pixfrac)
    # tdriz.run()
    # weights = tdriz.totWeightCube.sum(axis=0)[0]


