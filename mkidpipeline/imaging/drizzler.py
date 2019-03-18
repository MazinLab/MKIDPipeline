"""
TODO
Add astroplan, ephem, drizzle, SharedArray to setup.py/yml. ephem can be conda installed, drizzle and SharedArray need to be pip
installed. I found that  astroplan needed to be pip installed otherwise some astropy import fails

Get con2pix calibration from Isabel's code and remove from here

Better handling of savestate of photonlists

Move plotting functionality to another module

Usage
-----

python drizzle.py

Author: Rupert Dodkins, Julian van Eyken            Date: Jan 2019


This code is adapted from Julian's testImageStack from the ARCONS pipeline.
"""
import matplotlib
import os
import numpy as np
import time
import multiprocessing as mp
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker
from scipy.ndimage import gaussian_filter
import ephem
from astropy import wcs
from astropy.coordinates import EarthLocation, Angle, SkyCoord
import astropy.units as u
from astroplan import Observer
import astropy
from astropy.io import fits
from drizzle import drizzle as stdrizzle
from mkidcore import pixelflags
from mkidpipeline.hdf.photontable import ObsFile
from mkidcore.corelog import getLogger
from mkidpipeline.config import MKIDObservingDataDescription, MKIDObservingDither
import cPickle as pickle
import mkidpipeline
import numpy.ma as ma
import pkg_resources as pkg
from mkidreadout.hardware.conex import CONEX2PIXEL
import argparse


def ditherp_2_pixel(positions):
    """ A function to convert the connex offset to pixel displacement"""
    positions = np.asarray(positions)
    pix = np.asarray(CONEX2PIXEL(positions[:, 0], positions[:, 1])) - np.array(CONEX2PIXEL(0, 0)).reshape(2, 1)
    return pix


class DitherDescription(object):
    """
    Info on the dither

    rotate determines if the effective integrations are pupil stablised or not

    TODO implement center of rotation
    """

    def __init__(self, dither, virPixCen=(25, 0), observatory='Subaru', target='* kap And', rot_rate_multiplier=1):
        self.description = dither
        self.target = target
        try:
            self.coords = dither.obs[0].lookup_coodinates(queryname=target)
        except astropy.coordinates.name_resolve.NameResolveError:
            getLogger(__name__).warning(f'Unable to resolve coordinates for target name {target}, using (0,0).')
            self.coords = self.SkyCoord('0 deg', '0 deg')
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
        rot_rate *= rot_rate_multiplier

        self.dithHAs = [np.trapz(rot_rate[:ix], x=times[:ix] - times[0]) for ix in range(1, len(times) + 1)]

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
        self.nPixRA, self.nPixDec = 300, 300  # 250, 250

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
        self.w.wcs.crpix = np.array([self.nPixRA / 2., self.nPixDec / 2.]) + np.array(
            [self.virPixCen[0][0], self.virPixCen[1][0]])
        self.w.wcs.cdelt = np.array([self.vPlateScale, self.vPlateScale])
        self.w.wcs.crval = [self.cenRA, self.cenDec]
        self.w.wcs.ctype = ["RA-----", "DEC----"]
        self.w._naxis1 = self.nPixRA
        self.w._naxis2 = self.nPixDec
        print(self.w)


class SpectralDrizzler(Drizzler):
    """ Generate a spatially dithered fits dataacube from a set dithered dataset """

    def __init__(self, photonlists, metadata, pixfrac=1.):
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

    def __init__(self, photonlists, metadata, pixfrac=1., nwvlbins=2, timestep=0.1,
                 wvlMin=0, wvlMax=np.inf, startt=0, intt=10):

        super().__init__(photonlists, metadata)

        self.nwvlbins = nwvlbins
        self.timestep = timestep  # seconds

        self.ndithers = len(self.files)
        self.pixfrac = pixfrac
        self.wvlbins = np.linspace(wvlMin, wvlMax, self.nwvlbins + 1)
        self.ntimebins = int(intt / self.timestep)
        self.timebins = np.linspace(startt,
                                    startt + intt,
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

    def makeHyper(self, file, applyweights=False, applymask=False, maxCountsCut=200):
        if applyweights:
            weights = file['weight']
        else:
            weights = None
        sample = np.vstack((file['timestamps'], file['wavelengths'], file['photDecRad'], file['photRARad']))
        bins = np.array([self.timebins, self.wvlbins, self.ypix, self.xpix])
        hypercube, bins = np.histogramdd(sample.T, bins, weights=weights, )

        if applymask:
            usablemask = np.int_(np.rot90(file['usablemask']))
            hypercube *= usablemask

        if maxCountsCut:
            hypercube *= np.int_(hypercube < maxCountsCut)

        times, wavelengths, thisGridDec, thisGridRA = bins

        w = wcs.WCS(naxis=2)
        w.wcs.crpix = [self.xpix / 2., self.ypix / 2.]
        w.wcs.cdelt = np.array([thisGridRA[1] - thisGridRA[0], thisGridDec[1] - thisGridDec[0]])
        w.wcs.crval = [thisGridRA[self.xpix // 2], thisGridDec[self.ypix // 2]]
        w.wcs.ctype = ["RA-----", "DEC----"]
        w._naxis1 = self.xpix
        w._naxis2 = self.ypix

        return hypercube, w


class SpatialDrizzler(Drizzler):
    """ Generate a spatially dithered fits image from a set dithered dataset """

    def __init__(self, photonlists, metadata, pixfrac=1.):
        Drizzler.__init__(self, photonlists, metadata)
        self.driz = stdrizzle.Drizzle(outwcs=self.w, pixfrac=pixfrac)

    def identify_hotpix(self, metadata, dithfrac=0.1, min_count=500, plot=True):
        ndithers = len(metadata.dithHAs)
        hot_cube = np.zeros((ndithers, metadata.ypix, metadata.xpix))
        dith_cube = np.zeros_like(hot_cube)
        for ix, file in enumerate(self.files):
            dith_cube[ix], _ = self.makeImage(file)
            # plt.imshow(dith_cube[ix], origin='lower')
            # plt.show()
        # hot_cube[dith_cube > min_count] = ma.masked
        hot_cube[dith_cube > min_count] = 1
        hot_amount_map = np.sum(hot_cube, axis=0)  # hot_cube.count(axis=0)
        self.hot_mask = hot_amount_map / ndithers > dithfrac
        if plot:
            plt.imshow(self.hot_mask, origin='lower')
            plt.show(block=True)

    def run(self, save_file=None, applymask=False):
        for ix, file in enumerate(self.files):
            getLogger(__name__).debug('Processing %s', file)
            tic = time.clock()
            insci, inwcs = self.makeImage(file)
            if applymask:
                insci *= np.logical_not(self.hot_mask)
            getLogger(__name__).debug('Image load done. Time taken (s): %s', time.clock() - tic)
            inwht = (insci != 0).astype(int)
            self.driz.add_image(insci, inwcs, inwht=inwht)
        if save_file:
            self.driz.write(save_file)

        # TODO introduce total_exp_time variable and complete these steps

    def makeImage(self, file, applyweights=True, applymask=False, maxCountsCut=10000):

        weights = file['weight'] if applyweights else None

        # TODO mixing pixels and radians per variable names
        thisImage, thisGridDec, thisGridRA = np.histogram2d(file['photDecRad'], file['photRARad'],
                                                            weights=weights,
                                                            bins=[self.ypix, self.xpix],
                                                            normed=False)

        if applymask:
            usablemask = np.rot90(file['usablemask']).astype(int)
            # usablemask = file['usablemask'].T.astype(int)
            # thisImage *= ~usablemask
            thisImage *= usablemask

        if maxCountsCut:
            thisImage *= thisImage < maxCountsCut

        w = wcs.WCS(naxis=2)
        w.wcs.crpix = [self.xpix / 2., self.ypix / 2.]
        w.wcs.cdelt = np.array([thisGridRA[1] - thisGridRA[0], thisGridDec[1] - thisGridDec[0]])
        w.wcs.crval = [thisGridRA[self.xpix // 2], thisGridDec[self.ypix // 2]]
        w.wcs.ctype = ["RA-----", "DEC----"]
        w._naxis1 = self.xpix
        w._naxis2 = self.ypix
        return thisImage, w


def get_wcs(x, y, ditherdesc, ditherind, nxpix=146, nypix=140, toa_rotation=False, randoffset=False, nPhot=1,
            platescale=0.01 / 3600.):
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
        earthrate = 1. / 86164.1 * 2 * np.pi  # * 500
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
        vgrid_photons = np.array([ditherdesc.centroids[0][ditherind] + x - nxpix / 2,
                                  ditherdesc.centroids[1][ditherind] + y - nypix / 2])

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


def annotate_axis(im, ax, width, platescale, cenCoords):
    rad = platescale * width / 2.

    xticks = np.linspace(-rad, rad, 5) + cenCoords[0]
    yticks = np.linspace(-rad, rad, 5) + cenCoords[1]
    xticklabels = ["{:0.4f}".format(i) for i in xticks]
    yticklabels = ["{:0.4f}".format(i) for i in yticks]
    ax.set_xticks(np.linspace(-0.5, width - 0.5, 5))
    ax.set_yticks(np.linspace(-0.5, width - 0.5, 5))
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)
    im.axes.tick_params(color='white', direction='in', which='both', right=True, top=True, width=1,
                        length=10)  # , labelcolor=fg_color)
    im.axes.tick_params(which='minor', length=5, width=0.5)
    ax.xaxis.set_minor_locator(ticker.FixedLocator(np.linspace(-0.5, width - 0.5, 33)))
    ax.yaxis.set_minor_locator(ticker.FixedLocator(np.linspace(-0.5, width - 0.5, 33)))

    ax.set_xlabel('RA ($^{\circ}$)')
    ax.set_ylabel('Dec ($^{\circ}$)')


def pretty_plot(image, platescale, cenCoords, log_scale=False, vmin=None, vmax=None):
    if log_scale:
        norm = LogNorm()
    else:
        norm = None
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(image, origin='lower', vmin=vmin, vmax=vmax, norm=norm)
    annotate_axis(cax, ax, image.shape[0], platescale, cenCoords)
    cb = plt.colorbar(cax)
    cb.ax.set_title('Counts')
    plt.show(block=True)


def write_fits(image, filename):
    hdu = fits.PrimaryHDU(image)
    hdu.writeto(filename, clobber=True)


def load_data(ditherdesc, wvlMin, wvlMax, startt, intt, tempfile='drizzler_tmp_{target}.pkl',
              tempdir='', usecache=True, clearcache=False):
    ndither = len(ditherdesc.description.obs)  # len(dither.obs)
    target = ditherdesc.target

    pkl_save = os.path.join(tempdir, tempfile.format(target))
    if clearcache:  # TODO the cache must be autocleared if the query parameters would alter the contents
        os.remove(pkl_save)
    try:
        if not usecache:
            raise FileNotFoundError
        with open(pkl_save, 'rb') as f:
            data = pickle.load(f)
        print('loaded', pkl_save)
    except FileNotFoundError:
        begin = time.time()
        filenames = [o.h5 for o in ditherdesc.description.obs]
        if not filenames:
            # TODO use logging!
            print('No obsfiles found')

        def mp_worker(file, q, startt=startt, intt=intt, startw=wvlMin, stopw=wvlMax):
            obsfile = ObsFile(file)
            usableMask = np.array(obsfile.beamFlagImage) == pixelflags.GOODPIXEL

            photons = obsfile.query(startw=startw, stopw=stopw, startt=startt, intt=intt)
            weights = photons['SpecWeight'] * photons['NoiseWeight']
            getLogger(__name__).info("Fetched {} photons from {}".format(len(photons), file))

            x, y = obsfile.xy(photons)
            del obsfile

            q.put({'file': file, 'timestamps': photons["Time"], 'xPhotonPixels': x, 'yPhotonPixels': y,
                   'wavelengths': photons["Wavelength"], 'weight': weights, 'usablemask': usableMask})

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

        # Wait for all of the processes to finish fetching their data, this should hang until all the data has been
        # fetched
        for j in jobs:
            j.join()

        data = []
        for t in range(ndither):
            data.append(data_q.get())
        data.sort(key=lambda k: filenames.index(k['file']))

        getLogger(__name__).debug('Time spent: %f' % (time.time() - begin))

        with open(pkl_save, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Do the dither
    for i, d in enumerate(data):
        radec = get_wcs(d['xPhotonPixels'], d['yPhotonPixels'], ditherdesc, i,
                        nxpix=ditherdesc.xpix.ncols, nypix=ditherdesc.ypix, toa_rotation=False,
                        randoffset=False, nPhot=1, platescale=ditherdesc.platescale)
        d['photRARad'], d['photDecRad'] = radec

    return data


def form(dither, dim=2, rotrate_mult=1, target='', wvlMin=850, wvlMax=1100, startt=0, intt=60, pixfrac=.5):
    """

    :param dim: 2->image, 3->spectral cube, 4->sequence of spectral cubes. If drizzle==False then dim is ignored
    :param rotate: 0 or 1
    :param target:
    :param ditherlog:
    :param obsdir:
    :param wvlMin:
    :param wvlMax:
    :param startt:
    :param intt:
    :param pixfrac:
    :param drizzle bool for output format

    :return:
    """

    target = dither.name if not target else target
    ditherdesc = DitherDescription(dither, rot_rate_multiplier=rotrate_mult, target=target)
    data = load_data(ditherdesc, wvlMin, wvlMax, startt, intt)

    if dim not in range(2, 5):
        raise ValueError('dim must be either 2, 3 or 4')

    elif dim == 2:
        driz = SpatialDrizzler(data, ditherdesc, pixfrac=pixfrac)
        # driz.identify_hotpix(ditherdesc)
        driz.run(applymask=False)
        outsci = driz.driz.outsci
        outwcs = driz.w

    elif dim == 3:
        # TODO implement, is this even necessary. On hold till interface specification and dataproduct definition
        raise NotImplementedError

    elif dim == 4:
        tdriz = TemporalDrizzler(data, ditherdesc, pixfrac=pixfrac, nwvlbins=1, timestep=1.,
                                 wvlMin=wvlMin, wvlMax=wvlMax, startt=startt, intt=intt)
        tdriz.run()
        outsci = tdriz.totHypCube
        outwcs = tdriz.w
        # weights = tdriz.totWeightCube.sum(axis=0)[0]
        # TODO: While we can still have a reference-point WCS solution this class needs a drizzled WCS helper as the
        # WCS solution changes with time, right?

    return outsci, outwcs


if __name__ == '__main__':
    matplotlib.use('QT5Agg', force=True)
    matplotlib.rcParams['backend'] = 'Qt5Agg'
    import matplotlib.pylab as plt

    parser = argparse.ArgumentParser(description='MKID Wavelength Calibration Utility')
    parser.add_argument('cfg', type=str, help='The configuration file')
    parser.add_argument('-wl', type=float, dest='wvlMin', help='', default=850)
    parser.add_argument('-wh', type=float, dest='wvlMax', help='', default=1100)
    parser.add_argument('-t0', type=int, dest='startt', help='', default=0)
    parser.add_argument('-it', type=int, dest='intt', help='', default=60)
    args = parser.parse_args()

    # set up logging
    mkidpipeline.logtoconsole()

    # load as a task configuration
    cfg = mkidpipeline.config.load_task_config(args.cfg)

    wvlMin = args.wvlMin
    wvlMax = args.wvlMax
    startt = args.startt
    intt = args.intt
    pixfrac = cfg.pixfrac
    dither = cfg.dither

    image, drizwcs = form(dither, 'spatial', wvlMin=wvlMin, wvlMax=wvlMax, startt=startt, intt=intt, pixfrac=pixfrac)

    pretty_plot(image, drizwcs.wcs.cdelt[0] * 3600, drizwcs.wcs.crval, vmin=100, vmax=600)
    write_fits(image, cfg.dither.target + '_mean.fits')

    tess, drizwcs = form(dither, 4, wvlMin=wvlMin, wvlMax=wvlMax, startt=startt, intt=intt, pixfrac=pixfrac)

    y = np.ma.masked_where(tess[:, 0] == 0, tess[:, 0])
    medDither = np.ma.median(y, axis=0).filled(0)

    pretty_plot(medDither, drizwcs.wcs.cdelt[0] * 3600, drizwcs.wcs.crval, vmin=1, vmax=10)
    pretty_plot(medDither, drizwcs.wcs.cdelt[0] * 3600, drizwcs.wcs.crval, log_scale=True)
    write_fits(gaussian_filter(medDither, 1), cfg.dither.target + '_med1.fits')  # TODO WHY!!!!!!???
    write_fits(gaussian_filter(medDither, 0.5), cfg.dither.target + '_meddot5.fits')
