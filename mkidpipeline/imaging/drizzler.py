"""
*** Warning ***

    The STScI drizzle module currently appears to have a bug which has been raised under issue #50.
    In the meantime increment_id() patches the bug

Examples

    $ python drizzle.py /path/to/mkidpipeline/mkidpipeline/imaging/drizzler.yml

    or

    >>> dither = MKIDDitheredObservation(**kwargs)
    >>> drizzled = drizzler.form(dither, mode='spatial')
    >>> drizzled.writefits('output.fits')

Classes

    DrizzleParams   : Calculates and stores the relevant info for Drizzler
    Canvas          : Called by the "Drizzler" classes. Generates the canvas that is drizzled onto
    SpatialDrizzler : Generate a spatially dithered image from a set dithered dataset
    TemporalDrizzler: Generates a spatially dithered 4-cube (xytw)
    ListDrizzler    : Not implemented yet
    DrizzledData    : Saves the drizzled data as FITS

Functions

    increment_id    : Monkey patch for STScI drizzle class of drizzle package
    mp_worker       : Genereate a reduced, reformated photonlist
    load_data       : Consolidate all dither positions
    form            : Takes in a MKIDDitheredObservation object and drizzles the dithers onto a common sky grid
    get_star_offset : Get the rotation_center offset parameter for a dither


TODO:
    * Add astroplan, drizzle, to setup.py/yml. drizzle need to be pip installed. I found that astroplan needed to be pip
    installed otherwise some astropy import fails
    * Make and test ListDrizzler


Author: Rupert Dodkins,                                 Date: Jan 2020

"""
import os
import numpy as np
import time
import multiprocessing as mp
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
import pickle
from astropy import wcs
from astropy.coordinates import EarthLocation, Angle, SkyCoord
import astropy.units as u
from astroplan import Observer
import astropy
from astropy.utils.data import Conf
from astropy.io import fits
from drizzle import drizzle as stdrizzle
from mkidcore import pixelflags
from mkidpipeline.hdf.photontable import ObsFile
from mkidcore.corelog import getLogger
import mkidcore.corelog as pipelinelog
import mkidpipeline
from mkidcore.instruments import CONEX2PIXEL
import argparse
from mkidpipeline.utils.utils import get_device_orientation

def increment_id(self):
    """
    monkey patch for STScI drizzle class of drizzle package

    Increment the id count and add a plane to the context image if needed

    Drizzle tracks which input images contribute to the output image
    by setting a bit in the corresponding pixel in the context image.
    The uniqid indicates which bit. So it must be incremented each time
    a new image is added. Each plane in the context image can hold 32 bits,
    so after each 32 images, a new plane is added to the context.
    """

    getLogger(__name__).warning('Using increment_id monkey patch')

    # Compute what plane of the context image this input would
    # correspond to:

    planeid = int(self.uniqid / 32)

    # Add a new plane to the context image if planeid overflows

    if self.outcon.shape[0] == planeid:
        plane = np.zeros_like(self.outcon[0])
        self.outcon = np.append(self.outcon, [plane], axis=0)

    # Increment the id
    self.uniqid += 1

stdrizzle.Drizzle.increment_id = increment_id

class DrizzleParams(object):
    """
    Calculates and stores the relevant info for Drizzler

    """
    def __init__(self, dither, used_inttime, wcs_timestep=None, pixfrac=1.):
        self.dither = dither
        self.used_inttime = used_inttime
        self.pixfrac = pixfrac
        self.get_coords()

        if wcs_timestep:
            self.wcs_timestep = wcs_timestep
        else:
            self.wcs_timestep = self.non_blurring_timestep()

    def get_coords(self, simbad_lookup=False):
        """
        Get the SkyCoord type coordinates to use for center of sky grid that is drizzled onto

        :param simbad_lookup: use dither.target to get the J2000 coordinates
        """

        if simbad_lookup:
            self.coords = SkyCoord.from_name(self.dither.target)
            getLogger(__name__).warning('Using J2000 coordinates at: {} {} for target {} from Simbad for canvas wcs'.format(self.coords.ra, self.coords.dec, self.dither.target))
        else:
            self.coords = SkyCoord(self.dither.ra, self.dither.dec, unit=('hourangle', 'deg'))
            getLogger(__name__).info('Using coordinates at: {} {} for target {} from h5 header for canvas wcs'.format(self.coords.ra, self.coords.dec, self.dither.target))

    def dither_pixel_vector(self, center=(0, 0)):
        """
        A function to convert a list of conex offsets to pixel displacement

        :param center: the origin for the vector
        :return:
        """
        positions = np.asarray(self.dither.pos)
        pix = np.asarray(CONEX2PIXEL(positions[:, 0], positions[:, 1])) - np.array(CONEX2PIXEL(*center)).reshape(2, 1)
        return pix

    def non_blurring_timestep(self, allowable_pixel_smear=1):
        """
        [1] Smart, W. M. 1962, Spherical Astronomy, (Cambridge: Cambridge University Press), p. 55

        :param allowable_pixel_smear: the resolution element threshold
        """

        # get the field rotation rate at the start of each dither
        dith_start_times = np.array([o.start for o in self.dither.obs])

        site = astropy.coordinates.EarthLocation.of_site(self.dither.observatory)
        apo = Observer.at_site(self.dither.observatory)

        altaz = apo.altaz(astropy.time.Time(val=dith_start_times, format='unix'), self.coords)
        earthrate = 2 * np.pi / astropy.units.sday.to(astropy.units.second)

        lat = site.geodetic.lat.rad
        az = altaz.az.radian
        alt = altaz.alt.radian

        # Smart 1962
        dith_start_rot_rates = earthrate * np.cos(lat) * np.cos(az) / np.cos(alt)

        dith_pix_offset = self.dither_pixel_vector()
        # get the minimum required timestep. One that would produce 1 pixel displacement at the
        # center of furthest dither
        dith_dists = np.sqrt(dith_pix_offset[0]**2 + dith_pix_offset[1]**2)
        dith_angle = np.arctan(allowable_pixel_smear/dith_dists)
        max_timestep = min(dith_angle/abs(dith_start_rot_rates))

        getLogger(__name__).debug("Maximum non-blurring time step calculated to be {}".format(max_timestep))

        return(max_timestep)

def metadata_config_check(filename, conf_wcs):
    """ Checks a photontable metadata and data.yml agree on the wcs params """
    ob = ObsFile(filename)
    md = ob.metadata()
    for attribute in ['dither_home', 'dither_ref', 'platescale', 'device_orientation']:
        if getattr(md, attribute) != getattr(conf_wcs,attribute):
            getLogger(__name__).warning(f'{attribute} is different in config and obsfile metadata. metadata should be reapplied')

def mp_worker(file, startw, stopw, startt, intt, derotate, wcs_timestep, first_time=None, flags=None):
    """
    Genereate the reduced, reformated photonlists

    :param file:
    :param startw:
    :param stopw:
    :param startt:
    :param intt:
    :param derotate:
    :param wcs_timestep:

    :param flags: None or a flag bitmask as per file.flag_bitmask()
    :return:
    """
    obsfile = ObsFile(file)
    duration = obsfile.duration

    photons = obsfile.query(startw=startw, stopw=stopw, startt=startt, intt=intt)
    weights = photons['SpecWeight'] * photons['NoiseWeight']
    getLogger(__name__).info("Fetched {} photons from dither {}".format(len(photons), file))
    if len(photons) == 0:
        getLogger(__name__).warning(f'No photons found using wavelength range {startw}-{stopw} nm and time range '
                                    f'{startt}-{intt} s. Is the obsfile not wavelength calibrated causing a mismatch '
                                    f'in the units?')

    x, y = obsfile.xy(photons)

    if flags is not None:
        print(pixelflags.PROBLEM_FLAGS)
        #TODO @dodkins fixme
        usablelist = obsfile.flagMask(flags, (x, y))
        getLogger(__name__).info("Removed {} photons from {} total from bad pix"
                                 .format(len(photons) - len(photons[usablelist]), len(photons)))
        photons = photons[usablelist]
    x, y = obsfile.xy(photons)

    # ob.get_wcs returns all wcs solutions (including those after intt), so just pass then remove post facto()
    wcs = obsfile.get_wcs(derotate=derotate, wcs_timestep=wcs_timestep, first_time=first_time) #1545626973
    nwcs = int(np.ceil(intt/wcs_timestep))
    wcs = wcs[:nwcs]
    del obsfile

    return {'file': file, 'timestamps': photons["Time"], 'xPhotonPixels': x, 'yPhotonPixels': y,
            'wavelengths': photons["Wavelength"], 'weight': weights, 'obs_wcs_seq': wcs, 'duration': duration}


def load_data(dither, wvlMin, wvlMax, startt, used_inttime, wcs_timestep, tempfile='drizzler_tmp_{}.pkl',
              tempdir=None, usecache=True, clearcache=False, derotate=True, ncpu=1, flags=None):
    """
    Load the photons either by querying the obsfiles in parrallel or loading from pkl if it exists. The wcs
    solutions are added to this photon data dictionary but will likely be integrated into photontable.py directly

    :param dither:
    :param wvlMin:
    :param wvlMax:
    :param startt:
    :param intt:
    :param wcs_timestep:
    :param tempfile:
    :param tempdir:
    :param usecache:
    :param clearcache:
    :param derotate:
    :return:
    """

    if tempdir is None:
        try:
            tempdir = mkidpipeline.config.config.paths.tmp
        except KeyError:
            getLogger(__name__).warning('No tempdir specified. Setting usecache to False')
            usecache = False

    if usecache:
        if not os.path.exists(tempdir):
            os.mkdir(tempdir)

        pkl_save = os.path.join(tempdir, tempfile.format(dither.name))

        if clearcache:  # TODO the cache must be autocleared if the query parameters would alter the contents
            os.remove(pkl_save)

    try:
        if not usecache:
            raise FileNotFoundError
        with open(pkl_save, 'rb') as f:
            dithers_data = pickle.load(f)
            getLogger(__name__).info('loaded {}'.format(pkl_save))
    except FileNotFoundError:
        begin = time.time()
        filenames = [o.h5 for o in dither.obs]
        if not filenames:
            getLogger(__name__).info('No obsfiles found')

        if derotate:
            first_time = None
        else:
            first_time = ObsFile(filenames[0]).startTime

        metadata_config_check(filenames[0], dither.wcscal)

        ncpu = min(mkidpipeline.config.n_cpus_available(), ncpu)
        p = mp.Pool(ncpu)
        processes = [p.apply_async(mp_worker, (file, wvlMin, wvlMax, startt, used_inttime, derotate, wcs_timestep, first_time,
                                               flags)) for file in filenames]
        dithers_data = [res.get() for res in processes]

        dithers_data.sort(key=lambda k: filenames.index(k['file']))

        getLogger(__name__).debug('Time spent: %f' % (time.time() - begin))

        if usecache:
            with open(pkl_save, 'wb') as handle:
                getLogger(__name__).info('Saved data cache to {}'.format(pkl_save))
                pickle.dump(dithers_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return dithers_data


class Canvas(object):
    def __init__(self, dithers_data, ob, coords):
        """
        Class common to SpatialDrizzler, TemporalDrizzler and ListDrizzler. It generates the canvas that is
        drizzled onto

        TODO determine appropirate value from area coverage of dataset and oversampling, even longerterm there
        the oversampling should be selected to optimize total phase coverage to extract the most resolution at a
        desired minimum S/N

        :param dithers_data:
        :param ob:
        :param coords:
        """

        self.nPixRA = None
        self.nPixDec = None
        self.square_grid = True

        self.config = None
        self.dithers_data = dithers_data

        self.xpix = ob.beammap.ncols
        self.ypix = ob.beammap.nrows
        self.vPlateScale = ob.instrument_info.platescale.to(u.deg).value
        self.centerRA = coords.ra.deg
        self.centerDec = coords.dec.deg

        if self.nPixRA is None or self.nPixDec is None:
            dith_cellestial_min = np.zeros((len(dithers_data), 2))
            dith_cellestial_max = np.zeros((len(dithers_data), 2))
            for ip, photonlist in enumerate(dithers_data):
                # find the max and min coordinate for each dither (assuming those occur at the beginning/end of
                # the dither)
                dith_cellestial_span = np.vstack((wcs.WCS(photonlist['obs_wcs_seq'][0]).wcs.crpix,
                                                  wcs.WCS(photonlist['obs_wcs_seq'][-1]).wcs.crpix))
                dith_cellestial_min[ip] = np.min(dith_cellestial_span, axis=0)  # takes the min of both ra and dec
                dith_cellestial_max[ip] = np.max(dith_cellestial_span, axis=0)

            # find the min and max coordinate of all dithers
            raMin = min(dith_cellestial_min[:, 0])
            raMax = max(dith_cellestial_max[:, 0])
            decMin = min(dith_cellestial_min[:, 1])
            decMax = max(dith_cellestial_max[:, 1])

            # Set size of virtual grid to accommodate the limits of the offsets.
            # max_detector_dist = np.sqrt(self.xpix ** 2 + self.ypix **2)
            self.nPixRA = (2 * np.max((raMax-raMin)) + self.xpix).astype(int)
            self.nPixDec = (2 * np.max((decMax-decMin)) + self.ypix).astype(int)

        if self.square_grid:
            nPix = max((self.nPixRA, self.nPixDec))
            self.nPixRA, self.nPixDec = nPix, nPix

        # check the size of the grid is sensible
        max_grid_width = max(self.xpix, self.ypix) * len(dithers_data)  # equivalent dithering along a line
        assert self.nPixDec < max_grid_width and self.nPixRA < max_grid_width, \
            'Canvas grid exceeds maximum possible extent of dithers'

        self.get_canvas_wcs()

    def get_canvas_wcs(self):
        """

        :return:
        """

        self.wcs = wcs.WCS(naxis = 2)
        self.wcs.wcs.crpix = np.array([self.nPixRA / 2., self.nPixDec / 2.])
        self.wcs.wcs.crval = [self.centerRA, self.centerDec]
        self.wcs.wcs.ctype = ["RA--TAN", "DEC-TAN"]
        self.wcs.pixel_shape = (self.nPixRA, self.nPixDec)
        self.wcs.wcs.pc = np.array([[1,0],[0,1]])
        self.wcs.wcs.cdelt = [self.vPlateScale,self.vPlateScale]
        self.wcs.wcs.cunit = ["deg", "deg"]
        getLogger(__name__).debug(self.wcs)


class ListDrizzler(Canvas):
    """
    Drizzle individual photons onto the celestial grid
    """

    def __init__(self, dithers_data, drizzle_params):
        super().__init__(self, dithers_data, drizzle_params.dither.obs[0], drizzle_params.coords)

        getLogger(__name__).critical('This has not been tested')
        raise NotImplementedError

        inttime = drizzle_params.used_inttime

        # if inttime is say 100 and wcs_timestep is say 60 then this yields [0,60,100]
        # meaning the positions don't have constant integration time
        self.wcs_times = np.append(np.arange(0, inttime, drizzle_params.wcs_timestep), inttime) * 1e6
        self.run(pixfrac=drizzle_params.pixfrac)

    def run(self, save_file=None, pixfrac=1.):
        for ix, dither_photons in enumerate(self.dithers_data):
            getLogger(__name__).debug('Processing %s', dither_photons)

            tic = time.clock()
            # driz = stdrizzle.Drizzle(outwcs=self.wcs, pixfrac=pixfrac)
            for t, inwcs in enumerate(dither_photons['obs_wcs_seq']):
                inwcs = wcs.WCS(header=inwcs)
                inwcs.pixel_shape = (self.xpix, self.ypix)

                inds = [(yp, xp) for yp, xp in np.ndindex(self.ypix, self.xpix)]
                allpix2world = []
                for i in range(self.xpix*self.ypix):
                    insci = np.ones((self.ypix, self.xpix))

                    driz = stdrizzle.Drizzle(outwcs=self.wcs, pixfrac=pixfrac, wt_scl='')
                    inwht = np.zeros((self.ypix, self.xpix))
                    # print(inds[i])
                    inwht[inds[i]] = 1
                    driz.add_image(insci, inwcs, inwht=inwht)
                    sky_inds = np.where(driz.outsci == 1)
                    # print(sky_inds, np.shape(sky_inds), len(sky_inds), sky_inds is [], sky_inds == [])
                    if np.shape(sky_inds)[1] == 0:
                        pix2world = [[], []]
                    else:
                        pix2world = inwcs.all_pix2world(sky_inds[1], sky_inds[0], 1)
                    # print(allpix2world)
                    allpix2world.append(pix2world)

                    # plt.imshow(driz.outsci)
                    # plt.show(block=True)

                radecs = []
                for i, (xp, yp) in enumerate(zip(dither_photons['xPhotonPixels'], dither_photons['yPhotonPixels'])):
                    ind = xp + yp * self.xpix
                    # print(xp, yp, ind, )
                    # print(allpix2world[ind])
                    radecs.append(allpix2world[ind])

                dither_photons['radecs'] = radecs  # list of [npix, npix]

            getLogger(__name__).debug('Image load done. Time taken (s): %s', time.clock() - tic)


class TemporalDrizzler(Canvas):
    """
    Generate a spatially dithered fits 4D hypercube from a set dithered dataset. The cube size is
    ntimebins * ndithers X nwvlbins X nPixRA X nPixDec.

    exp_timestep or ntimebins argument accepted. ntimebins takes priority
    """

    def __init__(self, dithers_data, drizzle_params, nwvlbins=1, exp_timestep=1, wvlMin=0, wvlMax=np.inf):
        super().__init__(dithers_data, drizzle_params.dither.obs[0], drizzle_params.coords)

        self.nwvlbins = nwvlbins
        self.exp_timestep = exp_timestep  # seconds

        self.ndithers = len(self.dithers_data)
        self.pixfrac = drizzle_params.pixfrac
        self.wvlbins = np.linspace(wvlMin, wvlMax, self.nwvlbins + 1)

        inttime = drizzle_params.used_inttime
        self.wcs_times = np.append(np.arange(0, inttime, drizzle_params.wcs_timestep), inttime)
        self.wcs_times_ms = self.wcs_times * 1e6

        self.ntimebins = int(inttime / self.exp_timestep)
        if self.ntimebins < len(self.dithers_data[0]['obs_wcs_seq']):
            getLogger(__name__).warning('Increasing the number of time bins beyond the user request')
            self.ntimebins = len(self.dithers_data[0]['obs_wcs_seq'])

        self.timebins = np.linspace(0, inttime, self.ntimebins + 1) * 1e6  # timestamps are in microseconds
        self.totHypCube = None
        self.totWeightCube = None

    def run(self):
        tic = time.clock()

        self.totHypCube = np.zeros((self.ntimebins * self.ndithers, self.nwvlbins, self.nPixDec, self.nPixRA))
        self.expmap = np.zeros((self.ntimebins * self.ndithers, self.nwvlbins, self.nPixDec, self.nPixRA))
        for ix, dither_photons in enumerate(self.dithers_data):

            thishyper = np.zeros((self.ntimebins, self.nwvlbins, self.nPixDec, self.nPixRA), dtype=np.float32)

            it = 0
            for t, inwcs in enumerate(dither_photons['obs_wcs_seq']):
                inwcs = wcs.WCS(header=inwcs)
                inwcs.pixel_shape = (self.xpix, self.ypix)

                # the sky grid ref and dither ref should match (crpix varies between dithers)
                if not np.all(np.round(inwcs.wcs.crval, decimals=4) == np.round(self.wcs.wcs.crval, decimals=4)):
                    getLogger(__name__).critical('sky grid ref and dither ref do not match (crpix varies between dithers)!')
                    raise RuntimeError('sky grid ref and dither ref do not match (crpix varies between dithers)!')

                counts = self.makeTess(dither_photons, (self.wcs_times_ms[t], self.wcs_times_ms[t+1]))
                cps = counts/(self.wcs_times[t + 1] - self.wcs_times[t])  # scale this frame by its exposure time

                for ia, iw in np.ndindex(len(counts), self.nwvlbins):
                    # create a new drizzle object for each time (and wavelength) frame
                    drizhyper = stdrizzle.Drizzle(outwcs=self.wcs, pixfrac=self.pixfrac, wt_scl='')
                    inwht = np.int_(np.logical_not(cps[ia, iw] == 0))

                    drizhyper.add_image(cps[ia, iw], inwcs, inwht=inwht)  # in_units='cps' shouldn't have any affect
                                                                          # for a single drizzle
                    thishyper[it, iw] = drizhyper.outsci
                    # print(ix, t, ia, it, (ix*self.ntimebins)+it, self.wcs_times[t + 1] - self.wcs_times[t], (
                    #             self.wcs_times[t + 1] - self.wcs_times[t]) / self.ntimebins)
                    self.expmap[(ix*self.ntimebins)+t, iw] = drizhyper.outwht * (
                                self.wcs_times[t + 1] - self.wcs_times[t]) / self.ntimebins

                    if iw == self.nwvlbins-1:

                        it += 1

            self.totHypCube[ix * self.ntimebins: (ix + 1) * self.ntimebins] = thishyper

        getLogger(__name__).debug('Image load done. Time taken (s): %s', time.clock() - tic)

        self.header_4d()

        self.cps = self.totHypCube
        self.counts = self.cps * self.expmap
        self.outwcs = self.wcs

    def makeTess(self, dither_photons, timespan, applyweights=False, maxCountsCut=False):
        """

        :param dither_photons:
        :param timespan:
        :param applyweights:
        :param maxCountsCut:
        :return:
        """

        weights = dither_photons['weight'] if applyweights else None

        timespan_mask = (dither_photons['timestamps'] >= timespan[0]) & (dither_photons['timestamps'] <= timespan[1])

        sample = np.vstack((dither_photons['timestamps'][timespan_mask],
                            dither_photons['wavelengths'][timespan_mask],
                            dither_photons['xPhotonPixels'][timespan_mask],
                            dither_photons['yPhotonPixels'][timespan_mask]))

        timebins = self.timebins[(self.timebins >= timespan[0]) & (self.timebins <= timespan[1])]

        bins = np.array([timebins, self.wvlbins, range(self.ypix+1), range(self.xpix+1)])
        hypercube, _ = np.histogramdd(sample.T, bins, weights=weights, )

        if maxCountsCut:
            getLogger(__name__).debug("Applying max pixel count cut")
            hypercube *= np.int_(hypercube < maxCountsCut)

        # hypercube /= (timespan[1] - timespan[0])*1e-6

        return hypercube

    def header_4d(self):
        """
        Add to the extra elements to the header

        Its not clear how to increase the number of dimensions of a 2D wcs.WCS() after its created so just create
        a new object, read the original parameters where needed, and overwrite

        :return:
        """

        w4d = wcs.WCS(naxis=4)
        w4d.wcs.crpix = [self.wcs.wcs.crpix[0], self.wcs.wcs.crpix[1], 1, 1]
        w4d.wcs.crval = [self.wcs.wcs.crval[0], self.wcs.wcs.crval[1], self.wvlbins[0]/1e9, self.timebins[0]/1e6]
        w4d.wcs.ctype = [self.wcs.wcs.ctype[0], self.wcs.wcs.ctype[1], "WAVE", "TIME"]
        w4d.pixel_shape = (self.wcs.pixel_shape[0], self.wcs.pixel_shape[1], self.nwvlbins, self.ntimebins)
        w4d.wcs.pc = np.eye(4)
        w4d.wcs.cdelt = [self.wcs.wcs.cdelt[0], self.wcs.wcs.cdelt[1],
                         (self.wvlbins[1] - self.wvlbins[0])/1e9,
                         (self.timebins[1] - self.timebins[0])/1e6]
        w4d.wcs.cunit = [self.wcs.wcs.cunit[0], self.wcs.wcs.cunit[1], "m", "s"]

        self.wcs = w4d
        getLogger(__name__).debug('4D wcs {}'.format(w4d))


class SpatialDrizzler(Canvas):
    """ Generate a spatially dithered fits image from a set dithered dataset """

    def __init__(self, dithers_data, drizzle_params, stack=False):
        super().__init__(dithers_data, drizzle_params.dither.obs[0], drizzle_params.coords)

        self.cps = None
        self.outwcs = None
        self.counts = None
        self.expmap = None

        self.stack = stack

        self.driz = stdrizzle.Drizzle(outwcs=self.wcs, pixfrac=drizzle_params.pixfrac, wt_scl='')
        self.wcs_timestep = drizzle_params.wcs_timestep
        inttime = drizzle_params.used_inttime

        # if inttime is say 100 and wcs_timestep is say 60 then this yeilds [0,60,100]
        # meaning the positions don't have constant integration time
        self.wcs_times = np.append(np.arange(0, inttime, self.wcs_timestep), inttime)
        self.wcs_times_ms = self.wcs_times * 1e6
        self.stackedim = np.zeros((len(drizzle_params.dither.obs) * (len(self.wcs_times)-1), self.ypix, self.xpix))
        self.stacked_wcs = []

    def run(self, save_file=None):
        for ix, dither_photons in enumerate(self.dithers_data):

            tic = time.clock()
            for t, inwcs in enumerate(dither_photons['obs_wcs_seq']):
                inwcs = wcs.WCS(header=inwcs)
                inwcs.pixel_shape = (self.xpix, self.ypix)

                # the sky grid ref and dither ref should match (crpix varies between dithers)
                if not np.all(np.round(inwcs.wcs.crval, decimals=4) == np.round(self.wcs.wcs.crval, decimals=4)):
                    getLogger(__name__).critical(
                        'sky grid ref and dither ref do not match (crpix varies between dithers)!')
                    raise RuntimeError('sky grid ref and dither ref do not match (crpix varies between dithers)!')

                counts = self.makeImage(dither_photons, (self.wcs_times_ms[t], self.wcs_times_ms[t+1]))
                cps = counts/(self.wcs_times[t + 1] - self.wcs_times[t])  # scale this frame by its exposure time

                self.stackedim[ix*len(dither_photons['obs_wcs_seq']) + t] = counts
                self.stacked_wcs.append(inwcs)

                getLogger(__name__).debug('Image load done. Time taken (s): %s', time.clock() - tic)
                getLogger(__name__).warning('Bad pix identified using zero counts. Needs to be properly coded')
                # todo make inwht using bad pix maps
                inwht = (cps != 0).astype(int)

                # Input units as cps means Drizzle() will scale the output image by the # of contributions to each pix
                self.driz.add_image(cps, inwcs, in_units='cps', inwht=inwht)
            if save_file:
                self.driz.write(save_file)

        if self.stack:
            self.counts = self.stackedim
            self.outwcs = self.stacked_wcs
        else:
            self.cps = self.driz.outsci
            self.counts = self.driz.outwht * self.cps * np.mean(self.wcs_times)
            self.outwcs = self.wcs

    def makeImage(self, dither_photons, timespan, applyweights=False, maxCountsCut=10000):
        # TODO mixing pixels and radians per variable names

        weights = dither_photons['weight'] if applyweights else None
        timespan_ind = np.where(np.logical_and(dither_photons['timestamps'] >= timespan[0],
                                               dither_photons['timestamps'] <= timespan[1]))[0]

        thisImage, _, _ = np.histogram2d(dither_photons['xPhotonPixels'][timespan_ind],
                                         dither_photons['yPhotonPixels'][timespan_ind],  weights=weights,
                                         bins=[range(self.ypix+1), range(self.xpix+1)], normed=False)

        if maxCountsCut:
            getLogger(__name__).info("Applying max pixel count cut")
            thisImage *= thisImage < maxCountsCut

        return thisImage


class DrizzledData(object):
    def __init__(self, driz, mode, drizzle_params):
        """

        :param driz:
        :param mode:
        :param drizzle_params:
        """

        self.cps = driz.cps
        self.counts = driz.counts
        self.wcs = driz.outwcs
        self.expmap = driz.expmap
        self.mode = mode
        self.header_keys = ['WCSTIME', 'PIXFRAC']
        self.header_vals = [drizzle_params.wcs_timestep, drizzle_params.pixfrac]
        self.header_comments = ['[s] Time between calculated wcs (different PAs)', '']

    def header_additions(self, header):
        for key, val, comment in zip(self.header_keys, self.header_vals, self.header_comments):
            header[key] = (val, comment)
        return header

    def writefits(self, filename, overwrite=True, compress=False, dashboard_orient=False):
        """

        :param filename:
        :param overwrite:
        :param compress:
        :param dashboard_orient:
        :return:
        """

        if self.mode == 'stack':
            if dashboard_orient:
                getLogger(__name__).info('Transposing image stack to match dashboard orientation')
                getLogger(__name__).warning('Has not been verified')
                self.data = np.transpose(self.data, (0,2,1))
                for w in self.wcs:
                    w.wcs.pc = w.wcs.pc.T

            fits_header = [w.to_header() for w in self.wcs]
            [self.header_additions(hdr) for hdr in fits_header]

            hdus = [fits.ImageHDU(data=d, header=h) for d, h in zip(self.counts, fits_header)]
            hdul = fits.HDUList(list(np.append(fits.PrimaryHDU(), hdus)))

        else:
            fits_header = self.wcs.to_header()
            fits_header = self.header_additions(fits_header)

            hdul = fits.HDUList([fits.PrimaryHDU(header=fits_header),
                                 fits.ImageHDU(name='cps', data=self.cps, header=fits_header),
                                 fits.ImageHDU(name='variance', data=self.counts, header=fits_header)])

        if compress:
            filename = filename+'.gz'

        assert filename[-5:] == '.fits', 'Please enter valid filename'

        hdul.writeto(filename, overwrite=overwrite)
        getLogger(__name__).info('FITS file {} saved'.format(filename))


def form(dither, mode='spatial', derotate=True, wvlMin=None, wvlMax=None, startt=0., intt=60., pixfrac=.5, nwvlbins=1,
         wcs_timestep=1., exp_timestep=1., fitsname=None, usecache=True, ncpu=1, flags=None):
    """
    Takes in a MKIDDitheredObservation object and drizzles the dithers onto a common sky grid.

    Parameters
    ----------
    dither : MKIDDitheredObservation
        Contains the lists of observations and metadata for a set of dithers
    mode : str
        Format for the output (spatial | stack | temporal | list (not implemented yet))
    derotate : bool
        True means all dithers (and integrations within) are rotated to their orientation on the sky during their observation
        False aligns all dithers and integrations to the orientation at the beginning of the observation
    wvlMin : float or None
        Lower bound on the photons used in the drizzle. See photontable.query()
    wvlMax : float or None
        Upper bound on the photons used in the drizzle. See photontable.query()
    startt : float or None
        Starttime for photons used in each dither. See photontable.query()
    intt : float or None
        startt + int = upper bound on the photons used in each dither. See photontable.query()
    pixfrac : float (0<= pixfrac <=1)
        pixfrac parameter used in drizzle algorithm. See stsci.drizzle()
    nwvlbins : int
        Number of bins to group photon data by wavelength for all dithers when using mode == 'temporal'
    wcs_timestep : float (0<= wcs_timestep <=intt)
        Time between different wcs parameters (eg orientations). 0 will use the calculated non-blurring min
    exp_timestep : float (0<= exp_timestep <=intt)
        Duration of the time bins in the output cube when using mode == 'temporal'
    fitsname : str
        Output FITS file name
    usecache : bool
        True means the output of load_data() is stored and reloaded for subsequent runs of form
    ncpu : int
        Number of cpu used when loading and reformatting the dither obsfiles
    flags : int
        Bitmask containing the various flags on each pixel from previous steps

    Returns
    -------
    drizzle : DrizzledData
        Contains maps and metadata from the drizzled data


    There are 4 output modes: stack, spatial, temporal, list (not implemented yet). stack does no drizzling and just
    appends time integrated MKID images on oneanother with associated wcs solutions at that time. spatial drizzles all
    the selected times and wavelengths onto a single map. temporal bins the selected times and wavelengths into a 4D
    hypercube (2 spatial, 1 wavelength, 1 time) The number of time bins is ndither * intt/exp_timestep. list will be
    assigning an RA and dec to each photon

    exp_timestep vs wcs_timestep. Both parameters are independent -- several wcs solutions (orientations) can contribute
    to one effective exposure (taking an image of the zenith), and you can have one wcs object for lots of exposures
    (doing binned SSD on a target that barely rotates).
    """

    # ensure the user input is shorter than the dither or that wcs are just calculated for the requested timespan
    if intt > dither.inttime:
        getLogger(__name__).info('Used integration time is set by dither duration to be {}s'.format(dither.inttime))
    if dither.inttime > intt:
        getLogger(__name__).info('Used integration time is set by user defined integration time to be {}s'.format(intt))

    used_inttime = min(intt, dither.inttime)

    if dither.target == 'WL':
        getLogger(__name__).warning('Changing some of the wcs params to white light mode')
        derotate = False

    drizzle_params = DrizzleParams(dither, used_inttime, wcs_timestep, pixfrac)

    dithers_data = load_data(dither, wvlMin, wvlMax, startt, used_inttime, drizzle_params.wcs_timestep,
                             derotate=derotate, usecache=usecache, ncpu=ncpu, flags=flags)
    total_photons = sum([len(dither_data['timestamps']) for dither_data in dithers_data])

    if total_photons == 0:
        getLogger(__name__).critical('No photons found in any of the dithers. Check your wavelength and time ranges')
        raise ValueError

    if mode not in ['stack', 'spatial', 'temporal', 'list']:
        raise ValueError('Not calling one of the available functions')

    elif mode == 'list':
        driz = ListDrizzler(dithers_data, drizzle_params)
        return driz.dithers_data

    elif mode == 'spatial' or mode == 'stack':
        stack = mode=='stack'
        driz = SpatialDrizzler(dithers_data, drizzle_params, stack=stack)

    elif mode == 'temporal':
        driz = TemporalDrizzler(dithers_data, drizzle_params, nwvlbins=nwvlbins, exp_timestep=exp_timestep,
                                 wvlMin=wvlMin, wvlMax=wvlMax)
    driz.run()
    drizzle = DrizzledData(driz, mode, drizzle_params=drizzle_params)

    if fitsname:
        drizzle.writefits(file=fitsname + '.fits')  # unless path specified, save in cwd

    getLogger(__name__).info('Finished forming drizzled data')

    return drizzle


def get_star_offset(dither, wvlMin, wvlMax, startt, intt, start_guess=(0,0), zoom=2.):
    """
    Get the rotation_center offset parameter for a dither

    :param dither:
    :param wvlMin:
    :param wvlMax:
    :param startt:
    :param intt:
    :param start_guess:
    :param zoom: after each loop the figure zooms on the centre of the image. zoom==2 yields the middle quadrant on 1st iteration
    :return:
    """

    update = True

    rotation_center = start_guess

    def onclick(event):
        xlocs.append(event.xdata)
        ylocs.append(event.ydata)
        running_mean = [np.mean(xlocs), np.mean(ylocs)]
        getLogger(__name__).info('xpix=%i, ypix=%i. Running mean=(%i,%i)'
                 % (event.xdata, event.ydata, running_mean[0], running_mean[1]))

    iteration = 0
    while update:

        drizzle = form(dither=dither, mode='spatial', wvlMin=wvlMin,
                        wvlMax=wvlMax, startt=startt, intt=intt, pixfrac=1, derotate=None, usecache=False)

        image = drizzle.data
        fig, ax = plt.subplots()

        print("Click on the four satellite speckles and the star")
        cax = ax.imshow(image, origin='lower', norm=LogNorm())
        lims = np.array(image.shape) / zoom**iteration
        ax.set_xlim((image.shape[0]//2 - lims[0]//2, image.shape[0]//2 + lims[0]//2 - 1))
        ax.set_ylim((image.shape[1]//2 - lims[1]//2, image.shape[1]//2 + lims[1]//2 - 1))
        cb = plt.colorbar(cax)
        cb.ax.set_title('Counts')

        xlocs, ylocs = [], []

        fig.canvas.mpl_connect('button_press_event', onclick)
        plt.show(block=True)

        if not xlocs:  # if the user doesn't click on the figure don't change rotation_center's value
            xlocs, ylocs = np.array(image.shape)//2, np.array(image.shape)//2
        star_pix = np.array([np.mean(xlocs), np.mean(ylocs)]).astype(int)
        rotation_center += (star_pix - np.array(image.shape)//2)[::-1] #* np.array([1,-1])
        getLogger(__name__).info('rotation_center: {}'.format(rotation_center))

        user_input = input(' *** INPUT REQUIRED *** \nDo you wish to continue looping [Y/n]: \n')
        if user_input == 'n':
            update = False

        iteration += 1

    getLogger(__name__).info('rotation_center: {}'.format(rotation_center))

    return rotation_center


def drizzler_cfg_descr_str(drizzlercfg):
    return 'TODO_drizzler_cfg_descr'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Photon Drizzling Utility')
    parser.add_argument('cfg', type=str, help='The configuration file')
    parser.add_argument('-wl', type=float, dest='wvlMin', help='minimum wavelength', default=850)
    parser.add_argument('-wh', type=float, dest='wvlMax', help='maximum wavelength', default=1100)
    parser.add_argument('-t0', type=int, dest='startt', help='start time', default=0)
    parser.add_argument('-it', type=float, dest='intt', help='end time', default=60)
    parser.add_argument('-p', action='store_true', dest='plot', help='Plot the result', default=False)
    parser.add_argument('--get-offset', nargs=2, type=int, dest='gso', help='Runs get_star_offset eg 0 0 ')
    # changed this to bool so that the filename from drizzler_cfg_descr_str(cfg.drizzler) could be used
    parser.add_argument('--get-orientation', type=bool, dest='gdo',
                        help='Run get_device_orientation on a fits file, first created with the default orientation.',
                        default=None)

    args = parser.parse_args()

    # timeout limit for SkyCoord.from_name
    Conf.remote_timeout.set(10)

    # set up logging
    mkidpipeline.logtoconsole()
    pipelinelog.create_log('mkidpipeline.imaging.drizzler', console=True, level="INFO")

    # load as a task configuration
    cfg = mkidpipeline.config.load_task_config(args.cfg)

    wvlMin = args.wvlMin
    wvlMax = args.wvlMax
    startt = args.startt
    intt = args.intt
    pixfrac = cfg.drizzler.pixfrac
    dither = cfg.dither

    if args.gso and isinstance(args.gso, list):
        rotation_origin = get_star_offset(dither, wvlMin, wvlMax, startt, intt, start_guess=np.array(args.gso))

    fitsname = '{}_{}.fits'.format(cfg.dither.name, drizzler_cfg_descr_str(cfg.drizzler))

    # main function of drizzler
    scidata = form(dither, mode='spatial', wvlMin=wvlMin,
                   wvlMax=wvlMax, startt=startt, intt=intt, pixfrac=pixfrac,
                   derotate=True, fitsname=fitsname)

    if args.gdo:
        if not os.path.exists(fitsname):
            getLogger(__name__).info(("Can't find {} Create the fits image "
                                      "using the default orientation first").format(fitsname))
        else:
            target = dither.target
            if target is None or target == 'None':
                getLogger(__name__).error('Please enter a valid target name')
                raise TypeError
            elif isinstance(target, (list, np.array)):
                target = [float(tar.value) * u.deg for tar in target]  # assume list of coord values in degrees
                coords = SkyCoord(target[0], target[1])
            elif isinstance(target, SkyCoord):
                coords = target
            else:
                coords = SkyCoord.from_name(target)
            getLogger(__name__).info('Found coordinates {} for target {}'.format(coords, target))
            get_device_orientation(coords, fitsname)
