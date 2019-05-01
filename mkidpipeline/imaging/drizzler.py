"""
*** Warning ***
The STScI drizzle module appears to have a bug. Line 474 (as of 4/23/19) should change from

self.outcon = np.append(self.outcon, plane, axis=0)

to

self.outcon = np.append(self.outcon, [plane], axis=0)


TODO
Add astroplan, drizzle, to setup.py/yml. drizzle need to be pip installed. I found that astroplan needed to be pip
installed otherwise some astropy import fails

Make ListDrizzler

Usage
-----

python drizzle.py /mnt/data0/dodkins/src/mkidpipeline/mkidpipeline/imaging/drizzler.yml

Author: Rupert Dodkins,                                 Date: April 2019

"""
import os
import numpy as np
import time
import multiprocessing as mp
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
from matplotlib import gridspec
import pickle
from scipy.misc import imrotate
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
import mkidcore.corelog as pipelinelog
import mkidpipeline
from mkidcore.instruments import CONEX2PIXEL
import argparse

# timeout limit for SkyCoord.from_name
Conf.remote_timeout.set(10)

# set up logging
mkidpipeline.logtoconsole()

log = pipelinelog.create_log('mkidpipeline.imaging.drizzler', console=True, level="INFO")




def dither_pixel_vector(positions, center=(0, 0)):
    """
    A function to convert a list of conex offsets to pixel displacement

    :param positions: list of length 2 arrays
    :param center: the origin for the vector
    :return:
    """
    positions = np.asarray(positions)
    pix = np.asarray(CONEX2PIXEL(positions[:, 0], positions[:, 1])) - np.array(CONEX2PIXEL(*center)).reshape(2, 1)
    return pix


def get_device_orientation(ditherdesc, fits_filename='Theta1 Orionis B_mean.fits', separation=0.938, pa=253):
    """
    Given the position angle and offset of secondary calculate its RA and dec then
    continually update the FITS with different rotation matricies to tune for device orientation

    Default pa and offset for Trap come from https://arxiv.org/pdf/1308.4155.pdf figures 7 and 11

    B1 vs B2B3 barycenter separation is 0.938 and the position angle is 253 degrees

    :param ditherdesc:
    :param fits_filename:
    :param separation:
    :param pa:
    :return:
    """

    angle_from_east = 270-pa

    companion_ra_arcsec = np.cos(np.deg2rad(angle_from_east)) * separation
    companion_ra_offset = (companion_ra_arcsec * u.arcsec).to(u.deg).value
    companion_ra = ditherdesc.coords.ra.deg + companion_ra_offset

    companion_dec_arcsec = np.sin(np.deg2rad(angle_from_east)) * separation
    companion_dec_offset = (companion_dec_arcsec * u.arcsec).to(u.deg).value
    #minus sign here since reference object is below central star
    companion_dec = ditherdesc.coords.dec.deg - companion_dec_offset

    log.info('Target RA {} and dec {}'.format(Angle(companion_ra*u.deg).hms, Angle(companion_dec*u.deg).dms))

    update = True
    device_orientation = 0
    hdu1 = fits.open(fits_filename)[1]

    field = hdu1.data
    while update:

        pipelinelog.getLogger(__name__).info('Close this figure')
        ax1 = plt.subplot(111, projection=wcs.WCS(hdu1.header))
        ax1.imshow(field, norm=LogNorm(), origin='lower', vmin=1)
        # plt.colorbar()
        plt.show()

        user_input = input(' *** INPUT REQUIRED *** \nEnter new angle (deg) or F to end: ')
        if user_input == 'F':
            update = False
        else:
            device_orientation += float(user_input)

        field = imrotate(hdu1.data, device_orientation, interp='bilinear')

    pipelinelog.getLogger(__name__).info('Using position angle {} deg for device'.format(device_orientation))

    return np.deg2rad(device_orientation)


class DitherDescription(object):
    """
    Info on the dither

    rotate determines if the effective integrations are pupil stablised or not


    [1] Smart, W. M. 1962, Spherical Astronomy, (Cambridge: Cambridge University Press), p. 55

    """

    def __init__(self, dither, rotation_center=None, observatory='Subaru', target=None, target_radec=(0,0),
                 use_min_timestep=True, suggested_time_step=.1):
        """
        lookup_coordiantes may get a name error on correct target names leading to spurious results.
        Increasing timeout time to 60s does not fix
        TODO Try looping a fixed number of times
        Require a target for now

        :param dither:
        :param rotation_center: the vector that transforms the origin of connex frame to the center of rotation frame
        :param observatory:
        :param target:
        :param use_min_timestep:
        :param suggested_time_step:
        """
        self.description = dither
        self.target = target

        if target_radec == (0, 0):
            if target:
                # for i in range(attempts):
                self.coords = dither.obs[0].lookup_coodinates(queryname=target)
            else:
                pipelinelog.getLogger(__name__).warning('Unable to resolve coordinates for target name {target}, using (0,0).')
                self.coords = SkyCoord(0*u.deg, 0*u.deg)
            log.info('Found coordinated {}'.format(self.coords))
        else:
            self.coords = SkyCoord(target_radec[0]*u.deg, target_radec[1]*u.deg)
            log.info('Using coordinated {} in yml'.format(self.coords))

        self.starRA, self.starDec = self.coords.ra.deg, self.coords.dec.deg
        if rotation_center is None:
            rotation_center = (0, 0)
        assert suggested_time_step <= dither.inttime, 'You must have at least a time sample per dither'

        self.rotation_center = np.array([list(rotation_center)]).T  #neccessary hideous reformatting
        self.dith_pix_offset = dither_pixel_vector(dither.pos) - self.rotation_center  # TODO verify this

        inst_info = dither.obs[0].instrument_info
        self.xpix = inst_info.beammap.ncols
        self.ypix = inst_info.beammap.nrows
        self.platescale = inst_info.platescale.to(u.deg).value  # 10 mas
        self.apo = Observer.at_site(observatory)
        self.observatory = observatory

        if use_min_timestep:
            min_timestep = self.calc_min_timesamp(dither.obs)

            # sometimes the min timestep can be ~100s of seconds. We need it to be at least shorter
            # than the dith exposure time
            self.wcs_timestep = min(dither.inttime, min_timestep)
        else:
            self.wcs_timestep = suggested_time_step

        log.debug(
            "Timestep to be used {}".format(self.wcs_timestep))

    def calc_min_timesamp(self, obs, max_pix_disp=1.):
        """

        :param max_pix_disp: the resolution element threshold
        :return: min_timestep:
        """
        # get the field rotation rate at the start of each dither
        dith_start_times = np.array([o.start for o in obs])

        site = EarthLocation.of_site(self.observatory)
        altaz = self.apo.altaz(astropy.time.Time(val=dith_start_times, format='unix'), self.coords)
        earthrate = 2 * np.pi / u.sday.to(u.second)

        lat = site.geodetic.lat.rad
        az = altaz.az.radian
        alt = altaz.alt.radian

        # Smart 1962
        dith_start_rot_rates = earthrate * np.cos(lat) * np.cos(az) / np.cos(alt)

        # get the minimum required timestep. One that would produce 1 pixel displacement at the
        # center of furthest dither
        dith_dists = np.sqrt(self.dith_pix_offset[0]**2 + self.dith_pix_offset[1]**2)
        dith_angle = np.arctan(max_pix_disp/dith_dists)
        min_timestep = min(dith_angle/abs(dith_start_rot_rates))

        log.debug("Minimum required time step calculated to be {}".format(min_timestep))

        return min_timestep


class Drizzler(object):
    def __init__(self, photonlists, metadata):
        """
        TODO determine appropirate value from area coverage of dataset and oversampling, even longerterm there
        the oversampling should be selected to optimize total phase coverage to extract the most resolution at a
        desired minimum S/N

        :param photonlists:
        :param metadata:
        """

        self.nPixRA = None
        self.nPixDec = None
        self.square_grid = True

        self.config = None
        self.files = photonlists

        self.xpix = metadata.xpix
        self.ypix = metadata.ypix
        self.starRA = metadata.coords.ra.deg
        self.starDec = metadata.coords.dec.deg
        self.vPlateScale = metadata.platescale
        self.rotation_center = metadata.rotation_center

        if self.nPixRA is None or self.nPixDec is None:
            dith_cellestial_min = np.zeros((len(photonlists), 2))
            dith_cellestial_max = np.zeros((len(photonlists), 2))
            for ip, photonlist in enumerate(photonlists):
                # find the max and min coordinate for each dither (assuming those occur at the beginning/end of
                # the dither)
                dith_cellestial_span = np.vstack((photonlist['obs_wcs_seq'][0].wcs.crpix,
                                                  photonlist['obs_wcs_seq'][-1].wcs.crpix))
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

        self.generate_coordinate_grid()

        self.get_header()

    def generate_coordinate_grid(self):
        """
        Establish RA and dec coordinates for pixel boundaries in the virtual pixel grid,
        given the number of pixels in each direction (self.nPixRA and self.nPixDec), the
        location of the centre of the array (self.starRA, self.starDec), and the plate scale
        (self.vPlateScale).
        """

        self.gridRA = self.starRA + (self.vPlateScale * (np.arange(self.nPixRA + 1) - ((self.nPixRA + 1) // 2)))
        self.gridDec = self.starDec + (self.vPlateScale * (np.arange(self.nPixDec + 1) - ((self.nPixDec + 1) // 2)))

    def get_header(self, center_on_star=False):
        """
        TODO implement something like this
        w = mkidcore.buildwcs(self.nPixRA, self.nPixDec, self.vPlateScale, self.starRA, self.starDec)

        :param center_on_star:
        :return:
        """


        self.w = wcs.WCS(naxis = 2)
        self.w.wcs.crpix = np.array([self.nPixRA / 2., self.nPixDec / 2.])
        if center_on_star:
            self.w.wcs.crpix += np.array([self.rotation_center[0][0], self.rotation_center[1][0]])
        self.w.wcs.crval = [self.starRA, self.starDec]
        self.w.wcs.ctype = ["RA--TAN", "DEC-TAN"]
        self.w._naxis1 = self.nPixRA
        self.w._naxis2 = self.nPixDec
        self.w.wcs.pc = np.array([[1,0],[0,1]])
        self.w.wcs.cdelt = [self.vPlateScale,self.vPlateScale]
        self.w.wcs.cunit = ["deg", "deg"]
        log.debug(self.w)


class SpectralDrizzler(Drizzler):
    """ Generate a spatially dithered fits dataacube from a set dithered dataset """

    def __init__(self, photonlists, metadata, pixfrac=1.):
        self.nwvlbins = 3
        self.wvlbins = np.linspace(metadata.wvlMin, metadata.wvlMax, self.nwvlbins + 1)
        super().__init__(photonlists, metadata)
        self.drizcube = [stdrizzle.Drizzle(outwcs=self.w, pixfrac=pixfrac)] * self.nwvlbins

    def run(self, save_file=None):
        for ix, file in enumerate(self.files):

            log.debug('Processing %s', file)
            tic = time.clock()
            insci, inwcs = self.makeCube(file)
            log.debug('Image load done. Time taken (s): %s', time.clock() - tic)
            for iw in range(self.nwvlbins):
                self.drizcube[iw].add_image(insci[iw], inwcs, inwht=np.int_(np.logical_not(insci[iw] == 0)))

        self.cube = [d.outsci for d in self.drizcube]

        # TODO add the wavelength WCS

    def makeCube(self, file):
        sample = np.vstack((file['wavelengths'], file['photDecRad'], file['photRARad']))
        bins = np.array([self.wvlbins, self.ypix, self.xpix])

        datacube, (wavelengths, thisGridDec, thisGridRA) = np.histogramdd(sample.T, bins)

        w = wcs.WCS(naxis=2)
        w.wcs.crpix = [0., 0.]
        w.wcs.cdelt = np.array([thisGridRA[1] - thisGridRA[0], thisGridDec[1] - thisGridDec[0]])
        w.wcs.crval = [thisGridRA[0], thisGridDec[0]]
        w.wcs.ctype = ["RA---AIR", "DEC--AIR"]
        w._naxis1 = len(thisGridRA) - 1
        w._naxis2 = len(thisGridDec) - 1

        return datacube, w


class ListDrizzler(Drizzler):
    """
    Drizzle individual photons onto the celestial grid
    """
    def __init__(self, photonlists, metadata, pixfrac=1., nwvlbins=2, timestep=0.1, ntimebins=1,
                 wvlMin=0, wvlMax=np.inf):

        super().__init__(photonlists, metadata)

class TemporalDrizzler(Drizzler):
    """
    Generate a spatially dithered fits 4D hypercube from a set dithered dataset. The cube size is
    ntimebins * ndithers X nwvlbins X nPixRA X nPixDec.

    timestep or ntimebins argument accepted. ntimebins takes priority
    """

    def __init__(self, photonlists, metadata, pixfrac=1., nwvlbins=2, timestep=0.1, ntimebins=1,
                 wvlMin=0, wvlMax=np.inf):

        super().__init__(photonlists, metadata)

        self.nwvlbins = nwvlbins
        self.timestep = timestep  # seconds

        self.ndithers = len(self.files)
        self.pixfrac = pixfrac
        self.wvlbins = np.linspace(wvlMin, wvlMax, self.nwvlbins + 1)

        self.wcs_timestep = metadata.wcs_timestep
        inttime = metadata.description.inttime
        self.wcs_times = np.append(np.arange(0, inttime, self.wcs_timestep), inttime) * 1e6
        # self.wcs_times = np.arange(0, inttime + 1, self.wcs_timestep) * 1e6

        if ntimebins:
            self.ntimebins = ntimebins
        else:
            self.ntimebins = int(inttime / self.timestep)
        self.nchunktimebins = self.ntimebins // (len(self.wcs_times) - 1)
        self.timebins = np.linspace(0, inttime, self.ntimebins + 1) * 1e6  # timestamps are in microseconds
        self.totHypCube = None
        self.totWeightCube = None

        self.stackedim = np.zeros((len(metadata.description.obs) * (len(self.wcs_times)-1), nwvlbins,
                                    metadata.ypix, metadata.xpix))
        self.stacked_wcs = []

    def run(self, save_file=None):
        """

        :param save_file:
        :return:
        """
        tic = time.clock()

        self.totHypCube = np.zeros((self.ntimebins * self.ndithers, self.nwvlbins, self.nPixDec, self.nPixRA))
        self.totWeightCube = np.zeros((self.ntimebins, self.nwvlbins, self.nPixDec, self.nPixRA))
        for ix, file in enumerate(self.files):

            log.debug('Processing %s', file)

            thishyper = np.zeros((self.ntimebins, self.nwvlbins, self.nPixDec, self.nPixRA), dtype=np.float32)

            for t, inwcs in enumerate(file['obs_wcs_seq']):
                # set this here since _naxis1,2 are reinitialised during pickle
                inwcs._naxis1, inwcs._naxis2 = inwcs.naxis1, inwcs.naxis2

                insci = self.makeTess(file, (self.wcs_times[t], self.wcs_times[t+1]), applymask=False)

                self.stackedim[ix*len(file['obs_wcs_seq']) + t] = insci
                self.stacked_wcs.append(inwcs)

                for it, iw in np.ndindex(self.nchunktimebins, self.nwvlbins):
                    drizhyper = stdrizzle.Drizzle(outwcs=self.w, pixfrac=self.pixfrac)
                    drizhyper.add_image(insci[it, iw], inwcs, inwht=np.int_(np.logical_not(insci[it, iw] == 0)))
                    thishyper[it + t*self.nchunktimebins, iw] = drizhyper.outsci
                    self.totWeightCube[it + t*self.nchunktimebins, iw] += thishyper[it + t*self.nchunktimebins, iw] != 0

            self.totHypCube[ix * self.ntimebins: (ix + 1) * self.ntimebins] = thishyper

        log.debug('Image load done. Time taken (s): %s', time.clock() - tic)
        # TODO add the wavelength WCS

    def makeTess(self, file, timespan, applyweights=False, applymask=True, maxCountsCut=False):
        """

        :param file:
        :param timespan:
        :param applyweights:
        :param applymask:
        :param maxCountsCut:
        :return:
        """

        weights = file['weight'] if applyweights else None

        timespan_ind = np.where(np.logical_and(file['timestamps'] >= timespan[0],
                                               file['timestamps'] <= timespan[1]))[0]

        sample = np.vstack((file['timestamps'][timespan_ind],
                            file['wavelengths'][timespan_ind],
                            file['xPhotonPixels'][timespan_ind],
                            file['yPhotonPixels'][timespan_ind]))
        bins = np.array([self.timebins, self.wvlbins, self.ypix, self.xpix])
        hypercube, _ = np.histogramdd(sample.T, bins, weights=weights, )

        # hypercube = hypercube[:, :, :, ::-1]

        if applymask:
            log.debug("Applying bad pixel mask")
            usablemask = file['usablemask'].T.astype(int)
            hypercube *= usablemask

        if maxCountsCut:
            log.debug("Applying max pixel count cut")
            hypercube *= np.int_(hypercube < maxCountsCut)

        return hypercube

    def header_4d(self):
        """
        Add to the extra elements to the header

        Its not clear how to increase the number of dimensions of a 2D wcs.WCS() after its created so just create
        a new object, read the original parameters where needed, and overwrite

        :return:
        """

        w4d = wcs.WCS(naxis=4)
        w4d.wcs.crpix = [self.w.wcs.crpix[0], self.w.wcs.crpix[1], 1, 1]
        w4d.wcs.crval = [self.w.wcs.crval[0], self.w.wcs.crval[1], self.wvlbins[0]/1e9, self.timebins[0]/1e6]
        w4d.wcs.ctype = [self.w.wcs.ctype[0], self.w.wcs.ctype[1], "WAVE", "TIME"]
        w4d._naxis1 = self.w._naxis1
        w4d._naxis2 = self.w._naxis2
        w4d._naxis3 = self.nwvlbins
        w4d._naxis4 = self.ntimebins
        w4d.wcs.pc = np.eye(4)
        w4d.wcs.cdelt = [self.w.wcs.cdelt[0], self.w.wcs.cdelt[1],
                         (self.wvlbins[1] - self.wvlbins[0])/1e9,
                         (self.timebins[1] - self.timebins[0])/1e6]
        w4d.wcs.cunit = [self.w.wcs.cunit[0], self.w.wcs.cunit[1], "m", "sec"]

        self.w = w4d
        log.debug('4D wcs {}'.format(w4d))


class SpatialDrizzler(Drizzler):
    """ Generate a spatially dithered fits image from a set dithered dataset """

    def __init__(self, photonlists, metadata, pixfrac=1.):
        Drizzler.__init__(self, photonlists, metadata)
        self.driz = stdrizzle.Drizzle(outwcs=self.w, pixfrac=pixfrac)
        self.wcs_timestep = metadata.wcs_timestep
        inttime = metadata.description.inttime

        # if inttime is say 100 and wcs_timestep is say 60 then this yeilds [0,60,100]
        # meaning the positions don't have constant integration time
        # TODO verify this with different effective integration times
        self.wcs_times = np.append(np.arange(0, inttime, self.wcs_timestep), inttime) * 1e6

        self.stackedim = np.zeros((len(metadata.description.obs) * (len(self.wcs_times)-1),
                                    metadata.ypix, metadata.xpix))
        self.stacked_wcs = []

    def run(self, save_file=None, applymask=False):
        for ix, file in enumerate(self.files):
            log.debug('Processing %s', file)

            tic = time.clock()
            for t, inwcs in enumerate(file['obs_wcs_seq']):
                # set this here since _naxis1,2 are reinitialised during pickle
                inwcs._naxis1, inwcs._naxis2 = inwcs.naxis1, inwcs.naxis2
                insci = self.makeImage(file, (self.wcs_times[t], self.wcs_times[t+1]), applymask=False)

                self.stackedim[ix*len(file['obs_wcs_seq']) + t] = insci
                self.stacked_wcs.append(inwcs)

                if applymask:
                    insci *= ~self.hot_mask
                log.debug('Image load done. Time taken (s): %s', time.clock() - tic)
                inwht = (insci != 0).astype(int)
                self.driz.add_image(insci, inwcs, inwht=inwht)
            if save_file:
                self.driz.write(save_file)

        # TODO introduce total_exp_time variable and complete these steps

    def makeImage(self, file, timespan, applyweights=False, applymask=False, maxCountsCut=10000):

        weights = file['weight'] if applyweights else None

        # TODO mixing pixels and radians per variable names

        timespan_ind = np.where(np.logical_and(file['timestamps'] >= timespan[0],
                                               file['timestamps'] <= timespan[1]))[0]

        thisImage, _, _ = np.histogram2d(file['xPhotonPixels'][timespan_ind], file['yPhotonPixels'][timespan_ind],
                                         weights=weights, bins=[self.ypix, self.xpix], normed=False)

        if applymask:
            log.debug("Applying bad pixel mask")
            # usablemask = np.rot90(file['usablemask']).astype(int)
            usablemask = file['usablemask'].T.astype(int)
            # thisImage *= ~usablemask
            thisImage *= usablemask

        if maxCountsCut:
            log.debug("Applying max pixel count cut")
            thisImage *= thisImage < maxCountsCut

        return thisImage

    def get_persistant_bad(self, metadata, dithfrac=0.1, min_count=500, plot=True):
        """
        TODO is this function worth keeping?

        Could never really get this to work well. Requires a lot of tuning dithfrac vs min_count. Remove?

        Compare the same pixels at different dithers to determine if they are bad

        :param metadata:
        :param dithfrac:
        :param min_count:
        :param plot:
        :return:

        """
        ndithers = len(metadata.parallactic_angles)
        hot_cube = np.zeros((ndithers, metadata.ypix, metadata.xpix))
        dith_cube = np.zeros_like(hot_cube)
        for ix, file in enumerate(self.files):
            dith_cube[ix], _ = self.makeImage(file, applymask=False)
        # hot_cube[dith_cube > min_count] = ma.masked
        hot_cube[dith_cube > min_count] = 1
        hot_amount_map = np.sum(hot_cube, axis=0)  # hot_cube.count(axis=0)
        self.hot_mask = hot_amount_map / ndithers > dithfrac
        if plot:
            plt.imshow(self.hot_mask, origin='lower')
            plt.show(block=True)


def load_data(ditherdesc, wvlMin, wvlMax, startt, intt, tempfile='drizzler_tmp_{}.pkl',
              tempdir='', usecache=True, clearcache=False, derotate=True, device_orientation=-43):
    """
    Load the photons either by querying the obsfiles in parrallel or loading from pkl if it exists. The wcs
    solutions are added to this photon data dictionary but will likely be integrated into photontable.py directly

    TODO intergrate get_wcs into photontable

    :param ditherdesc:
    :param wvlMin:
    :param wvlMax:
    :param startt:
    :param intt:
    :param tempfile:
    :param tempdir:
    :param usecache:
    :param clearcache:
    :param derotate:
    :param device_orientation:
    :return:
    """
    ndither = len(ditherdesc.description.obs)

    pkl_save = os.path.join(tempdir, tempfile.format(ditherdesc.target))
    if clearcache:  # TODO the cache must be autocleared if the query parameters would alter the contents
        os.remove(pkl_save)
    try:
        if not usecache:
            raise FileNotFoundError
        with open(pkl_save, 'rb') as f:
            data = pickle.load(f)
            log.info('loaded {}'.format(pkl_save))
    except FileNotFoundError:
        begin = time.time()
        filenames = [o.h5 for o in ditherdesc.description.obs]
        if not filenames:
            log.info('No obsfiles found')

        def mp_worker(file, pos, q, startt=startt, intt=intt, startw=wvlMin, stopw=wvlMax):
            obsfile = ObsFile(file)
            usableMask = np.array(obsfile.beamFlagImage) == pixelflags.GOODPIXEL

            photons = obsfile.query(startw=startw, stopw=stopw, startt=startt, intt=intt)
            weights = photons['SpecWeight'] * photons['NoiseWeight']
            log.info("Fetched {} photons from {}".format(len(photons), file))

            x, y = obsfile.xy(photons)

            wcs = obsfile.get_wcs(platescale=ditherdesc.platescale, derotate=derotate,
                                  device_orientation=device_orientation,
                                  timestep=ditherdesc.wcs_timestep,
                                  target_coordinates=ditherdesc.coords, observatory='Subaru',
                                  target_center_at_ref=ditherdesc.rotation_center,
                                  conex_ref=(0, 0), conex_pos=pos)

            del obsfile

            q.put({'file': file, 'timestamps': photons["Time"], 'xPhotonPixels': x, 'yPhotonPixels': y,
                   'wavelengths': photons["Wavelength"], 'weight': weights, 'usablemask': usableMask,
                   'obs_wcs_seq': wcs})

        log.info('stacking number of dithers: %i'.format(ndither))

        jobs = []
        data_q = mp.Queue()

        if ndither > 25:
            raise RuntimeError('Needs rewrite, will use too many cores')

        for f, p in zip(filenames[:ndither], ditherdesc.description.pos):
            p = mp.Process(target=mp_worker, args=(f, p, data_q))
            jobs.append(p)
            p.daemon = True
            p.start()

        data = []
        for t in range(ndither):
            data.append(data_q.get())

        # Wait for all of the processes to finish fetching their data, this should hang until all the data has been
        # fetched
        for j in jobs:
            j.join()

        data.sort(key=lambda k: filenames.index(k['file']))

        log.debug('Time spent: %f' % (time.time() - begin))

        with open(pkl_save, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return data


class DrizzledData(object):
    def __init__(self, scidata, outwcs, stackedim, stacked_wcs, dither, image_weights=None):
        self.dither = dither
        self.data = scidata
        self.wcs = outwcs
        self.dumb_stack = stackedim
        self.stacked_wcs = stacked_wcs
        self.fits_header = self.wcs.to_header()
        if image_weights is not None:
            self.image_weights = image_weights

    def writefits(self, file, overwrite=True, save_stack=False, save_image=False):

        hdul = fits.HDUList([fits.PrimaryHDU(header=self.fits_header),
                             fits.ImageHDU(data=self.data, header=self.fits_header)])

        if self.data.ndim > 2 and save_image:
            image = np.sum(self.data, axis=(0, 1)) / self.image_weights
            hdul.append(fits.ImageHDU(data=image, header=self.fits_header))

        if save_stack:
            [hdul.append(fits.ImageHDU(data=dithim, header=self.stacked_wcs[i].to_header())) for i, dithim in
             enumerate(self.dumb_stack)]

        hdul.writeto(file, overwrite=overwrite)

    def quick_pretty_plot(self, log_scale=True, vmin=None, vmax=None, show=True, max_times=8):
        """
        Make an image (or array of images) with celestial coordinates (deg)

        :param scidata: image, spectralcube, or sequence of spectralcubes
        :param inwcs: single wcs solution
        :param log_scale:
        :param vmin:
        :param vmax:
        :param show:
        :param max_times: only display the first max_times frames
        :return:
        """
        if log_scale:
            norm = LogNorm()
        else:
            norm = None
        fig = plt.figure()

        # a way of identifying the non-spatial axes
        dims = len(self.data.shape)
        dim_ind = np.arange(dims)
        multiplots = np.where(dim_ind < dims - 2)[0]

        if len(multiplots) == 0:
            ax = fig.add_subplot(111, projection=self.wcs)
            axes = [ax]
            ind = [...]
        else:
            print(' *** Only displaying first {} timesteps ***'.format(max_times))
            scidata = self.data[:max_times]
            [ntimes, nwaves] = np.array(scidata.shape)[multiplots]
            gs = gridspec.GridSpec(nwaves, ntimes)
            for n in range(ntimes * nwaves):
                fig.add_subplot(gs[n], projection=self.wcs)
            axes = np.array(fig.axes)  # .reshape(ntimes, nwaves)
            ind = [(t, w) for t in range(ntimes) for w in range(nwaves)]

        for ia, ax in enumerate(axes):
            im = ax.imshow(self.data[ind[ia]], origin='lower', vmin=vmin, vmax=vmax, norm=norm)
            ax.coords.grid(True, color='white', ls='solid')
            ax.coords[0].set_axislabel('Right Ascension (J2000)')
            ax.coords[1].set_axislabel('Declination (J2000)')

        cax = fig.add_axes([0.92, 0.09 + 0.277, 0.025, 0.25])
        cb = plt.colorbar(im, cax=cax)
        cb.ax.set_title('Counts')
        plt.tight_layout()
        if show:
            plt.show(block=True)


def form(dither, mode='spatial', derotate=True, rotation_center=None, wvlMin=850, wvlMax=1100, startt=0, intt=60,
         pixfrac=.5, nwvlbins=1, timestep=1., ntimebins=0, device_orientation=-43, target_radec=(0, 0), fitsname='fits'):
    """

    :param dither:
    :param nwvlbins:
    :param timestep:
    :param device_orientation:
    :param mode: stack|spatial|spectral|temporal|list
    :param derotate: False|True|None
    :param rotation_center: None or array/tuple
    :param wvlMin:
    :param wvlMax:
    :param startt:
    :param intt:
    :param pixfrac:
    :return:
    """

    # ensure the user input is shorter than the dither or that wcs are just calculated for the relavant timespan
    if intt > dither.inttime:
        # log.warning(f'Reduced the effective integration time from {args.intt}s to {dither.inttime}s')
        log.warning('Reduced the effective integration time from {}s to {}s'.format(intt, dither.inttime))
    if dither.inttime > intt:
        # log.warning(f'Reduced the duration of each dither {dither.inttime}s to {args.intt}s')
        log.warning('Reduced the duration of each dither from {}s to {}s'.format(dither.inttime, intt))

    # redefining these variables in the middle of the code might not be good practice since form() is run multiple
    # times but once they've been equated it shouldn't have an effect?
    intt, dither.inttime = [min(intt, dither.inttime)] * 2

    ditherdesc = DitherDescription(dither, target=dither.name, target_radec=target_radec, rotation_center=rotation_center)
    data = load_data(ditherdesc, wvlMin, wvlMax, startt, intt, derotate=derotate,
                     device_orientation=device_orientation)

    if mode not in ['stack', 'spatial', 'spectral', 'temporal', 'list']:
        raise ValueError('Not calling one of the available functions')

    elif mode == 'spatial':
        driz = SpatialDrizzler(data, ditherdesc, pixfrac=pixfrac)
        driz.run(applymask=False)
        outsci = driz.driz.outsci
        outwcs = driz.w
        stackedim = driz.stackedim
        stacked_wcs = driz.stacked_wcs
        image_weights = None

    elif mode == 'spectral':
        # TODO implement, is this even necessary. On hold till interface specification and dataproduct definition
        raise NotImplementedError

    elif mode == 'temporal':
        tdriz = TemporalDrizzler(data, ditherdesc, pixfrac=pixfrac, nwvlbins=nwvlbins, timestep=timestep,
                                 ntimebins=ntimebins, wvlMin=wvlMin, wvlMax=wvlMax)
        tdriz.run()
        tdriz.header_4d()
        outsci = tdriz.totHypCube
        outwcs = tdriz.w
        image_weights = tdriz.totWeightCube.sum(axis=0)[0]
        # TODO: While we can still have a reference-point WCS solution this class needs a drizzled WCS helper as the
        # WCS solution changes with time, right?

        stackedim = tdriz.stackedim
        stacked_wcs = tdriz.stacked_wcs

    elif mode == 'list':
        ldriz = ListDrizzler(data, ditherdesc, pixfrac=pixfrac, nwvlbins=nwvlbins, timestep=timestep,
                                ntimebins=ntimebins)

    drizzle = DrizzledData(scidata=outsci, outwcs=outwcs, stackedim=stackedim, stacked_wcs=stacked_wcs, dither=dither,
                           image_weights=image_weights)
    # drizzle.quick_pretty_plot()

    if fitsname:
        drizzle.writefits(file=fitsname + '.fits')

    return drizzle


def get_star_offset(dither, wvlMin, wvlMax, startt, intt, start_guess=(0,0), zoom=2.):
    """
    Get the rotation_center offset parameter for DitherDescription

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
        print('xpix=%i, ypix=%i. Running mean=(%i,%i)'
              % (event.xdata, event.ydata, running_mean[0], running_mean[1]))

    iteration = 0
    while update:

        drizzle = form(dither=dither, mode='spatial', rotation_center=rotation_center, wvlMin=wvlMin,
                        wvlMax=wvlMax, startt=startt, intt=intt, pixfrac=1, derotate=None)

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

        if xlocs == []:  # if the user doesn't click on the figure don't change rotation_center's value
            xlocs, ylocs = np.array(image.shape)//2, np.array(image.shape)//2
        star_pix = np.array([np.mean(xlocs), np.mean(ylocs)]).astype(int)
        rotation_center += (star_pix - np.array(image.shape)//2)[::-1] #* np.array([1,-1])
        log.info('rotation_center: {}'.format(rotation_center))

        user_input = input(' *** INPUT REQUIRED *** \nDo you wish to continue looping [Y/n]: \n')
        if user_input == 'n':
            update = False

        iteration += 1

    log.info('rotation_center: {}'.format(rotation_center))

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

    # load as a task configuration
    cfg = mkidpipeline.config.load_task_config(args.cfg)

    wvlMin = args.wvlMin
    wvlMax = args.wvlMax
    startt = args.startt
    intt = args.intt
    pixfrac = cfg.drizzler.pixfrac
    dither = cfg.dither
    rotation_origin = cfg.drizzler.rotation_center
    device_orientation = cfg.drizzler.device_orientation
    target_radec = cfg.drizzler.target_radec

    if args.gso and type(args.gso) is list:
        rotation_origin = get_star_offset(dither, wvlMin, wvlMax, startt, intt, start_guess=np.array(args.gso))

    fitsname = '{}_{}.fits'.format(cfg.dither.name, drizzler_cfg_descr_str(cfg.drizzler))

    # main function of drizzler
    scidata = form(dither, mode=cfg.drizzler.mode, rotation_center=rotation_origin, wvlMin=wvlMin,
                   wvlMax=wvlMax, startt=startt, intt=intt, pixfrac=pixfrac, target_radec=target_radec,
                   device_orientation=device_orientation, derotate=True, fitsname=fitsname)

    if args.gdo:
        if not os.path.exists(fitsname):
            log.info("Can't find {} Create the fits image "
                                                 "using the default orientation first".format(fitsname))
        else:
            ditherdesc = DitherDescription(dither, target=dither.name, rotation_center=rotation_origin)
            get_device_orientation(ditherdesc, fitsname)

