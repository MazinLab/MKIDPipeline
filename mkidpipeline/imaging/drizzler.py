"""
TODO
Add astroplan, drizzle, to setup.py/yml. drizzle need to be pip installed. I found that astroplan needed to be pip
installed otherwise some astropy import fails

Move plotting functionality to another module

Update temporaldrizzler (and spectral?) to the new several wcs per dither format

Usage
-----

python drizzle.py path/to/drizzler.yml

Author: Rupert Dodkins,                                 Date: Mar 2019

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
from astropy.io import fits
from drizzle import drizzle as stdrizzle
from mkidcore import pixelflags
from mkidpipeline.hdf.photontable import ObsFile
import mkidcore.corelog as pipelinelog
import mkidpipeline
from mkidcore.instruments import CONEX2PIXEL
import argparse


# set up logging
mkidpipeline.logtoconsole()

# log = pipelinelog.getLogger('mkidpipeline.imaging.drizzler', setup=False)
log = pipelinelog.create_log('mkidpipeline.imaging.drizzler', console=True, level="INFO")


def ditherp_2_pixel(positions):
    """ A function to convert the connex offset to pixel displacement"""
    positions = np.asarray(positions)
    pix = np.asarray(CONEX2PIXEL(positions[:, 0], positions[:, 1])) - np.array(CONEX2PIXEL(0, 0)).reshape(2, 1)
    return pix


def get_device_orientation(ditherdesc, fits_filename='Theta1 Orionis B_mean.fits', separation=0.938, pa=253):
    """
    Given the position angle and offset of secondary calculate its RA and dec then
    continually update the FITS with different rotation matricies to tune for device orientation

    Default pa and offset for Trap come from https://arxiv.org/pdf/1308.4155.pdf figures 7 and 11

    B1 vs B2B3 barycenter separation is 0.938 and the position angle is 253 degrees
    """

    angle_from_east = 270-pa

    companion_ra_arcsec = np.cos(np.deg2rad(angle_from_east)) * separation
    companion_ra_offset = (companion_ra_arcsec * u.arcsec).to(u.deg).value
    companion_ra = ditherdesc.coords.ra.deg + companion_ra_offset

    companion_dec_arcsec = np.sin(np.deg2rad(angle_from_east)) * separation
    companion_dec_offset = (companion_dec_arcsec * u.arcsec).to(u.deg).value
    #minus sign here since reference object is below central star
    companion_dec = ditherdesc.coords.dec.deg - companion_dec_offset

    print('Target RA {} and dec {}'.format(Angle(companion_ra*u.deg).hms,Angle(companion_dec*u.deg).dms))

    update = True
    device_orientation = 0
    hdu1 = fits.open(fits_filename)[0]

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


def get_wcs(x, y, ditherdesc, ditherind, nxpix=146, nypix=140, platescale=0.01 / 3600., derotate=True,
            device_orientation=-48):
    """
    TODO adapt and make this a class function of photontable?

    Default device_orientation obtained with get_device_orientation using Trapezium observations

    :param device_orientation:
    :param x:
    :param y:
    :param ditherdesc:
    :param ditherind:
    :param nxpix:
    :param nypix:
    :param platescale:
    :param derotate: [True, False, None]
                     True:  align each wcs solution to position angle = 0
                     False: rotate all wcs solutions so the position angle of the middle wcs solution matches its
                            parallactic angle
                     None:  no correction angle
    :return:
    """
    dither = ditherdesc.description
    sample_times = np.arange(dither.obs[ditherind].start,
                             # -1 so that when timestep == stop-start there is one sample_time and inttime for
                             # consistancy between dithers
                             dither.obs[ditherind].start + dither.inttime - 1,
                             ditherdesc.wcs_timestep)

    device_orientation = np.deg2rad(device_orientation)
    if derotate is True:
        reformatted_times = astropy.time.Time(val=sample_times, format='unix')
        parallactic_angles = ditherdesc.apo.parallactic_angle(reformatted_times, ditherdesc.coords).value  # radians
        corrected_sky_angles = -parallactic_angles -device_orientation
    elif derotate is False:
        reformatted_mid_time = astropy.time.Time(val=(dither.obs[0].start + dither.obs[-1].stop)/2, format='unix')
        parallactic_angle = ditherdesc.apo.parallactic_angle(reformatted_mid_time, ditherdesc.coords).value
        corrected_sky_angles = np.ones_like(sample_times) * -(parallactic_angle + device_orientation)
    else:
        corrected_sky_angles = np.zeros_like(sample_times)

    log.debug("Correction angles: %s", corrected_sky_angles)

    obs_wcs_seq = []
    for t, ca in enumerate(corrected_sky_angles):
        rotation_matrix = np.array([[np.cos(ca), -np.sin(ca)],
                                   [np.sin(ca), np.cos(ca)]])

        offset_connex_frame = ditherdesc.dith_pix_offset[:, ditherind]
        offset_COR_frame = offset_connex_frame - ditherdesc.ConnexOrigin2COR.reshape(2)

        # TODO not sure why the axis flip and only one axis flip is neccessary?
        offset_inv_frame = offset_COR_frame[::-1] * np.array([1, -1])

        w = wcs.WCS(naxis=2)
        w.wcs.ctype = ["RA--TAN", "DEC-TAN"]
        w._naxis1 = nxpix
        w._naxis2 = nypix

        w.wcs.crval = np.array([ditherdesc.coords.ra.deg, ditherdesc.coords.dec.deg])
        w.wcs.crpix = np.array([nxpix / 2., nypix / 2.]) + offset_inv_frame

        w.wcs.pc = rotation_matrix
        w.wcs.cdelt = [platescale,platescale]
        w.wcs.cunit = ["deg", "deg"]

        obs_wcs_seq.append(w)

    return obs_wcs_seq


class DitherDescription(object):
    """
    Info on the dither

    rotate determines if the effective integrations are pupil stablised or not


    [1] Smart, W. M. 1962, Spherical Astronomy, (Cambridge: Cambridge University Press), p. 55

    """

    def __init__(self, dither, ConnexOrigin2COR=None, observatory='Subaru', target='* kap And',
                 use_min_timestep=True, suggested_time_step=1):
        """

        :param dither:
        :param ConnexOrigin2COR: the vector that transforms the origin of connex frame to the center of rotation frame
        :param observatory:
        :param target:
        :param use_min_timestep:
        :param suggested_time_step:
        """
        self.description = dither
        self.target = target
        try:
            self.coords = dither.obs[0].lookup_coodinates(queryname=target)
        except astropy.coordinates.name_resolve.NameResolveError:
            pipelinelog.getLogger(__name__).warning('Unable to resolve coordinates for target name {target}, using (0,0).')
            self.coords = self.SkyCoord('0 deg', '0 deg')
        self.starRA, self.starDec = self.coords.ra.deg, self.coords.dec.deg
        if ConnexOrigin2COR is None:
            ConnexOrigin2COR = (0, 0)
        assert suggested_time_step < dither.inttime, 'You must have at least a time sample per dither'

        self.ConnexOrigin2COR = np.array([list(ConnexOrigin2COR)]).T  #neccessary hideous reformatting

        self.dith_pix_offset = ditherp_2_pixel(dither.pos)
        self.cenRes = self.dith_pix_offset - self.ConnexOrigin2COR
        self.xCenRes, self.yCenRes = self.cenRes
        inst_info = dither.obs[0].instrument_info
        self.xpix = inst_info.beammap.ncols
        self.ypix = inst_info.beammap.nrows
        self.platescale = inst_info.platescale.to(u.deg).value  # 10 mas
        self.apo = Observer.at_site(observatory)
        self.observatory = observatory

        if use_min_timestep:
            min_timestep = self.calc_min_timesamp()

            # sometimes the min timestep can be ~100s of seconds. We need it to be at least shorter
            # than the dith exposure time
            self.wcs_timestep = min(dither.inttime, min_timestep)
        else:
            self.wcs_timestep = suggested_time_step

        log.debug(
            "Timestep to be used {}".format(self.wcs_timestep))

    def calc_min_timesamp(self, max_pix_disp=1.):
        """

        :param max_pix_disp: the resolution element threshold
        :return: min_timestep:
        """
        # get the field rotation rate at the start of each dither
        dith_start_times = np.array([o.start for o in dither.obs])

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
        dith_dists = np.sqrt(self.xCenRes**2 + self.yCenRes**2)
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
        self.ConnexOrigin2COR = metadata.ConnexOrigin2COR

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
            print(self.nPixRA, self.nPixDec, 'here')

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
        # TODO implement something like this
        # w = mkidcore.buildwcs(self.nPixRA, self.nPixDec, self.vPlateScale, self.starRA, self.starDec)
        # TODO implement the PV distortion?
        # eg w.wcs.set_pv([(2, 1, 45.0)])

        self.w = wcs.WCS(naxis=2)
        self.w.wcs.crpix = np.array([self.nPixRA / 2., self.nPixDec / 2.])
        if center_on_star:
            self.w.wcs.crpix -= np.array([self.ConnexOrigin2COR[0][0], self.ConnexOrigin2COR[1][0]])
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

        self.wcs_timestep = metadata.wcs_timestep
        inttime = metadata.description.inttime
        self.wcs_times = np.arange(0, inttime + 1, self.wcs_timestep) * 1e6

        self.ntimebins = int(intt / self.timestep)
        self.nchunktimebins =self.ntimebins // (len(self.wcs_times) - 1)
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

            log.debug('Processing %s', file)

            thishyper = np.zeros((self.ntimebins, self.nwvlbins, self.nPixDec, self.nPixRA), dtype=np.float32)

            for t, inwcs in enumerate(file['obs_wcs_seq']):
                print(t, self.wcs_times[t], self.wcs_times[t+1], self.ntimebins)
                insci = self.makeTess(file, (self.wcs_times[t], self.wcs_times[t+1]), applymask=False)


                for it, iw in np.ndindex(self.nchunktimebins, self.nwvlbins):
                    print(it, iw)
                    drizhyper = stdrizzle.Drizzle(outwcs=self.w, pixfrac=self.pixfrac)
                    drizhyper.add_image(insci[it, iw], inwcs, inwht=np.int_(np.logical_not(insci[it, iw] == 0)))
                    thishyper[it + t*self.nchunktimebins, iw] = drizhyper.outsci
                    self.totWeightCube[it + t*self.nchunktimebins, iw] += thishyper[it + t*self.nchunktimebins, iw] != 0

            self.totHypCube[ix * self.ntimebins: (ix + 1) * self.ntimebins] = thishyper

        log.debug('Image load done. Time taken (s): %s', time.clock() - tic)
        # TODO add the wavelength WCS

    def makeTess(self, file, timespan, applyweights=False, applymask=True, maxCountsCut=50):

        weights = file['weight'] if applyweights else None

        timespan_ind = np.where(np.logical_and(file['timestamps'] >= timespan[0],
                                               file['timestamps'] <= timespan[1]))[0]

        sample = np.vstack((file['timestamps'][timespan_ind],
                            file['wavelengths'][timespan_ind],
                            file['xPhotonPixels'][timespan_ind],
                            file['yPhotonPixels'][timespan_ind]))
        bins = np.array([self.timebins, self.wvlbins, self.ypix, self.xpix])
        hypercube, _ = np.histogramdd(sample.T, bins, weights=weights, )

        hypercube = hypercube[:, :, :, ::-1]

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
        print(w4d)

class SpatialDrizzler(Drizzler):
    """ Generate a spatially dithered fits image from a set dithered dataset """

    def __init__(self, photonlists, metadata, pixfrac=1.):
        Drizzler.__init__(self, photonlists, metadata)
        self.driz = stdrizzle.Drizzle(outwcs=self.w, pixfrac=pixfrac)
        self.wcs_timestep = metadata.wcs_timestep
        inttime = metadata.description.inttime
        self.wcs_times = np.arange(0, inttime + 1, self.wcs_timestep) * 1e6

    def run(self, save_file=None, applymask=False):
        for ix, file in enumerate(self.files):
            log.debug('Processing %s', file)

            tic = time.clock()
            for t, inwcs in enumerate(file['obs_wcs_seq']):
                insci = self.makeImage(file, (self.wcs_times[t], self.wcs_times[t+1]), applymask=False)
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
        thisImage = thisImage[:, ::-1]

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
            # plt.imshow(dith_cube[ix], origin='lower')
            # plt.show()
        # hot_cube[dith_cube > min_count] = ma.masked
        hot_cube[dith_cube > min_count] = 1
        hot_amount_map = np.sum(hot_cube, axis=0)  # hot_cube.count(axis=0)
        self.hot_mask = hot_amount_map / ndithers > dithfrac
        if plot:
            plt.imshow(self.hot_mask, origin='lower')
            plt.show(block=True)


def quick_pretty_plot(scidata, inwcs, log_scale=True, vmin=None, vmax=None, show=True, max_times=8):
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
    dims = len(scidata.shape)
    dim_ind = np.arange(dims)
    multiplots = np.where(dim_ind < dims-2)[0]

    if len(multiplots) == 0:
        ax = fig.add_subplot(111, projection=inwcs)
        axes = [ax]
        ind = [...]
    else:
        print(' *** Only displaying first {} timesteps ***'.format(max_times))
        scidata = scidata[:max_times]
        [ntimes, nwaves] = np.array(scidata.shape)[multiplots]
        gs = gridspec.GridSpec(nwaves,ntimes)
        for n in range(ntimes*nwaves):
            fig.add_subplot(gs[n], projection=inwcs)
        axes = np.array(fig.axes)#.reshape(ntimes, nwaves)
        ind = [(t,w) for t in range(ntimes) for w in range(nwaves)]

    for ia, ax in enumerate(axes):
        im = ax.imshow(scidata[ind[ia]], origin='lower', vmin=vmin, vmax=vmax, norm=norm)
        ax.coords.grid(True, color='white', ls='solid')
        ax.coords[0].set_axislabel('Right Ascension (J2000)')
        ax.coords[1].set_axislabel('Declination (J2000)')

    cax = fig.add_axes([0.92, 0.09 + 0.277, 0.025, 0.25])
    cb = plt.colorbar(im, cax=cax)
    cb.ax.set_title('Counts')
    plt.tight_layout()
    if show:
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

        def mp_worker(file, q, startt=startt, intt=intt, startw=wvlMin, stopw=wvlMax):
            obsfile = ObsFile(file)
            usableMask = np.array(obsfile.beamFlagImage) == pixelflags.GOODPIXEL

            photons = obsfile.query(startw=startw, stopw=stopw, startt=startt, intt=intt)
            weights = photons['SpecWeight'] * photons['NoiseWeight']
            log.info("Fetched {} photons from {}".format(len(photons), file))

            x, y = obsfile.xy(photons)
            del obsfile

            q.put({'file': file, 'timestamps': photons["Time"], 'xPhotonPixels': x, 'yPhotonPixels': y,
                   'wavelengths': photons["Wavelength"], 'weight': weights, 'usablemask': usableMask})

        log.info('stacking number of dithers: %i'.format(ndither))

        jobs = []
        data_q = mp.Queue()

        if ndither > 25:
            raise RuntimeError('Needs rewrite, will use too many cores')

        for f in filenames[:ndither]:
            p = mp.Process(target=mp_worker, args=(f, data_q))
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

    # Do the dither
    for i, d in enumerate(data):
        d['obs_wcs_seq'] = get_wcs(d['xPhotonPixels'], d['yPhotonPixels'], ditherdesc, i,
                                   nxpix=ditherdesc.xpix, nypix=ditherdesc.ypix,
                                   platescale=ditherdesc.platescale, derotate=derotate,
                                   device_orientation=device_orientation)

    return data


def form(dither, mode='spatial', derotate=True, ConnexOrigin2COR=None, wvlMin=850, wvlMax=1100, startt=0, intt=60,
         pixfrac=.5, nwvlbins=1, timestep=1., device_orientation=-43):
    """

    :param dither:
    :param nwvlbins:
    :param timestep:
    :param device_orientation:
    :param mode: 2->image, 3->spectral cube, 4->sequence of spectral cubes. If drizzle==False then mode is ignored
    :param derotate: 0 or 1
    :param ConnexOrigin2COR: None or array/tuple
    :param wvlMin:
    :param wvlMax:
    :param startt:
    :param intt:
    :param pixfrac:
    :return:
    """

    ditherdesc = DitherDescription(dither, target=dither.name, ConnexOrigin2COR=ConnexOrigin2COR)
    data = load_data(ditherdesc, wvlMin, wvlMax, startt, intt, derotate=derotate,
                     device_orientation=device_orientation)

    if mode not in ['spatial', 'spectral', 'temporal']:
        raise ValueError('Not calling one of the available functions')

    elif mode == 'spatial':
        driz = SpatialDrizzler(data, ditherdesc, pixfrac=pixfrac)
        # driz.get_persistant_bad(ditherdesc)
        driz.run(applymask=False)
        outsci = driz.driz.outsci
        outwcs = driz.w

        return outsci, outwcs

    elif mode == 'spectral':
        # TODO implement, is this even necessary. On hold till interface specification and dataproduct definition
        raise NotImplementedError

    elif mode == 'temporal':
        tdriz = TemporalDrizzler(data, ditherdesc, pixfrac=pixfrac, nwvlbins=nwvlbins, timestep=timestep,
                                 wvlMin=wvlMin, wvlMax=wvlMax, startt=startt, intt=intt)
        tdriz.run()
        tdriz.header_4d()
        outsci = tdriz.totHypCube
        outwcs = tdriz.w
        weights = tdriz.totWeightCube.sum(axis=0)[0]
        # TODO: While we can still have a reference-point WCS solution this class needs a drizzled WCS helper as the
        # WCS solution changes with time, right?

        return (outsci, weights), outwcs




def get_star_offset(dither, wvlMin, wvlMax, startt, intt, start_guess=(0,0), zoom=2.):
    """
    Get the ConnexOrigin2COR offset parameter for DitherDescription

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

    ConnexOrigin2COR = start_guess

    def onclick(event):
        xlocs.append(event.xdata)
        ylocs.append(event.ydata)
        running_mean = [np.mean(xlocs), np.mean(ylocs)]
        print('xpix=%i, ypix=%i. Running mean=(%i,%i)'
              % (event.xdata, event.ydata, running_mean[0], running_mean[1]))

    iteration = 0
    while update:
        fig, ax = plt.subplots()

        image, _ = form(dither=dither, mode='spatial', ConnexOrigin2COR=ConnexOrigin2COR, wvlMin=wvlMin,
                        wvlMax=wvlMax, startt=startt, intt=intt, pixfrac=1, derotate=None)

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

        if xlocs == []:  # if the user doesn't click on the figure don't change connexOrigin2COR's value
            xlocs, ylocs = np.array(image.shape)//2, np.array(image.shape)//2
        star_pix = np.array([np.mean(xlocs), np.mean(ylocs)]).astype(int)
        ConnexOrigin2COR += (star_pix - np.array(image.shape)//2)[::-1] * np.array([1,-1])
        log.info('ConnexOrigin2COR: {}'.format(ConnexOrigin2COR))

        user_input = input(' *** INPUT REQUIRED *** \nDo you wish to continue looping [Y/n]: \n')
        if user_input == 'n':
            update = False

        iteration += 1

    log.info('ConnexOrigin2COR: {}'.format(ConnexOrigin2COR))

    return ConnexOrigin2COR


def drizzler_cfg_descr_str(drizzlercfg):
    return 'TODO_drizzler_cfg_descr'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Photon Drizzling Utility')
    parser.add_argument('cfg', type=str, help='The configuration file')
    parser.add_argument('-wl', type=float, dest='wvlMin', help='minimum wavelength', default=850)
    parser.add_argument('-wh', type=float, dest='wvlMax', help='maximum wavelength', default=1100)
    parser.add_argument('-t0', type=int, dest='startt', help='start time', default=0)
    parser.add_argument('-it', type=int, dest='intt', help='end time', default=60)
    parser.add_argument('-p', action='store_true', dest='plot', help='Plot the result', default=False)
    parser.add_argument('--get-offset', nargs=2, type=int, dest='gso', help='Runs get_star_offset eg 0 0 ')
    parser.add_argument('--get-orientation', type=lambda x: os.path.isfile(x), dest='gdo',
                        help='Run get_device_orientation on a fits file, first created with the default orientation.',
                        default='Theta1 Orionis B_mean.fits')
    # parser.add_argument('--write', type=str, dest='write', help='Save a fits file with this name appended',
    #                     default='drizzle-mean')

    args = parser.parse_args()

    # load as a task configuration
    cfg = mkidpipeline.config.load_task_config(args.cfg)

    wvlMin = args.wvlMin
    wvlMax = args.wvlMax
    startt = args.startt
    pixfrac = cfg.drizzler.pixfrac
    dither = cfg.dither
    device_orientation = cfg.drizzler.device_orientation

    intt = min(dither.inttime, args.intt)  # user input is longer than the duration of each dither

    if args.gso:
        connexOrigin2COR = get_star_offset(dither, wvlMin, wvlMax, startt, intt, start_guess=args.gso)
    else:
        connexOrigin2COR = cfg.drizzler.connexorigin2cor

    if args.gdo:
        ditherdesc = DitherDescription(dither, target=dither.name, ConnexOrigin2COR=connexOrigin2COR)
        get_device_orientation(ditherdesc, args.gdo)
        exit()

    # main function of drizzler
    scidata, drizwcs = form(dither, mode=cfg.drizzler.mode, ConnexOrigin2COR=connexOrigin2COR, wvlMin=wvlMin,
                            wvlMax=wvlMax, startt=startt, intt=intt, pixfrac=pixfrac,
                            device_orientation=device_orientation)

    if cfg.drizzler.mode == 'temporal':
        image = np.sum(scidata[0], axis=(0, 1)) / scidata[1]
        hdul = fits.HDUList([fits.PrimaryHDU(header=drizwcs.to_header()),
                             fits.ImageHDU(data=scidata[0], header=drizwcs.to_header()),
                             fits.ImageHDU(data=image, header=drizwcs.to_header())])
    else:
        hdul = fits.HDUList([fits.PrimaryHDU(header=drizwcs.to_header()),
                             fits.ImageHDU(data=scidata, header=drizwcs.to_header())])

    hdul.writeto('{}_{}.fits'.format(cfg.dither.name, drizzler_cfg_descr_str(cfg.drizzler), overwrite=True)

    if args.plot:
        quick_pretty_plot(image, drizwcs, vmin=5, vmax=600)


