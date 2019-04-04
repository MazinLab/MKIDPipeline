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


def ditherp_2_pixel(positions):
    """ A function to convert the connex offset to pixel displacement"""
    positions = np.asarray(positions)
    pix = np.asarray(CONEX2PIXEL(positions[:, 0], positions[:, 1])) - np.array(CONEX2PIXEL(0, 0)).reshape(2, 1)
    return pix


class DitherDescription(object):
    """
    Info on the dither

    rotate determines if the effective integrations are pupil stablised or not


    [1] Smart, W. M. 1962, Spherical Astronomy, (Cambridge: Cambridge University Press), p. 55

    """

    def __init__(self, dither, ConnexOrigin2COR=None, observatory='Subaru', target='* kap And',
                 use_min_timestep=True, suggested_time_step=1):
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

        self.ConnexOrigin2COR = np.array([list(ConnexOrigin2COR)]).T

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
            self.timestep = min(dither.inttime, min_timestep)
        else:
            self.timestep = suggested_time_step

        pipelinelog.getLogger(__name__).debug(
            "Timestep to be used {}".format(self.timestep))

    def calc_min_timesamp(self, max_pix_disp=1.):
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

        pipelinelog.getLogger(__name__).debug("Minimum required time step calculated to be {}".format(min_timestep))

        return min_timestep


def get_device_orientation(ditherdesc, fits_filename, separation = 0.938, pa = 253):
    """
    Given the position angle and offset of secondary calculate its RA and dec then
    continually update the FITS with different rotation matricies to tune for device orientation

    Default pa and offset for Trap come from https://arxiv.org/pdf/1308.4155.pdf figures 7 and 11

    B1 vs B2B3 barycenter separation is 0.938 and the position angle is 253 degrees"""

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
    position_angle = 0
    hdu1 = fits.open(fits_filename)[0]

    field = hdu1.data
    while update:

        ax1 = plt.subplot(111, projection=wcs.WCS(hdu1.header))
        ax1.imshow(field, norm=LogNorm(), origin='lower', vmin=1)
        # plt.colorbar()
        plt.show()

        user_input = input('Enter new angle (deg) or F to end: ')
        if user_input == 'F':
            update = False
        else:
            position_angle += float(user_input)

        field = imrotate(hdu1.data, position_angle, interp='bilinear')
        print(position_angle)

    return position_angle


def get_wcs(x, y, ditherdesc, ditherind, nxpix=146, nypix=140, platescale=0.01 / 3600., derotate=True,
            device_orientation=-48*np.pi/180):
    """
    Default device_orientation obtained with get_device_orientation using Trapezium observations

    :param x:
    :param y:
    :param ditherdesc:
    :param ditherind:
    :param nxpix:
    :param nypix:
    :param platescale:
    :param derotate:
    :param device_orientation:
    :return:
    """
    dither = ditherdesc.description
    sample_times = np.arange(dither.obs[ditherind].start,
                             # -1 so that when timestep == stop-start there is one sample_time and inttime for
                             # consistancy between dithers
                             dither.obs[ditherind].start + dither.inttime - 1,
                             ditherdesc.timestep)

    if derotate:
        reformatted_times = astropy.time.Time(val=sample_times, format='unix')
        parallactic_angles = ditherdesc.apo.parallactic_angle(reformatted_times, ditherdesc.coords).value # radians
        corrected_sky_angles = -parallactic_angles -device_orientation

    else:
        reformatted_mid_time = astropy.time.Time(val=(dither.obs[0].start + dither.obs[-1].stop)/2, format='unix')
        corrected_sky_angles = np.ones_like(sample_times) * -(reformatted_mid_time + device_orientation)

    pipelinelog.getLogger(__name__).debug("Correction angles: %s", corrected_sky_angles)

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

        w.wcs.cd = rotation_matrix * platescale

        obs_wcs_seq.append(w)

    return obs_wcs_seq


class Drizzler(object):
    def __init__(self, photonlists, metadata):

        # TODO determine appropirate value from area coverage of dataset and oversampling, even longerterm there
        # the oversampling should be selected to optimize total phase coverage to extract the most resolution at a
        # desired minimum S/N

        # self.randoffset = False # apply random spatial offset to each photon
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
                dith_cellestial_span = np.vstack((photonlist['obs_wcs_seq'][0].wcs.crval,
                                                  photonlist['obs_wcs_seq'][-1].wcs.crval))
                dith_cellestial_min[ip] = np.min(dith_cellestial_span, axis=0)  # takes the min of both ra and dec
                dith_cellestial_max[ip] = np.max(dith_cellestial_span, axis=0)

            # find the min and max coordinate of all dithers
            raMin = min(dith_cellestial_min[:, 0])
            raMax = max(dith_cellestial_max[:, 0])
            decMin = min(dith_cellestial_min[:, 1])
            decMax = max(dith_cellestial_max[:, 1])

            # Set size of virtual grid to accommodate.
            max_detector_dist = np.sqrt(self.xpix ** 2 + self.ypix **2)
            self.nPixRA = (2 * np.max((raMax-self.starRA, self.starRA-raMin))/self.vPlateScale + max_detector_dist).astype(int)
            self.nPixDec = (2 * np.max((decMax-self.starDec, self.starDec-decMin))/self.vPlateScale + max_detector_dist).astype(int)



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

    def get_header(self, center_on_star=True):
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
        self.w.wcs.cd = np.array([[1,0],[0,1]]) * self.vPlateScale
        self.w.wcs.cunit = ["deg", "deg"]
        pipelinelog.getLogger(__name__).debug(self.w)


class SpectralDrizzler(Drizzler):
    """ Generate a spatially dithered fits dataacube from a set dithered dataset """

    def __init__(self, photonlists, metadata, pixfrac=1.):
        self.nwvlbins = 3
        self.wvlbins = np.linspace(metadata.wvlMin, metadata.wvlMax, self.nwvlbins + 1)
        super().__init__(photonlists, metadata)
        self.drizcube = [stdrizzle.Drizzle(outwcs=self.w, pixfrac=pixfrac)] * self.nwvlbins

    def run(self, save_file=None):
        for ix, file in enumerate(self.files):

            pipelinelog.getLogger(__name__).debug('Processing %s', file)
            tic = time.clock()
            insci, inwcs = self.makeCube(file)
            pipelinelog.getLogger(__name__).debug('Image load done. Time taken (s): %s', time.clock() - tic)
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
    TODO this needs to be updated to the new several WCS per dither format

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

            pipelinelog.getLogger(__name__).debug('Processing %s', file)

            insci, inwcs = self.makeHyper(file)

            thishyper = np.zeros((self.ntimebins, self.nwvlbins, self.nPixDec, self.nPixRA), dtype=np.float32)

            for it, iw in np.ndindex(self.ntimebins, self.nwvlbins):
                drizhyper = stdrizzle.Drizzle(outwcs=self.w, pixfrac=self.pixfrac)
                drizhyper.add_image(insci[it, iw], inwcs, inwht=np.int_(np.logical_not(insci[it, iw] == 0)))
                thishyper[it, iw] = drizhyper.outsci
                self.totWeightCube[it, iw] += thishyper[it, iw] != 0

            self.totHypCube[ix * self.ntimebins: (ix + 1) * self.ntimebins] = thishyper

        pipelinelog.getLogger(__name__).debug('Image load done. Time taken (s): %s', time.clock() - tic)
        # TODO add the wavelength WCS

    def makeHyper(self, file, applyweights=False, applymask=True, maxCountsCut=10):
        if applyweights:
            weights = file['weight']
        else:
            weights = None
        sample = np.vstack((file['timestamps'], file['wavelengths'], file['photDecRad'], file['photRARad']))
        bins = np.array([self.timebins, self.wvlbins, self.ypix, self.xpix])
        hypercube, bins = np.histogramdd(sample.T, bins, weights=weights, )

        if applymask:
            usablemask = file['usablemask'].T.astype(int)
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
        self.timestep = metadata.timestep
        self.inttime = metadata.description.inttime
        self.times = np.arange(0, self.inttime + 1, self.timestep) * 1e6

    def run(self, save_file=None, applymask=False):
        for ix, file in enumerate(self.files):
            pipelinelog.getLogger(__name__).debug('Processing %s', file)

            tic = time.clock()
            for t, inwcs in enumerate(file['obs_wcs_seq']):
                insci = self.makeImage(file, (self.times[t], self.times[t+1]), applymask=False)
                if applymask:
                    insci *= ~self.hot_mask
                pipelinelog.getLogger(__name__).debug('Image load done. Time taken (s): %s', time.clock() - tic)
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
            pipelinelog.getLogger(__name__).debug("Applying bad pixel mask")
            # usablemask = np.rot90(file['usablemask']).astype(int)
            usablemask = file['usablemask'].T.astype(int)
            # thisImage *= ~usablemask
            thisImage *= usablemask

        if maxCountsCut:
            pipelinelog.getLogger(__name__).debug("Applying max pixel count cut")
            thisImage *= thisImage < maxCountsCut

        return thisImage

    def get_persistant_bad(self, metadata, dithfrac=0.1, min_count=500, plot=True):
        '''
        Could never really get this to work well. Requires a lot of tuning dithfrac vs min_count. Remove?

        Compare the same pixels at different dithers to determine if they are bad

        :param metadata:
        :param dithfrac:
        :param min_count:
        :param plot:
        :return:

        '''
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


def pretty_plot(image, inwcs, log_scale=False, vmin=None, vmax=None, show=True):
    if log_scale:
        norm = LogNorm()
    else:
        norm = None
    fig = plt.figure()
    ax = fig.add_subplot(111, projection=inwcs)
    cax = ax.imshow(image, origin='lower', vmin=vmin, vmax=vmax, norm=norm)
    ax.coords.grid(True, color='white', ls='solid')
    ax.coords[0].set_axislabel('Right Ascension (J2000)')
    ax.coords[1].set_axislabel('Declination (J2000)')
    cb = plt.colorbar(cax)
    cb.ax.set_title('Counts')
    if show:
        plt.show(block=True)


def load_data(ditherdesc, wvlMin, wvlMax, startt, intt, tempfile='drizzler_tmp_{}.pkl',
              tempdir='', usecache=True, clearcache=False, derotate=True):
    ndither = len(ditherdesc.description.obs)

    pkl_save = os.path.join(tempdir, tempfile.format(ditherdesc.target))
    if clearcache:  # TODO the cache must be autocleared if the query parameters would alter the contents
        os.remove(pkl_save)
    try:
        if not usecache:
            raise FileNotFoundError
        with open(pkl_save, 'rb') as f:
            data = pickle.load(f)
            pipelinelog.getLogger(__name__).info('loaded', pkl_save)
    except FileNotFoundError:
        begin = time.time()
        filenames = [o.h5 for o in ditherdesc.description.obs]
        if not filenames:
            pipelinelog.getLogger(__name__).info('No obsfiles found')

        def mp_worker(file, q, startt=startt, intt=intt, startw=wvlMin, stopw=wvlMax):
            obsfile = ObsFile(file)
            usableMask = np.array(obsfile.beamFlagImage) == pixelflags.GOODPIXEL

            photons = obsfile.query(startw=startw, stopw=stopw, startt=startt, intt=intt)
            weights = photons['SpecWeight'] * photons['NoiseWeight']
            pipelinelog.getLogger(__name__).info("Fetched {} photons from {}".format(len(photons), file))

            x, y = obsfile.xy(photons)
            del obsfile

            q.put({'file': file, 'timestamps': photons["Time"], 'xPhotonPixels': x, 'yPhotonPixels': y,
                   'wavelengths': photons["Wavelength"], 'weight': weights, 'usablemask': usableMask})

        pipelinelog.getLogger(__name__).info('stacking number of dithers: %i'.format(ndither))

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

        pipelinelog.getLogger(__name__).debug('Time spent: %f' % (time.time() - begin))

        with open(pkl_save, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Do the dither
    for i, d in enumerate(data):
        d['obs_wcs_seq'] = get_wcs(d['xPhotonPixels'], d['yPhotonPixels'], ditherdesc, i,
                                   nxpix=ditherdesc.xpix, nypix=ditherdesc.ypix,
                                   platescale=ditherdesc.platescale, derotate=derotate)
        # d['photRARad'], d['photDecRad'] = radec

    return data


def form(dither, dim='spatial', derotate=True, ConnexOrigin2COR=None, wvlMin=850, wvlMax=1100, startt=0, intt=60, pixfrac=.5,
         nwvlbins=1, timestep=1.):
    """

    :param dim: 2->image, 3->spectral cube, 4->sequence of spectral cubes. If drizzle==False then dim is ignored
    :param derotate: 0 or 1
    :param ConnexOrigin2COR: None or array/tuple
    :param wvlMin:
    :param wvlMax:
    :param startt:
    :param intt:
    :param pixfrac:
    :param drizzle: Bool for output format

    :return:
    """

    ditherdesc = DitherDescription(dither, target=dither.name, ConnexOrigin2COR=ConnexOrigin2COR)
    data = load_data(ditherdesc, wvlMin, wvlMax, startt, intt, derotate=derotate)

    if dim not in ['spatial', 'spectral', 'temporal']:
        raise ValueError('Not calling one of the available functions')

    elif dim == 'spatial':
        driz = SpatialDrizzler(data, ditherdesc, pixfrac=pixfrac)
        # driz.get_persistant_bad(ditherdesc)
        driz.run(applymask=False)
        outsci = driz.driz.outsci
        outwcs = driz.w

    elif dim == 'spectral':
        # TODO implement, is this even necessary. On hold till interface specification and dataproduct definition
        raise NotImplementedError

    elif dim == 'temporal':
        tdriz = TemporalDrizzler(data, ditherdesc, pixfrac=pixfrac, nwvlbins=nwvlbins, timestep=timestep,
                                 wvlMin=wvlMin, wvlMax=wvlMax, startt=startt, intt=intt)
        tdriz.run()
        outsci = tdriz.totHypCube
        outwcs = tdriz.w
        # weights = tdriz.totWeightCube.sum(axis=0)[0]
        # TODO: While we can still have a reference-point WCS solution this class needs a drizzled WCS helper as the
        # WCS solution changes with time, right?

    return outsci, outwcs


def get_star_offset(dither, wvlMin, wvlMax, startt, intt):
    '''
    Get the ConnexOrigin2COR offset parameter for DitherDescription

    :param dither:
    :param wvlMin:
    :param wvlMax:
    :param startt:
    :param intt:
    :return:
    '''

    fig, ax = plt.subplots()

    image, _ = form(dither=dither, dim='spatial', ConnexOrigin2COR=(0,0), derotate=False,
                    wvlMin=wvlMin, wvlMax=wvlMax, startt=startt, intt=intt, pixfrac=1)

    print("Click on the four satellite speckles and the star")
    cax = ax.imshow(image, origin='lower', norm=None)
    cb = plt.colorbar(cax)
    cb.ax.set_title('Counts')

    xlocs, ylocs = [], []
    def onclick(event):
        xlocs.append(event.xdata)
        ylocs.append(event.ydata)
        running_mean = [np.mean(xlocs), np.mean(ylocs)]
        print('xpix=%i, ypix=%i. Running mean=(%i,%i)'
              % (event.xdata, event.ydata, running_mean[0], running_mean[1]))

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show(block=True)

    star_pix = np.array([np.mean(xlocs), np.mean(ylocs)]).astype(int)
    ConnexOrigin2COR = star_pix - np.array(image.shape)//2

    return ConnexOrigin2COR


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Photon Drizzling Utility')
    parser.add_argument('cfg', type=str, help='The configuration file')
    parser.add_argument('-wl', type=float, dest='wvlMin', help='', default=850)
    parser.add_argument('-wh', type=float, dest='wvlMax', help='', default=1100)
    parser.add_argument('-t0', type=int, dest='startt', help='', default=0)
    parser.add_argument('-it', type=int, dest='intt', help='', default=60)
    args = parser.parse_args()

    # set up logging
    mkidpipeline.logtoconsole()

    pipelinelog.getLogger('mkidpipeline.hdf.photontable').setLevel('INFO')

    # load as a task configuration
    cfg = mkidpipeline.config.load_task_config(args.cfg)

    wvlMin = args.wvlMin
    wvlMax = args.wvlMax
    startt = args.startt
    intt = args.intt
    pixfrac = cfg.drizzler.pixfrac
    dither = cfg.dither

    # ConnexOrigin2COR = get_star_offset(dither, wvlMin, wvlMax, startt, intt)
    ConnexOrigin2COR = (20,20)

    # ditherdesc = DitherDescription(dither, target=dither.name, ConnexOrigin2COR=ConnexOrigin2COR)
    # get_device_orientation(ditherdesc, 'Theta1 Orionis B_mean.fits')

    image, drizwcs = form(dither, 'spatial', ConnexOrigin2COR=ConnexOrigin2COR, wvlMin=wvlMin,
                          wvlMax=wvlMax, startt=startt, intt=intt, pixfrac=pixfrac)

    pretty_plot(image, drizwcs, vmin=5, vmax=600)

    hdu = fits.PrimaryHDU(image, header=drizwcs.to_header())
    hdu.writeto(cfg.dither.name + '_mean.fits', overwrite=True)

    # # fits.writeto(cfg.dither.name + '_mean.fits', image, drizwcs.to_header(), overwrite=True)
    #
    # # tess, drizwcs = form(dither, 'temporal', ConnexOrigin2COR=ConnexOrigin2COR, wvlMin=wvlMin,
    # #                      wvlMax=wvlMax, startt=startt, intt=intt, pixfrac=pixfrac, nwvlbins=1)
    # #
    # # #Get median collpased image
    # # mask_cube = np.ma.masked_where(tess[:, 0] == 0, tess[:, 0])
    # # medDither = np.ma.median(mask_cube, axis=0).filled(0)
    # #
    # # pretty_plot(medDither, drizwcs.wcs.cdelt[0], drizwcs.wcs.crval, vmin=1, vmax=10)
