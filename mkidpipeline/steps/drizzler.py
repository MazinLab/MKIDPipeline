"""
Classes

    DrizzleParams   : Calculates and stores the relevant info for Drizzler
    Canvas          : Called by the "Drizzler" classes. Generates the canvas that is drizzled onto
    Drizzler        : Generates a spatially dithered 4-cube (xytw)
    DrizzledData    : Saves the drizzled data as FITS

Functions

    _increment_id    : Monkey patch for STScI drizzle class of drizzle package
    mp_worker       : Genereate a reduced, reformated photonlist
    load_data       : Consolidate all dither positions
    form            : Takes in a MKIDDither object and drizzles the dithers onto a common sky grid
"""
import os
import numpy as np
import time
import multiprocessing as mp
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
import pickle
import hashlib
from glob import glob
import getpass
from mkidcore.metadata import MetadataSeries
import astropy
from astropy.io import fits
from astropy import wcs
from astropy.coordinates import EarthLocation
import astropy.units as u
#from drizzle import drizzle as stdrizzle
import mkidcore.corelog
import mkidcore.pixelflags
import mkidcore.metadata
from mkidcore.utils import mjd_to
from mkidcore.corelog import getLogger
from mkidcore.instruments import CONEX2PIXEL
from mkidpipeline.photontable import Photontable
import mkidpipeline.config
from mkidcore.utils import astropy_observer

EXCLUDE = ('pixcal.dead', 'pixcal.hot', 'pixcal.cold', 'beammap.noDacTone', 'wavecal.bad', 'wavecal.failed_convergence',
           'wavecal.no_histograms', 'wavecal.not_attempted', 'flatcal.bad')  # fill with undesired flags
PROBLEM_FLAGS = tuple()  # fill with flags that will break drizzler


class StepConfig(mkidpipeline.config.BaseStepConfig):
    yaml_tag = u'!drizzler_cfg'
    REQUIRED_KEYS = (('plots', 'summary', 'Which plots to generate: none|summary'),
                     ('pixfrac', 0.5, 'The drizzle algorithm pixel fraction'),
                     ('wcs_timestep', None, 'Seconds between different WCS (eg orientations). If None, the the '
                                            'non-blurring minimum (1 pixel at furthest dither center) will be used'),
                     ('whitelight', False, 'If True will not expect an OBJECT, RA, or DEC in the header and will only '
                                           'use the CONEX position to calculate the WCS. Used for bench tests where '
                                           'data is not taken'),
                     ('save_steps', False, 'Save intermediate fits files where possible (only some modes)'),
                     ('usecache', False, 'Cache photontable for subsequent runs'),
                     ('ncpu', 1, 'Number of CPUs to use'),
                     ('clearcache', False, 'Clear user cache on next run'))


class DrizzleParams:
    """Calculates and stores the relevant parameters for Drizzler"""
    def __init__(self, dither, inttime, wcs_timestep=None, pixfrac=1.0, simbad=False, startt=0, whitelight=False):
        """
        :param dither: MKIDDither, contains the lists of observations and metadata for a set of dithers
        :param inttime: duration in seconds
        :param wcs_timestep: cadence at which to calculate discrete WCS solutions
        :param pixfrac: ratio of how much each input pixel is shrunk by before drizzling that data onto the output grid
        (see also STScI DrizzlePac documentation)
        :param simbad: If True will use simbad to query the target coordinates instead of using the RA and Dec saved in
        the metadata
        :param startt: start time in relative seconds
        :param whitelight: If True will run drizzler in a simplified mode so as to avoid errors associated with not
        having required on-sky metadata. To be used with any data taken off-sky (i.e. using the SCExAO bench)
        """
        self.n_dithers = len(dither.obs)
        self.image_shape = dither.obs[0].beammap.shape
        self.platescale = [v.platescale.to(u.deg).value if v.platescale else
                           dither.obs[0].photontable.query_header('E_PLTSCL') for v in dither.wcscal.values()][0]
        if isinstance(self.platescale, u.Quantity):
            self.platescale = self.platescale.to(u.deg).value
        self.inttime = inttime
        self.pixfrac = pixfrac
        self.startt = startt
        # Get the SkyCoord type coordinates to use for center of sky grid that is drizzled onto
        if not whitelight:
            self.coords = mkidcore.metadata.skycoord_from_metadata(dither.obs[0].metadata_at(), force_simbad=simbad)
        else:
            self.coords = astropy.coordinates.SkyCoord(0, 0, unit=('hourangle', 'deg'))
        instrument = dither.obs[0].photontable.query_header('INSTRUME', last_if_series=True)
        default_tel = mkidcore.metadata.INSTRUMENT_KEY_MAP[instrument.lower()]['card']['TELESCOP'].value
        self.telescope = dither.obs[0].photontable.query_header('TELESCOP', last_if_series=True) or default_tel

        self.canvas_shape = (None, None)
        self.dith_start_times = np.array([o.start for o in dither.obs])
        self.dither_pos = np.asarray(dither.pos).T
        self.wcs_timestep = wcs_timestep or self.non_blurring_timestep(
            ref_pix=(dither.obs[0].photontable.query_header('E_PREFX'),
                     dither.obs[0].photontable.query_header('E_PREFY')),
            ref_con=(dither.obs[0].photontable.query_header('E_CXREFX'),
                     dither.obs[0].photontable.query_header('E_CXREFY')),
            conex_slopes=(dither.obs[0].photontable.query_header('E_DPDCX'),
                          dither.obs[0].photontable.query_header('E_DPDCY')))

    def non_blurring_timestep(self, allowable_pixel_smear=1, center=(0, 0), ref_pix=(0, 0), ref_con=(0, 0),
                              conex_slopes=(0, 0)):
        """
        Calculates the minimum required timestep that would produce allowable_pixel_smear pixel displacement at the
        center of the furthest dither
        [1] Smart, W. M. 1962, Spherical Astronomy, (Cambridge: Cambridge University Press), p. 55

        :param allowable_pixel_smear: resolution element threshold
        :param center: center tip-tilt mirror position
        :param ref_pix: (x, y) pixel reference position
        :param ref_con: tip-tilt mirror position at which ref_pix was determined
        :param conex_slopes: (dx, dy) how much the tip-tilt mirror moves in pixels for a mirror displacement of 1
        :return: maximum non-blurring timestep in seconds
        """
        # get the field rotation rate at the start of each dither
        site, apo = astropy_observer(self.telescope)
        altaz = apo.altaz(astropy.time.Time(val=self.dith_start_times, format='unix'), self.coords)
        earthrate = 2 * np.pi / astropy.units.sday.to(astropy.units.second)

        dith_start_rot_rates = (earthrate * np.cos(site.geodetic.lat.rad) * np.cos(altaz.az.radian) /
                                np.cos(altaz.alt.radian))
        dith_pix_offset = (CONEX2PIXEL(*self.dither_pos, ref_pix=ref_pix, slopes=conex_slopes, ref_con=ref_con) -
                           CONEX2PIXEL(*center, ref_pix=ref_pix, slopes=conex_slopes, ref_con=ref_con).reshape(2, 1))
        angle = np.arctan2(allowable_pixel_smear, np.linalg.norm(dith_pix_offset))
        max_timestep = np.abs(angle / dith_start_rot_rates).min()

        getLogger(__name__).debug(f"Maximum non-blurring time step calculated to be {max_timestep:.1f} s")
        return max_timestep


class Canvas:
    """Creates the canvas on which the drizzled data is placed"""

    def __init__(self, dithers_data, drizzle_params, canvas_shape=(None, None), force_square_grid=True, buffer=10,
                 rate=True):
        """
        :param dithers_data: list of dictionaries of relevant input data and parameters (see output of load_data)
        :param drizzle_params: DrizzleParams object
        :param canvas_shape: (x, y) shape of the Canvas on which to place the drizzled output. Note that if any part of
        the image falls of of the canvas this can result in the STScI drizzle code to return all 0's
        :param force_square_grid: If True will force the x and y dimensions of the drizzled output to be the same
        :param buffer: buffer (in pixels) to add to canvas grid to ensure no image falls off the edge of the grid
        """

        # TODO determine appropriate value from area coverage of dataset and oversampling, even longer term there
        #  the oversampling should be selected to optimize total phase coverage to extract the most resolution at a
        #  desired minimum S/N

        self.canvas_shape = canvas_shape
        self.dithers_data = dithers_data
        self.drizzle_params = drizzle_params
        self.shape = drizzle_params.image_shape
        self.vPlateScale = drizzle_params.platescale
        self.center = drizzle_params.coords
        self.header = None
        self.rate = rate

        if canvas_shape[0] is None or canvas_shape[1] is None:
            dith_pix_min = np.zeros((len(dithers_data), 2))
            dith_pix_max = np.zeros((len(dithers_data), 2))
            for ip, photonlist in enumerate(dithers_data):
                # find the max and min coordinate for each dither (assuming those occur at the beginning/end of
                # the dither)
                dith_pix_span = np.vstack((photonlist['obs_wcs_seq'][0].wcs.crpix,
                                           photonlist['obs_wcs_seq'][-1].wcs.crpix))
                dith_pix_min[ip] = np.min(dith_pix_span, axis=0)  # takes the min of both ra and dec
                dith_pix_max[ip] = np.max(dith_pix_span, axis=0)

            # find the min and max coordinate of all dithers
            pix_ra_span = [min(dith_pix_min[:, 0]), max(dith_pix_max[:, 0])]
            pix_dec_span = [min(dith_pix_min[:, 1]), min(dith_pix_min[:, 1])]

            # Set size of virtual grid to accommodate the limits of the offsets.
            # max_detector_dist = np.linalg.norm(self.shape)
            self.canvas_shape = ((2 * (pix_ra_span[1] - pix_ra_span[0]) + self.shape[0] + buffer).astype(int),
                                 (2 * (pix_dec_span[1] - pix_dec_span[0]) + self.shape[1] + buffer).astype(int))

        if force_square_grid:
            self.canvas_shape = tuple([max(self.canvas_shape)] * 2)

        # check the size of the grid is sensible. equivalent dithering along a line
        if max(self.canvas_shape) > max(self.shape) * len(dithers_data):
            getLogger(__name__).warning(f'Canvas grid {self.canvas_shape} exceeds maximum nominal extent of dithers '
                                        f'({max(self.shape) * len(dithers_data)})')
        # Create the canvas WCS
        self.wcs = wcs.WCS(naxis=2)
        self.wcs.wcs.pc = np.eye(2)
        self.wcs.pixel_shape = self.canvas_shape
        self.wcs.wcs.crpix = np.array(self.canvas_shape) / 2
        self.wcs.wcs.crval = [self.center.ra.deg, self.center.dec.deg]
        self.wcs.wcs.ctype = ["RA--TAN", "DEC-TAN"]
        self.wcs.wcs.cdelt = [self.vPlateScale, self.vPlateScale]
        self.wcs.wcs.cunit = ["deg", "deg"]

        self.header = self.canvas_header()

    def canvas_header(self):
        """
        Creates the majority of the header for the drizzled data. Some keys are added after drizzling.

        combine metadata from all input data into a single dict and use that to build ahead with mkidcore.build_header
        (to get defaults)

        All keys use the first value of the dither except for time keys (UT, HST, MJD, UNIX), which take the start,
        mean, or last as appropriate.

        Keys 'UT' and 'HST' will correspond to 'MJD'
        """
        combined_meta = self.dithers_data[0]['metadata'].copy()
        for entry in combined_meta:
            if not isinstance(combined_meta[entry], MetadataSeries):
                combined_meta[entry] = MetadataSeries(times=[self.dithers_data[0]['metadata']['UNIXSTR']],
                                                      values=[combined_meta[entry]])
        meta = [data['metadata'] for data in self.dithers_data[1:]]
        for md in meta:
            for entry in md:
                if not isinstance(md[entry], MetadataSeries):
                    assert isinstance(combined_meta[entry], MetadataSeries)
                    combined_meta[entry].add(md['UNIXSTR'], md[entry])
                else:
                    combined_meta[entry] += md[entry]

        metadata = combined_meta
        meta = {}

        mean = ('MJD', )
        last = ('UNIXEND', 'MJD-END', 'UT-END', 'HST-END')

        for key, series in metadata.items():
            if not len(series.values):
                continue
            if key in last:
                val = series.values[-1]
            elif key in mean:
                val = np.mean(series.values)
            else:
                val = series.values[0]
            meta[key] = val
        try:
            meta['UT'] = mjd_to(meta['MJD'], 'UTC').strftime('%H:%M:%S.%f')[:-4]
            meta['HST'] = mjd_to(meta['MJD'], 'HST').strftime('%H:%M:%S.%f')[:-4]
        except KeyError:
            getLogger(__name__).warning('MJD not present in metadata - make sure obslog is properly populated!')
        return mkidcore.metadata.build_header(meta, unknown_keys='warn')

    def write(self, filename, overwrite=True, compress=False, cube_type=None, time_bin_edges=None, wvl_bin_edges=None):
        """
        Writes the drizzled output to a FITS file
        :param filename: fully qualified file path
        :param overwrite: if True will overwrite existing file of the same name
        :param compress: If True will output a compressed GZ file instead of a FITS file
        :param cube_type: Type of output cube, options are 'both', 'time', or 'wave'
        :param time_bin_edges: temporal bin edges (in seconds) if cube_type is 'both' or 'time'
        :param wvl_bin_edges: wavelength bin edges (in nm) if cube_type is 'both' or 'wave'
        """
        primary_header = self.header
        primary_header.update(self.wcs.to_header())
        primary_header['EXPTIME'] = self.average_nonzero_exp_time
        science_header = primary_header.copy()
        science_header['WCSTIME'] = (self.drizzle_params.wcs_timestep, '')
        science_header['PIXFRAC'] = (self.drizzle_params.pixfrac, '')
        science_header['UNIT'] = 'photon/s' if self.rate else 'photons'

        variance_header = science_header.copy()
        variance_header['UNIT'] = 'photon'

        bin_hdu = []
        if cube_type in ('both', 'time'):
            hdu = fits.TableHDU.from_columns(np.recarray(shape=time_bin_edges.shape, buf=time_bin_edges,
                                                              dtype=np.dtype([('edges', time_bin_edges.dtype)])),
                                             name='CUBE_EDGES')
            hdu.header.append(fits.Card('UNIT', 's', comment='Bin unit'))
            bin_hdu.append(hdu)

        if cube_type in ('both', 'wave'):
            hdu = fits.TableHDU.from_columns(np.recarray(shape=wvl_bin_edges.shape, buf=wvl_bin_edges,
                                                         dtype=np.dtype([('edges', wvl_bin_edges.dtype)])),
                                             name='CUBE_EDGES')
            hdu.header.append(fits.Card('UNIT', 'nm', comment='Bin unit'))
            bin_hdu.append(hdu)
        if self.rate:
            hdul = fits.HDUList([fits.PrimaryHDU(header=primary_header),
                                 fits.ImageHDU(name='cps', data=self.cps, header=science_header),
                                 fits.ImageHDU(name='variance', data=self.counts, header=variance_header)] + bin_hdu)
        else:
            hdul = fits.HDUList([fits.PrimaryHDU(header=primary_header),
                                 fits.ImageHDU(name='counts', data=self.counts, header=science_header),
                                 fits.ImageHDU(name='variance', data=self.counts, header=variance_header)] + bin_hdu)

        if compress:
            filename = filename + '.gz'

        if not (filename.lower().endswith('.fits') or filename.lower().endswith('.fits.gz')):
            filename += '.fits'

        hdul.writeto(filename, overwrite=overwrite)
        getLogger(__name__).info('FITS file {} saved'.format(filename))


class Drizzler(Canvas):
    """
    Generate a 2D-4D hypercube from a set dithered dataset. The cube size is ntimes * ndithers * nwvlbins * nPixRA * nPixDec.
    """
    def __init__(self, dithers_data, drizzle_params, wvl_bin_width=0.0 * u.nm, time_bin_width=0.0, wvl_min=700.0 * u.nm,
                 wvl_max=1500 * u.nm, adi_mode=False, rate=True):
        """
        :param dithers_data: list of dictionaries of relevant input data and parameters (see output of load_data)
        :param drizzle_params: DrizzleParams object
        :param wvl_bin_width: wavelength bin width (in nm)
        :param time_bin_width: time bin width (in seconds)
        :param wvl_min: minimum wavelength to use
        :param wvl_max: maximum wavelength to use
        :param adi_mode: If True will not subtract off the calculated parallactic angle to preserve field rotation
        :param rate: If True output will be in photons/s else in photons
        """
        super().__init__(dithers_data, drizzle_params=drizzle_params, canvas_shape=drizzle_params.canvas_shape,
                         rate=rate)
        self.drizzle_params = drizzle_params
        self.pixfrac = drizzle_params.pixfrac
        self.time_bin_width = time_bin_width
        wvl_span = wvl_max.to(u.nm).value - wvl_min.to(u.nm).value
        self.timebins = None
        self.wvl_bin_edges = None
        # get wavelength bins to use
        if wvl_bin_width.to(u.nm).value > wvl_span:
            getLogger(__name__).info('Wavestep larger than entire wavelength range - using whole wavelength range '
                                     'instead')
            self.wvl_bin_edges = np.array([wvl_min.to(u.nm).value, wvl_max.to(u.nm).value])
        elif wvl_bin_width.value != 0 and wvl_span % wvl_bin_width.to(u.nm).value != 0:
            mod = wvl_span % wvl_bin_width.to(u.nm).value
            use_max = wvl_max.to(u.nm).value - mod
            n_steps = (use_max - wvl_min.to(u.nm).value) / wvl_bin_width.to(u.nm).value
            getLogger(__name__).warning(f'Specified wavelength range not evenly divisible by wavestep, using {n_steps} '
                                        f'wavelength steps of size {wvl_bin_width}')
            self.wvl_bin_edges = np.arange(wvl_min.to(u.nm).value, use_max, wvl_bin_width.to(u.nm).value)
        else:
            self.wvl_bin_edges = np.arange(wvl_min.to(u.nm).value, wvl_max.to(u.nm).value, wvl_bin_width.to(u.nm).value) if \
                wvl_bin_width.to(u.nm).value != 0.0 else np.array([wvl_min.to(u.nm).value, wvl_max.to(u.nm).value])

        # get time bins to use
        startt = drizzle_params.startt
        if startt + time_bin_width >= (drizzle_params.inttime * len(self.dithers_data)):
            getLogger(__name__).info('Timestep larger than entire duration - using whole duration instead')
            self.timebins = np.array([startt, drizzle_params.inttime])
        elif time_bin_width != 0 and (startt + drizzle_params.inttime) % time_bin_width != 0:
            mod = (startt + drizzle_params.inttime) % time_bin_width
            inttime = (startt + drizzle_params.inttime) - mod
            n_steps = inttime / time_bin_width
            getLogger(__name__).warning(f'Specified duration not evenly divisible by timestep, using {n_steps} '
                                        f'time steps of length {time_bin_width}s for each dither position ')
            self.timebins = np.append(np.arange(startt, inttime, time_bin_width), inttime)
        else:
            self.timebins = np.append(np.arange(startt, startt + drizzle_params.inttime,
                                                time_bin_width if time_bin_width != 0 else drizzle_params.inttime),
                                      startt + drizzle_params.inttime)

        self.wcs_times = np.append(np.arange(startt, self.timebins[-1], drizzle_params.wcs_timestep),
                                   self.timebins[-1]) if not adi_mode else self.timebins
        self.cps = None
        self.counts = None
        self.expmap = None

    def run(self, apply_weight=True):
        """
        Runs the drizzling code
        :param apply_weight: If True will weight each pixel by its weight from the photon table.
        """
        tic = time.clock()

        nexp_time = len(self.timebins) - 1
        nwvls = len(self.wvl_bin_edges) - 1
        ndithers = len(self.dithers_data)

        # use exp_timestep for final spacing
        # TODO this looks like it might be backwards from docs in ra/dec of canvas shape
        self.cps = np.zeros((nexp_time * ndithers, nwvls) + self.canvas_shape[::-1])
        expmap = np.zeros((nexp_time * ndithers, nwvls) + self.canvas_shape[::-1])
        for pos, dither_photons in enumerate(self.dithers_data):  # iterate over dithers
            dithhyper = np.zeros((nexp_time, nwvls) + self.canvas_shape[::-1], dtype=np.float32)
            dithexp = np.zeros((nexp_time, nwvls) + self.canvas_shape[::-1], dtype=np.float32)

            for wcs_i, wcs_sol in enumerate(dither_photons['obs_wcs_seq']):  # iterate through each of the wcs time spacing
                if wcs_i >= len(self.wcs_times) - 1 and len(self.wcs_times) != 1:
                    break
                wcs_sol.pixel_shape = self.shape
                # the sky grid ref and dither ref should match (crpix varies between dithers)
                if not np.allclose(wcs_sol.wcs.crval, self.wcs.wcs.crval, rtol=1e-4):
                    getLogger(__name__).critical('sky grid ref and dither ref do not match '
                                                 '(crval varies between dithers)!')
                    raise RuntimeError('sky grid ref and dither ref do not match (crval varies between dithers)!')

                if len(self.timebins) <= len(self.wcs_times):
                    time_bins = np.array([self.wcs_times[wcs_i], self.wcs_times[wcs_i + 1]])
                else:
                    if len(self.wcs_times) == 1:
                        time_bins = self.timebins
                    else:
                        idx = np.where(
                            (self.timebins >= self.wcs_times[wcs_i]) & (self.timebins <= self.wcs_times[wcs_i + 1]))
                        time_bins = self.timebins[idx]
                counts = self.make_cube(dither_photons, time_bins, self.wvl_bin_edges, apply_weight=apply_weight)
                expin = time_bins[1] - time_bins[0]
                cps = counts / expin  # scale this frame by its exposure time
                # get exposure bin of current wcs time
                wcs_time = self.wcs_times[wcs_i]
                iwcs = np.where([(wcs_time >= self.timebins[i]) & (wcs_time < self.timebins[i + 1]) for i in
                                 range(len(self.timebins) - 1)])[0][0]
                for it in range(len(time_bins) - 1):  # iterate over time step - > 1 if timestep < wcs_timestep
                    for n_wvl in range(nwvls):  # iterate over wavelengths
                        # create a new drizzle object for each time (and wavelength) frame
                        #TODO Add README disclaimer or go to multi extension:
                        # in adi mode the companion will appear to move on sky because a common wcs is being used
                        # in reality the detector mapping is changing
                        driz = stdrizzle.Drizzle(outwcs=self.wcs, pixfrac=self.pixfrac)
                        inwht = cps[it, n_wvl].astype(bool).astype(int)
                        driz.add_image(cps[it, n_wvl], wcs_sol, expin=expin, inwht=inwht, in_units='cps')
                        # for a single wcs timestep
                        dithhyper[iwcs + it, n_wvl, :, :] += driz.outsci * driz.outexptime  # sum all counts in same exposure bin
                        used_exptimes = np.full(np.shape(driz.outsci), driz.outexptime)
                        whtmask = driz.outsci == 0
                        used_exptimes[whtmask] = 0
                        dithexp[iwcs + it, n_wvl, :, :] += used_exptimes

            # for the whole dither pos
            if len(self.wcs_times) > len(self.timebins):
                wcs_per_timebin = (len(self.wcs_times) - 1) / nexp_time
                #TODO check why/if this is necessary?
                dithhyper = dithhyper / wcs_per_timebin

            self.cps[pos * nexp_time: (pos + 1) * nexp_time] = dithhyper / dithexp
            expmap[pos * nexp_time: (pos + 1) * nexp_time] = dithexp

        self.cps[np.isnan(self.cps)] = 0
        expmap[np.isnan(expmap)] = 0

        getLogger(__name__).debug(f'Image load done in {time.clock() - tic:.1f} s')

        if nexp_time == 1 and self.time_bin_width == 0:
            counts = np.sum(self.cps * expmap, axis=0)
            expmap = np.sum(expmap, axis=0)
            self.cps = counts / expmap
            self.cps[np.isnan(self.cps)] = 0

        if nwvls == 1:
            self.cps = np.squeeze(self.cps)
            expmap = expmap[0, :, :] if nexp_time == 1 else expmap[:, 0, :, :]

        self.wcs = self.generate_wcs(wave=nwvls != 1, time=nexp_time != 1)
        self.counts = self.cps * expmap
        self.average_nonzero_exp_time = expmap[expmap > 0].mean()

    def make_cube(self, dither_photons, time_bins, wvl_bins, apply_weight=False):
        """
        Creates a 4D image cube for the duration of the wcs timestep range or finer sampled if timestep is
        shorter
        :param dither_photons: dictionary of relevant input data for a single dither position
        :param time_bins: array of time bin edges (in seconds)
        :param wvl_bins: array of wavelength bin edges (in nm)
        :param apply_weight: If True will weight each pixel by its weight from the photon table.
        :return: 4D data cube
        """
        time_bins = time_bins * 1e6
        weights = dither_photons['weight'] if apply_weight else None
        timespan_mask = ((dither_photons['timestamps'] >= time_bins[0]) &
                         (dither_photons['timestamps'] <= time_bins[-1]))
        if weights is not None:
            weights = weights[timespan_mask]
        sample = np.vstack((dither_photons['timestamps'][timespan_mask],
                            dither_photons['wavelengths'][timespan_mask],
                            dither_photons['photon_pixels'][0][timespan_mask],
                            dither_photons['photon_pixels'][1][timespan_mask]))

        bins = np.array([time_bins, wvl_bins, range(self.shape[1] + 1), range(self.shape[0] + 1)])
        hypercube, _ = np.histogramdd(sample.T, bins, weights=weights)
        return hypercube

    def generate_wcs(self, wave=True, time=True):
        """
        Return a WCS object appropriate for the resulting cube to the extra elements to the header
        Its not clear how to increase the number of dimensions of a 2D wcs.WCS() after its created so just create
        a new object, read the original parameters where needed
        :param wave: If True creates a spectral dimension
        :param time: If True creates a temporal dimension
        :return: Astropy.wcs.WCS object
        """
        naxis = 2 + int(wave) + int(time)
        w = wcs.WCS(naxis=naxis)
        w.wcs.pc = np.eye(naxis)
        w.wcs.crpix[0], w.wcs.crpix[1] = self.wcs.wcs.crpix[0], self.wcs.wcs.crpix[1]
        w.wcs.crval[0], w.wcs.crval[1] = self.wcs.wcs.crval[0], self.wcs.wcs.crval[1]
        w.wcs.ctype[0], w.wcs.ctype[1] = self.wcs.wcs.ctype[0], self.wcs.wcs.ctype[1]
        w.wcs.cdelt[0], w.wcs.cdelt[1] = self.wcs.wcs.cdelt[0], self.wcs.wcs.cdelt[1]
        w.wcs.cunit[0], w.wcs.cunit[1] = self.wcs.wcs.cunit[0], self.wcs.wcs.cunit[1]
        pixel_shape = [self.wcs.pixel_shape[0], self.wcs.pixel_shape[1]]
        if wave:
            typ = wtype = "WAVE"
            val = wval = self.wvl_bin_edges[0] / 1e9
            shape = wshape = len(self.wvl_bin_edges) - 1
            delt = wdelt = (self.wvl_bin_edges[1] - self.wvl_bin_edges[0]) / 1e9
            unit = wunit = "m"

        if time:
            typ = ttype = "TIME"
            val = tval = self.timebins[0]
            shape = tshape = len(self.timebins) - 1
            delt = tdelt = (self.timebins[1] - self.timebins[0])
            unit = tunit = "s"

        if naxis == 4:
            pixel_shape.extend([wshape, tshape])
            w.wcs.crpix[2], w.wcs.crpix[3] = 1, 1
            w.wcs.crval[2], w.wcs.crval[3] = wval, tval
            w.wcs.ctype[2], w.wcs.ctype[3] = wtype, ttype
            w.wcs.cdelt[2], w.wcs.cdelt[3] = wdelt, tdelt
            w.wcs.cunit[2], w.wcs.cunit[3] = wunit, tunit
        elif naxis == 3:
            pixel_shape.extend([shape])
            w.wcs.crpix[2] = 1
            w.wcs.crval[2] = val
            w.wcs.ctype[2] = typ
            w.wcs.cdelt[2] = delt
            w.wcs.cunit[2] = unit

        w.pixel_shape = tuple(pixel_shape)
        getLogger(__name__).debug(f'{naxis}D wcs {w}')
        return w


def debug_dither_image(dithers_data, drizzle_params, save=None, weight=True):
    """
    Plot the location of frames with simple boxes for calibration/debugging purposes.
    :param dithers_data: list of dictionaries of relevant input data and parameters (see output of load_data)
    :param drizzle_params: DrizzleParams object
    :param weight: If True will weight each pixel by its weight from the photon table.
    """
    drizzle_params.canvas_shape = 500, 500  # hand set to large number to ensure all frames are captured
    driz = Drizzler(dithers_data, drizzle_params)
    driz.run(apply_weight=weight)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(driz.cps, cmap='viridis', origin='lower', norm=LogNorm())

    output_image = np.zeros_like(driz.cps)
    canvas_wcs = driz.wcs
    shape = driz.shape
    del driz
    drizzle_params.canvas_shape = None, None
    angles = []
    for pos, dither_photons in enumerate(dithers_data):
        # make a new driz object so the color of each frame is uniform
        driz = stdrizzle.Drizzle(outwcs=canvas_wcs, wt_scl='')
        for t, inwcs in enumerate(dither_photons['obs_wcs_seq']):
            # inwcs = wcs.WCS(header=inwcs)
            pc_matrix = inwcs.wcs.pc
            inwcs.pixel_shape = shape
            image = np.zeros(shape[::-1])  # create a simple image consisting of the array boarder and the diagonals
            image[[0, -1]] = 1
            image[:, [0, -1]] = 1
            eye = np.eye(*shape[::-1]).astype(bool)
            image[eye] = 1
            image[eye[::-1]] = 1

            driz.add_image(image, inwcs)
            driz.outsci = driz.outsci.astype(bool)
            output_image[driz.outsci] = pos
            angles.append((np.arccos(pc_matrix[0][0]) * u.rad).to(u.deg).value)

    output_image[output_image == 0] = np.nan

    axes[0].grid(True, color='k', which='both', axis='both')
    im = axes[1].imshow(output_image, cmap='Reds', origin='lower')
    divider = make_axes_locatable(axes[1])
    axes[1].grid(True, color='k', which='both', axis='both')
    cax = divider.append_axes("right", size="5%", pad=0.05)
    clb = fig.colorbar(im, cax=cax)
    clb.ax.set_title('Dither index')

    total_rotation = abs(angles[-1] - angles[0])
    rotation_offset = angles[0]
    col_labels = ['total relative rotation (deg)', 'first frame rotation offset (deg)']
    table_data = [[np.round(total_rotation, decimals=2), np.round(rotation_offset, decimals=2)]]
    table = axes[1].table(cellText=table_data, colLabels=col_labels, cellLoc='center', loc='upper right',
                          colWidths=[0.3] * 2)
    table.set_zorder(5)
    table.auto_set_font_size(False)
    table.set_fontsize(6)
    if save:
        plt.savefig(save)
        plt.show(block=True)
    else:
        plt.show(block=True)

def mp_worker(file, startw, stopw, startt, intt, adi_mode, wcs_timestep, md, exclude_flags=()):
    """
    Uses photontable.query to retrieve all photons in a wavelength range defined by startw and stopw in a timerange
    defined by start time startt and duration intt. Culls photons affected by exclude_flags.

    Determines WCS solutions for the interval at wcs_timestep cadence calling with adi_mode
    :param file: h5 file name
    :param startw: start wavelenght (nm)
    :param stopw: stop wavelength (nm)
    :param startt: start time in relative seconds
    :param intt: duration in seconds
    :param adi_mode: if True will not subtract off the calculated parallactic angle to preserve field rotation
    :param wcs_timestep: cadence at which to calculate discrete WCS solutions
    :param md: observational metadata
    :param exclude_flags: list of pixel flags to exclude from analysis
    :return: dictionary of relevant data and parameters
    """
    getLogger(__name__).debug(f'Fetching data from {file}')
    pt = Photontable(file)
    if startt + intt > pt.duration:
        getLogger(__name__).warning(f'Specified start ({startt}s) and duration ({intt}s) exceed the full length of the '
                                    f'photontable ({pt.duration}s).')
    photons = pt.query(startw=startw, stopw=stopw, start=startt, intt=intt)
    num_unfiltered = len(photons)
    if not len(photons):
        getLogger(__name__).warning(f'No photons found using wavelength range {startw}-{stopw} nm and time range '
                                    f'{startt}-{intt} s. Is the photontable not wavelength calibrated causing a mismatch '
                                    f'in the units?')
    else:
        getLogger(__name__).info("Fetched {} photons from dither {}".format(len(photons), file))

    exclude_flags += EXCLUDE
    photons = pt.filter_photons_by_flags(photons, disallowed=exclude_flags)
    getLogger(__name__).info(f"Removed {num_unfiltered - len(photons)} photons "
                             f"from {num_unfiltered} total from bad pix")
    xy = pt.xy(photons)
    wcs_times = pt.start_time + np.arange(startt, startt + intt, wcs_timestep)  # This is in unixtime
    wcs = pt.get_wcs(derotate=not adi_mode, sample_times=wcs_times)
    del pt
    return {'file': file, 'timestamps': photons["time"], 'wavelengths': photons["wavelength"],
            'weight': photons['weight'], 'photon_pixels': xy, 'obs_wcs_seq': wcs, 'duration': intt, 'metadata': md}


def load_data(dither, wvl_min, wvl_max, startt, duration, wcs_timestep, adi_mode=False, ncpu=1,
              exclude_flags=()):
    """
    Load the photons either by querying the photontables in parrallel or loading from pkl if it exists. The wcs
    solutions are added to this photon data dictionary but will likely be integrated into photontable.py directly
    :param dither: MKIDDither, contains the lists of observations and metadata for a set of dithers
    :param wvl_min: minimum wavelength (in nm)
    :param wvl_max: maximum wavelength (in nm)
    :param startt: start time relative ot the beginning of the file (in seconds)
    :param duration: duration of file to use (in seconds)
    :param wcs_timestep: cadence at which to calculate discrete WCS solutions
    :param adi_mode: If True will not subtract off the calculated parallactic angle to preserve field rotation. Useful
    for ADI analysis
    :param ncpu: number of CPUs to use for multiprocessing
    :param exclude_flags: list of pixelflags to be excluded from analysis
    :return: list of dictionaries of relevant data and parameters
    """
    begin = time.time()
    filenames = [o.h5 for o in dither.obs]
    meta = [Photontable(o.h5).metadata() for o in dither.obs]
    if not filenames:
        getLogger(__name__).info('No photontables found')

    offsets = [o.start - int(o.start) for o in dither.obs]  # How many seconds into the h5 does valid data start
    if ncpu < 2:
        dithers_data = []
        for file, offset, md in zip(filenames, offsets, meta):
            data = mp_worker(file, wvl_min, wvl_max, startt + offset, duration, adi_mode, wcs_timestep, md,
                             exclude_flags)
            dithers_data.append(data)
    else:
        # TODO result of mp_worker too big, causes issues with multiprocessing when pickling
        p = mp.Pool(ncpu)
        processes = [p.apply_async(mp_worker, (file, wvl_min, wvl_max, startt + offsett, duration, adi_mode,
                                               wcs_timestep, md, exclude_flags))
                     for file, offsett, md in zip(filenames, offsets, meta)]
        dithers_data = [res.get() for res in processes]

    dithers_data.sort(key=lambda k: filenames.index(k['file']))

    getLogger(__name__).debug(f'Loading data took {time.time() - begin:.0f} s')

    return dithers_data


def form(dither, mode='drizzler', wave_start=None, wave_stop=None, start=0, duration=None, pixfrac=.5,
         wvl_bin_width=0.0 * u.nm, time_bin_width=0.0, wcs_timestep=1., usecache=True, ncpu=None,
         exclude_flags=PROBLEM_FLAGS + EXCLUDE, whitelight=False, adi_mode=False, debug_dither_plot=False,
         rate=True, output_file='', weight=False, **kwargs):
    """
    Takes in a MKIDDither object and drizzles each frame onto a common sky grid.
    :param dither: MKIDDither, contains the lists of observations and metadata for a set of dithers
    :param mode: 'drizzler' only currently accepted mode
    :param wave_start: start wavelength. See photontable.query()
    :param wave_stop: stop wavelength. See photontable.query()
    :param start: start offset (in seconds) for photons used in each dither.
    :param duration: upper bound on the photons used in each dither. See photontable.query()
    :param pixfrac: pixfrac parameter used in drizzle algorithm. See stsci.drizzle()
    :param wvl_bin_width: astropy.units.Quantity - size of wavelength bins. If 0.0 will use full wavelength extent of
    the instrument
    :param time_bin_width: size of time bins (in seconds). If 0.0 will use full duration
    :param wcs_timestep: Time between different wcs parameters (e.g. orientations). 0 will use the calculated
    non-blurring min
    :param usecache: True means the output of load_data() is stored and reloaded for subsequent runs of form
    :param ncpu: Number of cpu used when loading and reformatting the dither photontables
    :param exclude_flags: 'ist of pixelflags to be excluded from analysis
    :param whitelight: Relevant parameters are updated to perform a whitelight dither.
    :param adi_mode: If True will not subtract off the calculated parallactic angle to preserve field rotation. Useful
    for ADI analysis
    :param debug_dither_plot: Plot the location of frames with simple boxes for calibration/debugging purposes
    :param output_file: Name of the output save file
    :param weight: If True will apply weight column of the photontable to the dither frames
    :returns: drizzle : DrizzledData. Contains maps and metadata from the drizzled data
    """
    dcfg = mkidpipeline.config.PipelineConfigFactory(step_defaults=dict(drizzler=StepConfig()), ncpu=ncpu, cfg=None,
                                                     copy=True)
    ncpu = mkidpipeline.config.n_cpus_available(max=dcfg.get('ncpu', inherit=True))

    if mode not in ('drizzle'):
        raise ValueError('mode must be: drizzle')

    dither_inttime = min(dither.inttime)

    # if no duration specified, use whole dither integration time
    if duration is None:
        duration = dither_inttime
    # ensure the user input is shorter than the dither or that wcs are just calculated for the requested timespan
    if duration > dither_inttime:
        getLogger(__name__).info(f'User integration time of {duration:.1f} too long, '
                                 f'using shortest dither dwell time ({dither_inttime:.1f} s)')
        used_inttime = dither_inttime
    else:
        getLogger(__name__).info(f'Using user specified integration time of {duration:.1f} s')
        used_inttime = duration

    if dither_inttime < time_bin_width and adi_mode:
        getLogger(__name__).error(f" Temporal bin widths below the dither frame duration are not presently supported\n"
                                  f"\t for ADI drizzles. To correct this the PA at each time bin center needs to\n"
                                  f"\t be computed and then that PA added back to the WCS for each bin.\n"
                                  f"\t WCS solutions for the bins need to be computed with their PA subtracted\n"
                                  f"\t (i.e. with PA subtraction enabled)")
        getLogger(__name__).error(f'Unable to form {dither}')
        return

    getLogger(__name__).debug('Parsing Params')
    drizzle_params = DrizzleParams(dither, used_inttime, wcs_timestep, pixfrac, startt=start, whitelight=whitelight)

    getLogger(__name__).debug('Loading data')
    dithers_data = None
    if usecache:
        settings = (tuple(o.h5 for o in dither.obs), dither.name, wave_start.value, wave_stop.value, start,
                    drizzle_params.inttime, drizzle_params.wcs_timestep, exclude_flags, adi_mode)
        setting_hash = hashlib.md5(str(settings).encode()).hexdigest()
        pkl_save = os.path.join(mkidpipeline.config.config.paths.tmp,
                                f'drizzler_{getpass.getuser()}_{dither.name}_{setting_hash}.pkl')

        if dcfg.drizzler.clearcache:
            getLogger(__name__).info('Clearing drizzler cache')
            for f in glob(os.path.join(mkidpipeline.config.config.paths.tmp, f'drizzler_{getpass.getuser()}_*.pkl')):
                try:
                    os.remove(f)
                except IOError:
                    getLogger(__name__).error(f'Unable to remove {f} from cache')
        else:
            try:
                with open(pkl_save, 'rb') as f:
                    dithers_data = pickle.load(f)
                    getLogger(__name__).info(f'Using cached data {pkl_save}')
            except IOError:
                pass

    if dithers_data is None:
        dithers_data = load_data(dither, wave_start, wave_stop, start, drizzle_params.inttime,
                                 drizzle_params.wcs_timestep, ncpu=ncpu, exclude_flags=exclude_flags,
                                 adi_mode=adi_mode)
        if usecache:
            try:
                with open(pkl_save, 'wb') as handle:
                    getLogger(__name__).info(f'Saved data cache to {pkl_save}')
                    pickle.dump(dithers_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            except IOError:
                getLogger(__name__).warning(f'Unable to write cache {pkl_save}', exc_info=True)

    total_photons = sum([len(dither_data['timestamps']) for dither_data in dithers_data])

    if not total_photons:
        getLogger(__name__).critical('No photons found in any of the dithers. Check your wavelength and time ranges')
        return None

    if dcfg.drizzler.plots == 'summary':
        getLogger(__name__).debug('Generating debug image')
        debug_dither_image(dithers_data, drizzle_params, save=dcfg.paths.database + '/drizzler_debug.pdf')

    getLogger(__name__).debug('Initializing drizzler core')
    getLogger(__name__).debug('Running Drizzler')
    driz = Drizzler(dithers_data, drizzle_params, wvl_bin_width=wvl_bin_width, time_bin_width=time_bin_width,
                    wvl_min=wave_start, wvl_max=wave_stop, adi_mode=adi_mode, rate=rate)

    if time_bin_width != 0.0 and wvl_bin_width != 0.0 * u.nm:
        cube_type = 'both'
    elif time_bin_width != 0.0:
        cube_type = 'time'
    elif wvl_bin_width != 0.0 * u.nm:
        cube_type = 'wave'
    else:
        cube_type = None
    time_bin_edges = driz.timebins
    wvl_bin_edges = driz.wvl_bin_edges
    getLogger(__name__).debug('Drizzling...')
    driz.run(apply_weight=weight)
    if output_file:
        getLogger(__name__).debug(f'Writing fits cube of type {cube_type}.')
        driz.write(output_file, cube_type=cube_type, time_bin_edges=time_bin_edges, wvl_bin_edges=wvl_bin_edges)
    getLogger(__name__).info('Finished')
    return driz
