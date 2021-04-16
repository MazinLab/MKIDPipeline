#!/bin/python
"""
Author: Matt Strader        Date: August 19, 2012
Modified 2017 for Darkness/MEC
Authors: Seth Meeker, Neelay Fruitwala, Alex Walter

The class Photontable is an interface to observation files.  It provides methods
for typical ways of accessing photon list observation data.  It can also load 
and apply wavelength and flat calibration.  
"""
from __future__ import print_function
import os
import warnings
import time
from datetime import datetime
import multiprocessing as mp
import functools
from interval import interval
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from matplotlib.backends.backend_pdf import PdfPages
from regions import CirclePixelRegion, PixCoord
from progressbar import *
from mkidcore.headers import PhotonCType, PhotonNumpyType, METADATA_BLOCK_BYTES
from mkidcore.corelog import getLogger
from mkidcore.pixelflags import FlagSet
import mkidcore.pixelflags as pixelflags
from mkidcore.config import yaml, StringIO

from mkidcore.instruments import compute_wcs_ref_pixel

import SharedArray
from mkidpipeline.steps import lincal
import tables
import tables.parameters
import tables.file

import astropy.constants
from astropy.coordinates import SkyCoord
from astropy import wcs
from astropy.io import fits
from astroplan import Observer
import astropy.units as u


# These are little better than blind guesses and don't seem to impact performaace, but still need benchmarking
# tables.parameters.CHUNK_CACHE_SIZE = 2 * 1024 * 1024 * 1024
# tables.parameters.TABLE_MAX_SIZE = 2 * 1024 * 1024 * 1024  # default 1MB
# This governs the chunk cache that will store table data, if a row is
# 20 bytes (as is our present state)
# nslots = TABLE_MAX_SIZE / (chunksize * rowsize_bytes)
# so number of rows in the cache is ~ TABLE_MAX_SIZE/rowsize_bytes
# one 1s of 20k pix data @ 2500c/s is 0.95GB

# These are all used by the c code backing pytables and it isn't clear their importance yet
# tables.parameters.CHUNK_CACHE_NELMTS *= 10
# tables.parameters.SORTEDLR_MAX_SIZE = 1 *1024*1024*1024 # default 8MB
# tables.parameters.SORTED_MAX_SIZE = 1 *1024*1024*1024 # default 1MB
# tables.parameters.LIMBOUNDS_MAX_SIZE = 1*1024*1024*1024
# tables.parameters.SORTEDLR_MAX_SLOTS *= 10

# tables.parameters.CHUNK_CACHE_SIZE *=10
# tables.parameters.TABLE_MAX_SIZE *=10
# tables.parameters.CHUNK_CACHE_NELMTS *= 10
# tables.parameters.SORTEDLR_MAX_SIZE *=10
# tables.parameters.SORTED_MAX_SIZE *=10
# tables.parameters.LIMBOUNDS_MAX_SIZE *=10
# tables.parameters.SORTEDLR_MAX_SLOTS *= 10


class ThreadsafeFileRegistry(tables.file._FileRegistry):
    lock = mp.RLock()

    @property
    def handlers(self):
        return self._handlers.copy()

    def add(self, handler):
        with self.lock:
            return super().add(handler)

    def remove(self, handler):
        with self.lock:
            return super().remove(handler)

    def close_all(self):
        with self.lock:
            return super().close_all()


class ThreadsafeFile(tables.file.File):
    def __init__(self, *args, **kargs):
        with ThreadsafeFileRegistry.lock:
            super().__init__(*args, **kargs)

    def close(self):
        with ThreadsafeFileRegistry.lock:
            super().close()


@functools.wraps(tables.open_file)
def synchronized_open_file(*args, **kwargs):
    with ThreadsafeFileRegistry.lock:
        return tables.file._original_open_file(*args, **kwargs)


# monkey patch the tables package
tables.file._original_open_file = tables.file.open_file
tables.file.open_file = synchronized_open_file
tables.open_file = synchronized_open_file
tables.file._original_File = tables.File
tables.file.File = ThreadsafeFile
tables.File = ThreadsafeFile
tables.file._open_files = ThreadsafeFileRegistry()


class SharedTable(object):
    """multiprocessingsafe shared photon table, readonly!!!!"""

    def __init__(self, shape):
        self._shape = shape
        self._X = mp.RawArray(PhotonCType, int(np.prod(shape)))
        self.data = np.frombuffer(self._X, dtype=PhotonNumpyType).reshape(self._shape)

    def __getstate__(self):
        d = dict(__dict__)
        d.pop('data', None)
        return d

    def __setstate__(self, state):
        self.__dict__ = state
        self.data = np.frombuffer(self._X, dtype=PhotonNumpyType).reshape(self._shape)


def load_shareable_photonlist(file):
    # TODO Add arguments to pass through a query
    tic = time.time()
    f = Photontable(file)
    table = SharedTable(f.photonTable.shape)
    f.photonTable.read(out=table.data)
    ram = table.data.size * table.data.itemsize / 1024 / 1024 / 1024.0
    msg = 'Created a shared table with {size} rows from {file} in {time:.2f} s, using {ram:.2f} GB'
    getLogger(__name__).info(msg.format(file=file, size=f.photonTable.shape[0], time=time.time() - tic, ram=ram))
    return table


class SharedPhotonList(object):
    """multiprocessingsafe shared photon table, readonly!!!!"""

    def __init__(self, file):

        file = os.path.abspath(file)
        self._primary = True
        self._name = file.replace(os.path.sep, '_')
        try:
            SharedArray.delete("shm://{}".format(self._name))
            getLogger(__name__).debug("Deleted existing shared memory store {}".format(self._name))
        except FileNotFoundError:
            pass
        f = Photontable(file)
        tic = time.time()
        self.data = np.frombuffer(SharedArray.create("shm://{}".format(self._name), f.photonTable.shape,
                                                     dtype=PhotonNumpyType), dtype=PhotonNumpyType)
        f.photonTable.read(out=self.data)
        toc = time.time()
        ram = self.data.size * self.data.itemsize / 1024 / 1024 / 1024.0
        msg = 'Created a shared table with {size} rows from {file} in {time:.2f} s, using {ram:.2f} GB'
        getLogger(__name__).debug(msg.format(file=file, size=f.photonTable.shape[0], time=toc - tic, ram=ram))

    def __getstate__(self):
        d = dict(__dict__)
        d.pop('data', None)
        d.pop('raw')
        return d

    def __setstate__(self, state):
        self.__dict__ = state
        self._primary = False
        getLogger(__name__).debug('Attaching to ' + self._name)
        self.data = np.frombuffer(SharedArray.attach("shm://{}".format(self._name), dtype=PhotonNumpyType))

    def __del__(self):
        if self._primary:
            getLogger(__name__).debug('Deleting ' + self._name)
            SharedArray.delete("shm://{}".format(self._name))


class Photontable(object):
    h = astropy.constants.h.to('eV s').value
    c = astropy.constants.c.to('m/s').value
    ticks_per_sec = int(1.0 / 1e-6)  # each integer value is 1 microsecond

    def __init__(self, file_name, mode='read', verbose=False):
        """
        Create Photontable object and load in specified HDF5 file.

        Parameters
        ----------
            file_name: String
                Path to HDF5 File
            mode: String
                'read' or 'write'. File should be opened in 'read' mode 
                unless you are applying a calibration.
            verbose: bool
                Prints debug messages if True
        Returns
        -------
            Photontable instance
                
        """
        self.mode = 'write' if mode.lower() in ('write', 'w', 'a', 'append') else 'read'
        self.verbose = verbose
        self.photonTable = None
        self.filename = None
        self.header = None
        self.nominal_wavelength_bins = None
        self.beamImage = None
        self._flagArray = None
        self.nXPix = None
        self.nYPix = None
        self._mdcache = None
        self._load_file(file_name)

    def __del__(self):
        """
        Closes the obs file and any cal files that are open
        """
        try:
            self.file.close()
        except:
            pass

    def __str__(self):
        return 'Photontable: ' + self.filename

    def _load_file(self, file):
        """ Opens file and loads obs file attributes and beammap """
        self.filename = file
        getLogger(__name__).debug("Loading {} in {} mode.".format(self.filename, self.mode))
        try:
            self.file = tables.open_file(self.filename, mode='a' if self.mode == 'write' else 'r')
        except (IOError, OSError):
            raise

        # get the header
        self.header = self.file.root.header.header

        # get important cal params
        self.nominal_wavelength_bins = self.wavelength_bins(width=self.query_header('energyBinWidth'),
                                                            start=self.query_header('wvlBinStart'),
                                                            stop=self.query_header('wvlBinEnd'))

        # get the beam image
        self.beamImage = self.file.get_node('/BeamMap/Map').read()
        self._flagArray = self.file.get_node('/BeamMap/Flag')  # The absence of .read() here is correct
        self.nXPix, self.nYPix = self.beamImage.shape
        self.photonTable = self.file.get_node('/Photons/PhotonTable')

    def _apply_column_weight(self, resid, weights, column):
        """
        Applies a weight calibration to the column specified by colName.

        Parameters
        ----------
        resid: int
            resID of desired pixel
        weights: array of floats
            Array of cal weights. Multiplied into the specified column column.
        column: string
            Name of weight column. Should be either 'SpecWeight' or 'NoiseWeight'
        """
        if self.mode != 'write':
            raise Exception("Must open file in write mode to do this!")

        if column not in ('SpecWeight' or 'NoiseWeight'):
            raise ValueError(f"{column} is not 'SpecWeight' or 'NoiseWeight'")

        indices = self.photonTable.get_where_list('ResID==resid')

        if not (np.diff(indices) == 1).all():
            raise NotImplementedError('Table is not sorted by Res ID!')

        if len(indices) != len(weights):
            raise ValueError('weights length does not match length of photon list for resID!')

        new = self.query(resid=resid)[column] * np.asarray(weights)
        self.photonTable.modify_column(start=indices[0], stop=indices[-1] + 1, column=new, colname=column)
        self.photonTable.flush()

    def _update_extensible_header_store(self, extensible_header):
        if not isinstance(extensible_header, dict):
            raise TypeError('extensible_header must be of type dict')
        out = StringIO()
        yaml.dump(extensible_header, out)
        emdstr = out.getvalue().encode()
        if len(emdstr) > METADATA_BLOCK_BYTES:  # this should match mkidcore.headers.ObsHeader.metadata
            raise ValueError("Too much metadata! {} KB needed, {} allocated".format(len(emdstr) // 1024,
                                                                                    METADATA_BLOCK_BYTES // 1024))
        self._mdcache = extensible_header
        self.update_header('metadata', emdstr)

    def _parse_query_range_info(self, startw=None, stopw=None, start=None, stop=None, intt=None):
        """ return a dict with info about the data returned by query with a particular set of args

        None says don't place this limit.

        intt takes precedence over stop

        start times beyond the duration of the file are interpreted as absolute start times. Start times that would
        still be beyond the end of the file are treated as None

        stop times beyond the end of the file are treated as absolute times.

        startw and stopw are overridden with None if they are more than an order of magnitude beyond the edges of
        nominal_wavelength_bins. if not astropy.units.Qunatity, nm is assumed
        """
        try:
            if start > self.duration:
                start = start
                relstart = start - self.start_time
            else:
                relstart = start
                start = self.start_time + start

            qstart = int(relstart * self.ticks_per_sec)

            if start <= 0 or start > self.duration:
                raise TypeError

        except TypeError:
            start = self.start_time
            relstart = 0
            qstart = None

        if intt is not None:
            stop = relstart + intt

        try:
            if stop >= self.duration:
                stop -= self.start_time
                if stop < 0 or stop > self.duration:
                    raise TypeError
            relstop = stop
            stop += self.start_time
            qstop = int(relstop * self.ticks_per_sec)

        except TypeError:
            stop = self.duration + self.start_time
            relstop = self.duration
            qstop = None

        if startw is None:
            qminw = None
        else:
            v = u.Quantity(startw, u.nm).value
            qminw = None if v <= self.nominal_wavelength_bins[0] / 10 else v

        if stopw is None:
            qmaxw = None
        else:
            v = u.Quantity(stopw, u.nm).value
            qmaxw = None if v >= self.nominal_wavelength_bins[-1] * 10 else v

        return dict(start=start, stop=stop, relstart=relstart, relstop=relstop, duration=relstop - relstart,
                    qstart=qstart, qstop=qstop, minw=startw, maxw=stopw, qminw=qminw, qmaxw=qmaxw)

    def enablewrite(self):
        """USE CARE IN A THREADED ENVIRONMENT"""
        if self.mode == 'write':
            return
        self.file.close()
        self.mode = 'write'
        self._load_file(self.filename)

    def disablewrite(self):
        """USE CARE IN A THREADED ENVIRONMENT"""
        if self.mode == 'read':
            return
        self.file.close()
        self.mode = 'read'
        self._load_file(self.filename)

    def detailed_str(self):
        t = self.photonTable.read()
        tinfo = repr(self.photonTable).replace('\n', '\n\t\t')
        if np.all(t['Time'][:-1] <= t['Time'][1:]):
            sort = 'Time '
        elif np.all(t['ResID'][:-1] <= t['ResID'][1:]):
            sort = 'ResID '
        else:
            sort = 'Un'

        msg = ('{file}:\n'
               '\t{nphot:.3g} photons, {sort}sorted\n'
               '\tT= {start} - {stop} ({dur} s)\n'
               '\tTable repr: {tbl}\n'
               '\t{dirty}\n'
               '\twave: {wave}\n'
               '\tflat: {flat}\n')

        dirty = ', '.join([n for n in self.photonTable.colnames if self.photonTable.cols._g_col(n).index is not None
                           and self.photonTable.cols._g_col(n).index.dirty])

        s = msg.format(file=self.filename, nphot=len(self.photonTable), sort=sort, tbl=tinfo,
                       start=t['Time'].min(), stop=t['Time'].max(), dur=self.duration,
                       dirty='Column(s) {} have dirty indices.'.format(dirty) if dirty else 'No columns dirty',
                       wave=self.query_header('wvlCalFile'), flat=self.query_header('fltCalFile'))
        return s

    def xy(self, photons):
        """Return a tuple of two arrays corresponding to the x & y pixel positions of the given photons"""
        flatbeam = self.beamImage.flatten()
        beamsorted = np.argsort(flatbeam)
        ind = np.searchsorted(flatbeam[beamsorted], photons["ResID"])
        return np.unravel_index(beamsorted[ind], self.beamImage.shape)

    @property
    def duration(self):
        return self.query_header('expTime')

    @property
    def start_time(self):
        return self.query_header('startTime')

    @property
    def stop_time(self):
        return self.start_time + self.duration

    @property
    def bad_pixel_mask(self):
        """A boolean image with true where pixel data has problems """
        from mkidpipeline.pipeline import PROBLEM_FLAGS  # This must be here to prevent a circular import!
        return self.flagged(PROBLEM_FLAGS)

    @property
    def wavelength_calibrated(self):
        return self.query_header('isWvlCalibrated')

    @property
    def extensible_header_store(self):
        if self._mdcache is None:
            try:
                d = yaml.load(self.header[0]['metadata'].decode())
                self._mdcache = {} if d is None else dict(d)
            except TypeError:
                msg = ('Could not restore a dict of extensible metadata, '
                       'purging and repairing (file will not be changed until write).'
                       'Metadata must be reattached to {}.')
                getLogger(__name__).warning(msg.format(type(self._mdcache), self.filename))
                self._mdcache = {}
        return self._mdcache

    def flagged(self, flags, pixel=(slice(None), slice(None)), allow_unknown_flags=True, all_flags=False,
                resid=None):
        """
        Test to see if a flag is set on a given pixel or set of pixels

        if resid is set it takes precedence over pixel
        :param pixel: (x,y) of pixel, 2d slice, list of (x,y), if not specified all pixels are used
        :param flags: if an empty set/None it the pixel(s) are considered unflagged
        :param allow_unknown_flags:
        :param all_flags: Require all specified flags to be set for the mask to be True
        :return:
        """

        if resid is not None:
            pixel = tuple(map(tuple, np.argwhere(resid == self.beamImage)))

        x, y = zip(*pixel) if isinstance(pixel[0], tuple) else pixel
        if not flags:
            return False if isinstance(x, int) else np.zeros_like(self._flagArray[x, y], dtype=bool)

        f = self.flags
        if len(set(f.names).difference(flags)) and not allow_unknown_flags:
            return False if isinstance(x, int) else np.zeros_like(self._flagArray[x, y], dtype=bool)

        bitmask = f.bitmask(flags, unknown='ignore')
        bits = self._flagArray[x, y] & bitmask
        return bits == bitmask if all_flags else bits.astype(bool)

    @property
    def flags(self):
        """
        The flags associated with the with the file.

        Changing this once it is initialized is at your own peril!
        """
        from mkidpipeline.pipeline import PIPELINE_FLAGS  # This must be here to prevent a circular import!

        names = self.extensible_header_store.get('flags', [])
        if not names:
            getLogger(__name__).warning('Flag names were not attached at time of H5 creation. '
                                        'If beammap flags have changed since then things WILL break. '
                                        'You must recreate the H5 file.')
            names = PIPELINE_FLAGS.names
            self.enablewrite()
            self.update_header('flags', names)
            self.disablewrite()

        f = FlagSet(*[(n, i, PIPELINE_FLAGS.flags[n].description if n in PIPELINE_FLAGS.flags else '')
                      for i, n in enumerate(names)])
        return f

    def flag(self, flag, pixel=(slice(None), slice(None))):
        """
        Applies a flag to the selected pixel on the BeamFlag array. Flag is a bitmask;
        new flag is bitwise OR between current flag and provided flag. Flag definitions
        can be found in mkidcore.pixelflags, flags extant when file was created are in self.flag_names

        Named flags must be converged to bitmask via self.flag_bitmask(flag names) first

        Parameters
        ----------
        pixel: 2-tuple of int/slice denoting x,y pixel location
        flag: int
            Flag to apply to pixel
        """
        if self.mode != 'write':
            raise Exception("Must open file in write mode to do this!")

        x, y = pixel
        flag = np.asarray(flag)
        self.flags.valid(flag, error=True)
        if not np.isscalar(flag) and self._flagArray[y, x].shape != flag.shape:
            raise ValueError('flag must be scalar or match the desired region selected by x & y coordinates')
        self._flagArray[y, x] |= flag
        self._flagArray.flush()

    def unflag(self, flag, pixel=(slice(None, slice(None)))):
        """
        Resets the specified flag in the BeamFlag array to 0. Flag is a bitmask;
        only the bit specified by 'flag' is reset. Flag definitions
        can be found in Headers/pipelineFlags.py.

        Named flags must be converged to bitmask via self.flag_bitmask(flag names) first

        Parameters
        ----------
        pixel: 2-tuple of ints/slices
            xy-coordinate of pixel
        flag: int
            Flag to undo
        """
        if self.mode != 'write':
            raise Exception("Must open file in write mode to do this!")

        x, y = pixel
        flag = np.asarray(flag)
        self.flags.valid(flag, error=True)
        if not np.isscalar(flag) and self._flagArray[y, x].shape != flag.shape:
            raise ValueError('flag must be scalar or match the desired region selected by x & y coordinates')
        self._flagArray[y, x] &= ~flag
        self._flagArray.flush()

    def query(self, startw=None, stopw=None, start=None, stopt=None, resid=None, intt=None, pixel=None):
        """
        intt takes precedence, all none is the full file

        :param start:
        :param pixel:
        :param stopt:
        :param intt:
        :param startw: number or none
        :param stopw: number or none
        :param resid: number, list/array or None

        pixel may be used and will be converted to the appropriate resid via the beamamp, resid takes precedence
        use caution with slices and large numbers of pixels!

        :return:
        """
        if pixel and not resid:
            resid = tuple(self.beamImage[pixel].ravel())

        query_nfo = self._parse_query_range_info(startw=startw, stopw=stopw, start=start, stop=stopt, intt=intt)
        start = query_nfo['qstart']
        stopt = query_nfo['qstop']
        startw = query_nfo['qminw']
        stopw = query_nfo['qmaxw']

        if resid is None:
            resid = tuple()

        try:
            iter(resid)
        except TypeError:
            resid = (resid,)

        if startw is None and stopw is None and start is None and stopt is None and not resid:
            return self.photonTable.read()  # we need it all!

        res = '|'.join(['(ResID=={})'.format(r) for r in map(int, resid)])
        res = '(' + res + ')' if '|' in res and res else res
        tp = '(Time < stopt)'
        tm = '(Time >= start)'
        wm = '(Wavelength >= startw)'
        wp = '(Wavelength < stopw)'
        # should follow '{res} & ( ({time}) & ({wave}))'

        if startw is not None:
            if stopw is not None:
                wave = '({} & {})'.format(wm, wp)
            else:
                wave = wm
        elif stopw is not None:
            wave = wp
        else:
            wave = ''

        if start is not None:
            if stopt is not None:
                timestr = '({} & {})'.format(tm, tp)
            else:
                timestr = tm
        elif stopt is not None:
            timestr = tp
        else:
            timestr = ''

        query = res
        if res and (timestr or wave):
            query += '&'
        if res and timestr and wave:
            query += '('

        query += timestr
        if timestr and wave:
            query += '&'

        query += wave
        if res and timestr and wave:
            query += ')'

        if not query:
            # TODO make dtype pull from mkidcore.headers
            # TODO check whether dtype for Time should be u4 or u8. u4 loses
            #  a bunch of leading bits, but it may not matter.
            getLogger(__name__).debug('Null Query, returning empty result')
            return np.array([], dtype=[('ResID', '<u4'), ('Time', '<u4'), ('Wavelength', '<f4'),
                                       ('SpecWeight', '<f4'), ('NoiseWeight', '<f4')])
        else:
            tic = time.time()
            try:
                q = self.photonTable.read_where(query)
            except SyntaxError:
                raise
            toc = time.time()
            msg = 'Fetched {}/{} rows in {:.3f}s using indices {} for query {} \n\t st:{} et:{} sw:{} ew:{}'
            getLogger(__name__).debug(msg.format(len(q), len(self.photonTable), toc - tic,
                                                 tuple(self.photonTable.will_query_use_indexing(query)), query,
                                                 *map(lambda x: '{:.2f}'.format(x) if x is not None else 'None',
                                                      (start, stopt, startw, stopw))))

            return q

    def filter_photons_by_flags(self, photons, allowed=(), disallowed=()):
        """
        Parameters
        ----------
        photons: structured array
            photon list to be filtered
        allowed: tuple
            collection of pixel flags to keep photons for
        disallowed: tuple
            collection of pixel flags to remove photons for

        Return
        ------
        photons filtered by flags
        """

        if len(allowed) > 0:
            raise NotImplementedError

        filtered = self.flagged(disallowed, self.xy(photons))
        return photons[np.invert(filtered)]

    def get_wcs(self, derotate=True, wcs_timestep=None, target_coordinates=None, cube_type=None, single_pa_time=None,
                bins=None):
        """
        Parameters
        ----------
        derotate : bool
             True:  align each wcs solution to position angle = 0
             False:  no correction angle
        wcs_timestep : float
            Determines the time between each wcs solution (each new position angle)
        target_coordinates : SkyCoord, str
            Used to get parallatic angles. string is used to query simbad.
        cube_type : str
            None: wcs solution is calculated for ra/dec
            'wave':  wcs solution is calculated for ra/dec/wavelength
            'time':  wcs solution is calculated for ra/dec/time
        bins: array
            must be passed with the evenly spaced bins of the cube if cube_type is wave or time
        single_pa_time : float
            Time at which to orient all non-derotated frames

        See instruments.compute_wcs_ref_pixel() for information on wcscal parameters

        Returns
        -------
        List of wcs headers at each position angle
        :param bins:
        """

        # TODO add target_coordinates=None
        # 0) to header info during HDF creation, also add all fits header info from dashboard log

        md = self.metadata()
        ditherHome = md.dither_home
        ditherReference = md.dither_ref
        ditherPos = md.dither_pos
        platescale = md.platescale  # units should be mas/pix
        getLogger(__name__).debug(f'ditherHome: {md.dither_home} (conex units -3<x<3), '
                                  f'ditherReference: {md.dither_ref} (pix 0<x<150), '
                                  f'ditherPos: {md.dither_home} (conex units -3<x<3), '
                                  f'platescale: {md.platescale} (mas/pix ~10)')

        platescale = platescale.to(u.mas) if isinstance(platescale, u.Quantity) else platescale * u.mas

        if target_coordinates is not None and not isinstance(target_coordinates, SkyCoord):
            target_coordinates = SkyCoord.from_name(target_coordinates)
        else:
            target_coordinates = SkyCoord(md.ra, md.dec, unit=('hourangle', 'deg'))

        apo = Observer.at_site(md.observatory)

        if wcs_timestep is None:
            wcs_timestep = self.duration

        # sample_times upper boundary is limited to the user defined end time
        sample_times = np.arange(self.start_time, self.stop_time, wcs_timestep)
        getLogger(__name__).debug("sample_times: %s", sample_times)

        device_orientation = np.deg2rad(md.device_orientation)
        if derotate is True:
            times = astropy.time.Time(val=sample_times, format='unix')
            parallactic_angles = apo.parallactic_angle(times, target_coordinates).value  # radians
            corrected_sky_angles = -parallactic_angles - device_orientation
        else:
            if single_pa_time is not None:
                single_time = np.full_like(sample_times, fill_value=single_pa_time)
                getLogger(__name__).info(f"Derotate off. Using single PA at time: {single_time[0]}")
                single_times = astropy.time.Time(val=single_time, format='unix')
                single_parallactic_angle = apo.parallactic_angle(single_times, target_coordinates).value  # radians
                corrected_sky_angles = -single_parallactic_angle - device_orientation
            else:
                corrected_sky_angles = np.zeros_like(sample_times)

        getLogger(__name__).debug("Correction angles: %s", corrected_sky_angles)

        obs_wcs_seq = []
        for t, ca in enumerate(corrected_sky_angles):
            rotation_matrix = np.array([[np.cos(ca), -np.sin(ca)],
                                        [np.sin(ca), np.cos(ca)]])

            ref_pixel = compute_wcs_ref_pixel(ditherPos, ditherHome, ditherReference)

            if cube_type in ('wave',):
                obs_wcs = wcs.WCS(naxis=3)
                obs_wcs.wcs.crpix = [ref_pixel[0], ref_pixel[1], 1]
                obs_wcs.wcs.crval = [target_coordinates.ra.deg, target_coordinates.dec.deg, bins[0]]
                obs_wcs.wcs.ctype = ["RA--TAN", "DEC-TAN", "WAVE"]
                obs_wcs.naxis3 = obs_wcs._naxis3 = bins.size
                obs_wcs.wcs.pc = np.eye(3)
                obs_wcs.wcs.pc[:2, :2] = rotation_matrix
                obs_wcs.wcs.cdelt = [platescale.to(u.deg).value, platescale.to(u.deg).value, (bins[1] - bins[0])]
                obs_wcs.wcs.cunit = ["deg", "deg", "nm"]
            elif cube_type in ('time',):
                obs_wcs = wcs.WCS(naxis=3)
                obs_wcs.wcs.crpix = [ref_pixel[0], ref_pixel[1], 1]
                obs_wcs.wcs.crval = [target_coordinates.ra.deg, target_coordinates.dec.deg, bins[0]]
                obs_wcs.wcs.ctype = ["RA--TAN", "DEC-TAN", "TIME"]
                obs_wcs.naxis3 = obs_wcs._naxis3 = bins.size
                obs_wcs.wcs.pc = np.eye(3)
                obs_wcs.wcs.pc[:2, :2] = rotation_matrix
                obs_wcs.wcs.cdelt = [platescale.to(u.deg).value, platescale.to(u.deg).value, (bins[1] - bins[0])]
                obs_wcs.wcs.cunit = ["deg", "deg", "s"]
            else:
                obs_wcs = wcs.WCS(naxis=2)
                obs_wcs.wcs.crpix = ref_pixel
                obs_wcs.wcs.crval = [target_coordinates.ra.deg, target_coordinates.dec.deg]
                obs_wcs.wcs.ctype = ["RA--TAN", "DEC-TAN"]

                obs_wcs.wcs.pc = rotation_matrix
                obs_wcs.wcs.cdelt = [platescale.to(u.deg).value, platescale.to(u.deg).value]
                obs_wcs.wcs.cunit = ["deg", "deg"]

            header = obs_wcs.to_header()
            obs_wcs_seq.append(header)

        return obs_wcs_seq

    def get_pixel_spectrum(self, pixel, start=None, duration=None, spec_weight=False, noise_weight=False,
                           wave_start=None, wave_stop=None, bin_width=None, bin_edges=None, bin_type='energy'):
        """
        returns a spectral histogram of a given pixel integrated from start to start+duration,
        and an array giving the cutoff wavelengths used to bin the wavelength values

        Wavelength Bin Specification:
        Depends on parameters: wave_start, stop, wvlBinWidth, energyBinWidth, wvlBinEdges.
        Can only specify one of: wvlBinWidth, energyBinWidth, or wvlBinEdges. If none of these are specified,
        default wavelength bins are used. If flat calibration exists and is applied, flat cal wavelength bins
        must be used.

        Parameters
        ----------
        xCoord: int
            x-coordinate of desired pixel.
        yCoord: int
            y-coordinate index of desired pixel.
        start: float
            Start time of integration, in seconds relative to beginning of file
        duration: float
            Total integration time in seconds. If -1, everything after start is used
        spec_weight: bool
            If True, weights counts by spectral/flat/linearity weight
        noise_weight: bool
            If True, weights counts by true positive fraction (noise weight)
        wave_start: float
            Start wavelength of histogram. Only used if wvlBinWidth or energyBinWidth is
            specified (otherwise it's redundant). Defaults to self.wvlLowerLimit or 7000.
        wvlEnd: float
            End wavelength of histogram. Only used if wvlBinWidth or energyBinWidth is
            specified. Defaults to self.wvlUpperLimit or 15000
        wvlBinWidth: float
            Width of histogram wavelength bin. Used for fixed wavelength bin widths.
        energyBinWidth: float
            Width of histogram energy bin. Used for fixed energy bin widths.
        wvlBinEdges: numpy array of floats
            Specifies histogram wavelength bins. wave_start and wvlEnd are ignored.
        timeSpacingCut: ????
            Legacy; unused


        Returns
        -------
        Dictionary with keys:
            'spectrum' - spectral histogram of given pixel.
            'wvlBinEdges' - edges of wavelength bins
            'effIntTime' - the effective integration time for the given pixel
                           after accounting for hot-pixel time-masking.
            'rawCounts' - The total number of photon triggers (including from
                            the noise tail) during the effective exposure.
                            :param spec_weight:
                            :param noise_weight:
                            :param wave_start:
                            :param wave_stop:
                            :param pixel:
                            :param start:
                            :param duration:
                            :param bin_width:
                            :param bin_edges:
                            :param bin_type:
        """
        if (bin_edges or bin_width) and spec_weight and self.query_header('isFlatCalibrated'):
            # TODO is this even accurate anymore
            raise ValueError('Using flat cal, so flat cal bins must be used')

        photons = self.query(pixel=pixel, start=start, intt=duration, startw=wave_start, stopw=wave_stop)

        weights = np.ones(len(photons))
        if spec_weight:
            weights *= photons['SpecWeight']
        if noise_weight:
            weights *= photons['NoiseWeight']

        if bin_edges and bin_width:
            getLogger(__name__).warning('Both bin_width and bin_edges provided. Using edges')
        elif not bin_edges and bin_width:
            bin_edges = self.wavelength_bins(width=bin_width, start=wave_start, stop=wave_stop,
                                             energy=bin_type == 'energy')
        elif not bin_edges and not bin_width:
            bin_edges = self.nominal_wavelength_bins

        spectrum, _ = np.histogram(photons['Wavelength'], bins=bin_edges, weights=weights)

        return {'spectrum': spectrum, 'wavelengths': bin_edges, 'nphotons': len(photons)}

    def get_fits(self, start=None, duration=None, spec_weight=False, noise_weight=False, wave_start=None,
                 wave_stop=None, rate=True, cube_type=None, bin_width=None, bin_edges=None, bin_type='energy',
                 exclude_flags=pixelflags.PROBLEM_FLAGS):
        """
        Return a fits hdul of the data.

        if cube_type is time or wave a spectral or temporal cube will be returned

        bin_type is for computing bins off bin_width: is width in energy or wavelength anything other than
        'energy' will be treated as wavelength requested. if cube_type is 'time' bin_width/edges are in seconds and
        one of the two is required. bin_type is ignored.

        Temporal bin_edges will take precedence over start and duration

        :param wave_stop:
        :param rate:
        :param cube_type:
        :param bin_width:
        :param bin_edges:
        :param bin_type:
        :param start: float
            Photon list start time, in seconds relative to beginning of file
        :param duration: float
            Photon list end time, in seconds relative to start.
            If None, goes to end of file
        :param spec_weight: bool
            If True, applies the spectral/flat/linearity weight
        :param noise_weight: bool
            If True, applies the true positive fraction (noise) weight
        :param exclude_flags: int
            Specifies flags to exclude from the tallies
            flag definitions see 'h5FileFlags' in Headers/pipelineFlags.py
        :param wave_start:
        """
        cube_type = cube_type.lower()
        bin_type = bin_type.lower()

        tic = time.time()
        ridbins = sorted(self.beamImage.ravel())
        ridbins = np.append(ridbins, ridbins[-1] + 1)

        if cube_type is 'time':
            ycol = 'Time'
            if not bin_edges:
                t0 = 0 if start is None else start
                itime = self.duration if duration is None else duration
                bin_edges = np.linspace(t0, t0 + itime, int(itime / bin_width) + 1)

            start = bin_edges[0]
            duration = bin_edges[-1] - bin_edges[0]
            bin_edges = (bin_edges * 1e6)  # .astype(int)

        elif cube_type is 'wave':
            ycol = 'Wavelength'
            if bin_edges and bin_width:
                getLogger(__name__).warning('Both bin_width and bin_edges provided. Using edges')
            elif not bin_edges and bin_width:
                bin_edges = self.wavelength_bins(width=bin_width, start=wave_start, stop=wave_stop,
                                                 energy=bin_type == 'energy')
            elif not bin_edges and not bin_width:
                bin_edges = self.nominal_wavelength_bins
        else:
            ycol = None
            bin_edges = self.nominal_wavelength_bins[[0, -1]]

        # Retrieval rate is about 2.27Mphot/s for queries in the 100-200M photon range
        photons = self.query(start=start, intt=duration, startw=wave_start, stopw=wave_stop)

        if spec_weight and noise_weight:
            weights = photons['SpecWeight'] * photons['NoiseWeight']
        elif noise_weight:
            weights = photons['NoiseWeight']
        elif spec_weight:
            weights = photons['SpecWeight']
        else:
            weights = None

        if cube_type in ('time', 'wave'):
            data = np.zeros((self.nXPix, self.nYPix, bin_edges.size - 1))
            hist, xedg, yedg = np.histogram2d(photons['ResID'], photons[ycol], bins=(ridbins, bin_edges),
                                              weights=weights)
        else:
            data = np.zeros((self.nXPix, self.nYPix))
            hist, xedg = np.histogram(photons['ResID'], bins=ridbins, weights=weights)

        toc = time.time()
        xe = xedg[:-1]
        for (x, y), resID in np.ndenumerate(self.beamImage):
            if self.flagged(exclude_flags, (x, y)):
                continue
            data[x, y] = hist[xe == resID]

        toc2 = time.time()
        getLogger(__name__).debug(f'Histogram completed in {toc2 - tic:.2f} s, reformatting in {toc2 - toc:.2f}')
        hdu = fits.PrimaryHDU()
        header = hdu.header

        # TODO flesh this out and integrate the non metadata keys
        time_nfo = self._parse_query_range_info(start=start, intt=duration)
        header['START'] = time_nfo['start']
        header['RELSTART'] = time_nfo['relstart']
        header['STOP'] = time_nfo['stop']
        header['EXPTIME'] = time_nfo['duration']
        header['RELSTOP'] = time_nfo['relstop']
        # header['SPECCAL']
        # header['WAVECAL']
        # header['FLATCAL']
        # header['BADPIX']
        # header['COSMIC']
        # header['LINCAL']
        header['MINWAVE'] = wave_start
        header['MAXWAVE'] = wave_stop
        # header['EXFLAG'] = exclude_flags
        header['UNIT'] = 'photons/s' if rate else 'photons'
        header['H5.FILENAME'] = self.filename

        md = self.metadata(timestamp=start)
        if md is not None:
            for k, v in md.items():
                if k.lower() == 'comments':
                    for c in v:
                        header['comment'] = c
                else:
                    try:
                        header[k] = v
                    except ValueError:
                        header[k] = str(v).replace('\n', '_')
        else:
            getLogger(__name__).warning('No metadata found to add to fits header')

        header.update(self.get_wcs(cube_type=cube_type, bins=bin_edges)[0])

        # TODO ensure the following are present
        header['integrationTime'] = duration
        hdul = fits.HDUList([fits.PrimaryHDU(header=header),
                             fits.ImageHDU(data=data / duration if rate else data,
                                           header=header, name='SCIENCE'),
                             fits.ImageHDU(data=np.sqrt(data), header=header, name='VARIANCE'),
                             fits.ImageHDU(data=self._flagArray, header=header, name='FLAGS'),
                             fits.TableHDU(data=bin_edges, name='CUBE_BINS')])
        hdul['CUBE_BINS'].header['UNIT'] = 'us' if cube_type is 'time' else 'nm'
        return hdul

    def query_header(self, name):
        """
        Returns a requested entry from the obs file header
        """
        # header = self.file.root.header.header
        # titles = header.colnames
        # info = header[0]
        # return info[titles.index(name)]
        return self.file.root.header.header[0][self.file.root.header.header.colnames.index(name)]

    def update_header(self, key, value):
        """
        Modifies an entry in the header. Useful for indicating whether wavelength cals,
        flat cals, etc are applied

        Parameters
        ----------
        key: string
            Name of entry to be modified
        value: depends on title
            New value of entry
        """
        if self.mode != 'write':
            raise IOError("Must open file in write mode to do this!")

        if key not in self.header.colnames:
            extensible_header = self.extensible_header_store
            if key not in extensible_header:
                msg = 'Creating a header entry for {} during purported modification to {}'
                getLogger(__name__).warning(msg.format(key, value))
            extensible_header[key] = value
            self._update_extensible_header_store(extensible_header)
        else:
            self.header.modify_column(column=value, colname=key)
            self.header.flush()

    def metadata(self, timestamp=None, include_h5_header=True):
        """ Return an object with attributes containing the the available observing metadata,
        also supports dict access (Presently returns a mkidcore.config.ConfigThing)

        if no timestamp is specified the first record is returned.
        if a timestamp is specified then the first record
        before or equal the time is returned unless there is only one record, and then that is returned

        None if there are no records, ValueError if there is not matching timestamp
        """

        omd = self.extensible_header_store.get('obs_metadata', [])

        H5_FITS_KEYMAP = {'target': 'H5TARGET',
                          'beammapFile': 'H5BEAMMAP',
                          'isWvlCalibrated': 'H5WAVECAL',
                          'isFlatCalibrated': 'H5FLATCAL',
                          'isFluxCalibrated': 'H5SPECCAL',
                          'isLinearityCorrected': 'H5LINCAL',
                          # 'isPhaseNoiseCorrected': 'H5PHASECALED',
                          # 'isPhotonTailCorrected': 'H5TAILCOR',
                          # 'timeMaskExists': 'H5TIMEMASK',
                          'startTime': 'H5START',
                          'expTime': 'H5LENGTH',
                          # 'wvlBinStart': 'H5WAVESTART',
                          # 'wvlBinEnd': 'H5WAVEEND',
                          # 'energyBinWidth': 'H5EBIN'
                          }
        #     wvlCalFile = StringCol(255)
        #     fltCalFile = StringCol(255)

        if not omd:
            return None

        if timestamp is None or len(omd) == 1:
            ret = omd[0]
        else:
            utc = datetime.fromtimestamp(timestamp)
            time_after = np.array([(m.utc - utc).total_seconds() for m in omd])
            if not (time_after <= 0).any():
                times = [m.utc.timestamp() for m in omd]
                msg = 'No metadata available for {:.0f}, {} metadata records from {:.0f} to {:.0f}'
                raise ValueError(msg.format(timestamp, len(omd), min(times), max(times)))

            to_use, = np.where(time_after[time_after <= 0].max() == time_after)
            ret = omd[to_use[0]]

        if include_h5_header:
            titles = self.file.root.header.header.colnames
            info = self.file.root.header.header[0]
            for x in titles:
                if x in H5_FITS_KEYMAP:
                    ret.register(H5_FITS_KEYMAP[x], info[titles.index(x)], update=True)

        return ret

    def attach_observing_metadata(self, metadata):
        self.update_header('obs_metadata', metadata)

    @staticmethod
    def wavelength_bins(width=.1, start=700, stop=1500, energy=True):
        """
        returns an array of wavlength bin edges, with a fixed energy bin width
        withing the limits given in wave_start and wave_stop
        Args:
            width: bin width in eV or wavelength
            start: Lower wavelength edge in Angstrom
            stop: Upper wavelength edge in Angstrom
            energy: (True) set to false
        Returns:
            an array of wavelength bin edges that can be used with numpy.histogram(bins=wvlBinEdges)
        """
        if not energy:
            return np.linspace(start, stop, int((stop - start) / width) + 1)
        const = Photontable.h * Photontable.c * 1e9
        # Calculate upper and lower energy limits from wavelengths, note that start and stop switch when going to energy
        e_stop = const / start
        e_start = const / stop
        n = int((e_stop - e_start) / width)
        # Construct energy bin edges (reversed) and convert back to wavelength
        return const / np.linspace(e_stop, e_start, n + 1)

    def mask_timestamps(self, timestamps, inter=interval(), otherListsToFilter=[]):
        """
        Masks out timestamps that fall in an given interval
        inter is an interval of time values to mask out
        otherListsToFilter is a list of parallel arrays to timestamps that should be masked in the same way
        returns a dict with keys 'timestamps','otherLists'
        """
        # first special case:  inter masks out everything so return zero-length
        # numpy arrays
        raise NotImplementedError('Out of date, update for cosmics')
        if inter == self.intervalAll:
            filteredTimestamps = np.arange(0)
            otherLists = [np.arange(0) for list in otherListsToFilter]
        else:
            if inter == interval() or len(timestamps) == 0:
                # nothing excluded or nothing to exclude
                # so return all unpacked values
                filteredTimestamps = timestamps
                otherLists = otherListsToFilter
            else:
                # there is a non-trivial set of times to mask.
                slices = calculate_slices(inter, timestamps)
                filteredTimestamps = repack_array(timestamps, slices)
                otherLists = []
                for eachList in otherListsToFilter:
                    filteredList = repack_array(eachList, slices)
                    otherLists.append(filteredList)
        # return the values filled in above
        return {'timestamps': filteredTimestamps, 'otherLists': otherLists}

    def updateWavelengths(self, wvlCalArr, xCoord=None, yCoord=None, resID=None, flush=True):
        """
        Changes wavelengths for a single pixel. Overwrites "Wavelength" column w/
        contents of wvlCalArr. NOT reversible unless you have a copy of the original contents.
        Photontable must be open in "write" mode to use.

        Parameters
        ----------
        resID: int
            resID of pixel to overwrite
        wvlCalArr: array of floats
            Array of calibrated wavelengths. Replaces "Wavelength" column of this pixel's
            photon list.
            :param wvlCalArr:
            :param resID:
            :param xCoord:
            :param yCoord:
            :param flush:
        """
        if resID is None:
            resID = self.beamImage[xCoord, yCoord]

        if self.mode != 'write':
            raise Exception("Must open file in write mode to do this!")
        if self.query_header('isWvlCalibrated'):
            getLogger(__name__).warning("Wavelength calibration already exists!")
            warnings.warn("Wavelength calibration already exists!")

        tic = time.time()

        indices = self.photonTable.get_where_list('ResID==resID')
        assert len(indices) == len(wvlCalArr), 'Calibrated wavelength list does not match length of photon list!'

        if not indices:
            return

        if (np.diff(indices) == 1).all():
            getLogger(__name__).debug('Using modify_column')
            self.photonTable.modify_column(start=indices[0], stop=indices[-1] + 1, column=wvlCalArr,
                                           colname='Wavelength')
        else:
            getLogger(__name__).debug('Using modify_coordinates')
            rows = self.photonTable.read_coordinates(indices)
            rows['Wavelength'] = wvlCalArr
            self.photonTable.modify_coordinates(indices, rows)

        if flush:
            self.photonTable.flush()

        getLogger(__name__).debug('Wavelengths updated in {:.2f}s'.format(time.time() - tic))

    def apply_wavecal(self, solution):
        """
        loads the wavelength cal coefficients from a given file and applies them to the
        wavelengths table for each pixel. Photontable must be loaded in write mode. Dont call updateWavelengths !!!

        Note that run-times longer than ~330s for a full MEC dither (~70M photons, 8kpix) is a regression and
        something is wrong. -JB 2/19/19
        """
        # check file_name and status of obsFile
        if self.query_header('isWvlCalibrated'):
            getLogger(__name__).info('Data already calibrated using {}'.format(self.query_header('wvlCalFile')))
            return
        getLogger(__name__).info('Applying {} to {}'.format(solution, self.filename))
        self.photonTable.autoindex = False  # Don't reindex every time we change column
        # apply waveCal
        tic = time.time()
        for (row, column), resID in np.ndenumerate(self.beamImage):

            if not solution.has_good_calibration_solution(res_id=resID):
                continue

            indices = self.photonTable.get_where_list('ResID==resID')
            if not indices.size:
                continue

            flags = self.flags
            self.unflag(flags.bitmask([f for f in flags.names if f.startswith('wavecal')]), pixel=(column, row))
            self.flag(flags.bitmask([f'wavecal.{f.name}' for f in solution.get_flag(res_id=resID)]),
                      pixel=(column, row))

            calibration = solution.calibration_function(res_id=resID, wavelength_units=True)

            if (np.diff(indices) == 1).all():  # This takes ~475s for ALL photons combined on a 70Mphot file.
                # getLogger(__name__).debug('Using modify_column')
                phase = self.photonTable.read(start=indices[0], stop=indices[-1] + 1, field='Wavelength')
                self.photonTable.modify_column(start=indices[0], stop=indices[-1] + 1, column=calibration(phase),
                                               colname='Wavelength')
            else:  # This takes 3.5s on a 70Mphot file!!!
                # raise NotImplementedError('This code path is impractically slow at present.')
                getLogger(__name__).debug('Using modify_coordinates')
                rows = self.photonTable.read_coordinates(indices)
                rows['Wavelength'] = calibration(rows['Wavelength'])
                self.photonTable.modify_coordinates(indices, rows)
            tic2 = time.time()
            getLogger(__name__).debug('Wavelength updated in {:.2f}s'.format(time.time() - tic2))

        self.update_header('isWvlCalibrated', True)
        self.update_header('wvlCalFile', str.encode(solution.name))
        self.photonTable.reindex_dirty()  # recompute "dirty" wavelength index
        self.photonTable.autoindex = True  # turn on auto-indexing
        self.photonTable.flush()
        getLogger(__name__).info('Wavecal applied in {:.2f}s'.format(time.time() - tic))

    def apply_flatcal(self, calsolFile, use_wavecal=True, startw=800, stopw=1375):
        """
        Applies a flat calibration to the "SpecWeight" column of a single pixel.

        Weights are multiplied in and replaced; if "weights" are the contents
        of the "SpecWeight" column, weights = weights*weightArr. NOT reversible
        unless the original contents (or weightArr) is saved.

        Parameters
        ----------
        calSolnPath: string
            Path to the location of the FlatCal solution files
            should contain the base filename of these solution files
            e.g '/mnt/data0/isabel/DeltaAnd/Flats/DeltaAndFlatSoln.h5')
            Code will grab all files titled DeltaAndFlatSoln#.h5 from that directory
        if verbose: will print resID, row, and column of pixels which have a successful FlatCal
                    will print averaged flat weights of pixels which have a successful FlatCal.
        Will write plots of flatcal solution (5 second increments over a single flat exposure) 
             with average weights overplotted to a pdf for pixels which have a successful FlatCal.
             Written to the calSolnPath+'FlatCalSolnPlotsPerPixel.pdf'
             :param use_wavecal:
             :param startw:
             :param stopw:
             :param calsolFile:
             :param save_plots:
        """

        if self.query_header('isFlatCalibrated'):
            getLogger(__name__).info("H5 {} is already flat calibrated".format(self.filename))
            return

        getLogger(__name__).info('Applying {} to {}'.format(calsolFile, self.filename))

        tic = time.time()

        flat_cal = tables.open_file(calsolFile, mode='r')
        calsoln = flat_cal.root.flatcal.calsoln.read()

        for pixel, resID in np.ndenumerate(self.beamImage):

            soln = calsoln[resID == calsoln['resid']]

            # if len(soln) > 1:
            #     msg = 'Flatcal {} has non-unique resIDs'.format(calsolFile)
            #     getLogger(__name__).critical(msg)
            #     raise RuntimeError(msg)

            if not len(soln) and not self.flagged(pixelflags.PROBLEM_FLAGS, pixel=pixel):
                getLogger(__name__).warning('No flat calibration for good pixel {}'.format(resID))
                continue

            if np.any([sol['bad'] for sol in soln]):
                getLogger(__name__).debug('No flat calibration bad for pixel {}'.format(resID))
                continue

            # TODO set pixel flags to include flatcal flags and handle all the various pixel edge cases

            indices = self.photonTable.get_where_list('ResID==resID')
            if not indices.size:
                continue

            if (np.diff(indices) == 1).all():  # This takes ~300s for ALL photons combined on a 70Mphot file.
                # getLogger(__name__).debug('Using modify_column')
                wavelengths = self.photonTable.read(start=indices[0], stop=indices[-1] + 1, field='Wavelength')
                wavelengths[wavelengths < startw] = 0
                wavelengths[wavelengths > stopw] = 0
                coeffs = soln['coeff'].flatten()
                weights = soln['weight'].flatten()
                errors = soln['err'].flatten()
                if self.wavelength_calibrated and not any(
                        [self.flagged(pixelflags.PROBLEM_FLAGS, pixel=pixel)]):
                    weightArr = np.poly1d(coeffs)(wavelengths)
                    if any(weightArr > 100) or any(weightArr < 0.01):
                        getLogger(__name__).debug('Unreasonable fitted weight of for resID {}'.format(resID))

                elif not use_wavecal:
                    weighted_avg, sum_weight_arr = np.ma.average(weights, axis=0, weights=errors ** -2.,
                                                                 returned=True)  # should be a weighted average
                    weightArr = np.ones_like(wavelengths) * weighted_avg
                    if any(weightArr > 100) or any(weightArr < 0.01):
                        getLogger(__name__).debug('Unreasonable averaged weight for resID {}'.format(resID))

                else:
                    assert use_wavecal
                    getLogger(__name__).debug(
                        'No wavecal for pixel with resID {} so no flatweight applied'.format(resID))
                    weightArr = np.zeros(len(wavelengths))

                # enforce positive weights only
                weightArr[weightArr < 0] = 0
                self.photonTable.modify_column(start=indices[0], stop=indices[-1] + 1, column=weightArr,
                                               colname='SpecWeight')
            else:  # This takes 3.5s on a 70Mphot file!!!
                raise NotImplementedError('This code path is impractically slow at present.')
                getLogger(__name__).debug('Using modify_coordinates')
                rows = self.photonTable.read_coordinates(indices)
                rows['SpecWeight'] = np.poly1d(coeffs)(rows['Wavelength'])
                self.photonTable.modify_coordinates(indices, rows)
                getLogger(__name__).debug('Flat weights updated in {:.2f}s'.format(time.time() - tic2))

        self.update_header('isFlatCalibrated', True)
        self.update_header('fltCalFile', calsolFile.encode())
        getLogger(__name__).info('Flatcal applied in {:.2f}s'.format(time.time() - tic))

    def apply_badpix(self, mask, metadata: dict = None):
        """
        loads the wavelength cal coefficients from a given file and applies them to the
        wavelengths table for each pixel. Photontable must be loaded in write mode. Dont call updateWavelengths !!!

        metadata should be a dict of key: value pairs describing the masking
        maks shoe be a boolean array of shape self.beamImage.shape+(3,) where the third axis is: host, cold, unstable

        Note that run-times longer than ~330s for a full MEC dither (~70M photons, 8kpix) is a regression and
        something is wrong. -JB 2/19/19
        """
        tic = time.time()
        getLogger(__name__).info('Applying a bad pixel mask to {}'.format(self.filename))

        f = self.flags
        self.flag(f.bitmask('pixcal.hot') * mask[:,:,0])
        self.flag(f.bitmask('pixcal.cold') * mask[:,:,1])
        self.flag(f.bitmask('pixcal.unstable') * mask[:,:,2])
        self.update_header('isBadPixMasked', True)
        if metadata is not None:
            for k, v in metadata.items():
                self.update_header(f'BADPIX.{k}', v)
        getLogger(__name__).info('Mask applied in {:.3f}s'.format(time.time() - tic))

    def apply_lincal(self, dt=1000, tau=0.000001):
        tic = time.time()
        if self.query_header('isLinearityCorrected'):
            getLogger(__name__).info("H5 {} is already linearity calibrated".format(self.filename))
            return
        bar = ProgressBar(maxval=20439).start()
        for i, resid in enumerate(self.beamImage):
            if self.flagged(pixelflags.PROBLEM_FLAGS, resid=resid):
                continue
            photons = self.query(resid=resid)
            self._apply_column_weight(resid, lincal.calculate_weights(photons['TIME'], dt, tau), 'SpecWeight')
            bar.update(i)
        bar.finish()
        getLogger(__name__).info('Lincal applied to {} in {:.2f}s'.format(self.filename, time.time() - tic))
        self.update_header('isLinearityCorrected', True)
        self.update_header('LINCAL.DT', dt)
        self.update_header('LINCAL.TAU', tau)

    def apply_speccal(self, spectralcal, power=3):
        """
        :param spectralcal: a spectralcal solution. must have id and response_curve attributes, the latter
         a 2xN columnss are wavelength in angstroms and the spectral response
        :return:
        """
        if self.query_header('isFluxCalibrated'):
            getLogger(__name__).info(f"{self.filename} previously calibrated with {self.query_header('SPECCAL.ID')}, "
                                     f"skipping")
            return

        response_curve = spectralcal.response_curve

        getLogger(__name__).info('Applying {} to {}'.format(response_curve, self.filename))
        ind = np.isfinite(response_curve.curve[1])  # dont include nan or inf values
        coeffs = np.polyfit(response_curve.curve[0][ind] / 10.0, response_curve.curve[1][ind], power)
        func = np.poly1d(coeffs)
        tic = time.time()
        for resid in self.beamImage:
            if self.flagged(pixelflags.PROBLEM_FLAGS, resid=resid):
                continue
            self._apply_column_weight(resid, func(self.query(resid=resid)['Wavelength']), 'SpecWeight')

        self.update_header('isFluxCalibrated', True)
        self.update_header('SPECCAL.ID', spectralcal.id)
        self.update_header('SPECCAL.POW', power)
        getLogger(__name__).info('spectralcal applied in {:.2f}s'.format(time.time() - tic))


def calculate_slices(inter, timestamps):
    """
    Hopefully a quicker version of  the original calculate_slices. JvE 3/8/2013

    Returns a list of strings, with format i0:i1 for a python array slice
    inter is the interval of values in timestamps to mask out.
    The resulting list of strings indicate elements that are not masked out

    inter must be a single pyinterval 'interval' object (can be multi-component)
    timestamps is a 1D array of timestamps (MUST be an *ordered* array).

    If inter is a multi-component interval, the components must be unioned and sorted
    (which is the default behaviour when intervals are defined, and is probably
    always the case, so shouldn't be a problem).
    """
    timerange = interval([timestamps[0], timestamps[-1]])
    slices = []
    slce = "0:"  # Start at the beginning of the timestamps array....
    imax = 0  # Will prevent error if inter is an empty interval
    for eachComponent in inter.components:
        # Check if eachComponent of the interval overlaps the timerange of the
        # timestamps - if not, skip to the next component.

        if eachComponent & timerange == interval(): continue
        # [
        # Possibly a bit faster to do this and avoid interval package, but not fully tested:
        # if eachComponent[0][1] < timestamps[0] or eachComponent[0][0] > timestamps[-1]: continue
        # ]

        imin = np.searchsorted(timestamps, eachComponent[0][0], side='left')  # Find nearest timestamp to lower bound
        imax = np.searchsorted(timestamps, eachComponent[0][1], side='right')  # Nearest timestamp to upper bound
        # As long as we're not about to create a wasteful '0:0' slice, go ahead
        # and finish the new slice and append it to the list
        if imin != 0:
            slce += str(imin)
            slices.append(slce)
        slce = str(imax) + ":"
    # Finish the last slice at the end of the timestamps array if we're not already there:
    if imax != len(timestamps):
        slce += str(len(timestamps))
        slices.append(slce)
    return slices


def repack_array(array, slices):
    """
    returns a copy of array that includes only the element defined by slices
    """
    nIncluded = 0
    for slce in slices:
        s0 = int(slce.split(":")[0])
        s1 = int(slce.split(":")[1])
        nIncluded += s1 - s0
    retval = np.zeros(nIncluded)
    iPt = 0;
    for slce in slices:
        s0 = int(slce.split(":")[0])
        s1 = int(slce.split(":")[1])
        iPtNew = iPt + s1 - s0
        retval[iPt:iPtNew] = array[s0:s1]
        iPt = iPtNew
    return retval
