#!/bin/python
"""
Author: Matt Strader        Date: August 19, 2012
Modified 2017 for Darkness/MEC
Authors: Seth Meeker, Neelay Fruitwala, Alex Walter

The class Photontable is an interface to observation files.  It provides methods
for typical ways of accessing photon list observation data.  It can also load 
and apply wavelength and flat calibration.  


Class Obsfile:
====Helper functions====
__init__(self,fileName,mode='read',verbose=False)
__del__(self)
_load_file(self, fileName)
query_header(self, name)
pixelIsBad(self, xCoord, yCoord, forceWvl=False, forceWeights=False, forceTPFWeights=False)

====Single pixel access functions====
getPixelPhotonList(self, xCoord, yCoord, firstSec=0, integrationTime= -1, wvlStart=None,wvlStop=None, forceRawPhase=False)
getListOfPixelsPhotonList(self, posList, **kwargs)
getPixelCount(self, *args, applyWeight=True, applyTPFWeight=True, applyTimeMask=False, **kwargs)
getPixelLightCurve(self,*args,lastSec=-1, cadence=1, scaleByEffInt=True, **kwargs)

====Array access functions====
!!!!These functions need some work!!!!
!!!!They aren't robust to edge cases yet!!!!
getPixelCountImage(self, firstSec=0, integrationTime= -1, wvlStart=None,wvlStop=None,applyWeight=True, applyTPFWeight=True, applyTimeMask=False, scaleByEffInt=False, flagToUse=0)
getCircularAperturePhotonList(self, centerXCoord, centerYCoord, radius, firstSec=0, integrationTime=-1, wvlStart=None,wvlStop=None, flagToUse=0)
_makePixelSpectrum(self, photonList, **kwargs)
getSpectralCube(self, firstSec=0, integrationTime=-1, applyWeight=False, applyTPFWeight=False, wvlStart=700, wvlStop=1500,wvlBinWidth=None, energyBinWidth=None, wvlBinEdges=None, timeSpacingCut=None, flagToUse=0)
getPixelSpectrum(self, xCoord, yCoord, firstSec=0, integrationTime= -1,applyWeight=False, applyTPFWeight=False, wvlStart=None, wvlStop=None,wvlBinWidth=None, energyBinWidth=None, wvlBinEdges=None,timeSpacingCut=None)
wavelength_bins(energy_width=.1, start=700, stop=1500)

====Data write functions for calibrating====
apply_wavecal(self, file_name)
updateWavelengths(self, wvlCalArr, xCoord=None, yCoord=None, resid=None)
_apply_column_weight(self, resID, weightArr, colName)
applySpecWeight(self, resID, weightArr)
applyTPFWeight(self, resID, weightArr)
apply_flatcal(self, calSolnPath,verbose=False)
undoFlag(self, xCoord, yCoord, flag)
update_header(self, headerTitle, headerValue)
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


#These are little better than blind guesses and don't seem to impact performaace, but still need benchmarking
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
    #TODO Add arguments to pass through a query
    tic = time.time()
    f = Photontable(file)
    table = SharedTable(f.photonTable.shape)
    f.photonTable.read(out=table.data)
    ram = table.data.size*table.data.itemsize/1024/1024/1024.0
    msg = 'Created a shared table with {size} rows from {file} in {time:.2f} s, using {ram:.2f} GB'
    getLogger(__name__).info(msg.format(file=file, size=f.photonTable.shape[0], time=time.time()-tic, ram=ram))
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
            getLogger(__name__).debug('Deleting '+self._name)
            SharedArray.delete("shm://{}".format(self._name))


class Photontable(object):
    h = astropy.constants.h.to('eV s').value
    c = astropy.constants.c.to('m/s').value
    nCalCoeffs = 3
    tickDuration = 1e-6  # each integer value is 1 microsecond

    def __init__(self, fileName, mode='read', verbose=False):
        """
        Create Photontable object and load in specified HDF5 file.

        Parameters
        ----------
            fileName: String
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
        self.tickDuration = Photontable.tickDuration
        self.noResIDFlag = 2 ** 32 - 1  # All pixels should have a unique resID. But if for some reason it doesn't, it'll have this resID
        self.wvlLowerLimit = None
        self.wvlUpperLimit = None
        self.filterIsApplied = False
        # self.timeMaskExists = False
        # self.makeMaskVersion = None
        self.ticksPerSec = int(1.0 / self.tickDuration)
        self.intervalAll = interval[0.0, (1.0 / self.tickDuration) - 1]
        self.photonTable = None
        self.fullFileName = fileName
        self.fileName = os.path.basename(fileName)
        self.header = None
        self.defaultWvlBins = None
        self.beamImage = None
        self._flagArray = None  # set in _load_file
        self.nXPix = None
        self.nYPix = None
        self._mdcache = None
        self._load_file(fileName)

    def __del__(self):
        """
        Closes the obs file and any cal files that are open
        """
        try:
            self.file.close()
        except:
            pass

    def __str__(self):
        return 'Photontable: '+self.fullFileName

    def _load_file(self, fileName):
        """ Opens file and loads obs file attributes and beammap """
        self.fileName = os.path.basename(fileName)
        self.fullFileName = fileName
        getLogger(__name__).debug("Loading {} in {} mode.".format(self.fullFileName, self.mode))
        try:
            self.file = tables.open_file(self.fullFileName, mode='a' if self.mode == 'write' else 'r')
        except (IOError, OSError):
            raise

        # get the header
        self.header = self.file.root.header.header

        # get important cal params
        self.defaultWvlBins = Photontable.wavelength_bins(self.query_header('energyBinWidth'), self.query_header('wvlBinStart'),
                                                          self.query_header('wvlBinEnd'))
        # get the beam image.
        self.beamImage = self.file.get_node('/BeamMap/Map').read()
        self._flagArray = self.file.get_node('/BeamMap/Flag')  #The absence of .read() here is correct
        self.nXPix, self.nYPix = self.beamImage.shape
        self.photonTable = self.file.get_node('/Photons/PhotonTable')

    def _makePixelSpectrum(self, photonList, **kwargs):
        """
        Makes a histogram using the provided photon list
        """
        applyWeight = kwargs.pop('applyWeight', False)
        applyTPFWeight = kwargs.pop('applyTPFWeight', False)
        wvlStart = kwargs.pop('wvlStart', None)
        wvlStop = kwargs.pop('stop', None)
        wvlBinWidth = kwargs.pop('wvlBinWidht', None)
        energyBinWidth = kwargs.pop('energyBinWidth', None)
        wvlBinEdges = kwargs.pop('wvlBinEdges', None)

        wvlStart = wvlStart if (wvlStart != None and wvlStart > 0.) else (
            self.wvlLowerLimit if (self.wvlLowerLimit != None and self.wvlLowerLimit > 0.) else 700)
        wvlStop = wvlStop if (wvlStop != None and wvlStop > 0.) else (
            self.wvlUpperLimit if (self.wvlUpperLimit != None and self.wvlUpperLimit > 0.) else 1500)

        wvlList = photonList['Wavelength']
        rawCounts = len(wvlList)

        weights = np.ones(len(wvlList))

        if applyWeight:
            weights *= photonList['SpecWeight']

        if applyTPFWeight:
            weights *= photonList['NoiseWeight']

        if (wvlBinWidth is None) and (energyBinWidth is None) and (
                wvlBinEdges is None):  # use default/flat cal supplied bins
            spectrum, wvlBinEdges = np.histogram(wvlList, bins=self.defaultWvlBins, weights=weights)

        else:  # use specified bins
            if applyWeight and self.query_header('isFlatCalibrated'):
                raise ValueError('Using flat cal, so flat cal bins must be used')
            elif wvlBinEdges is not None:
                assert wvlBinWidth is None and energyBinWidth is None, 'Histogram bins are overspecified!'
                spectrum, wvlBinEdges = np.histogram(wvlList, bins=wvlBinEdges, weights=weights)
            elif energyBinWidth is not None:
                assert wvlBinWidth is None, 'Cannot specify both wavelength and energy bin widths!'
                wvlBinEdges = Photontable.wavelength_bins(energy_width=energyBinWidth, start=wvlStart, stop=wvlStop)
                spectrum, wvlBinEdges = np.histogram(wvlList, bins=wvlBinEdges, weights=weights)
            elif wvlBinWidth is not None:
                nWvlBins = int((wvlStop - wvlStart) / wvlBinWidth)
                spectrum, wvlBinEdges = np.histogram(wvlList, bins=nWvlBins, range=(wvlStart, wvlStop), weights=weights)

            else:
                raise Exception('Something is wrong with getPixelSpectrum...')

        if self.filterIsApplied == True:
            if not np.array_equal(self.filterWvlBinEdges, wvlBinEdges):
                raise ValueError("Synthetic filter wvlBinEdges do not match pixel spectrum wvlBinEdges!")
            spectrum *= self.filterTrans
        # if getEffInt is True:
        return {'spectrum': spectrum, 'wvlBinEdges': wvlBinEdges, 'rawCounts': rawCounts}

    def _apply_column_weight(self, resID, weightArr, colName):
        """
        Applies a weight calibration to the column specified by colName.

        Parameters
        ----------
        resID: int
            resID of desired pixel
        weightArr: array of floats
            Array of cal weights. Multiplied into the "SpecWeight" column.
        colName: string
            Name of weight column. Should be either 'SpecWeight' or 'NoiseWeight'
        """
        if self.mode != 'write':
            raise Exception("Must open file in write mode to do this!")

        if colName not in ('SpecWeight' or 'NoiseWeight'):
            raise ValueError(f"{colName} is not 'SpecWeight' or 'NoiseWeight'")

        indices = self.photonTable.get_where_list('ResID==resID')

        if not (np.diff(indices) == 1).all():
            raise NotImplementedError('Table is not sorted by Res ID!')
        if len(indices) != len(weightArr):
            raise ValueError('weightArr length does not match length of photon list for resID!')

        newWeights = self.query(resid=resID)['SpecWeight'] * np.asarray(weightArr)
        self.photonTable.modify_column(start=indices[0], stop=indices[-1] + 1, column=newWeights, colname=colName)
        self.photonTable.flush()

    def enablewrite(self):
        """USE CARE IN A THREADED ENVIRONMENT"""
        if self.mode == 'write':
            return
        self.file.close()
        self.mode = 'write'
        self._load_file(self.fullFileName)

    def disablewrite(self):
        """USE CARE IN A THREADED ENVIRONMENT"""
        if self.mode == 'read':
            return
        self.file.close()
        self.mode = 'read'
        self._load_file(self.fullFileName)

    def detailed_str(self):
        t=self.photonTable.read()
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

        dirty = ', '.join([n for n in self.photonTable.colnames
                           if self.photonTable.cols._g_col(n).index is not None and
                           self.photonTable.cols._g_col(n).index.dirty])

        s = msg.format(file=self.fullFileName, nphot=len(self.photonTable), sort=sort, tbl=tinfo,
                       start=t['Time'].min(), stop=t['Time'].max(), dur=self.query_header('expTime'),
                       dirty='Column(s) {} have dirty indices.'.format(dirty) if dirty else 'No columns dirty',
                       wave=self.query_header('wvlCalFile'), #self.query_header('isWvlCalibrated') else 'None'
                       flat=self.query_header('fltCalFile') if 'fltCalFile' in self.info.dtype.names else 'None')
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
    def startTime(self):
        return self.query_header('startTime')

    @property
    def stopTime(self):
        return self.query_header('startTime') + self.query_header('expTime')

    @property
    def flag_names(self):
        """
        The ordered list of flag names associated with the file.

        Changing this once it is initialized is at your own peril!
        """
        ret = self.extensible_header_store.get('flags', [])
        if not ret:
            getLogger(__name__).warning('Flag names were not attached at time of H5 creation. '
                                        'If beammap flags have changed since then things WILL break. '
                                        'You must recreate the H5 file.')
            ret = tuple(pixelflags.FLAG_LIST)
            self.enablewrite()
            self.update_header('flags', ret)
            self.disablewrite()
        return ret

    @property
    def bad_pixel_mask(self):
        """A boolean image with true where pixel data has probllems """
        return self.flagMask(pixelflags.PROBLEM_FLAGS)

    def flagMask(self, flag_set, pixel=(slice(None), slice(None)), allow_unknown_flags=True, all_flags=False):
        """
        Test to see if a flag is set on a given pixel or set of pixels

        :param pixel: (x,y) of pixel, 2d slice, list of (x,y), if not specified all pixels are used
        :param flag_set:
        :param allow_unknown_flags:
        :param all_flags: Require all specified flags to be set for the mask to be True
        :return:
        """

        x, y = zip(*pixel) if isinstance(pixel[0], tuple) else pixel

        if len(set(pixelflags.FLAG_LIST).difference(flag_set)) and not allow_unknown_flags:
            return False if isinstance(x, int) else np.zeros_like(self._flagArray[x,y], dtype=bool)

        if not flag_set:
            return True if isinstance(x, int) else np.ones_like(self._flagArray[x, y], dtype=bool)

        bitmask = self.flag_bitmask(flag_set)
        bits = self._flagArray[x,y] & bitmask
        return bits == bitmask if all_flags else bits.astype(bool)

    def flag_bitmask(self, flag_names):
        return pixelflags.flag_bitmask(flag_names, flag_list=self.flag_names)

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
        pixelflags.valid(flag, error=True)
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
        pixelflags.valid(flag, error=True)
        if not np.isscalar(flag) and self._flagArray[y, x].shape != flag.shape:
            raise ValueError('flag must be scalar or match the desired region selected by x & y coordinates')
        self._flagArray[y, x] &= ~flag
        self._flagArray.flush()

    def query(self, startw=None, stopw=None, startt=None, stopt=None, resid=None, intt=None):
        """
        intt takes precedence, All none is a null result

        :param stopt:
        :param intt:
        :param startw: number or none
        :param stopw: number or none
        :param startt: number or none
        :param resid: number, list/array or None
        :return:
        """

        startt = startt if startt else None  # don't query on 0 its slower

        try:
            startt = int(startt * self.ticksPerSec)  # convert to us
        except TypeError:
            pass

        try:
            stopt = int(stopt * self.ticksPerSec)
        except TypeError:
            pass

        if intt is not None:
            stopt = (startt if startt is not None else 0) + int(intt * self.ticksPerSec)

        if resid is None:
            resid = tuple()

        try:
            iter(resid)
        except TypeError:
            resid = [resid]

        res = '|'.join(['(ResID=={})'.format(r) for r in map(int, resid)])
        res = '(' + res + ')' if '|' in res and res else res
        tp = '(Time < stopt)'
        tm = '(Time >= startt)'
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

        if startt is not None:
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
                                                      (startt, stopt, startw, stopw))))

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

        filtered = self.flagMask(disallowed, self.xy(photons))
        return photons[np.invert(filtered)]

    def get_wcs(self, derotate=True, wcs_timestep=None, target_coordinates=None, wave_axis=False, single_pa_time=None):
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
        wave_axis : bool
            False: wcs solution is calculated for ra/dec
            True:  wcs solution is calculated for ra/dec/wavelength
        single_pa_time : float
            Time at which to orientation all non-derotated frames

        See instruments.compute_wcs_ref_pixel() for information on wcscal parameters

        Returns
        -------
        List of wcs headers at each position angle
        """

        #TODO add target_coordinates=None
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

        if isinstance(platescale, u.Quantity):
            platescale = platescale.to(u.mas)
        else:
            platescale = platescale * u.mas

        # TODO remove this check once the relevant h5 files have been corrected
        # try:
        #     rough_mec_platescale_mas = 10*u.mas
        #     np.testing.assert_array_almost_equal(platescale.value, rough_mec_platescale_mas.value)
        # except AssertionError:
        #     getLogger(__name__).warning(f"Setting the platescale to MEC's {rough_mec_platescale_mas.value} mas/pix")
        #     platescale = rough_mec_platescale_mas

        if target_coordinates is not None:
            if not isinstance(target_coordinates, SkyCoord):
                target_coordinates = SkyCoord.from_name(target_coordinates)
        else:
            target_coordinates = SkyCoord(md.ra, md.dec, unit=('hourangle', 'deg'))

        apo = Observer.at_site(md.observatory)

        if wcs_timestep is None:
            wcs_timestep = self.query_header('expTime')

        # sample_times upper boundary is limited to the user defined end time
        sample_times = np.arange(self.query_header('startTime'), self.query_header('startTime')+self.query_header('expTime'), wcs_timestep)
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

            if wave_axis:
                obs_wcs = wcs.WCS(naxis=3)
                obs_wcs.wcs.crpix = [ref_pixel[0], ref_pixel[1], 1]
                obs_wcs.wcs.crval = [target_coordinates.ra.deg, target_coordinates.dec.deg, self.wvlbins[0] / 1e9]
                obs_wcs.wcs.ctype = ["RA--TAN", "DEC-TAN", "WAVE"]
                obs_wcs.naxis3 = obs_wcs._naxis3 = self.nwvlbins
                obs_wcs.wcs.pc = np.eye(3)
                obs_wcs.wcs.cdelt = [platescale.to(u.deg).value, platescale.to(u.deg).value, (self.wvlbins[1] - self.wvlbins[0]) / 1e9]
                obs_wcs.wcs.cunit = ["deg", "deg", "m"]
            else:
                obs_wcs = wcs.WCS(naxis=2)
                obs_wcs.wcs.ctype = ["RA--TAN", "DEC-TAN"]

                obs_wcs.wcs.crval = np.array([target_coordinates.ra.deg, target_coordinates.dec.deg])
                obs_wcs.wcs.crpix = ref_pixel

                obs_wcs.wcs.pc = rotation_matrix
                obs_wcs.wcs.cdelt = [platescale.to(u.deg).value, platescale.to(u.deg).value]
                obs_wcs.wcs.cunit = ["deg", "deg"]

            header = obs_wcs.to_header()
            obs_wcs_seq.append(header)

        return obs_wcs_seq

    def getPixelPhotonList(self, xCoord=None, yCoord=None, resid=None, firstSec=0, integrationTime=-1, wvlStart=None,
                           wvlStop=None, forceRawPhase=False):
        """
        Retrieves a photon list at xCoord,yCoord using the attached beammap.

        Parameters
        ----------
        xCoord: int or iterable valid to index arrays
            x-coordinate of pixel in beammap
        yCoord: int or iterable valid to index arrays
            y-coordinate index of pixel in beammap
        resID: int or iterable valid to index arrays, takes precidence over x, y
        firstSec: float
            Photon list start time, in seconds relative to beginning of file
        integrationTime: float
            Photon list end time, in seconds relative to firstSec.
            If -1, goes to end of file
        wvlStart: float
            Desired start wavelength range (inclusive). Must satisfy wvlStart <= stop.
            If None, includes all wavelengths less than stop. If file is not wavelength calibrated, this parameter
            specifies the range of desired phase heights.
        stop: float
            Desired stop wavelength range (non-inclusive). Must satisfy wvlStart <= stop.
            If None, includes all wavelengths greater than wvlStart. If file is not wavelength calibrated, this parameter
            specifies the range of desired phase heights.
        forceRawPhase: bool
            If the Photontable is not wavelength calibrated this flag does nothing.
            If the Photontable is wavelength calibrated (Photontable.query_header('isWvlCalibrated') = True) then:
             - forceRawPhase=True will return all the photons in the list (might be phase heights instead of wavelengths)
             - forceRawPhase=False is guaranteed to only return properly wavelength calibrated photons in the photon list

        Returns
        -------
        Structured Numpy Array
            Each row is a photon.
            Columns have the following keys: 'Time', 'Wavelength', 'SpecWeight', 'NoiseWeight'

        Time is in microseconds
        Wavelength is in degrees of phase height or nanometers
        SpecWeight is a float in [0,1]
        NoiseWeight is a float in [0,1]
        :param resid:

        """
        try:
            if wvlStop < wvlStart:
                raise ValueError('Invalid wavelength range')
        except TypeError:
            pass
        if firstSec is not None and firstSec > self.query_header('expTime'):
            raise ValueError('Start time not in file.')

        resid = self.beamImage[xCoord, yCoord] if resid is None else resid

        return self.query(startw=wvlStart, stopw=wvlStop, startt=firstSec if firstSec else None,
                          resid=resid, intt=None if integrationTime == -1 else integrationTime)

    def getFits(self, firstSec=0, integrationTime=None, applyWeight=False, applyTPFWeight=False,
                wvlStart=None, wvlStop=None, cube=False, countRate=True):
        """
        Return a time-flattened spectral cube of the counts integrated from firstSec to firstSec+integrationTime.
        If integration time is -1, all time after firstSec is used.
        If weighted is True, flat cal weights are applied.
        If fluxWeighted is True, spectral shape weights are applied.
        """


        if integrationTime is None:
            integrationTime = self.query_header('expTime')

        # Retrieval rate is about 2.27Mphot/s for queries in the 100-200M photon range
        masterPhotonList = self.query(startt=firstSec if firstSec else None, intt=integrationTime,
                                      startw=wvlStart, stopw=wvlStop)
        weights = None
        if applyWeight:
            weights = masterPhotonList['SpecWeight']
        if applyTPFWeight:
            if weights is not None:
                weights *= masterPhotonList['NoiseWeight']
            else:
                weights = masterPhotonList['NoiseWeight']

        tic = time.time()
        if cube:

            wvlBinEdges = self.defaultWvlBins.size
            nWvlBins = wvlBinEdges.size - 1
            data = np.zeros((self.nXPix, self.nYPix, nWvlBins))

            ridbins = sorted(self.beamImage.ravel())
            ridbins = np.append(ridbins, max(ridbins) + 1)
            hist, xedg, yedg = np.histogram2d(masterPhotonList['ResID'], masterPhotonList['Wavelength'],
                                              bins=(ridbins, wvlBinEdges), weights=weights)

            toc = time.time()
            xe = xedg[:-1]
            for (x, y), resID in np.ndenumerate(self.beamImage):  # 3% % of the time
                if self.flagMask(pixelflags.PROBLEM_FLAGS, (x, y)):
                    continue
                data[x, y, :] = hist[xe == resID]
        else:
            data = np.zeros((self.nXPix, self.nYPix))
            ridbins = sorted(self.beamImage.ravel())
            ridbins = np.append(ridbins, max(ridbins) + 1)
            hist, xedg = np.histogram(masterPhotonList['ResID'], bins=ridbins, weights=weights)

            toc = time.time()
            xe = xedg[:-1]
            for (x, y), resID in np.ndenumerate(self.beamImage):
                if self.flagMask(pixelflags.PROBLEM_FLAGS, (x, y)):
                    continue
                data[x, y] = hist[xe == resID]

        toc2 = time.time()
        getLogger(__name__).debug('Histogrammed data in {:.2f} s, reformatting in {:.2f}'.format(toc2 - tic,
                                                                                                toc2 - toc))
        hdu = fits.PrimaryHDU()
        header = hdu.header

        md = self.metadata(timestamp=firstSec)
        if md is not None:
            for k, v in md.items():
                if k.lower() == 'comments':
                    for c in v:
                        header['comment'] = c
                else:
                    try:
                        header[k] = v
                    except ValueError:
                        header[k] = str(v).replace('\n','_')
        else:
            getLogger(__name__).warning('No metadata found to add to fits header')

        wcs = self.get_wcs(wave_axis=cube)[0]
        header.update(wcs)

        #TODO set the header units for the extensions
        hdul = fits.HDUList([fits.PrimaryHDU(header=header),
                             fits.ImageHDU(data=data/integrationTime if countRate else data,
                                           header=header, name='SCIENCE'),
                             fits.ImageHDU(data=np.sqrt(data), header=header, name='VARIANCE'),
                             fits.ImageHDU(data=self._flagArray, header=header, name='FLAGS')])
        return hdul

    def getPixelCountImage(self, firstSec=0, integrationTime=None, wvlStart=None, wvlStop=None, applyWeight=False,
                            applyTPFWeight=False, scaleByEffInt=False, exclude_flags=pixelflags.PROBLEM_FLAGS, hdu=False):
        """
        Returns an image of pixel counts over the entire array between firstSec and firstSec + integrationTime. Can specify calibration weights to apply as
        well as wavelength range.

        Parameters
        ----------
        firstSec: float
            Photon list start time, in seconds relative to beginning of file
        integrationTime: float
            Photon list end time, in seconds relative to firstSec.
            If -1, goes to end of file
        wvlRange: (float, float)
            Desired wavelength range of photon list. Must satisfy wvlRange[0] < wvlRange[1].
            If None, includes all wavelengths. If file is not wavelength calibrated, this parameter
            specifies the range of desired phase heights.
        applyWeight: bool
            If True, applies the spectral/flat/linearity weight
        applyTPFWeight: bool
            If True, applies the true positive fraction (noise) weight
        scaleByEffInt: bool
            If True, scales each pixel by (total integration time)/(effective integration time)
        flagToUse: int
            Specifies (bitwise) pixel flags that are suitable to include in image. For
            flag definitions see 'h5FileFlags' in Headers/pipelineFlags.py

        Returns
        -------
        Dictionary with keys:
            'image': 2D numpy array, image of pixel counts
            'effIntTimes':2D numpy array, image effective integration times after time-masking is
           `          accounted for.
           :param wvlStart:
           :param wvlStop:
           :param hdu:
        """
        if integrationTime is None:
            integrationTime = self.query_header('expTime')

        image = np.zeros((self.nXPix, self.nYPix))

        # TODO Actually compute the effective integration time
        effIntTime = np.full((self.nXPix, self.nYPix), integrationTime)

        masterPhotonList = self.query(startt=firstSec if firstSec else None, startw=wvlStart,
                                      stopw=wvlStop, intt=integrationTime)

        weights = None
        if applyWeight:
            weights = masterPhotonList['SpecWeight']

        if applyTPFWeight:
            if weights is not None:
                weights *= masterPhotonList['NoiseWeight']
            else:
                weights = masterPhotonList['NoiseWeight']

        # 11.4s on MEC data with 173Mphot
        tic = time.time()
        ridbins = sorted(self.beamImage.ravel())
        ridbins = np.append(ridbins, max(ridbins) + 1)
        hist, ridbins = np.histogram(masterPhotonList['ResID'], bins=ridbins, weights=weights)

        toc = time.time()
        resIDs = ridbins[:-1]
        for (x, y), resID in np.ndenumerate(self.beamImage): # 4 % of the time
            if self.flagMask(exclude_flags, (x, y)) and any(exclude_flags):
                continue
            image[x, y] = hist[resIDs == resID]
        toc2 = time.time()
        getLogger(__name__).debug('Histogrammed data in {:.2f} s, reformatting in {:.2f}'.format(toc2 - tic,
                                                                                                 toc2 - toc))
        if hdu:
            ret = fits.ImageHDU(data=image)
            return ret
        else:
            return {'image': image, 'effIntTime': effIntTime, 'spec_weights': masterPhotonList['SpecWeight'],
                    'noise_weights': masterPhotonList['NoiseWeight']}

    def getCircularAperturePhotonList(self, centerXCoord, centerYCoord, radius,
                                      firstSec=0, integrationTime=-1, wvlStart=None,
                                      wvlStop=None, flagToUse=0):
        """
        Retrieves a photon list for the specified circular aperture.
        For pixels that partially overlap with the region, all photons
        are included, and the overlap fraction is multiplied into the
        'NoiseWeight' column.

        Parameters
        ----------
        centerXCoord: float
            x-coordinate of aperture center (pixel units)
        centerYCoord: float
            y-coordinate of aperture center (pixel units)
        radius: float
            radius of aperture
        firstSec: float
            Photon list start time, in seconds relative to beginning of file
        integrationTime: float
            Photon list end time, in seconds relative to firstSec.
            If -1, goes to end of file
        wvlRange: (float, float)
            Desired wavelength range of photon list. Must satisfy wvlRange[0] <= wvlRange[1].
            If None, includes all wavelengths.
        flagToUse: int
            Specifies (bitwise) pixel flags that are suitable to include in photon list. For
            flag definitions see 'h5FileFlags' in Headers/pipelineFlags.py

        Returns
        -------
        Dictionary with keys:
            photonList: numpy structured array
                Time ordered photon list. Adds resID column to keep track
                of individual pixels
            effQE: float
                Fraction of usable pixel area inside aperture
            apertureMask: numpy array
                Image of effective pixel weight inside aperture. "Pixel weight"
                for now is just the area of overlap w/ aperture, with dead
                pixels set to 0.
                :param wvlStart:
                :param wvlStop:

        """
        raise RuntimeError('Update this to query all at the same time and fix flags')
        center = PixCoord(centerXCoord, centerYCoord)
        apertureRegion = CirclePixelRegion(center, radius)
        exactApertureMask = apertureRegion.to_mask('exact').data
        boolApertureMask = exactApertureMask > 0
        apertureMaskCoords = np.transpose(
            np.array(np.where(boolApertureMask)))  # valid coordinates within aperture mask
        photonListCoords = apertureMaskCoords + np.array(
            [apertureRegion.bounding_box.ixmin, apertureRegion.bounding_box.iymin])  # pixel coordinates in image

        # loop through valid coordinates, grab photon lists and store in photonList
        photonList = None
        for i, coords in enumerate(photonListCoords):
            if coords[0] < 0 or coords[0] >= self.nXPix or coords[1] < 0 or coords[1] >= self.nYPix:
                exactApertureMask[apertureMaskCoords[i, 0], apertureMaskCoords[i, 1]] = 0
                continue
            if not self.flagMask(flagToUse, (x, y)):
                exactApertureMask[apertureMaskCoords[i, 0], apertureMaskCoords[i, 1]] = 0
                continue

            pixPhotonList = self.getPixelPhotonList(coords[0], coords[1], firstSec, integrationTime, wvlStart, wvlStop)
            pixPhotonList['NoiseWeight'] *= exactApertureMask[apertureMaskCoords[i, 0], apertureMaskCoords[i, 1]]
            if photonList is None:
                photonList = pixPhotonList
            else:
                photonList = np.append(photonList, pixPhotonList)

        photonList = np.sort(photonList, order='Time')
        return photonList, exactApertureMask

    def getTemporalCube(self, firstSec=None, integrationTime=None, applyWeight=False, applyTPFWeight=False,
                        startw=None, stopw=None, timeslice=None, timeslices=None,
                        exclude_flags=pixelflags.PROBLEM_FLAGS, hdu=False):
        """
        Return a wavelength-flattened spectral cube of the counts integrated from firstSec to firstSec+integrationTime.
        If stopt is None, all time after startt is used.
        If weighted is True, flat cal weights are applied.
        If fluxWeighted is True, spectral shape weights are applied.

        Timeslices is an optional array of cube time bin edges in seconds. If provided, timeslices takes precedence over startt,
        stopt, and timeslice.

        the timeslices returned will be in seconds

        [nx,ny,time]
        """
        if timeslices is not None:
            firstSec = timeslices.min()
            integrationTime = timeslices.max() - timeslices.min()
        else:
            # May not include data at tail end if timeslice does not evenly divide time window
            timeslices = np.arange(0 if firstSec is None else firstSec,
                                   (self.duration if integrationTime is None else integrationTime) + 1e-9, timeslice)
            firstSec = timeslices.min()
            integrationTime = timeslices.max() - timeslices.min()

        cube = np.zeros((self.nXPix, self.nYPix, timeslices.size-1))

        ## Retrieval rate is ~ ? Mphot/s for queries in the ~M photon range
        if not firstSec and not integrationTime and not startw and not stopw:
            masterPhotonList = self.photonTable.read()
        else:
            masterPhotonList = self.query(startt=firstSec, stopt=integrationTime, startw=startw, stopw=stopw)

        weights = None
        if applyWeight:
            weights = masterPhotonList['SpecWeight']
        if applyTPFWeight:
            weights = masterPhotonList['NoiseWeight'] if weights is None else weights*masterPhotonList['NoiseWeight']

        tic = time.time()
        ridbins = sorted(self.beamImage.ravel())
        ridbins = np.append(ridbins, max(ridbins) + 1)
        hist, xedg, yedg = np.histogram2d(masterPhotonList['ResID'], masterPhotonList['Time'],
                                          bins=(ridbins, timeslices*1e6), weights=weights)

        toc = time.time()
        xe = xedg[:-1]
        for (x, y), resID in np.ndenumerate(self.beamImage):  # 3% of the time
            if self.flagMask(exclude_flags, (x, y)):
                continue
            cube[x, y, :] = hist[xe == resID]
        toc2 = time.time()
        getLogger(__name__).debug('Histogramed data in {:.2f} s, reformatting in {:.2f}'.format(toc2 - tic, toc2 - toc))

        if hdu:
            ret = fits.ImageHDU(data=np.moveaxis(cube, 2, 0))
            getLogger(__name__).warning('Must integrate wavelength info into ImageHDU ctype kw and finish building hdu')
            #TODO finish returning hdu
            return ret
        else:
            return {'cube': cube, 'timeslices': timeslices, 'bad': self.bad_pixel_mask}

    def getSpectralCube(self, firstSec=0, integrationTime=None, applyWeight=False, applyTPFWeight=False,
                        wvlStart=700, wvlStop=1500, wvlBinWidth=None, energyBinWidth=None, wvlBinEdges=None,
                        exclude_flags=pixelflags.PROBLEM_FLAGS, hdu=False):
        """
        Return a time-flattened spectral cube of the counts integrated from firstSec to firstSec+integrationTime.
        If integration time is -1, all time after firstSec is used.
        If weighted is True, flat cal weights are applied.
        If fluxWeighted is True, spectral shape weights are applied.
        """
        if wvlBinEdges is not None:
            assert wvlBinWidth is None and energyBinWidth is None, 'Histogram bins are overspecified!'
        elif energyBinWidth is not None:
            assert wvlBinWidth is None, 'Cannot specify both wavelength and energy bin widths!'
            wvlBinEdges = self.wavelength_bins(energy_width=energyBinWidth, start=wvlStart, stop=wvlStop)
        elif wvlBinWidth is not None:
            wvlBinEdges = np.linspace(wvlStart, wvlStop, num=int((wvlStop - wvlStart) / wvlBinWidth) + 1)
        else:
            wvlBinEdges = self.defaultWvlBins

        nWvlBins = wvlBinEdges.size - 1

        if integrationTime == -1 or integrationTime is None:
            integrationTime = self.query_header('expTime')

        cube = np.zeros((self.nXPix, self.nYPix, nWvlBins))

        # TODO Actually compute the effective integration time
        effIntTime = np.full((self.nXPix, self.nYPix), integrationTime)

        ## Retrieval rate is about 2.27Mphot/s for queries in the 100-200M photon range
        masterPhotonList = self.query(startt=firstSec if firstSec else None, intt=integrationTime, startw=wvlStart,
                                      stopw=wvlStop)

        # tic=time.time()  #14.39
        # r=self.photonTable.read(field='ResID')
        # t=self.photonTable.read(field='Time')
        # tic2=time.time()
        # u=t<5000000
        # r[u]
        # t[u]
        # print(tic2-tic, time.time()-tic2)
        #
        # tic=time.time()
        # d=self.photonTable.read()
        # tic2=time.time()
        # r=np.array(d['ResID'])
        # t=np.array(d['Time'])
        # u=t<5000000
        # r[u]
        # t[u]
        # print(tic2-tic, time.time()-tic2)
        # 4.766175985336304 0.4749867916107178
        # 3.8692047595977783 2.829094648361206

        weights = None
        if applyWeight:
            weights = masterPhotonList['SpecWeight']
        if applyTPFWeight:
            if weights is not None:
                weights *= masterPhotonList['NoiseWeight']
            else:
                weights = masterPhotonList['NoiseWeight']

        # option one
        # self.photonTable.itersorted('ResID', checkCSI=True)

        # option 2 pytables
        # grouped = df.groupby('A')
        # for name, group in grouped:

        # option 3 numpy_iter package
        # GroupBy(masterPhotonList['ResID']).split_sequence_as_iterable(masterPhotonList)

        # Option 4 np.histogram 2d ~5.25s on MEC data with 30Mphot
        tic = time.time()
        ridbins = sorted(self.beamImage.ravel())
        ridbins = np.append(ridbins, max(ridbins) + 1)
        hist, xedg, yedg = np.histogram2d(masterPhotonList['ResID'], masterPhotonList['Wavelength'],
                                          bins=(ridbins, wvlBinEdges), weights=weights)

        toc = time.time()
        xe = xedg[:-1]
        for (x, y), resID in np.ndenumerate(self.beamImage):  # 3% % of the time
            if self.flagMask(exclude_flags, (x, y)):
                continue
            cube[x, y, :] = hist[xe == resID]
        toc2 = time.time()
        getLogger(__name__).debug('Histogramed data in {:.2f} s, reformatting in {:.2f}'.format(toc2 - tic, toc2 - toc))

        # Option 5: legacy 1183 s on MEC data with 30Mphot
        # cube2 = np.zeros((self.nXPix, self.nYPix, nWvlBins))
        # rawCounts = np.zeros((self.nXPix,self.nYPix))
        # tic = time.time()
        # resIDs = masterPhotonList['ResID']
        # for (xCoord, yCoord), resID in np.ndenumerate(self.beamImage): #162 ms/loop
        #     #all the time
        #     photonList = masterPhotonList[resIDs == resID] if self.flagMask(exclude_flags, (xCoord,yCoord)) else emptyPhotonList
        #     x = self._makePixelSpectrum(photonList, applyWeight=applyWeight, applyTPFWeight=applyTPFWeight,
        #                                 wvlBinEdges=wvlBinEdges)
        #     cube2[xCoord, yCoord, :] = x['spectrum']
        #     rawCounts[xCoord, yCoord] = x['rawCounts']
        # toc = time.time()
        # getLogger(__name__).debug(('Cubed data in {:.2f} s using old'
        #                           ' approach. Cubes same {}').format(toc - tic, (cube==cube2).all()))

        if hdu:
            ret = fits.ImageHDU(data=cube)
            getLogger(__name__).warning('Must integrate wavelength info into ImageHDU ctype kw and finish building hdu')
            #TODO finish returning hdu
            return ret
        else:
            return {'cube': cube, 'wvlBinEdges': wvlBinEdges, 'effIntTime': effIntTime}

    def getPixelSpectrum(self, xCoord, yCoord, firstSec=0, integrationTime=-1,
                         applyWeight=False, applyTPFWeight=False, wvlStart=None, wvlStop=None,
                         wvlBinWidth=None, energyBinWidth=None, wvlBinEdges=None, timeSpacingCut=None):
        """
        returns a spectral histogram of a given pixel integrated from firstSec to firstSec+integrationTime,
        and an array giving the cutoff wavelengths used to bin the wavelength values

        Wavelength Bin Specification:
        Depends on parameters: wvlStart, stop, wvlBinWidth, energyBinWidth, wvlBinEdges.
        Can only specify one of: wvlBinWidth, energyBinWidth, or wvlBinEdges. If none of these are specified,
        default wavelength bins are used. If flat calibration exists and is applied, flat cal wavelength bins
        must be used.

        Parameters
        ----------
        xCoord: int
            x-coordinate of desired pixel.
        yCoord: int
            y-coordinate index of desired pixel.
        firstSec: float
            Start time of integration, in seconds relative to beginning of file
        integrationTime: float
            Total integration time in seconds. If -1, everything after firstSec is used
        applyWeight: bool
            If True, weights counts by spectral/flat/linearity weight
        applyTPFWeight: bool
            If True, weights counts by true positive fraction (noise weight)
        wvlStart: float
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
            Specifies histogram wavelength bins. wvlStart and wvlEnd are ignored.
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
                            :param wvlStop:
        """

        photonList = self.getPixelPhotonList(xCoord, yCoord, firstSec=firstSec, integrationTime=integrationTime)
        return self._makePixelSpectrum(photonList, applyWeight=applyWeight,
                                       applyTPFWeight=applyTPFWeight, wvlStart=wvlStart, wvlStop=wvlStop,
                                       wvlBinWidth=wvlBinWidth, energyBinWidth=energyBinWidth,
                                       wvlBinEdges=wvlBinEdges, timeSpacingCut=timeSpacingCut)

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
        getLogger(__name__).info('Applying {} to {}'.format(solution, self.fullFileName))
        self.photonTable.autoindex = False  # Don't reindex every time we change column
        # apply waveCal
        tic = time.time()
        for (row, column), resID in np.ndenumerate(self.beamImage):
            self.unflag(self.flag_bitmask([f for f in pixelflags.FLAG_LIST if f.startswith('wavecal')]),
                        pixel=(column, row))
            self.flag(self.flag_bitmask(pixelflags.to_flag_names('wavecal', solution.get_flag(res_id=resID))),
                      pixel=(column, row))

            if not solution.has_good_calibration_solution(res_id=resID):
                continue

            calibration = solution.calibration_function(res_id=resID, wavelength_units=True)

            indices = self.photonTable.get_where_list('ResID==resID')
            if not indices.size:
                continue
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
        getLogger(__name__).info('Wavecal applied in {:.2f}s'.format(time.time()-tic))

    def applyBadPixelMask(self, hot_mask, cold_mask, unstable_mask):
        """
        loads the wavelength cal coefficients from a given file and applies them to the
        wavelengths table for each pixel. Photontable must be loaded in write mode. Dont call updateWavelengths !!!

        Note that run-times longer than ~330s for a full MEC dither (~70M photons, 8kpix) is a regression and
        something is wrong. -JB 2/19/19
        """
        tic = time.time()
        getLogger(__name__).info('Applying a bad pixel mask to {}'.format(self.fullFileName))
        self.flag(self.flag_bitmask('pixcal.hot') * hot_mask)
        self.flag(self.flag_bitmask('pixcal.cold') * cold_mask)
        self.flag(self.flag_bitmask('pixcal.unstable') * unstable_mask)
        self.update_header('isBadPixMasked', True)
        getLogger(__name__).info('Mask applied in {:.3f}s'.format(time.time()-tic))

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
                getLogger(__name__).warning(msg.format(type(self._mdcache), self.fileName))
                self._mdcache = {}
        return self._mdcache

    def _update_extensible_header_store(self, extensible_header):
        if not isinstance(extensible_header, dict):
            raise TypeError('extensible_header must be of type dict')
        out = StringIO()
        yaml.dump(extensible_header, out)
        emdstr = out.getvalue().encode()
        if len(emdstr) > METADATA_BLOCK_BYTES:  # this should match mkidcore.headers.ObsHeader.metadata
            raise ValueError("Too much metadata! {} KB needed, {} allocated".format(len(emdstr)//1024,
                                                                                    METADATA_BLOCK_BYTES//1024))
        self._mdcache = extensible_header
        self.update_header('metadata', emdstr)

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
            self.info = self.header[0]

    def metadata(self, timestamp=None):
        """ Return an object with attributes containing the the available observing metadata,
        also supports dict access (Presently returns a mkidcore.config.ConfigThing)

        if no timestamp is specified the first record is returned.
        if a timestamp is specified then the first record
        before or equal the time is returned unless there is only one record, and then that is returned

        None if there are no records, ValueError if there is not matching timestamp
        """

        omd = self.extensible_header_store.get('obs_metadata', [])

        # TODO integrate appropriate things in .info with the metadata returned so this is a one-stop-shop
        # infomd_keys = {'wavefile': 'wvlCalFile', 'flatfile':'fltCalFile',
        #                'beammap': 'beammapFile'}
        # #Other INFO keys that might should be included
        # target = StringCol(255)
        # dataDir = StringCol(255)
        # beammapFile = StringCol(255)
        # isWvlCalibrated = BoolCol()
        # isFlatCalibrated = BoolCol()
        # isFluxalibrated = BoolCol()
        # isLinearityCorrected = BoolCol()
        # isPhaseNoiseCorrected = BoolCol()
        # isPhotonTailCorrected = BoolCol()
        # timeMaskExists = BoolCol()
        # startTime = Int32Col()
        # expTime = Int32Col()
        # wvlBinStart = Float32Col()
        # wvlBinEnd = Float32Col()
        # energyBinWidth = Float32Col()
        #
        # for mdk, k in infomd_keys.items():
        #     md[mdk] = self.info[k]
        # OBS2FITS = dict(target='DASHTARG', dataDir='DATADIR', beammapFile='BEAMMAP', wvlCalFile='WAVECAL',
        #                 fltCalFile='FLATCAL')

        # for k in self.info:
        #     omd.register(k, self.inf[k], update=True)

        # header['NWEIGHT'] = (applyTPFWeight and self.query_header('isPhaseNoiseCorrected'), 'Noise weight corrected')
        # header['LWEIGHT'] = (applyWeight and self.query_header('isLinearityCorrected'), 'Linearity corrected')
        # header['FWEIGHT'] = (applyWeight and self.query_header('isFlatCalibrated'), 'Flatcal corrected')
        # header['SWEIGHT'] = (applyWeight and self.query_header('isFluxCalibrated'), 'QE corrected')

        if not omd:
            return None

        if timestamp is None or len(omd) == 1:
            ret = omd[0]
        else:
            utc = datetime.fromtimestamp(timestamp)
            time_after = np.array([(m.utc-utc).total_seconds() for m in omd])
            if not (time_after <= 0).any():
                times = [m.utc.timestamp() for m in omd]
                msg = 'No metadata available for {:.0f}, {} metadata records from {:.0f} to {:.0f}'
                raise ValueError(msg.format(timestamp, len(omd), min(times), max(times)))

            to_use, = np.where(time_after[time_after <= 0].max() == time_after)
            ret = omd[to_use[0]]

        return ret

    def attach_observing_metadata(self, metadata):
        self.update_header('obs_metadata', metadata)

    @staticmethod
    def wavelength_bins(energy_width=.1, start=700, stop=1500):
        """
        returns an array of wavlength bin edges, with a fixed energy bin width
        withing the limits given in wvlStart and wvlStop
        Args:
            energy_width: bin width in eV
            start: Lower wavelength edge in Angstrom
            stop: Upper wavelength edge in Angstrom
        Returns:
            an array of wavelength bin edges that can be used with numpy.histogram(bins=wvlBinEdges)
        """
        const = Photontable.h*Photontable.c*1e9
        # Calculate upper and lower energy limits from wavelengths, note that start and stop switch when going to energy
        e_stop = const / start
        e_start = const / stop
        n = int((e_stop - e_start) / energy_width)
        # Construct energy bin edges (reversed) and convert back to wavelength
        return const/np.linspace(e_stop, e_start, n + 1)

    def maskTimestamps(self, timestamps, inter=interval(), otherListsToFilter=[]):
        """
        Masks out timestamps that fall in an given interval
        inter is an interval of time values to mask out
        otherListsToFilter is a list of parallel arrays to timestamps that should be masked in the same way
        returns a dict with keys 'timestamps','otherLists'
        """
        # first special case:  inter masks out everything so return zero-length
        # numpy arrays
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
                slices = calculateSlices(inter, timestamps)
                filteredTimestamps = repackArray(timestamps, slices)
                otherLists = []
                for eachList in otherListsToFilter:
                    filteredList = repackArray(eachList, slices)
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

        if (np.diff(indices)==1).all():
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

        getLogger(__name__).debug('Wavelengths updated in {:.2f}s'.format(time.time()-tic))

    def applySpecWeight(self, resID, weightArr):
        """
        Applies a weight calibration to the "SpecWeight" column of a single pixel.

        This is where the flat cal, linearity cal, and spectral cal go.
        Weights are multiplied in and replaced; if "weights" are the contents
        of the "SpecWeight" column, weights = weights*weightArr. NOT reversible
        unless the original contents (or weightArr) is saved.

        Parameters
        ----------
        resID: int
            resID of desired pixel
        weightArr: array of floats
            Array of cal weights. Multiplied into the "SpecWeight" column.
        """
        self._apply_column_weight(resID, weightArr, 'SpecWeight')

    def applyTPFWeight(self, resID, weightArr):
        """
        Applies a weight calibration to the "SpecWeight" column.

        This is where the TPF (noise weight) goes.
        Weights are multiplied in and replaced; if "weights" are the contents
        of the "NoiseWeight" column, weights = weights*weightArr. NOT reversible
        unless the original contents (or weightArr) is saved.

        Parameters
        ----------
        resID: int
            resID of desired pixel
        weightArr: array of floats
            Array of cal weights. Multiplied into the "NoiseWeight" column.
        """
        self._apply_column_weight(resID, weightArr, 'NoiseWeight')

    def apply_flatcal(self, calsolFile, use_wavecal=True, save_plots=False, startw=800, stopw=1375):
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
             :param calsolFile:
             :param save_plots:
        """

        if self.query_header('isFlatCalibrated'):
            getLogger(__name__).info("H5 {} is already flat calibrated".format(self.fullFileName))
            return

        getLogger(__name__).info('Applying {} to {}'.format(calsolFile, self.fullFileName))
        timestamp = datetime.utcnow().timestamp()

        tic = time.time()

        pdfFullPath = calsolFile + '_flatplot_{}.pdf'.format(timestamp)
        nPlotsPerRow = 2
        nPlotsPerCol = 4
        iPlot = 0
        nPlotsPerPage = nPlotsPerRow * nPlotsPerCol
        if save_plots:
            matplotlib.rcParams['font.size'] = 4

        flat_cal = tables.open_file(calsolFile, mode='r')
        calsoln = flat_cal.root.flatcal.calsoln.read()
        bins = flat_cal.root.flatcal.wavelengthBins.read()
        minwavelength, maxwavelength = bins[0], bins[-1]

        import contextlib
        @contextlib.contextmanager
        def dummy_context_mgr():
            yield None

        with PdfPages(pdfFullPath) if save_plots else dummy_context_mgr() as pdf:
            for (row, column), resID in np.ndenumerate(self.beamImage):

                soln = calsoln[resID == calsoln['resid']]

                # if len(soln) > 1:
                #     msg = 'Flatcal {} has non-unique resIDs'.format(calsolFile)
                #     getLogger(__name__).critical(msg)
                #     raise RuntimeError(msg)

                if not len(soln) and not self.flagMask(pixelflags.PROBLEM_FLAGS, (x, y)):
                    getLogger(__name__).warning('No flat calibration for good pixel {}'.format(resID))
                    continue

                if np.any([sol['bad'] for sol in soln]):
                    getLogger(__name__).debug('No flat calibration bad for pixel {}'.format(resID))
                    continue

                #TODO set pixel flags to include flatcal flags and handle all the various pixel edge cases

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
                    if self.query_header('isWvlCalibrated') and not any([self.flagMask(pixelflags.PROBLEM_FLAGS,
                                                                               pixel=(row, column))]):
                        weightArr = np.poly1d(coeffs)(wavelengths)
                        if any(weightArr > 100) or any(weightArr < 0.01):
                            getLogger(__name__).debug('Unreasonable fitted weight of for resID {}'.format(resID))

                    elif not use_wavecal:
                        weighted_avg, sum_weight_arr = np.ma.average(weights, axis=0,
                                                                     weights=errors ** -2.,
                                                                     returned=True)  # should be a weighted average
                        weightArr = np.ones_like(wavelengths) * weighted_avg
                        if any(weightArr > 100) or any(weightArr < 0.01):
                            getLogger(__name__).debug('Unreasonable averaged weight for resID {}'.format(resID))

                    else:
                        assert use_wavecal
                        getLogger(__name__).debug('No wavecal for pixel with resID {} so no flatweight applied'.format(resID))
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

                if save_plots:  # TODO:plotting is inefficient, speed up, turn into single pixel plotting fxn maybe
                    if iPlot % nPlotsPerPage == 0:
                        fig = plt.figure(figsize=(10, 10), dpi=100)
                    ax = fig.add_subplot(nPlotsPerCol, nPlotsPerRow, iPlot % nPlotsPerPage + 1)
                    ax.set_ylim(0, 5)
                    ax.set_xlim(minwavelength, maxwavelength)
                    ax.plot(bins, weights, '-', label='weights')
                    ax.errorbar(bins, weights, yerr=errors, label='weights', fmt='o', color='green')
                    ax.plot(wavelengths, weightArr, '.', markersize=5)
                    ax.set_title('p rID:{} ({}, {})'.format(resID, row, column))
                    ax.set_ylabel('weight')

                    if iPlot % nPlotsPerPage == nPlotsPerPage - 1 or (
                            row == self.nXPix - 1 and column == self.nYPix - 1):
                        pdf.savefig()
                        plt.close()
                    iPlot += 1

        self.update_header('isFlatCalibrated', True)
        self.update_header('fltCalFile', calsolFile.encode())
        getLogger(__name__).info('Flatcal applied in {:.2f}s'.format(time.time()-tic))

    def applyLinearitycal(self, dt=1000, tau=0.000001):
        tic = time.time()
        if self.query_header('isLinearityCorrected'):
            getLogger(__name__).info("H5 {} is already linearity calibrated".format(self.fullFileName))
            return
        bar = ProgressBar(maxval=20439).start()
        bari = 0
        for (row, column), resID in np.ndenumerate(self.beamImage):
            if self.flagMask(pixelflags.PROBLEM_FLAGS, (row, column)) and any(pixelflags.PROBLEM_FLAGS):
                continue
            photon_list = self.getPixelPhotonList(xCoord=row, yCoord=column)
            time_stamps = photon_list['Time']
            weights = lincal.calculate_weights(time_stamps, dt, tau, pixel=(row, column))
            self.applySpecWeight(resID, weights)
            bari += 1
            bar.update(bari)
        bar.finish()
        getLogger(__name__).info('Linearitycal applied to {} in {:.2f}s'.format(self.fileName, time.time() - tic))
        self.update_header('isLinearityCorrected', True)

    def apply_spectralcal(self, spectralcal):
        """

        :param spectralcal: a spectralcal solution. must have id and response_curve attributes, the latter
         a 2xN columnss are wavelength in angstroms and the spectral response
        :return:
        """
        if self.query_header('isFluxCalibrated'):
            getLogger(__name__).info(f"{self.fullFileName} previously calibrated with {self.query_header('spectralcal')}, "
                                     f"skipping")
            return

        response_curve = spectralcal.response_curve
        # dont include nan or inf values
        ind = np.isfinite(response_curve.curve[1])

        getLogger(__name__).info('Applying {} to {}'.format(response_curve, self.fullFileName))

        coeffs = np.polyfit(response_curve.curve[0][ind]/10.0, response_curve.curve[1][ind], 3)
        func = np.poly1d(coeffs)
        tic = time.time()
        for (row, column), resID in np.ndenumerate(self.beamImage):
            if self.flagMask(pixelflags.PROBLEM_FLAGS, (row, column)) and any(pixelflags.PROBLEM_FLAGS):
                continue
            photon_list = self.getPixelPhotonList(xCoord=row, yCoord=column)
            weight_arr = func(photon_list['Wavelength'])
            self._apply_column_weight(resID, weight_arr, 'SpecWeight')

        self.update_header('isFluxCalibrated', True)
        self.update_header('spectralcal', spectralcal.id)

        getLogger(__name__).info('spectralcal applied in {:.2f}s'.format(time.time() - tic))


def calculateSlices(inter, timestamps):
    """
    Hopefully a quicker version of  the original calculateSlices. JvE 3/8/2013

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


def repackArray(array, slices):
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
