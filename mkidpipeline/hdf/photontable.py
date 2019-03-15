#!/bin/python
"""
Author: Matt Strader        Date: August 19, 2012
Modified 2017 for Darkness/MEC
Authors: Seth Meeker, Neelay Fruitwala, Alex Walter
Last Updated: April 16, 2018

The class ObsFile is an interface to observation files.  It provides methods 
for typical ways of accessing photon list observation data.  It can also load 
and apply wavelength and flat calibration.  


Class Obsfile:
====Helper functions====
__init__(self,fileName,mode='read',verbose=False)
__del__(self)
loadFile(self, fileName)
getFromHeader(self, name)
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
getSpectralCube(self, firstSec=0, integrationTime=-1, applySpecWeight=False, applyTPFWeight=False, wvlStart=700, wvlStop=1500,wvlBinWidth=None, energyBinWidth=None, wvlBinEdges=None, timeSpacingCut=None, flagToUse=0)
getPixelSpectrum(self, xCoord, yCoord, firstSec=0, integrationTime= -1,applySpecWeight=False, applyTPFWeight=False, wvlStart=None, wvlStop=None,wvlBinWidth=None, energyBinWidth=None, wvlBinEdges=None,timeSpacingCut=None)
makeWvlBins(energyBinWidth=.1, wvlStart=700, wvlStop=1500)

====Data write functions for calibrating====
applyWaveCal(self, file_name)
updateWavelengths(self, wvlCalArr, xCoord=None, yCoord=None, resid=None)
_applyColWeight(self, resID, weightArr, colName)
applySpecWeight(self, resID, weightArr)
applyTPFWeight(self, resID, weightArr)
applyFlatCal(self, calSolnPath,verbose=False)
applyFlag(self, xCoord, yCoord, flag)
undoFlag(self, xCoord, yCoord, flag)
modifyHeaderEntry(self, headerTitle, headerValue)





"""
import glob
import os
import warnings
import time
from datetime import datetime
import multiprocessing as mp

import astropy.constants
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tables
from interval import interval
from matplotlib.backends.backend_pdf import PdfPages
from regions import CirclePixelRegion, PixCoord

from mkidcore.headers import PhotonCType, PhotonNumpyType
from mkidcore.corelog import getLogger
import mkidcore.pixelflags as pixelflags
from mkidcore.pixelflags import h5FileFlags
from PyPDF2 import PdfFileMerger, PdfFileReader
import SharedArray

import tables.parameters
import tables.file
import mkidcore.utils

import functools

TBRERROR = RuntimeError('Function needs to be reviewed')


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
    f = ObsFile(file)
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
        f = ObsFile(file)
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

PLANK_CONSTANT_EV = astropy.constants.h.to('eV s').value
SPEED_OF_LIGHT_MS = astropy.constants.c.to('m/s').value


class ObsFile(object):
    h = PLANK_CONSTANT_EV # 4.135668e-15 #eV s
    c = SPEED_OF_LIGHT_MS  # '2.998e8 #m/s
    nCalCoeffs = 3
    tickDuration = 1e-6  # each integer value is 1 microsecond

    def __init__(self, fileName, mode='read', verbose=False):
        """
        Create ObsFile object and load in specified HDF5 file.

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
            ObsFile instance
                
        """
        self.mode = 'write' if mode.lower() in ('write', 'w', 'a', 'append') else 'read'
        self.verbose = verbose
        self.tickDuration = ObsFile.tickDuration
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
        self.titles = None
        self.info = None
        self.defaultWvlBins = None
        self.beamImage = None
        self.beamFlagImage = None
        self.nXPix = None
        self.nYPix = None
        self.loadFile(fileName)

    def __del__(self):
        """
        Closes the obs file and any cal files that are open
        """
        try:
            self.file.close()
        except:
            pass

    def enablewrite(self):
        """USE CARE IN A THREADED ENVIRONMENT"""
        if self.mode == 'write':
            return
        self.file.close()
        self.mode = 'write'
        self.loadFile(self.fullFileName)

    def disablewrite(self):
        """USE CARE IN A THREADED ENVIRONMENT"""
        if self.mode == 'read':
            return
        self.file.close()
        self.mode = 'read'
        self.loadFile(self.fullFileName)

    def loadFile(self, fileName):
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
        self.titles = self.header.colnames
        self.info = self.header[0]  # header is a table with one row

        # get important cal params
        self.defaultWvlBins = ObsFile.makeWvlBins(self.info['energyBinWidth'], self.info['wvlBinStart'],
                                                  self.info['wvlBinEnd'])
        # get the beam image.
        self.beamImage = self.file.get_node('/BeamMap/Map').read()
        self.beamFlagImage = self.file.get_node('/BeamMap/Flag')  #The absence of .read() here is correct
        self.nXPix, self.nYPix = self.beamImage.shape
        self.photonTable = self.file.get_node('/Photons/PhotonTable')

    def resIDs(self):
        return np.unique(self.beamImage.ravel())

    def getFromHeader(self, name):
        """
        Returns a requested entry from the obs file header
        eg. 'expTime'
        """
        entry = self.info[self.titles.index(name)]
        return entry

    def print(self):
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
                       start=t['Time'].min(), stop=t['Time'].max(), dur=self.info['expTime'],
                       dirty='Column(s) {} have dirty indices.'.format(dirty) if dirty else 'No columns dirty',
                       wave=self.info['wvlCalFile'], #self.info['isWvlCalibrated'] else 'None'
                       flat=self.info['fltCalFile'] if 'fltCalFile' in self.info.dtype.names else 'None')
        print(s)
        return s

    @property
    def pixelMask(self):
        """A boolean image with true where pixel data isn't perfect (i.e. any flag is set)"""
        return np.array(self.beamFlagImage) > 0

    def xy(self, photons):
        """Return a tuple of two arrays corresponding to the x & y pixel positions of the given photons"""
        flatbeam = self.beamImage.flatten()
        beamsorted = np.argsort(flatbeam)
        ind = np.searchsorted(flatbeam[beamsorted], photons["ResID"])
        return np.unravel_index(beamsorted[ind], self.beamImage.shape)

    def pixelIsBad(self, xCoord, yCoord, forceWvl=False, forceWeights=False, forceTPFWeights=False):
        """
        Returns True if the pixel wasn't read out or if a given calibration failed when needed
        
        Parameters
        ----------
        xCoord: int
        yCoord: int
        forceWvl: bool - If true, check that the wvlcalibration was good for this pixel
        forceWeights: bool - If true, check that the flat calibration was good
                           - Will also check if linearitycal, spectralcal worked correctly eventually?
        forceTPFWeights: bool - ignored for now
        
        Returns
        -------
            True if pixel is bad. Otherwise false
        """
        resID = self.beamImage[xCoord, yCoord]
        if resID == self.noResIDFlag:
            return True  # No resID was given during readout
        pixelFlags = self.beamFlagImage[xCoord, yCoord]
        deadFlags = h5FileFlags['noDacTone']
        if forceWvl and self.info['isWvlCalibrated']:
            deadFlags |= h5FileFlags['waveCalFailed']
        if forceWeights and self.info['isFlatCalibrated']:
            deadFlags |= h5FileFlags['flatCalFailed']
        # if forceWeights and self.info['isLinearityCorrected']:
        #   deadFlags+=h5FileFlags['linCalFailed']
        # if forceTPFWeights and self.info['isPhaseNoiseCorrected']:
        #   deadFlags+=h5FileFlags['phaseNoiseCalFailed']
        return (pixelFlags & deadFlags) > 0

    def query(self, startw=None, stopw=None, startt=None, stopt=None, resid=None, intt=None):
        """
        intt takes precedence, All none is a null result

        :param startw: number or none
        :param stopw: number or none
        :param startt: number or none
        :param endt: number or none
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
            return np.array([], dtype=[('ResID', '<u4'), ('Time', '<u4'), ('Wavelength', '<f4'),
                                       ('SpecWeight', '<f4'), ('NoiseWeight', '<f4')])
        else:
            tic = time.time()
            try:
                q = self.photonTable.read_where(query)
            except SyntaxError:
                raise
            toc = time.time()
            msg = 'Feteched {}/{} rows in {:.3f}s using indices {} for query {} \n\t st:{} et:{} sw:{} ew:{}'
            getLogger(__name__).debug(msg.format(len(q), len(self.photonTable), toc - tic,
                                                 tuple(self.photonTable.will_query_use_indexing(query)), query,
                                                 *map(lambda x: '{:.2f}'.format(x) if x is not None else 'None',
                                                      (startt, stopt, startw, stopw))))
            return q

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
            Desired start wavelength range (inclusive). Must satisfy wvlStart <= wvlStop.
            If None, includes all wavelengths less than wvlStop. If file is not wavelength calibrated, this parameter
            specifies the range of desired phase heights.
        wvlStop: float
            Desired stop wavelength range (non-inclusive). Must satisfy wvlStart <= wvlStop.
            If None, includes all wavelengths greater than wvlStart. If file is not wavelength calibrated, this parameter
            specifies the range of desired phase heights.
        forceRawPhase: bool
            If the ObsFile is not wavelength calibrated this flag does nothing.
            If the ObsFile is wavelength calibrated (ObsFile.info['isWvlCalibrated'] = True) then:
             - forceRawPhase=True will return all the photons in the list (might be phase heights instead of wavelengths)
             - forceRawPhase=False is guarenteed to only return properly wavelength calibrated photons in the photon list

        Returns
        -------
        Structured Numpy Array
            Each row is a photon.
            Columns have the following keys: 'Time', 'Wavelength', 'SpecWeight', 'NoiseWeight'

        Time is in microseconds
        Wavelength is in degrees of phase height or nanometers
        SpecWeight is a float in [0,1]
        NoiseWeight is a float in [0,1]

        """
        try:
            if wvlStop < wvlStart:
                raise ValueError('Invalid wavelength range')
        except TypeError:
            pass
        if firstSec is not None and firstSec > self.info['expTime']:
            raise ValueError('Start time not in file.')

        resid = self.beamImage[xCoord, yCoord] if resid is None else resid

        return self.query(startw=wvlStart, stopw=wvlStop, startt=firstSec if firstSec else None,
                          resid=resid, intt=None if integrationTime == -1 else integrationTime)

    def getPixelCount(self, x, y, applyWeight=True, applyTPFWeight=True, applyTimeMask=False, **kwargs):
        """
        Returns the number of photons received in a single pixel from firstSec to firstSec + integrationTime

        Parameters
        ----------
        *args: args from getPixelPhotonList 
            eg. xCoord, yCoord
        applyWeight: bool
            If True, applies the spectral/flat/linearity weight
        applyTPFWeight: bool
            If True, applies the true positive fraction (noise) weight
        applyTimeMask: bool
            If True, applies the included time mask (if it exists)
        **kwargs: keywords from getPixelPhotonList
            eg. firstSec, wvlStart

        Returns
        -------
        Dictionary with keys:
            'counts':int, number of photon counts
            'effIntTime':float, effective integration time after time-masking is
           `          accounted for.
        """
        raise RuntimeError("Clean up the arguemnts/kwarguments, bug Jeb if you aren't sure.")
        try:
            applyWvl = not kwargs['forceRawPhase']
        except KeyError:
            applyWvl = False
        try:
            xCoord = kwargs['xCoord']
        except KeyError:
            xCoord = x
        try:
            yCoord = kwargs['yCoord']
        except KeyError:
            yCoord = y
        if self.pixelIsBad(xCoord, yCoord, forceWvl=applyWvl, forceWeights=applyWeight, forceTPFWeights=applyTPFWeight):
            return 0, 0.0

        photonList = self.getPixelPhotonList(**kwargs)

        weights = np.ones(len(photonList))
        if applyWeight:
            weights *= photonList['SpecWeight']
        if applyTPFWeight:
            weights *= photonList['NoiseWeight']

        try:
            firstSec = kwargs['firstSec']
        except KeyError:
            firstSec = 0
        try:
            intTime = kwargs['integrationTime']
        except KeyError:
            intTime = self.info['expTime']
        intTime -= firstSec
        if applyTimeMask:
            raise NotImplementedError
            if self.info['timeMaskExists']:
                pass
                # Apply time mask to photon list
                # update intTime
            else:
                warnings.warn('Time mask does not exist!')

        return {'counts': np.sum(weights), 'effIntTime': intTime}

    def getPixelLightCurve(self, *args, lastSec=-1, cadence=1, scaleByEffInt=True, **kwargs):
        """
        Get a simple light curve for a pixel (basically a wrapper for getPixelCount).

        INPUTS:
            *args: args from getPixelCount
                 xCoord, yCoord
            lastSec: float
                 - end time (sec) within obsFile for the light curve. If -1, returns light curve to end of file.
            cadence: float
                 - cadence (sec) of light curve. i.e., return values integrated every 'cadence' seconds.
            scaleByEffInt: bool
                 - Scale by the pixel's effective integration time
            **kwargs: keywords for getPixelCount, including:
                firstSec
                applyWeight
                wvlStart
                Note: if integrationTime is given as a keyword it will be ignored


        OUTPUTS:
            A single one-dimensional array of flux counts integrated every 'cadence' seconds
            between firstSec and lastSec. Note if step is non-integer may return inconsistent
            number of values depending on rounding of last value in time step sequence (see
            documentation for numpy.arange() ).
        """
        if lastSec == -1:
            lastSec = self.info['expTime']
        firstSec = kwargs.get('firstSec', 0)
        if 'integrationTime' in kwargs.keys():
            warnings.warn("Integration time is being set to keyword 'cadence'")
        kwargs['integrationTime'] = cadence
        raise NotImplementedError('This function will not work as presently implemented. You have been assigned to '
                                  'fix it')
        data = [self.getPixelCount(*args, **kwargs) for x in range(firstSec, lastSec, cadence)]
        if scaleByEffInt:
            return np.asarray([1.0 * x['counts'] / x['effIntTime'] for x in data])
        else:
            return np.asarray([x['counts'] for x in data])

    # TODO standardize between applyWeight and applySpecWeight throughout file!
    def getPixelCountImage(self, firstSec=0, integrationTime=None, wvlStart=None, wvlStop=None, applyWeight=False,
                            applyTPFWeight=False, scaleByEffInt=False, flagToUse=0):
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
        """
        if integrationTime is None:
            integrationTime = self.info['expTime']

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
        for (x, y), resID in np.ndenumerate(self.beamImage):  # 4 % of the time
            if not (self.beamFlagImage[x, y] | flagToUse) == flagToUse:
                continue
            image[x, y] = hist[resIDs == resID]
        toc2 = time.time()
        getLogger(__name__).debug('Histogrammed data in {:.2f} s, reformatting in {:.2f}'.format(toc2 - tic,
                                                                                                 toc2 - toc))
        return {'image': image, 'effIntTime': effIntTime}

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

        """

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
            flag = self.beamFlagImage[coords[0], coords[1]]
            if (flag | flagToUse) != flagToUse:
                exactApertureMask[apertureMaskCoords[i, 0], apertureMaskCoords[i, 1]] = 0
                continue
            resID = self.beamImage[coords[0], coords[1]]
            raise RuntimeError('Update this to query all at the same time')
            pixPhotonList = self.getPixelPhotonList(coords[0], coords[1], firstSec, integrationTime, wvlStart, wvlStop)
            pixPhotonList['NoiseWeight'] *= exactApertureMask[apertureMaskCoords[i, 0], apertureMaskCoords[i, 1]]
            if photonList is None:
                photonList = pixPhotonList
            else:
                photonList = np.append(photonList, pixPhotonList)

        photonList = np.sort(photonList, order='Time')
        return photonList, exactApertureMask

    def _makePixelSpectrum(self, photonList, **kwargs):
        """
        Makes a histogram using the provided photon list
        """
        applySpecWeight = kwargs.pop('applySpecWeight', False)
        applyTPFWeight = kwargs.pop('applyTPFWeight', False)
        wvlStart = kwargs.pop('wvlStart', None)
        wvlStop = kwargs.pop('wvlStop', None)
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

        if applySpecWeight:
            weights *= photonList['SpecWeight']

        if applyTPFWeight:
            weights *= photonList['NoiseWeight']

        if (wvlBinWidth is None) and (energyBinWidth is None) and (
                wvlBinEdges is None):  # use default/flat cal supplied bins
            spectrum, wvlBinEdges = np.histogram(wvlList, bins=self.defaultWvlBins, weights=weights)

        else:  # use specified bins
            if applySpecWeight and self.info['isFlatCalibrated']:
                raise ValueError('Using flat cal, so flat cal bins must be used')
            elif wvlBinEdges is not None:
                assert wvlBinWidth is None and energyBinWidth is None, 'Histogram bins are overspecified!'
                spectrum, wvlBinEdges = np.histogram(wvlList, bins=wvlBinEdges, weights=weights)
            elif energyBinWidth is not None:
                assert wvlBinWidth is None, 'Cannot specify both wavelength and energy bin widths!'
                wvlBinEdges = ObsFile.makeWvlBins(energyBinWidth=energyBinWidth, wvlStart=wvlStart, wvlStop=wvlStop)
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

    def getSpectralCube(self, firstSec=0, integrationTime=None, applySpecWeight=False, applyTPFWeight=False,
                        wvlStart=700, wvlStop=1500, wvlBinWidth=None, energyBinWidth=None, wvlBinEdges=None,
                        flagToUse=0):
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
            wvlBinEdges = self.makeWvlBins(energyBinWidth=energyBinWidth, wvlStart=wvlStart, wvlStop=wvlStop)
        elif wvlBinWidth is not None:
            wvlBinEdges = np.linspace(wvlStart, wvlStop, num=int((wvlStop - wvlStart) / wvlBinWidth) + 1)
        else:
            wvlBinEdges = self.defaultWvlBins.size
        nWvlBins = wvlBinEdges.size - 1

        if integrationTime == -1 or integrationTime is None:
            integrationTime = self.info['expTime']

        cube = np.zeros((self.nXPix, self.nYPix, nWvlBins))

        # TODO Actually compute the effective integration time
        effIntTime = np.full((self.nXPix, self.nYPix), integrationTime)

        ## Retrieval rate is about 2.27Mphot/s for queries in the 100-200M photon range
        masterPhotonList = self.query(startt=firstSec if firstSec else None, intt=integrationTime)

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
        if applySpecWeight:
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
            if not (self.beamFlagImage[x, y] | flagToUse) == flagToUse:
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
        #     flag = self.beamFlagImage[xCoord, yCoord]
        #     #all the time
        #     photonList = masterPhotonList[resIDs == resID] if (flag | flagToUse) == flagToUse else emptyPhotonList
        #     x = self._makePixelSpectrum(photonList, applySpecWeight=applySpecWeight, applyTPFWeight=applyTPFWeight,
        #                                 wvlBinEdges=wvlBinEdges)
        #     cube2[xCoord, yCoord, :] = x['spectrum']
        #     rawCounts[xCoord, yCoord] = x['rawCounts']
        # toc = time.time()
        # getLogger(__name__).debug(('Cubed data in {:.2f} s using old'
        #                           ' approach. Cubes same {}').format(toc - tic, (cube==cube2).all()))
        return {'cube': cube, 'wvlBinEdges': wvlBinEdges, 'effIntTime': effIntTime}

    def getPixelSpectrum(self, xCoord, yCoord, firstSec=0, integrationTime=-1,
                         applySpecWeight=False, applyTPFWeight=False, wvlStart=None, wvlStop=None,
                         wvlBinWidth=None, energyBinWidth=None, wvlBinEdges=None, timeSpacingCut=None):
        """
        returns a spectral histogram of a given pixel integrated from firstSec to firstSec+integrationTime,
        and an array giving the cutoff wavelengths used to bin the wavelength values

        Wavelength Bin Specification:
        Depends on parameters: wvlStart, wvlStop, wvlBinWidth, energyBinWidth, wvlBinEdges.
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
        applySpecWeight: bool
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
        """

        photonList = self.getPixelPhotonList(xCoord, yCoord, firstSec=firstSec, integrationTime=integrationTime)
        return self._makePixelSpectrum(photonList, applySpecWeight=applySpecWeight,
                                       applyTPFWeight=applyTPFWeight, wvlStart=wvlStart, wvlStop=wvlStop,
                                       wvlBinWidth=wvlBinWidth, energyBinWidth=energyBinWidth,
                                       wvlBinEdges=wvlBinEdges, timeSpacingCut=timeSpacingCut)
        # else:
        #    return spectrum,wvlBinEdges

    def loadBeammapFile(self, beammapFileName):
        """
        Load an external beammap file in place of the obsfile's attached beammap
        Can be used to correct pixel location mistakes.
        
        Make sure the beamflag image is correctly transformed to the new beammap
        """
        raise NotImplementedError

    def write_bad_pixels(self, bad_pixel_mask):
        """
        Write the output hot-pixel time masks table to the obs file

        Required Input:
        :param bad_pixel_mask:    A 2D array of integers of the same shape as the input image, denoting locations
                                  of bad pixels and the reason they were flagged

        :return:
        Writes a 'bad pixel table' to an output h5 file titled 'badpixmask_--timestamp--.h5'.

        """
        badpixmask = self.file.create_group(self.file.root, 'badpixmap', 'Bad Pixel Map')
        tables.Array(badpixmask, 'badpixmap', obj=bad_pixel_mask,
                     title='Bad Pixel Mask')
        self.file.flush()
        self.file.close()

    def applyWaveCal(self, solution):
        """
        loads the wavelength cal coefficients from a given file and applies them to the
        wavelengths table for each pixel. ObsFile must be loaded in write mode. Dont call updateWavelengths !!!

        Note that run-times longer than ~330s for a full MEC dither (~70M photons, 8kpix) is a regression and
        something is wrong. -JB 2/19/19
        """
        # check file_name and status of obsFile
        if self.info['isWvlCalibrated']:
            getLogger(__name__).info('Data already calibrated using {}'.format(self.info['wvlCalFile']))
            return
        getLogger(__name__).info('Applying {} to {}'.format(solution, self.fullFileName))
        self.photonTable.autoindex = False  # Don't reindex every time we change column
        # apply waveCal
        tic = time.time()
        for (row, column), resID in np.ndenumerate(self.beamImage):

            if not solution.has_good_calibration_solution(res_id=resID):
                self.flag(pixelflags.h5FileFlags['waveCalFailed'], row, column)
                continue

            self.undoFlag(row, column, pixelflags.h5FileFlags['waveCalFailed'])
            calibration = solution.calibration_function(res_id=resID, wavelength_units=True)

            tic2 = time.time()
            indices = self.photonTable.get_where_list('ResID==resID')
            if not indices.size:
                continue
            if (np.diff(indices) == 1).all():  # This takes ~475s for ALL photons combined on a 70Mphot file.
                # getLogger(__name__).debug('Using modify_column')
                phase = self.photonTable.read(start=indices[0], stop=indices[-1] + 1, field='Wavelength')
                self.photonTable.modify_column(start=indices[0], stop=indices[-1] + 1, column=calibration(phase),
                                               colname='Wavelength')
            else:  # This takes 3.5s on a 70Mphot file!!!
                raise NotImplementedError('This code path is impractically slow at present.')
                getLogger(__name__).debug('Using modify_coordinates')
                rows = self.photonTable.read_coordinates(indices)
                rows['Wavelength'] = calibration(rows['Wavelength'])
                self.photonTable.modify_coordinates(indices, rows)
                getLogger(__name__).debug('Wavelength updated in {:.2f}s'.format(time.time() - tic2))

        self.modifyHeaderEntry(headerTitle='isWvlCalibrated', headerValue=True)
        self.modifyHeaderEntry(headerTitle='wvlCalFile', headerValue=str.encode(solution.name))
        self.photonTable.reindex_dirty()  # recompute "dirty" wavelength index
        self.photonTable.autoindex = True  # turn on auto-indexing
        self.photonTable.flush()
        getLogger(__name__).info('Wavecal applied in {:.2f}s'.format(time.time()-tic))

    @property
    def wavelength_calibrated(self):
        return self.info['isWvlCalibrated']

    @property
    def duration(self):
        return self.info['expTime']

    @staticmethod
    def makeWvlBins(energyBinWidth=.1, wvlStart=700, wvlStop=1500):
        """
        returns an array of wavlength bin edges, with a fixed energy bin width
        withing the limits given in wvlStart and wvlStop
        Args:
            energyBinWidth: bin width in eV
            wvlStart: Lower wavelength edge in Angstrom
            wvlStop: Upper wavelength edge in Angstrom
        Returns:
            an array of wavelength bin edges that can be used with numpy.histogram(bins=wvlBinEdges)
        """

        # Calculate upper and lower energy limits from wavelengths
        # Note that start and stop switch when going to energy
        energyStop = ObsFile.h * ObsFile.c * 1.e9 / wvlStart
        energyStart = ObsFile.h * ObsFile.c * 1.e9 / wvlStop
        nWvlBins = int((energyStop - energyStart) / energyBinWidth)
        # Construct energy bin edges
        energyBins = np.linspace(energyStart, energyStop, nWvlBins + 1)
        # Convert back to wavelength and reverse the order to get increasing wavelengths
        wvlBinEdges = np.array(ObsFile.h * ObsFile.c * 1.e9 / energyBins)
        wvlBinEdges = wvlBinEdges[::-1]
        return wvlBinEdges

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

    def setWvlCutoffs(self, wvlLowerLimit=700, wvlUpperLimit=1500):
        """
        Sets wavelength cutoffs so that if convertToWvl(excludeBad=True) or getPixelWvlList(excludeBad=True) is called
        wavelengths outside these limits are excluded.  To remove limits
        set wvlLowerLimit and/or wvlUpperLimit to None.  To use the wavecal limits
        for each individual pixel, set wvlLowerLimit and/or wvlUpperLimit to -1
        NB - changed defaults for lower/upper limits to None (from 3000 and 8000). JvE 2/22/13
        """
        self.wvlLowerLimit = wvlLowerLimit
        self.wvlUpperLimit = wvlUpperLimit

    def switchOffHotPixTimeMask(self):
        """
        Switch off hot pixel time masking - bad pixel times will no longer be
        removed (although the mask remains 'loaded' in ObsFile instance).
        """
        # self.hotPixIsApplied = False
        raise NotImplementedError

    def switchOnHotPixTimeMask(self, reasons=[]):
        """
        Switch on hot pixel time masking. Subsequent calls to getPixelCountImage
        etc. will have bad pixel times removed.
        """
        raise NotImplementedError
        if self.hotPixTimeMask is None:
            raise RuntimeError('No hot pixel file loaded')
        self.hotPixIsApplied = True
        if len(reasons) > 0:
            self.hotPixTimeMask.set_mask(reasons)

    def updateWavelengths(self, wvlCalArr, xCoord=None, yCoord=None, resID=None, flush=True):
        """
        Changes wavelengths for a single pixel. Overwrites "Wavelength" column w/
        contents of wvlCalArr. NOT reversible unless you have a copy of the original contents.
        ObsFile must be open in "write" mode to use.

        Parameters
        ----------
        resID: int
            resID of pixel to overwrite
        wvlCalArr: array of floats
            Array of calibrated wavelengths. Replaces "Wavelength" column of this pixel's
            photon list.
        """
        if resID is None:
            resID = self.beamImage[xCoord, yCoord]

        if self.mode != 'write':
            raise Exception("Must open file in write mode to do this!")
        if self.info['isWvlCalibrated']:
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

    def _applyColWeight(self, resID, weightArr, colName):
        """
        Applies a weight calibration to the column specified by colName.
        Call using applySpecWeight or applyTPFWeight.

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

        indices = self.photonTable.get_where_list('ResID==resID')

        if not (np.diff(indices) == 1).all():
            raise NotImplementedError('Table is not sorted by Res ID!')
        if len(indices) != len(weightArr):
            raise ValueError('weightArr length does not match length of photon list for resID!')

        newWeights = self.query(resid=resID)['SpecWeight'] * np.asarray(weightArr)
        self.photonTable.modify_column(start=indices[0], stop=indices[-1] + 1, column=newWeights, colname=colName)
        self.photonTable.flush()

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
        self._applyColWeight(resID, weightArr, 'SpecWeight')

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
        self._applyColWeight(resID, weightArr, 'NoiseWeight')

    def applyFlatCal(self, calsolFile, save_plots=False):
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
        """

        if self.info['isFlatCalibrated']:
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

                if len(soln) > 1:
                    msg = 'Flatcal {} has non-unique resIDs'.format(calsolFile)
                    getLogger(__name__).critical(msg)
                    raise RuntimeError(msg)

                if not len(soln) and self.beamFlagImage[row, column] == pixelflags.GOODPIXEL:
                    getLogger(__name__).warning('No flat calibration for good pixel {}'.format(resID))
                    continue

                if soln['bad']:
                    getLogger(__name__).debug('No flat calibration bad for pixel {}'.format(resID))
                    continue

                #TODO set pixel flags to include flatcal flags and handle all the various pixel edge cases

                indices = self.photonTable.get_where_list('ResID==resID')
                if not indices.size:
                    continue

                if (np.diff(indices) == 1).all():  # This takes ~300s for ALL photons combined on a 70Mphot file.
                    # getLogger(__name__).debug('Using modify_column')
                    phases = self.photonTable.read(start=indices[0], stop=indices[-1] + 1, field='Wavelength')

                    coeffs=soln['coeff'].flatten()
                    weights = soln['weight'].flatten()
                    errors = soln['err'].flatten()
                    weightArr = np.poly1d(coeffs)(phases)
                    self.photonTable.modify_column(start=indices[0], stop=indices[-1] + 1, column=weightArr,
                                                   colname='SpecWeight')
                else:  # This takes 3.5s on a 70Mphot file!!!
                    raise NotImplementedError('This code path is impractically slow at present.')
                    getLogger(__name__).debug('Using modify_coordinates')
                    rows = self.photonTable.read_coordinates(indices)
                    rows['SpecWeight'] = np.poly1d(coeffs)(rows['Wavelength'])
                    self.photonTable.modify_coordinates(indices, rows)
                    getLogger(__name__).debug('Flat weights updated in {:.2f}s'.format(time.time() - tic2))

                if save_plots:  #TODO:  plotting is inefficient, speed up, turn into single pixel plotting fxn maybe
                    if iPlot % nPlotsPerPage == 0:
                        fig = plt.figure(figsize=(10, 10), dpi=100)
                    ax = fig.add_subplot(nPlotsPerCol, nPlotsPerRow, iPlot % nPlotsPerPage + 1)
                    ax.set_ylim(0, 5)
                    ax.set_xlim(minwavelength, maxwavelength)
                    ax.plot(bins, weights, '-', label='weights')
                    ax.errorbar(bins, weights, yerr=errors, label='weights', fmt='o', color='green')
                    ax.plot(phases, weightArr, '.', markersize=5)
                    ax.set_title('p rID:{} ({}, {})'.format(resID, row, column))
                    ax.set_ylabel('weight')

                    if iPlot % nPlotsPerPage == nPlotsPerPage - 1 or (
                            row == self.nXPix - 1 and column == self.nYPix - 1):
                        pdf.savefig()
                        plt.close()
                    iPlot += 1

        self.modifyHeaderEntry(headerTitle='isFlatCalibrated', headerValue=True)
        #TODO add this line whe the metadata store of the hdf file is sorted out.
        # self.modifyHeaderEntry(headerTitle='fltCalFile', headerValue=str.encode(calsolFile))

        getLogger(__name__).info('Flatcal applied in {:.2f}s'.format(time.time()-tic))

    def flag(self, flag, xCoord=slice(None), yCoord=slice(None)):
        """
        Applies a flag to the selected pixel on the BeamFlag array. Flag is a bitmask;
        new flag is bitwise OR between current flag and provided flag. Flag definitions
        can be found in Headers/pipelineFlags.py.

        Parameters
        ----------
        xCoord: int
            x-coordinate of pixel or a slice
        yCoord: int
            y-coordinate of pixel or a slice
        flag: int
            Flag to apply to pixel
        """
        if self.mode != 'write':
            raise Exception("Must open file in write mode to do this!")
        flag = np.asarray(flag)
        pixelflags.valid(flag, error=True)
        if not np.isscalar(flag) and self.beamFlagImage[xCoord, yCoord].shape != flag.shape:
            raise RuntimeError('flag must be scalar or match the desired region selected by x & y coordinates')
        self.beamFlagImage[xCoord, yCoord] |= flag
        self.beamFlagImage.flush()

    def undoFlag(self, xCoord, yCoord, flag):
        """
        Resets the specified flag in the BeamFlag array to 0. Flag is a bitmask;
        only the bit specified by 'flag' is reset. Flag definitions
        can be found in Headers/pipelineFlags.py.

        Parameters
        ----------
        xCoord: int
            x-coordinate of pixel
        yCoord: int
            y-coordinate of pixel
        flag: int
            Flag to undo
        """
        if self.mode != 'write':
            raise Exception("Must open file in write mode to do this!")
        self.beamFlagImage[xCoord, yCoord] &= ~flag
        self.beamFlagImage.flush()

    def modifyHeaderEntry(self, headerTitle, headerValue):
        """
        Modifies an entry in the header. Useful for indicating whether wavelength cals,
        flat cals, etc are applied

        Parameters
        ----------
        headerTitle: string
            Name of entry to be modified
        headerValue: depends on title
            New value of entry
        """
        if self.mode != 'write':
            raise IOError("Must open file in write mode to do this!")
        self.header.modify_column(column=headerValue, colname=headerTitle)
        self.header.flush()
        self.info = self.header[0]


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


# Temporary test
if __name__ == "__main__":
    pass
