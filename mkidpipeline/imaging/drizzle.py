"""
TODO
Add ephem and pyguide to yml
Add compiling of boxer.f to setup.py

Usage
-----

python drizzle 'Trapezium/goodfiles/'

Ephem installed with pip install ephem
PyGuide is just downloaded from github (no installation

Author: Rupert Dodkins, Julian van Eyken            Date: Jan 2019

Reads in obsfiles for dither offsets (preferably fully reduced and calibrated) as well as a dither log file and creates
a stacked image.

This code is adapted from Julian's testImageStack from the ARCONS pipeline.
"""

import os
import numpy as np
import time
import mkidpipeline.imaging.RADecImage as rdi
from mkidpipeline.hdf.photontable import ObsFile
from mkidcore.corelog import getLogger
from mkidpipeline.config import MKIDObservingDataDescription, MKIDObservingDither
from astropy.io import fits
from datetime import datetime


def ditherp_2_pixel(positions):
    """ A function to convert the connex offset to pixel displacement"""  #
    con2pix =np.array([[-20, 20], [20,-20]])
    conPos = np.array(list(zip(*positions)))
    return np.int_(np.matmul(conPos.T, con2pix)).T


def form(observations, *args, cfg=None, **kwargs):
    """
    Form a (possibly drizzled) image from a set of observations. Forms a simple image for single observations.

    TODO Sort out what arguments and kw arguments are needed
    """
    try:
        iter(observations)
    except:
        observations = tuple(observations)


    #Determine what we are drizzling onto
    #e.g. 4d equivalent (sky_x, sky_y, wave, time) of an rdi.RADecImage


    for obs in observations:
        if isinstance(obs, MKIDObservingDataDescription):
            #TODO formImage on a single obs per
            raise NotImplementedError
        elif isinstance(obs, MKIDObservingDither):
            #TODO form from a dither
            drizzleDither(obs, *args, **kwargs)
        else:
            #TODO do we need to enable drizzling lists of arbitrary observations?
            raise ValueError('Unknown input')


def drizzleDither(dither, *args, **kwargs):
    """Form a drizzled image from a dither"""
    drizzleddata = drizzle([ObsFile(get_h5_filename(ob)) for ob in dither], **kwargs)
    drizzleddata.save(dither.name)


class DitherDescription(object):
    def __init__(self, mkid_observing_dither):
        self.description = mkid_observing_dither
        #TODO implement. Should take the place of ditherdict and more
        pixel_x, pixel_y = ditherp_2_pixel(mkid_observing_dither.pos)


class Drizzler(object):
    def __init__(self):
        self.gridRA = None
        self.gridDec = None

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


class SpectralDrizzler(Drizzler):
    pass


class TemporalDrizzler(Drizzler):
    pass


class SpatialDrizzler(Drizzler):
    """ Generate a spatially dithered fits image from a set dithered dataset """
    def __init__(self, obsfiles, metadata):
        #TODO Implement
        # Assume obsfiles either have their metadata or needed metadata is passed, e.g. WCS information, target info,
        # etc

        self.config = None
        self.files = obsfiles

        #TODO determine appropirate value from areal coverage of dataset and oversampling, even longerterm there
        # the oversampling should be selected to optimize total phase coverage to extract the most resolution at a
        # desired minimum S/N
        self.nPixRA = 250
        self.nPixDec = 250

        self.cenRA = metadata['cenRA']
        self.cenDec = metadata['cenDec']

        self.vPlateScale = 1
        self.detPlateScale = 1
        self.plateScale = 0.44

        self.vPlateScale = self.vPlateScale * 2 * np.pi / 1296000  # No. of radians on sky per virtual pixel.
        self.detPlateScale = self.detPlateScale * 2 * np.pi / 1296000

        self.generate_coordinate_grid()

    def run(self, save_file='out.fits'):

        wvlMin=-200
        wvlMax=-150
        doWeighted=False
        medCombine=False
        maxBadPixTimeFrac=None
        integrationTime=5

        dither = DitherDescription(self.meta.dither)

        # Initialise empty image
        # TODO use the Dither
        virtualImage = rdi.RADecImage(nPixRA=self.nPixRA, nPixDec=self.nPixDec, vPlateScale=self.vPlateScale,
                                      wcs_out=self.wcs, dither=dither)
        imageStack = []
        for ix, file in enumerate(self.files):
            getLogger(__name__).debug('Processing %s', file)
            obsfile = ObsFile(file)

            tic = time.clock()
            photons = virtualImage.loadObsFile(obsfile, ditherInd=ix, wvlMin=wvlMin, wvlMax=wvlMax,
                                               doWeighted=doWeighted, maxBadPixTimeFrac=maxBadPixTimeFrac,
                                               integrationTime=integrationTime)
            virtualImage.stackExposure(photons, ditherInd=ix, doStack=not medCombine, savePreStackImage=None)
            getLogger(__name__).debug('Image load done. Time taken (s): %s', time.clock() - tic)
            # Only makes sense if medCombine==True, otherwise will be ignored
            imageStack.append(virtualImage.image * virtualImage.expTimeWeights)

        # Save the results.
        results = {'vim': virtualImage, 'imstack': imageStack}

        effective_timestamp = self.meta.time+self.meta.total_exp_time/2  #TODO this is only approximate

        ret = fits.ImageHDU(data=virtualImage.image)
        ret.header['imgname'] = save_file
        ret.header['utc'] = datetime.utcfromtimestamp(effective_timestamp).strftime('%Y-%m-%d %H:%M:%S')
        ret.header['exptime'] = self.meta.total_exp_time
        hdul = fits.HDUList([fits.PrimaryHDU(), ret])
        hdul.writeto(os.path.join(self.config.paths.out, save_file))

        return results


def loadDitherCfg(fileName):
    logPath = os.getenv('MKID_PROC_PATH',default="/Scratch") + 'Trapezium/'
    log = os.path.join(logPath,fileName)
    ditherDict = {}
    with open(log) as f:
        XPIX = 140
        YPIX = 146
        binPath = '/mnt/data0/ScienceData/Subaru/20190112/'
        outPath = '/mnt/data0/isabel/highcontrastimaging/Jan2019Run/20190112/Trapezium/'
        beamFile = "/mnt/data0/MEC/20190111/finalMap_20181218.bmap"
        mapFlag = 1
        filePrefix = 'a'
        b2hPath = '/home/isabel/src/mkidpipeline/mkidpipeline/hdf'
        ditherDict['startTimes'] = np.int_(f.readline()[14:-2].split(','))
        ditherDict['endTimes'] = np.int_(f.readline()[12:-2].split(','))
        ditherDict['xPos'] = np.float_(f.readline()[8:-3].split(','))
        ditherDict['yPos'] = np.float_(f.readline()[8:-3].split(','))
        ditherDict['intTime'] = np.float(f.readline()[10:])
        ditherDict['nSteps'] = np.int(f.readline()[9:])

    firstSec = ditherDict['startTimes'][0]
    ditherDict['relStartTimes'] = ditherDict['startTimes'] - firstSec
    ditherDict['relEndTimes'] = ditherDict['endTimes'] - firstSec

    return ditherDict

if __name__ == '__main__':
    raise NotImplementedError()
