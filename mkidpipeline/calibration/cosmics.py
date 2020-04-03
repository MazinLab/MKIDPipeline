"""
Cosmic Cleaner is a class that will find the cosmic correction and will create a file that
has the associated ObsFile name (beginning timestamp) as well as a list of the timestamps
to be removed plus some metadata, such as the cutout times before and after the cosmic ray
event, the method used for peak finding (more on that in the code itself), a set of data for
creating a histogram of bincounts, and more as needed. This docstring will be edited as code
is updated and refined.
"""

import numpy as np
from mkidpipeline.hdf.photontable import ObsFile
import argparse
from scipy.stats import poisson

MEC_LASER_CAL_WAVELENGTH_RANGE = [850, 1375]
DARKNESS_LASER_CAL_WAVELENGTH_RANGE = [808, 1310]

class CosmicCleaner(object):
    def __init__(self, file, instrument=None, wavelengthCut=False, ):
        self.instrument = instrument if instrument is not None else "MEC"
        self.obs = ObsFile(file)
        self.wavelengthCut = wavelengthCut
        self.allphotons = None
        self.photons = None
        self.photonMask = None
        self.arrivaltimes = None
        self.timebins = None
        self.timestream = None
        self.trimmedtimestream = None
        self.arraycounts = None
        self.countsperbin = None
        self.countoccurrences = None
        self.pdf = None
        self.cutouttimes = None

    def get_photon_list(self):
        '''
        Reads the photon list from the ObsFile, also creates a mask if wavelengthCut=True and
        applies it. Modifies self.allphotons and self.photons. If wavelengthCut=False
        self.allphotons=self.photons. If wavelengthCut=True, all photons outside the laser calibrated
        wavelengths will be removed from self.photons, self.allphotons will remain unchanged.
        '''
        self.allphotons = self.obs.photonTable.read()
        if self.wavelengthCut:
            if self.instrument.upper() == "MEC":
                self.photonMask = (self.allphotons > MEC_LASER_CAL_WAVELENGTH_RANGE[0]) & \
                                  (self.allphotons < MEC_LASER_CAL_WAVELENGTH_RANGE[1])
                self.photons = self.allphotons[self.photonMask]
            elif self.instrument.upper() == "DARKNESS":
                self.photonMask = (self.allphotons > DARKNESS_LASER_CAL_WAVELENGTH_RANGE[0]) & \
                                  (self.allphotons < DARKNESS_LASER_CAL_WAVELENGTH_RANGE[1])
                self.photons = self.allphotons[self.photonMask]
            elif (self.instrument.upper() == "BLACKFRIDGE") or (self.instrument.upper() == "BF"):
                self.photonMask = (self.allphotons > DARKNESS_LASER_CAL_WAVELENGTH_RANGE[0]) & \
                                  (self.allphotons < DARKNESS_LASER_CAL_WAVELENGTH_RANGE[1])
                self.photons = self.allphotons[self.photonMask]
            else:
                print(f"WARNING: {self.instrument} is not a recognized instrument. No cut applied")
        else:
            self.photons = self.allphotons

    def get_time_info(self):
        '''
        Extracts the timestamps of all the remaining photons and generates the bin edges to make the
        photon timestream (counts over array vs. time)
        '''
        self.arrivaltimes = self.photons['Time']
        self.timebins = np.arange(0, int(np.ceil(self.arrivaltimes.max() / 10) * 10) + 10, 10)

    def make_timestream(self):
        self.arraycounts, timebins = np.histogram(self.arrivaltimes, self.timebins)
        assert np.setdiff1d(timebins, self.timebins).size == 0
        self.timestream = np.array((self.arraycounts, self.timebins[:-1]))

    def make_count_histogram(self):
        unique, counts = np.unique(self.arraycounts, return_counts=True)
        self.countsperbin = unique
        self.countoccurrences = counts

    def find_cutout_times(self):
        if self.wavelengthCut:
            pdf = poisson.pmf(self.countsperbin, np.average(self.arraycounts))
            mask = pdf <= 1 / np.sum(self.arraycounts)
            pdf[mask] = 1 / np.sum(self.arraycounts)
            self.pdf = pdf
            maxcounts = self.pdf[~mask].max()
            cutmask = self.arraycounts >= maxcounts
            # newcounts = self.arraycounts[~cutmask]
            # newtimes = self.timebins[:-1][~cutmask]
            # self.trimmedtimestream = np.array((newcounts, newtimes))

            cosmicraytimes = self.timebins[:-1][cutmask]
            cutouttimes = np.array([np.arange(i-self.removalrange[0],
                                              i+self.removalrange[1], 1) for i in cosmicraytimes]).flatten()
            cutouttimes = list(dict.fromkeys(cutouttimes))
            self.cutouttimes = cutouttimes
        else:
            #Do Clarissa's method