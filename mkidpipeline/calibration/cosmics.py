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
BF_LASER_CAL_WAVELENGTH_RANGE = DARKNESS_LASER_CAL_WAVELENGTH_RANGE

class CosmicCleaner(object):
    def __init__(self, file, instrument=None, wavelengthCut=False, removalRange=[-50,100]):
        self.instrument = instrument if instrument is not None else "MEC"
        self.obs = ObsFile(file)
        self.wavelengthCut = wavelengthCut  # Boolean that encodes the decision of which removal method to use:
                                            # peak finding (no cut) or poisson statistics of arrival times (cut)
        self.removalRange = removalRange
        self.allphotons = None  # Master photon list from the ObsFile, should not be modified at any point
        self.photons = None # The photon list used for the cosmic ray removal. Not modified from allphotons if no cut
        self.photonMask = None # The mask used to remove photons from the master list (e.g. wvl outside of allowed range)
        self.arrivaltimes = None  # Arrival times from photons in self.photons, taken from self.photons for convenience
        self.arraycounts = None  # Number of counts across the array for each time bin
        self.timebins = None  # Edges of the time bins (microseconds) used to bin photon arrival times and generate
                              # a photon timestream for the file. This is a property rather than an immediate step for
                              # eventual plotting tools and use between functions
        self.timestream = None  # A (2,N) array where N is the number of time bins and the elements are a timestamp and
                                # the number of counts across the array in that bin
        self.trimmedtimestream = None # Same as self.timestream, but with cosmic rays removed, again for plotting
        self.countsperbin = None  # The data which will form the x-axis in a histogram of frequency of counts per bin
                                  # in the ObsFile photon timestream
        self.countoccurrences = None  # The 'y-values' of the counts per bin histogram. If using Poisson statistics,
                                      # this will be used to calculate the PDF needed to get a threshold value
        self.pdf = None # The 'y-values' of the Poisson PDF, if used
        self.cutouttimes = None  # A list of all the timestamps affected by a cosmic ray event. This is the ultimate
                                 # information that is needed to clean the ObsFile cosmic rays and remove the
                                 # offending photons.

    def get_photon_list(self):
        '''
        Reads the photon list from the ObsFile, also creates a mask if wavelengthCut=True and
        applies it. Modifies self.allphotons and self.photons. If wavelengthCut=False
        self.allphotons=self.photons. If wavelengthCut=True, all photons outside the laser calibrated
        wavelengths will be removed from self.photons, self.allphotons will remain unchanged.
        '''
        self.allphotons = self.obs.photonTable.read()
        if self.wavelengthCut:
            # With the wavelength cut, this step creates a mask and applies it so that we only keep photons within the
            # laser calibrated wavelength range. At this point the black fridge and darkness laser boxes are the same,
            # and therefore have the same range. Any future instruments can be added to this by adding a dictionary
            # with their info at the top of the file.
            if self.instrument.upper() == "MEC":
                self.photonMask = (self.allphotons > MEC_LASER_CAL_WAVELENGTH_RANGE[0]) & \
                                  (self.allphotons < MEC_LASER_CAL_WAVELENGTH_RANGE[1])
                self.photons = self.allphotons[self.photonMask]
            elif self.instrument.upper() == "DARKNESS":
                self.photonMask = (self.allphotons > DARKNESS_LASER_CAL_WAVELENGTH_RANGE[0]) & \
                                  (self.allphotons < DARKNESS_LASER_CAL_WAVELENGTH_RANGE[1])
                self.photons = self.allphotons[self.photonMask]
            elif (self.instrument.upper() == "BLACKFRIDGE") or (self.instrument.upper() == "BF"):
                self.photonMask = (self.allphotons > BF_LASER_CAL_WAVELENGTH_RANGE[0]) & \
                                  (self.allphotons < BF_LASER_CAL_WAVELENGTH_RANGE[1])
                self.photons = self.allphotons[self.photonMask]
            else:
                print(f"WARNING: {self.instrument} is not a recognized instrument. No cut applied")
        else:
            # If there is no wavelength cut desired, that means the peak-finding algorithm will be used as opposed to
            # leveraging the poisson statistics when 'extrapolated-wl' photons are removed.
            self.photons = self.allphotons

    def get_time_info(self):
        '''
        Extracts the timestamps of all the remaining photons and generates the bin edges to make the
        photon timestream (counts over array vs. time)
        '''
        self.arrivaltimes = self.photons['Time']
        self.timebins = np.arange(0, int(np.ceil(self.arrivaltimes.max() / 10) * 10) + 10, 10)

    def make_timestream(self):
        '''
        Uses the numpy.histogram function to bin all of the photons into 10 microsecond bins. Then creates the
        self.timestream attribute so that we can plot the array counts vs. time. We chop off the last entry of the
        time bins in the timestream because that is the end of the final bin, and we are plotting these events based on
        the start times of each bin. We could also plot the midpoint of each bin, but because the removal range is 15
        times larger than the bin size, it ultimately won't matter.
        '''
        self.arraycounts, timebins = np.histogram(self.arrivaltimes, self.timebins)
        assert np.setdiff1d(timebins, self.timebins).size == 0
        self.timestream = np.array((self.arraycounts, self.timebins[:-1]))

    def make_count_histogram(self):
        '''
        This function will use the numpy.unique function on the self.arraycounts attribute to return 2 lists, the first
        will be all of the different values that occur in the self.arraycounts array, the second will be the corresponding
        number of times that each of those values appear. This will allow us to create a histogram for data visualization
        as well as create the Poisson PDF to determine the cosmic ray threshold value.
        '''
        unique, counts = np.unique(self.arraycounts, return_counts=True)
        self.countsperbin = unique
        self.countoccurrences = counts

    def find_cutout_times(self):
        '''
        The meat of the cosmic ray cleaning code.
        In the wavelength cut mode, a Poisson pdf is generated where the
        expected value is the mean number of counts from the time stream from the observation (empirically, the number
        of cosmic ray events in a given observation doesn't drastically change the value of the mean). This pdf
        is used to find the threshold of counts across the array that we will define as a cosmic ray. Once the threshold
        is found, the timestamps of bins with counts higher than the threshold are found and stored in a list. After
        that list is generated the full list of timestamps to be removed is made and the trimmedtimesteam attribute is
        saved, again for diagnostic/recordkeeping purposes.
        In the wavelength not cut mode ...
        TODO: Implement the original peak-finding algorithm from Clarissa's code
        '''
        if self.wavelengthCut:
            pdf = poisson.pmf(self.countsperbin, np.average(self.arraycounts))
            mask = pdf <= 1 / np.sum(self.arraycounts)
            pdf[mask] = 1 / np.sum(self.arraycounts)
            self.pdf = pdf
            maxcounts = self.pdf[~mask].max()
            cutmask = self.arraycounts >= maxcounts

            cosmicraytimes = self.timebins[:-1][cutmask]
            cutouttimes = np.array([np.arange(i-self.removalrange[0],
                                              i+self.removalrange[1], 1) for i in cosmicraytimes]).flatten()
            cutouttimes = np.array(list(dict.fromkeys(cutouttimes)))
            self.cutouttimes = cutouttimes

            trimmask = np.in1d(self.timebins[:-1], cutouttimes)
            trimmedcounts = self.arraycounts[trimmask]
            trimmedtimes = self.timebins[:-1][trimmask]
            self.trimmedtimestream = np.array((trimmedcounts,trimmedtimes))
        else:
            #Do Clarissa's method