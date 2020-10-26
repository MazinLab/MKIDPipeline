"""
Author: Clarissa Rizzo, Isabel Lipartito, Noah Swimmer
Cosmic Cleaner is a class that will find the cosmic correction and will create a file that
has the associated ObsFile name (beginning timestamp) as well as a list of the timestamps
to be removed plus some metadata, such as the cutout times before and after the cosmic ray
event, the method used for peak finding (more on that in the code itself), a set of data for
creating a histogram of bincounts, and more as needed. This docstring will be edited as code
is updated and refined.
TODO: logging
TODO: Integrate into pipeline
TODO: Create performance report, such as number of CR events, amount of time removed, time intervals removed, etc.
TODO: Remove CR photons from table
"""

import numpy as np
import os
from mkidpipeline.hdf.photontable import ObsFile
import argparse
from scipy.stats import poisson
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import mkidcore.corelog as pipelinelog
from datetime import datetime

MEC_LASER_CAL_WAVELENGTH_RANGE = [850, 1375]
DARKNESS_LASER_CAL_WAVELENGTH_RANGE = [808, 1310]
BF_LASER_CAL_WAVELENGTH_RANGE = DARKNESS_LASER_CAL_WAVELENGTH_RANGE

log = pipelinelog.getLogger('mkidpipeline.calibration.cosmiccal', setup=False)


def setup_logging(tologfile='', toconsole=True, time_stamp=None):
    """
    Set up logging for the wavelength calibration module for running from the command line.
    Args:
        tologfile: directory where the logs folder will be placed. If empty, no logfile is made.
        toconsole: boolean specifying if the console log is set up.
        time_stamp: utc time stamp to name the log file.
    """
    if time_stamp is None:
        time_stamp = int(datetime.utcnow().timestamp())
    if toconsole:
        log_format = "%(levelname)s : %(message)s"
        pipelinelog.create_log('mkidpipeline', console=True, fmt=log_format, level="INFO")

    if tologfile:
        log_directory = os.path.join(tologfile, 'logs')
        log_file = os.path.join(log_directory, '{:.0f}.log'.format(time_stamp))
        log_format = '%(asctime)s : %(funcName)s : %(levelname)s : %(message)s'
        pipelinelog.create_log('mkidpipeline', logfile=log_file, console=False, fmt=log_format, level="DEBUG")


class CosmicCleaner(object):
    def __init__(self, file, instrument=None, wavelengthCut=True, method="poisson", removalRange=(50, 100)):
        self.instrument = str(instrument) if instrument is not None else "MEC"
        self.obs = ObsFile(file)
        self.wavelengthCut = wavelengthCut
        self.method = method
        self.removalRange = removalRange
        self.allphotons = None  # Master photon list from the ObsFile, should not be modified at any point
        self.photons = None  # The photon list used for the cosmic ray removal. Not modified from allphotons if no cut
        self.photonMask = None  # The mask used to remove photons from the master list (e.g. wvl outside of allowed range)
        self.arrivaltimes = None  # Arrival times from photons in self.photons, taken from self.photons for convenience
        self.arraycounts = None  # Number of counts across the array for each time bin
        self.timebins = None  # Edges of the time bins (microseconds) used to bin photon arrival times and generate
        # a photon timestream for the file. This is a property rather than an immediate step for
        # eventual plotting tools and use between functions
        self.timestream = None  # A (2,N) array where N is the number of time bins and the elements are a timestamp and
        # the number of counts across the array in that bin
        self.trimmedtimestream = None  # Same as self.timestream, but with cosmic rays removed, again for plotting
        self.countsperbin = None  # The data which will form the x-axis in a histogram of frequency of counts per bin
        # in the ObsFile photon timestream
        self.countoccurrences = None  # The 'y-values' of the counts per bin histogram. If using Poisson statistics,
        # this will be used to calculate the PDF needed to get a threshold value
        self.countshistogram = None  # A container for the previous two attributes, for plotting/reporting ease.
        self.pdf = None  # The 'y-values' of the Poisson PDF, if used
        self.threshold = None  # The minimum number of counts per bin which will be able to be identified as a cosmic
        # ray event. If self.method='peak-finding', not all bins with values greater than this will necessarily be
        # marked as cosmic rays, although all bins with values greater than this should still be removed (due to their
        # proximity to the cosmic ray event.
        self.cosmictimes = None  # A list of all the timestamps of cosmic ray events. In the "poisson" method, there
        # may be multiple timestamps corresponding to the same peak, as this shows all bins that have more counts than
        # the calculated threshold. In "peak-finding method" this is not the case, and it should find the timestamp of
        # the peak of the cosmic ray, allowing the code to generate the times around it to cut out.
        self.cosmicpeaktimes = None  # A list of all of the timestamps of the peaks of cosmic ray events. This is needed
        # as single cosmic ray events can have sequential bins over the calculated threshold. This picks out the
        # timestamp of the highest number of counts in each cosmic ray.
        self.cutouttimes = None  # A list of all the timestamps affected by a cosmic ray event. This is the ultimate
        # information that is needed to clean the ObsFile cosmic rays and remove the
        # offending photons.

        if not self.wavelengthCut:
            log.warning("With the increased number of photons, cosmic ray removal may take significantly longer!")
            if self.method.lower() == "poisson":
                log.warning("The poisson method is not optimized for non-wavelength "
                            "cut data and may remove more time than desired!")

    def run(self):
        start = datetime.utcnow().timestamp()
        log.debug(f"Cosmic ray cleaning of {self.obs.fileName} began at {start}")
        self.get_photon_list()
        self.make_timestream()
        self.make_count_histogram()
        self.generate_poisson_pdf()
        self.find_cosmic_times()
        self.find_cutout_times()
        self.trim_timestream()
        end = datetime.utcnow().timestamp()
        log.debug(f"Cosmic ray cleaning of {self.obs.fileName} finished at {end}")
        log.info(f"Cosmic ray cleaning of {self.obs.fileName}took {end - start} s")

    def get_photon_list(self):
        """
        Reads the photon list from the ObsFile, also creates a mask if wavelengthCut=True and
        applies it. Modifies self.allphotons and self.photons. If wavelengthCut=False
        self.allphotons=self.photons. If wavelengthCut=True, all photons outside the laser calibrated
        wavelengths will be removed from self.photons, self.allphotons will remain unchanged.
        """
        self.allphotons = self.obs.photonTable.read()
        if self.wavelengthCut:
            # With the wavelength cut, this step creates a mask and applies it so that we only keep photons within the
            # laser calibrated wavelength range. At this point the black fridge and darkness laser boxes are the same,
            # and therefore have the same range. Any future instruments can be added to this by adding a dictionary
            # with their info at the top of the file.
            if self.instrument.upper() == "MEC":
                self.photonMask = (self.allphotons['Wavelength'] >= MEC_LASER_CAL_WAVELENGTH_RANGE[0]) & \
                                  (self.allphotons['Wavelength'] <= MEC_LASER_CAL_WAVELENGTH_RANGE[1])
                self.photons = self.allphotons[self.photonMask]
            elif self.instrument.upper() == "DARKNESS":
                self.photonMask = (self.allphotons['Wavelength'] >= DARKNESS_LASER_CAL_WAVELENGTH_RANGE[0]) & \
                                  (self.allphotons['Wavelength'] <= DARKNESS_LASER_CAL_WAVELENGTH_RANGE[1])
                self.photons = self.allphotons[self.photonMask]
            elif (self.instrument.upper() == "BLACKFRIDGE") or (self.instrument.upper() == "BF"):
                self.photonMask = (self.allphotons['Wavelength'] >= BF_LASER_CAL_WAVELENGTH_RANGE[0]) & \
                                  (self.allphotons['Wavelength'] <= BF_LASER_CAL_WAVELENGTH_RANGE[1])
                self.photons = self.allphotons[self.photonMask]
            else:
                print(f"WARNING: {self.instrument} is not a recognized instrument. No cut applied")
        else:
            self.photons = self.allphotons

    def get_time_info(self):
        """
        Extracts the timestamps of all the remaining photons and generates the bin edges to make the
        photon timestream (counts over array vs. time)
        """
        self.arrivaltimes = self.photons['Time']
        self.timebins = np.arange(0, int(np.ceil(self.arrivaltimes.max() / 10) * 10) + 10, 10)
        assert self.timebins[-1] >= self.arrivaltimes.max()

    def make_timestream(self):
        """
        Uses the numpy.histogram function to bin all of the photons into 10 microsecond bins. Then creates the
        self.timestream attribute so that we can plot the array counts vs. time. We chop off the last entry of the
        time bins in the timestream because that is the end of the final bin, and we are plotting these events based on
        the start times of each bin. We could also plot the midpoint of each bin but because the removal range is 15
        times larger than the bin size it ultimately doesn't matter.
        """
        self.get_time_info()
        self.arraycounts, timebins = np.histogram(self.arrivaltimes, self.timebins)
        assert np.setdiff1d(timebins, self.timebins).size == 0
        self.timestream = np.array((self.timebins[:-1], self.arraycounts))

    def make_count_histogram(self):
        """
        This function will use the numpy.unique function on the self.arraycounts attribute to return 2 lists, the first
        will be all of the different values that occur in the self.arraycounts array, the second will be the
        corresponding number of times that each of those values appear. This will allow us to create a histogram for
        data visualization as well as create the Poisson PDF to determine the cosmic ray threshold value.
        """
        unique, counts = np.unique(self.arraycounts, return_counts=True)
        unique_full = np.arange(np.min(unique), np.max(unique) + 1, 1)
        counts_full = np.zeros(len(unique_full))
        for i, j in enumerate(unique_full):
            if j in unique:
                m = unique == j
                counts_full[i] = counts[m]
        self.countsperbin = unique_full  # This is the x-axis of the histogram. It will be filled with all of the values
        # that occur in the self.arraycounts attribute.
        self.countoccurrences = counts_full  # The y-axis of the histogram. It will be filled with the correspongding number
        # of times that each value in self.countsperbin occurs.
        self.countshistogram = np.array((self.countsperbin, self.countoccurrences))

    def generate_poisson_pdf(self):
        """
        This function generates a Poisson probability density function fit to the histogram of counts per bin.
        Regardless of whether or not photon list has been cut for wavelength the distribution should be Poisson and
        should help provide a reasonable value for the number of counts per bin that can be classified as a cosmic ray.
        """
        avgcounts = np.average(self.arraycounts)  # Takes the average number of counts over the array in 10-microsecond
        # bins to be used in creating the Poisson PDF
        pdf = poisson.pmf(self.countshistogram[0], avgcounts)  # Creates a Poisson PDF of the number of counts per time
        # bin. This can be used to compare with wavelength cut or non-cut data, but more care must be taken with non-cut
        # data if there is a low probability of no counts over the array.
        mask = pdf < 1 / np.sum(self.arraycounts)  # Creates a mask where the probability of that number of countsperbin
        # happening is less than 1-in-all the counts during the observation.
        pdf[mask] = 0  # Applies the 'low-probability mask' to the PDF for (1) plotting and
        # (2) helping determine where the threshold for cosmic ray cuts should be made.
        self.pdf = pdf

    def _generate_poisson_threshold(self):
        """
        For the Poisson cosmic ray identification method generates the threshold of counts in a given bin which will be
        considered a cosmic ray event. For the Poisson method, that is the lowest value where the probability of that
        number of counts occurring is essentially less than expected from the data. It also sets a lower limit in the
        case that the photon list was not filtered based on wavelength and the Poisson statistics are trending towards
        being Gaussian (and in turn low count rates also have near 0 probability).
        :return:
        threshold - The number of photons/10-microsecond bin
        """
        thresholdmask = (self.pdf <= 1 / np.sum(self.arraycounts)) & (self.countshistogram[0] >= 2)
        threshold = self.countshistogram[0][thresholdmask].min()
        return threshold

    def _generate_signal_threshold(self):
        """
        For the peak-finding cosmic ray identification generates the threshold of counts per bin which will be used to
        find the cosmic ray events. For the peak finding method, that is a value which is >6 times higher than the
        average number of counts per bin. This value was found to be appropriate empirically and has been tested
        predominantly on data which was not wavelength-cut. However, it works on either filtered or non-filtered data
        and will be used to find appropriate cosmic ray 'signals' in self.find_cutout_times.
        :return:
        threshold - The number of photons/10-microsecond bin
        """
        avgcounts = np.average(self.arraycounts)
        threshold = np.ceil(6 * poisson.std(avgcounts, loc=0) + avgcounts)
        return threshold

    def find_cosmic_times(self):
        """
        """
        if self.method.lower() == "poisson":
            self.threshold = self._generate_poisson_threshold()
            cutmask = self.arraycounts >= self.threshold
            self.cosmictimes = self.timebins[:-1][cutmask]

        elif self.method.lower() == "peak-finding":
            self.threshold = self._generate_signal_threshold()
            self.cosmictimes = signal.find_peaks(self.arraycounts, height=self.threshold, threshold=10, distance=30)[
                                   0] * 10
        else:
            print("Invalid method!!")

        self._narrow_cosmic_times(500)

    def _narrow_cosmic_times(self, width):
        """
        Since the algorithm for determining the cosmic ray events does not distinguish if two 'cosmic rays' are really
        just two time bins with counts above the calculated CR threshold. So - for reporting - we find the peaks of each
        of these events and their associated timestamps so we can say how many CR events were removed and not
        artificially inflate those numbers.
        """
        cosmic_peak_times = []
        for i in self.cosmictimes:
            mask = abs(self.cosmictimes - i) <= width
            temp = self.cosmictimes[mask]
            locations = [np.where(self.timestream[0] == j) for j in temp]
            counts = [self.timestream[1][j] for j in locations]
            max_idx = np.argmax(counts)
            max_time_idx = np.where(self.timestream[0] == temp[max_idx])[0][0]
            peak_time = self.timestream[0][max_time_idx]
            cosmic_peak_times.append(peak_time)
        cosmic_peak_times = list(dict.fromkeys(cosmic_peak_times))
        self.cosmicpeaktimes = cosmic_peak_times

    def find_cutout_times(self):
        """
        Generates the timestamps in microseconds to be removed from the obsFile. Removes doubles for clarity.
        """
        cutouttimes = np.array([np.arange(i - self.removalRange[0],
                                          i + self.removalRange[1], 1) for i in self.cosmictimes]).flatten()
        cutouttimes = np.array(list(dict.fromkeys(cutouttimes)))
        self.cutouttimes = cutouttimes

    def trim_timestream(self):
        """
        Function designed to create a new timestream with the cosmic ray timestamped photos removed.
        """
        trimmask = np.in1d(self.arrivaltimes, self.cutouttimes)
        trimmedphotons = self.arrivaltimes[~trimmask]
        trimmedarraycounts, timebins = np.histogram(trimmedphotons, self.timebins)
        assert np.setdiff1d(timebins, self.timebins).size == 0
        self.trimmedtimestream = np.array((self.timebins[:-1], trimmedarraycounts))

    def animate_cr_event(self, timestamp, saveName=None, timeBefore=150, timeAfter=250, frameSpacing=5, frameIntTime=10,
                         wvlStart=None, wvlStop=None, fps=5, save=True):
        """
        Functionality to animate a cosmic ray event. This function is modeled off of the mkidpipeline.imaging.movies
        _make_movie() function, but adapted for the specific functionality of animating a cosmic ray event. There is a
        known bug that trying to animate the same event with different parameters multiple times in a row (e.g. with
        no wavelength cut then with wavelength boundaries) it will properly make the frames
        """
        Writer = animation.writers['imagemagick']
        writer = Writer(fps=fps, bitrate=-1)

        ctimes = np.arange(timestamp - timeBefore, timestamp + timeAfter, frameSpacing)
        frames = np.array([self.obs.getPixelCountImage(firstSec=i / 1e6, integrationTime=frameIntTime / 1e6,
                                                       wvlStart=wvlStart, wvlStop=wvlStop)['image'] for i in ctimes])

        fig = plt.figure()
        im = plt.imshow(frames[0])
        plt.tight_layout()
        plt.xlabel('Pixel')
        plt.ylabel('Pixel')

        if save:
            with writer.saving(fig, f"cosmic{timestamp}.gif" if saveName is None else str(saveName), frames.shape[2]):
                for i in frames:
                    im.set_array(i)
                    writer.grab_frame()
        del fig, im
        return frames