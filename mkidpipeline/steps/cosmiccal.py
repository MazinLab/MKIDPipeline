"""
Author: Clarissa Rizzo, Isabel Lipartito, Noah Swimmer
Cosmic Cleaner is a class that will find the cosmic correction and will create a file that
has the associated ObsFile name (beginning timestamp) as well as a list of the timestamps
to be removed plus some metadata, such as the cutout times before and after the cosmic ray
event, the method used for peak finding (more on that in the code itself), a set of data for
creating a histogram of bincounts, and more as needed. This docstring will be edited as code
is updated and refined.
TODO: Integrate into pipeline
TODO: Create performance report, such as number of CR events, amount of time removed, time intervals removed, etc.

TODO: Finalize best way to incorporate as pipeline step (flag photons, list of CR timestamps, new col in h5?)
"""
import numpy as np
from scipy.stats import poisson
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from logging import getLogger
from datetime import datetime

import mkidpipeline.config
from mkidpipeline.photontable import Photontable
import tables

NP_IMPACT_TYPE = np.dtype([('count', np.uint8), ('start', np.uint64), ('stop', np.uint64), ('rete', np.uint32)])


class CRImpact(tables.IsDescription):
    count = tables.UInt8Col()  # unsigned byte
    start = tables.UInt64Col()  # 64-bit integer
    stop = tables.UInt64Col()  # 64-bit integer
    rate = tables.UInt32Col()  # double (double-precision)


class StepConfig(mkidpipeline.config.BaseStepConfig):
    yaml_tag = u'!badpix_cfg'
    REQUIRED_KEYS = (('plots', 'all', 'Which plots to generate'),
                     ('wave_range', None, 'TODO'),
                     ('method', 'poisson', 'TODO'),
                     ('removal_range', (50, 100), 'TODO'))


class CosmicCleaner:
    def __init__(self, file, wave_range=(-np.inf, np.inf), method="poisson", removal_range=(50, 100)):
        self.obs = Photontable(file)
        self.wave_range = wave_range
        self.method = method
        self.removalRange = removal_range
        self.photons = None  # The photon list used for the cosmic ray removal. Not modified from allphotons if no cut
        self.photonMask = None  # The mask used to remove photons from the master list (e.g. wvl outside of allowed range)
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

        if not np.isfinite(np.sum(self.wave_range)):
            getLogger(__name__).warning("Consider using a wavelength cut to speed removal.")
            if self.method.lower() is "poisson":
                getLogger(__name__).warning("The Poisson method is not optimized for broadband data and may remove "
                                            "more time than desired!")

    def run(self):
        start = datetime.utcnow().timestamp()
        getLogger(__name__).debug(f"Starting cosmic ray detection on {self.obs.filename}")
        self.photons = self.obs.query(startw=self.wave_range[0], stopw=self.wave_range[1], column='Time')
        self.make_timestream()
        self.make_count_histogram()
        self.generate_poisson_pdf()
        self.find_cosmic_times()
        self.find_cutout_times()
        self.trim_timestream()
        end = datetime.utcnow().timestamp()
        getLogger(__name__).info(f"Cosmic ray cleaning of {self.obs.filename} took {end - start} s")

    def make_timestream(self):
        """
        Uses the numpy.histogram function to bin all of the photons into 10 microsecond bins. Then creates the
        self.timestream attribute so that we can plot the array counts vs. time. We chop off the last entry of the
        time bins in the timestream because that is the end of the final bin, and we are plotting these events based on
        the start times of each bin. We could also plot the midpoint of each bin but because the removal range is 15
        times larger than the bin size it ultimately doesn't matter.
        """
        self.get_time_info()
        self.timebins = np.arange(0, int(np.ceil(self.photons.max() / 10) * 10) + 10, 10)
        assert self.timebins[-1] >= self.photons.max()
        self.arraycounts, timebins = np.histogram(self.photons, self.timebins)
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
        else:
            self.threshold = self._generate_signal_threshold()

        # distance is in bin widths noah sasy thats 10ms
        # after peak double threshold for the decay time
        self.cosmictimes = signal.find_peaks(self.arraycounts, height=self.threshold, threshold=10,
                                             distance=50)[0] * self.bin_width

    def find_cutout_times(self):
        """
        Generates the timestamps in microseconds to be removed from the obsFile. Removes doubles for clarity.
        """
        # cutouttimes=np.arange(*self.removalRange)[:,None]*self.cosmictimes).flatten()
        #TODO
        cutouttimes = np.array([np.arange(i - self.removalRange[0],
                                          i + self.removalRange[1], 1) for i in self.cosmictimes]).flatten()
        self.cutouttimes = np.array(set(cutouttimes))

    def trim_timestream(self):
        """
        Function designed to create a new timestream with the cosmic ray timestamped photos removed.
        """
        trimmask = np.in1d(self.photons, self.cutouttimes)
        trimmedphotons = self.photons[~trimmask]
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
        # This should be much faster but it gets rid of the interpolation that is achieved by overlapping the query
        # intervals, the way around this would be to tread the frames as keyframes and then use a seperate
        # post-processing step to insert interpolated frames.
        # frames = self.obs.get_fits(bin_edges=ctimes, rate=False, wave_start=wave_start, wave_stop=wave_stop,
        #                            cube_type='time')['SCIENCE'].data
        # Interpolate and insert frames here

        frames = np.array([self.obs.get_fits(start=i / 1e6, duration=frameIntTime / 1e6, rate=False,
                                             wave_start=wvlStart, wave_stop=wvlStop)['SCIENCE'].data for i in ctimes])

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


def apply(o: mkidpipeline.config.MKIDTimerange, config=None):

    cfg = mkidpipeline.config.config.cosmiccal if config is None else config
    if cfg in None:
        cfg = StepConfig()

    #TODO
    exclude = [k[0] for k in StepConfig.REQUIRED_KEYS]
    methodkw = {k: cfg.get(k) for k in cfg.keys() if k not in exclude}
    cc = CosmicCleaner(o.h5, **methodkw)
    cc.determine_cosmic_intervals()

    impacts = np.zeros(len(cc.interval_starts), dtype=NP_IMPACT_TYPE)
    # for i, uniqe_exclude_region in enumerate(cc.cosmic_intervals):
    #     impacts[i]['start'] = uniqe_exclude_region.start
    #     impacts[i]['stop'] = uniqe_exclude_region.stop
    #     impacts[i]['count'] = len(uniqe_exclude_region.events)
    #     impacts[i]['rate'] = np.average([x.peak_val for x in uniqe_exclude_region.events])
    impacts[:]['start'] = cc.interval_starts
    impacts[:]['stop'] = cc.interval_stops
    impacts[:]['count'] = cc.interval_event_count
    impacts[:]['rate'] = cc.interval_avg_peak

    md = dict(region=cc.removalRange, thresh=cc.threshold, method=cc.method, wavecut=cc.wave_range)
    cc.obs.enablewrite()
    for k, v in md.items:
        cc.obs.update_header(f'COSMICCAL.{k}', v)
    cc.obs.attach_new_table('cosmics', 'Cosmic Ray Info', 'impacts', CRImpact, "Cosmic-Ray Hits", impacts)
    cc.obs.disablewrite()