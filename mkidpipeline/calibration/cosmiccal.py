'''
Author:  Isabel Lipartito and Clarissa Rizzo  Date: January 17, 2018

Identify synchronous photons
'''

import sys,os
import tables
import numpy as np
import matplotlib.pyplot as plt
import pyximport; pyximport.install()
###import mkidpipeline.calibration.ts_binner as ts_binner
from utils import utils
from mkidpipeline.hdf.photontable import ObsFile
import inspect
from interval import interval, inf, imath


from scipy.optimize import curve_fit
from scipy.stats import expon
import time


class Cosmic:
    def __init__(self):
        self.endtime=-1
        self.begintime=0
        self.tick_duration = 1e-6
        self.ticks_per_sec = int(1.0 / self.tick_duration)
        self.nRow=140
        self.nCol=146
        self.h5file='test'
        self.obs = ObsFile(self.h5file)



    def findCosmics(self, stride=10, threshold=100,
                    population_max=2000, nsigma=5, pps_stride=10000):
        """
        Find cosmics ray suspects.  Histogram the number of photons
        recorded at each timeStamp.  When the number of photons in a
        group of stride timeStamps is greater than threshold in second
        iSec, add (iSec,timeStamp) to cosmicTimeLists.  Also keep
        track of the histogram of the number of photons per stride
        timeStamps.
        return a dictionary of 'populationHg', 'cosmicTimeLists',
        'binContents', 'timeHgValues', 'interval', 'frameSum', and 'pps'

        populationHg is a histogram of the number of photons in each time bin.
        This is a poisson distribution with a long tail due to cosmic events

        cosmicTimeLists is a numpy array  of all the sequences that are
        suspects for cosmic rays
        binContents corresponds to cosmicTimeLists.  For each time in
        cosmicTimeLists, binContents is the number of photons detected
        at that time.

        timeHgValues is a histogram of the number of photons in each time
        interval

        frameSum is a two dimensional  numpy array of the number of photons
        detected by each pixel
        interval is the interval of data to be masked out
        pps is photons per second, calculated every ppsStride bins.
        """

        ###self.logger.info("findCosmics: begin stride=%d threshold=%d populationMax=%d nSigma=%d writeCosmicMask=%s")

        exptime = self.endtime - self.begintime
        nbins = int(np.round(self.ticks_per_sec * exptime + 1))
        bins = np.arange(0, nbins, 1)
        timehist_values, framesum = self.get_timehist_and_framesum(self.begintime, self.endtime)
        remainder = len(timehist_values) % pps_stride
        if remainder > 0:
            temp = timehist_values[:-remainder]
        else:
            temp = timehist_values
        pps_time = (pps_stride * self.tick_duration)
        pps = np.sum(temp.reshape(-1, pps_stride), axis=1) / pps_time
        cr_pop = Cosmic.population_from_timehist_values(timehist_values, population_max, stride, threshold)

        # now build up all of the intervals in seconds

        i = interval()
        icount = 0
        seconds_per_tick = self.tick_duration
        for cosmic_time in cr_pop['cosmic_time_list']:

            t0 = self.begintime + cosmic_time * seconds_per_tick
            dt = stride * seconds_per_tick
            t1 = t0 + dt
            left = max(self.begintime, t0 - nsigma * dt)
            right = min(self.endtime, t1 + 2 * nsigma * dt)
            i = i | interval[left, right]
            icount += 1

        times_masked = Cosmic.count_masked_bins(i)
        ppm_masked = 1000000 * times_masked / (self.endtime - self.begintime)

        return_val = {}
        return_val['timehist_values'] = timehist_values
        return_val['population_hist'] = cr_pop['population_hist']
        return_val['cosmic_timelist'] = cr_pop['cosmic_timelist']
        return_val['bin_contents'] = cr_pop['bin_contents']
        return_val['framesum'] = framesum
        return_val['interval'] = i
        return_val['ppm_masked'] = ppm_masked
        return_val['pps'] = pps
        return_val['pps_time'] = pps_time

        return return_val


    def get_timehist_and_framesum(self, begintime, endtime):
        integrationtime = endtime - begintime
        nbins = int(np.round(self.ticks_per_sec * integrationtime + 1))
        timehist_values = np.zeros(nbins, dtype=np.int64)
        framesum = np.zeros((self.nRow, self.nCol))
        for iRow in range(self.nRow):
            for iCol in range(self.nCol):
                gtpl = self.obs.getPixelPhotonList(iRow, iCol, firstSec=begintime, integrationtime=integrationtime)
                timestamps = gtpl['timestamps']
                if timestamps.size > 0:
                    timestamps = (timestamps - begintime) * self.ticks_per_sec
                    ts32 = np.round(timestamps).astype(np.uint32)
                    ts_binner.ts_binner32(ts32, timehist_values)
                    framesum[iRow, iCol] += ts32.size

        return timehist_values, framesum

    @staticmethod
    def count_masked_bins(mask_interval):
        return_val = 0
        for x in mask_interval:
            return_val += x[1]-x[0]
        return return_val

    @staticmethod
    def population_from_timehist_values(timehist_values,population_max,stride,threshold):
        """
        Rebin the timehist_values histogram by combining stride bins.  If
        stride > 1, then bin a second time after shifting by stride/2
        Create population_hist, a histogram of the number of photons in
        the large bins.  Also, create (and then sort) a list
        cosmic_timelist of the start of bins (in original time units)
        of overpopulated bins that have more than threshold number of
        photons.
        return a dictionary containing population_hist and cosmic_timelist
        """
        pop_range = (-0.5,population_max-0.5)
        if stride==1:
            population_hist = np.histogram(timehist_values, population_max, range=pop_range)
            cosmic_timelist = np.where(timehist_values > threshold)[0]
            bin_contents = np.extract(timehist_values > threshold, timehist_values)
        else:
            # rebin the timehist_values before counting the populations
            length = timehist_values.size
            remainder = length%stride
            if remainder == 0:
                end = length
            else:
                end = -remainder

            timehist_values_trimmed = timehist_values[0:end]

            timehist_values_rebinned_0 = np.reshape(timehist_values_trimmed, [length/stride, stride]).sum(axis=1)
            population_hist_0 = np.histogram(timehist_values_rebinned_0, population_max, range=pop_range)
            cosmic_timelist_0 = stride*np.where(timehist_values_rebinned_0 > threshold)[0]
            bin_contents_0 = np.extract(timehist_values_rebinned_0 > threshold, timehist_values_rebinned_0)

            timehist_values_rebinned_1 = np.reshape(timehist_values_trimmed[stride/2:-stride/2],[(length-stride)/stride, stride]).sum(axis=1)
            population_hist_1 = np.histogram(timehist_values_rebinned_1, population_max, range=pop_range)
            cosmic_timelist_1 = (stride/2)+stride*np.where(timehist_values_rebinned_1 > threshold)[0]
            bin_contents_1 = np.extract(timehist_values_rebinned_1 > threshold, timehist_values_rebinned_1)

            population_hist = (population_hist_0[0]+population_hist_1[0],population_hist_0[1])
            cosmic_timelist = np.concatenate((cosmic_timelist_0,cosmic_timelist_1))
            bin_contents = np.concatenate((bin_contents_0, bin_contents_1))
            args = np.argsort(cosmic_timelist)
            cosmic_timelist = cosmic_timelist[args]
            bin_contents = bin_contents[args]
            cosmic_timelist.sort()

        return_val = {}
        return_val['population_hist'] = population_hist
        return_val['cosmic_timelist'] = cosmic_timelist
        return_val['bin_contents'] = bin_contents
        return return_val