'''
Author:  Isabel Lipartito and Clarissa Rizzo  Date: January 17, 2018

Identify synchronous photons
'''

import sys,os
import tables
import numpy as np
import matplotlib.pyplot as plt
from utils import utils
from hdf.photontable import photontable
import inspect
from interval import interval, inf, imath
try:
    from cosmic import tsBinner
except ImportError:
    print "trouble importing tsBinner.  Follow directions in cosmic/README.txt"

from scipy.optimize import curve_fit
from scipy.stats import expon
import time
import pickle
import logging
class Cosmic:



def findCosmics(self, stride=10, threshold=100,
                population_max=2000, nSigma=5, write_cosmicmask=False,
                pps_stride=10000):
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

    self.logger.info("findCosmics: begin stride=%d threshold=%d populationMax=%d nSigma=%d writeCosmicMask=%s" % (
    stride, threshold, populationMax, nSigma, writeCosmicMask))

    exptime = self.endTime - self.beginTime
    nBins = int(np.round(self.file.ticksPerSec * exptime + 1))
    bins = np.arange(0, nBins, 1)
    timeHgValues, frameSum = self.getTimeHgAndFrameSum(self.beginTime, self.endTime)
    remainder = len(timeHgValues) % ppsStride
    if remainder > 0:
        temp = timeHgValues[:-remainder]
    else:
        temp = timeHgValues
    ppsTime = (ppsStride * self.file.tickDuration)
    pps = np.sum(temp.reshape(-1, ppsStride), axis=1) / ppsTime
    self.logger.info("findCosmics:  call populationFromTimeHgValues")
    pfthgv = Cosmic.populationFromTimeHgValues \
        (timeHgValues, populationMax, stride, threshold)
    # now build up all of the intervals in seconds

    self.logger.info("findCosmics:  build up intervals:  nCosmicTime=%d" % len(pfthgv['cosmicTimeList']))

    i = interval()
    iCount = 0
    secondsPerTick = self.file.tickDuration
    for cosmicTime in pfthgv['cosmicTimeList']:

        t0 = self.beginTime + cosmicTime * secondsPerTick
        dt = stride * secondsPerTick
        t1 = t0 + dt
        left = max(self.beginTime, t0 - nSigma * dt)
        right = min(self.endTime, t1 + 2 * nSigma * dt)
        i = i | interval[left, right]
        self.logger.debug("findCosmics:  iCount=%d t0=%f t1=%f left=%f right=%f" % (iCount, t0, t1, left, right))
        iCount += 1

    tMasked = Cosmic.countMaskedBins(i)
    ppmMasked = 1000000 * tMasked / (self.endTime - self.beginTime)

    retval = {}
    retval['timeHgValues'] = timeHgValues
    retval['populationHg'] = pfthgv['populationHg']
    retval['cosmicTimeList'] = pfthgv['cosmicTimeList']
    retval['binContents'] = pfthgv['binContents']
    retval['frameSum'] = frameSum
    retval['interval'] = i
    retval['ppmMasked'] = ppmMasked
    retval['pps'] = pps
    retval['ppsTime'] = ppsTime
    if writeCosmicMask:
        cfn = self.fn.cosmicMask()
        self.logger.info("findCosmics:  write masks to =%s" % cfn)
        ObsFile.writeCosmicIntervalToFile(i, self.file.ticksPerSec,
                                          cfn, self.beginTime, self.endTime,
                                          stride, threshold, nSigma, populationMax)
    self.logger.info("findCosmics:  end with ppm masked=%d" % ppmMasked)
    return retval


def getTimeHgAndFrameSum(self, beginTime, endTime):
    integrationTime = endTime - beginTime
    nBins = int(np.round(self.file.ticksPerSec * integrationTime + 1))
    timeHgValues = np.zeros(nBins, dtype=np.int64)
    frameSum = np.zeros((self.file.nRow, self.file.nCol))
    self.logger.info("get all time stamps for integrationTime=%f" % integrationTime)
    for iRow in range(self.file.nRow):
        for iCol in range(self.file.nCol):

            gtpl = self.file.getPackets(iRow, iCol,
                                        beginTime, integrationTime)
            timestamps = gtpl['timestamps']
            if timestamps.size > 0:
                timestamps = \
                    (timestamps - beginTime) * self.file.ticksPerSec
                ts32 = np.round(timestamps).astype(np.uint32)
                tsBinner.tsBinner32(ts32, timeHgValues)
                frameSum[iRow, iCol] += ts32.size

    return timeHgValues, frameSum