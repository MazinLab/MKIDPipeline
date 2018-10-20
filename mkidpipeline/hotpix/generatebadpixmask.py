import tables
import glob
import os.path
import sys
import warnings

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from math import *
import numpy as np
import numpy.ma as ma

import scipy.ndimage.filters as spfilters
import astropy.stats

from mkidpipeline.hdf.darkObsFile import ObsFile
from mkidpipeline.utils import utils
import mkidcore.corelog as pipelinelog
from mkidcore.corelog import getLogger
from matplotlib.backends.backend_pdf import PdfPages

"""
Bad Pixel Cal Flag Meanings
'good': 0,  # No flagging.
'hot': 1,  # Hot pixel
'cold': 2,  # Cold pixel
'dead': 3,  # Dead pixel
"""



def checkInterval(fwhm=2.5, boxSize=5, nSigmaHot=4.0, image=None,
                  maxIter=5, useLocalStdDev=False, bkgdPercentile=50.0):
    """

    :param fwhm:
    :param boxSize:
    :param nSigmaHot:
    :param image:
    :param maxIter:
    :param useLocalStdDev:
    :param bkgdPercentile:
    :return:

    """

    im = np.copy(image)

    # Approximate peak/median ratio for an ideal (Gaussian) PSF sampled at
    # pixel locations corresponding to the median kernal used with the real data.
    gauss_array = utils.gaussian_psf(fwhm, boxSize)
    maxRatio = np.max(gauss_array) / np.median(gauss_array)


    # Turn dead pixel values into NaNs.
    deadMask = im < 0.01  # Assume everything with 0 counts is a dead pixel
    im[deadMask] = np.nan

    # Initialise a mask for hot pixels (all False) for comparison on each iteration.
    oldHotMask = np.zeros(shape=np.shape(im), dtype=bool)
    hotMask = np.zeros(shape=np.shape(im), dtype=bool)

    # Initialise some arrays with nan's in case we don't get to fill them out for real,
    medFiltImage = np.zeros_like(im)
    medFiltImage.fill(np.nan)
    diff = np.zeros_like(im)
    diff.fill(np.nan)
    diffErr = np.zeros_like(im)
    diffErr.fill(np.nan)
    # Ditto for number of iterations
    iIter = -1

    if np.sum(im[np.where(np.isfinite(im))]) <= 0:  # Check to make sure not *all* the pixels are dead before doing further masking.
        print('Entire image consists of dead pixels')
        badMask=deadMask+2  #return a badMask that's just the whole image flagged as DEAD
    else:
        for iIter in range(maxIter):
            print('Iteration: ', iIter)
            # Calculate median filtered image
            # Each pixel takes the median of itself and the surrounding boxSize x boxSize box.
            # Do the median filter on a NaN-fixed version of im.
            nanFixedImage = utils.replaceNaN(im, mode='mean', boxsize=boxSize)
            assert np.all(np.isfinite(nanFixedImage))  # Just make sure there's nothing weird still in there.
            medFiltImage = spfilters.median_filter(nanFixedImage, boxSize, mode='mirror')

            overallMedian = np.median(im[~np.isnan(im)])
            overallBkgd = np.percentile(im[~np.isnan(im)], bkgdPercentile)

            stdFiltImage = utils.nearestNrobustSigmaFilter(im, n=boxSize ** 2 - 1)
            overallBkgdSigma = np.median(stdFiltImage[np.isfinite(stdFiltImage)])  #Estimate of the background std. dev.
            stdFiltImage[np.where(stdFiltImage < 1.)] = 1.
            if overallBkgdSigma < 0.01: overallBkgdSigma = 0.01  # Just so it's not 0

            # Calculate difference between flux in each pixel and maxRatio * the median in the enclosing box.
            # Also calculate the error that would exist in a measurment of a pixel that *was* at the peak of a real PSF
            # Condition for flagging is:
            #        (flux - background)/(box median - background) > maxRatio.
            # Or:
            #        flux > maxRatio*median - background*(maxRatio-1)   (... + n*sigma, where sigma is photon shot noise for the threshold level)
            # If the threshold is *lower* than the background, then set it equal to the background level instead (a pixel below the background level is unlikely to be hot!)
            print('overallMedian: ', overallMedian)
            print('overallBkgd: ', overallBkgd)
            print('overallBkgdSigma: ', overallBkgdSigma)
            print('maxRatio: ', maxRatio)
            threshold = np.maximum((maxRatio * medFiltImage - (maxRatio - 1.) * overallBkgd), overallBkgd)

            diff = im - threshold

            # Simple estimate, probably makes the most sense: photon error in the max value allowed. Neglect errors in the median itself here.
            if useLocalStdDev is False:
                # Consider imaginary photon noise in the expected threshold level and background
                # random noise, added in quadrature. Prob. far from perfect, but attempts to account
                # for whatever the extra noise is in the images.
                diffErr = np.sqrt(threshold + overallBkgdSigma ** 2)  # Note threshold = sqrt(threshold)**2

            else:
                diffErr = stdFiltImage

            # Any pixel that has a peak/median ratio more than nSigma above the maximum ratio should be flagged as hot:
            # (True=>bad pixel; False=> good pixel).
            hotMask = (diff > (nSigmaHot * diffErr)) | oldHotMask

            # If no change between between this and the last iteration then stop iterating
            if np.all(hotMask == oldHotMask): break

            # Otherwise update 'oldHotMask' and set all detected bad pixels to NaN for the next iteration
            oldHotMask = np.copy(hotMask)
            im[hotMask] = np.nan

    # Finished with loop
    assert np.all(hotMask & deadMask == False)  #Make sure a pixel is not simultaneously hot and dead

    deadMask=(deadMask*1)+2  # Convert bools into 0s and 1s, use correct bad pix flags
    hotMask=(hotMask*1)
    badMask=deadMask+hotMask

    return(badMask)

#TODO print-> logging
#TODO document this fxn