"""
Author: Isabel Lipartito
Date: October 2018

Routines for checking for hot and dead pixels in obs. files (input can be calibrated or uncalibrated).

The supplied obs. file is stepped through in short time steps, and a 2D mask
made for each step. The current detection algorithm compares the flux in each pixel
with that in each pixel in a surrounding box. If the difference is significantly
higher than could be expected for a stellar (point-source) PSF (allowing for
noise), then the flux level cannot be real and the pixel is flagged for the
time step under examination.

New PtSi devices do not seem to display hot pixel behavior from the same mechanism as
old TiN devices. As such, the full time masking technique used in the ARCONS pipeline
may not be necessary. Bad pixels largely remain "bad" with little switching behavior.

Functions available for use:

-------------
find_bad_pixels:  The main routine. Takes an obs. file  as input and writes a bad pixel mask as a table into the obsfile
                  The routine by which hot, dead, and cold pixels are identified can be swapped out for one of four options
                  Algorithms differ in how they flag hot pixels but they all are able to flag dead pixels

                  Bad Pixel Masking Via Image Arrays --> Best for wavelength calibrated + flat-fielded arrays
                    a.  hpm_flux_threshold:  Established PSF-comparison method to find hot pixels
                    b.  hpm_laplacian:  New method to find hot pixels using a Laplacian filter
                    c.  cps_cut_image:  Basic hot pixel search using a count-per-second threshold (not robust)

                  Bad Pixel Masking via Time Streams --> Best for uncalibrated obs data or calibration data
                    a.  hpm_poisson_dist:

-------------
Functions for .img data:

cps_cut_img:    Creates a 2D bad pixel mask for a given time interval within a given
                exposure using a count-per-second threshold routine.  USE THIS FOR DATA VIEWING/QUICK CHECKS

cps_cut_stack:  Creates a 2D bad pixel mask for a given time interval within a given
                exposure using a count-per-second threshold routine.  USE THIS FOR DATA VIEWING/QUICK CHECKS

save_mask_array:  Write bad pixel mask to desired output directory

load_mask_array:  Load and return a mask

plot_mask_array:  Plot a mask or masked image for viewing

-------------
Functions for obs data:

hpm_flux_threshold: Creates a 2D bad pixel mask for a given time interval within a given
                    exposure using a robust PSF-comparison method.  USE THIS FOR DATA REDUCTION

hpm_laplacian:  Creates a 2D bad pixel mask for a given time interval within a given exposure using a Laplacian filter
                based on approximate second derivatives.
                Untested on MKID data so far, but it worked on CHARIS data injected with fake hot pixels

-------------


"""

import tables
import os.path
import sys
from datetime import datetime
import warnings
import argparse

import matplotlib.pyplot as plt
import numpy as np

import scipy.ndimage.filters as spfilters
from mkidpipeline.speckle.binned_rician import *
from scipy.stats import poisson

from mkidpipeline.hdf.darkObsFile import ObsFile
from mkidpipeline.utils import utils
from mkidpipeline.utils.plottingTools import plot_array
from mkidcore.corelog import getLogger
import mkidcore.corelog


def hpm_flux_threshold(image, fwhm=2.5, box_size=5, nsigma_hot=4.0, max_iter=5,
                   use_local_stdev=False, bkgd_percentile=50.0):
    """
    Robust!  NOTE:  This is a routine that was ported over from the ARCONS pipeline.
    Finds the hot and dead pixels in a for a 2D input array.
    Compares the ratio of flux in each pixel to the median of the flux in an
    enclosing box. If the ratio is too high -- i.e. the flux is too tightly
    distributed compared to a Gaussian PSF of the expected FWHM -- then the
    pixel is flagged as HOT.

    If the pixel has counts less than 0.01, then the pixel is flagged as DEAD

    The HOT and DEAD masks are combined into a single BAD mask at the end

    Required Input:
    :param image:           A 2D image array of photon counts.

    Other Input:
    :param fwhm:               Scalar float. Estimated full-width-half-max of the PSF (in pixels).
    :param box_size:           Scalar integer. Size box used for calculating median flux in the region surrounding each pixel.
    :param nsigma_hot:         Scalar float. If the flux ratio for a pixel is (nsigma_hot x expected error)
                                             above the max expected given the PSF FWHM, then flag it as hot.
    :param max_iter:           Scalar integer. Maximum number of iterations allowed.
    :param use_local_stdev:    Bool.  If True, use the local (robust) standard deviation within the
                                      moving box for the sigma value in the hot pixel thresholding
                                      instead of Poisson statistics. Mainly intended for situations where
                                      you know there is no astrophysical source in the image (e.g. flatfields,
                                      laser calibrations), where you can also set fwhm=np.inf
    :param bkgd_percentile:    Scalar Integer.  Percentile level (in %) in image to use as an estimate of the background.
                                                In an ideal world, this will be 50% (i.e., the median of the image).
                                                For raw images, however, there is often a gradient across the field,
                                                in which case it's sensible to use something lower than 50%.
    :return:
    A dictionary containing the result and various diagnostics. Keys are:

    'bad_mask': the main output. Contains a 2D array of integers of the same shape as the input image, where:
            0 = Good pixel
            1 = Hot pixel
            2 = Cold Pixel
            3 = Dead Pixel
    'dead_mask': 2D array of Bools of the same shape as the input image, where:
                True = Dead Pixel
                False = Not Dead Pixel
    'hot_mask': 2D array of Bools of the same shape as the input image, where:
                True = Hot Pixel
                False = Not Hot Pixel
    'image': 2D array containing the input image
    'median_filter_image': The median-filtered image
    'max_ratio': 2D array - the maximum allowed ratio between pixel flux and the median-filtered image
    'difference_image': the difference between the input image and an image representing the max allowed flux in each pixel.
    'difference_image_error': the expected error in the difference calculation.
    'num_iter': number of iterations performed.
    """
    if max_iter is None: max_iter = 5
    if use_local_stdev is None: use_local_stdev = False

    raw_image = np.copy(image)

    # Approximate peak/median ratio for an ideal (Gaussian) PSF sampled at
    # pixel locations corresponding to the median kernel used with the real data.
    gauss_array = utils.gaussian_psf(fwhm, box_size)
    max_ratio = np.max(gauss_array) / np.median(gauss_array)


    # Assume everything with 0 counts is a dead pixel, turn dead pixel values into NaNs
    dead_mask = raw_image < 0.01
    raw_image[dead_mask] = np.nan

    # Initialise a mask for hot pixels (all False) for comparison on each iteration.
    initial_hot_mask = np.zeros(shape=np.shape(raw_image), dtype=bool)
    hot_mask = np.zeros(shape=np.shape(raw_image), dtype=bool)

    # Initialise some arrays with NaNs in case they don't get filled out during the iteration
    median_filter_image = np.zeros_like(raw_image)
    median_filter_image.fill(np.nan)
    difference_image = np.zeros_like(raw_image)
    difference_image.fill(np.nan)
    difference_image_error = np.zeros_like(raw_image)
    difference_image_error.fill(np.nan)
    iteration = -1

    # In the case that *all* the pixels are dead, return a bad_mask where all the pixels are flagged as DEAD
    if raw_image[np.isfinite(raw_image)].sum() <= 0:
        log.info('Entire image consists of dead pixels')
        bad_mask=dead_mask*3
        hot_mask=numpy.zeros_like(bad_mask, dtype=bool)
        dead_mask=numpy.ones_like(bad_mask, dtype=bool)
    else:
        for iteration in range(max_iter):
            log.info('Iteration: '.format(iteration))
            print('Iteration', iteration)
            # Remove all the NaNs in an image and calculate a median filtered image
            # each pixel takes the median of itself and the surrounding box_size x box_size box.
            nan_fixed_image = utils.replaceNaN(raw_image, mode='mean', boxsize=box_size)
            assert np.all(np.isfinite(nan_fixed_image))
            median_filter_image = spfilters.median_filter(nan_fixed_image, box_size, mode='mirror')

            overall_median = np.median(raw_image[~np.isnan(raw_image)])
            print('overall_median', overall_median)
            overall_bkgd = np.percentile(raw_image[~np.isnan(raw_image)], bkgd_percentile)

            # Estimate the background std. dev.
            standard_filter_image = utils.nearestNrobustSigmaFilter(raw_image, n=box_size ** 2 - 1)
            overall_bkgd_sigma = max(.01, np.median(standard_filter_image[np.isfinite(standard_filter_image)]))
            standard_filter_image[np.where(standard_filter_image < 1.)] = 1.

            # Calculate difference between flux in each pixel and max_ratio * the median in the enclosing box.
            # Also calculate the error that would exist in a measurement of a pixel that *was* at the peak of a real PSF
            # Condition for flagging is:
            #        (flux - background)/(box median - background) > max_ratio.
            # Or:
            #        flux > max_ratio*median - background*(max_ratio-1)
            # If the threshold is *lower* than the background, then set it equal to the background level instead
            # (a pixel below the background level is unlikely to be hot!)
            log.info('overall_median: '.format(overall_median))
            log.info('overall_bkgd: '.format(overall_bkgd))
            log.info('overall_bkgd_sigma: '.format(overall_bkgd_sigma))
            log.info('max_ratio: '.format(max_ratio))
            threshold = np.maximum((max_ratio * median_filter_image - (max_ratio - 1.) * overall_bkgd), overall_bkgd)
            difference_image = raw_image - threshold

            # Simple estimate, photon error in the max value allowed. Neglect errors in the median itself here.
            # Add in quadrature imaginary photon noise in the expected threshold level and background random noise
            if use_local_stdev is False:
                difference_image_error = np.sqrt(threshold + overall_bkgd_sigma ** 2)  # Note threshold = sqrt(threshold)**2
            else:
            #use the local (robust) standard deviation within the moving box
                difference_image_error = standard_filter_image

            # Any pixel that has a peak/median ratio more than nSigma above the maximum ratio should be flagged as hot:
            # True = bad pixel; False = good pixel.
            hot_mask = (difference_image > (nsigma_hot * difference_image_error)) | initial_hot_mask

            # If no change between between this and the last iteration then stop iterating
            if np.all(hot_mask == initial_hot_mask): break

            # Otherwise update 'initial_hot_mask' and set all detected bad pixels to NaN for the next iteration
            initial_hot_mask = np.copy(hot_mask)
            raw_image[hot_mask] = np.nan

        # Finished with loop, make sure a pixel is not simultaneously hot and dead
        assert np.all(hot_mask & dead_mask == False)

        # Convert bools into 0s and 1s, use correct bad pix flags
        dead_mask_return=dead_mask*3
        hot_mask_return=hot_mask*1
        bad_mask=dead_mask_return+hot_mask_return

    return {'bad_mask': bad_mask, 'dead_mask': dead_mask, 'hot_mask': hot_mask,  'image': raw_image,
            'median_filter_image': median_filter_image, 'max_ratio': max_ratio, 'difference_image': difference_image,
            'difference_image_error': difference_image_error, 'num_iter': iteration + 1}


def hpm_median_movingbox(image, box_size=5, nsigma_hot=4.0, max_iter=5):
    """
    New routine, developed to serve as a generic hot pixel masking method
    Finds the hot and dead pixels in a for a 2D input array.

    Passes a box_size by box_size moving box over the entire array and checks if the pixel at the center of that window
    has counts higher than the median plus nsigma_hot times the standard deviation of the pixels in that window

    If the pixel has counts less than 0.01, then the pixel is flagged as DEAD

    Dead pixels are excluded from the standard deviation calculation and the standard deviation is corrected for small
    sample sizes as per the function stddev_bias_corr(n)

    The HOT and DEAD masks are combined into a single BAD mask at the end

    Required Input:
    :param image:           A 2D image array of photon counts.

    Other Input:
    :param box_size:           Scalar integer. Size box used for calculating median counts in the region surrounding each pixel.
    :param nsigma_hot:         Scalar float. If the flux ratio for a pixel is nsigma_hot x standard deviation within the moving box
                                             above the max expected given the PSF FWHM, then flag it as hot.
    :param max_iter:           Scalar integer. Maximum number of iterations allowed.

    :return:
    A dictionary containing the result and various diagnostics. Keys are:

    'bad_mask': the main output. Contains a 2D array of integers of the same shape as the input image, where:
            0 = Good pixel
            1 = Hot pixel
            2 = Cold Pixel
            3 = Dead Pixel
    'dead_mask': 2D array of Bools of the same shape as the input image, where:
                True = Dead Pixel
                False = Not Dead Pixel
    'hot_mask': 2D array of Bools of the same shape as the input image, where:
                True = Hot Pixel
                False = Not Hot Pixel
    'image': 2D array containing the input image
    'median_filter_image': The median-filtered image
    'num_iter': number of iterations performed.
    """
    if max_iter is None: max_iter = 5

    raw_image = np.copy(image)

    # Assume everything with 0 counts is a dead pixel, turn dead pixel values into NaNs
    dead_mask = raw_image < 0.01
    raw_image[dead_mask] = np.nan

    # Initialise a mask for hot pixels (all False) for comparison on each iteration.
    initial_hot_mask = np.zeros(shape=np.shape(raw_image), dtype=bool)
    hot_mask = np.zeros(shape=np.shape(raw_image), dtype=bool)

    # Initialise some arrays with NaNs in case they don't get filled out during the iteration
    median_filter_image = np.zeros_like(raw_image)
    median_filter_image.fill(np.nan)
    iteration = -1

    # In the case that *all* the pixels are dead, return a bad_mask where all the pixels are flagged as DEAD
    if raw_image[np.isfinite(raw_image)].sum() <= 0:
        log.info('Entire image consists of dead pixels')
        bad_mask=dead_mask*3
        hot_mask=numpy.zeros_like(bad_mask, dtype=bool)
        dead_mask=numpy.ones_like(bad_mask, dtype=bool)
    else:
        for iteration in range(max_iter):
            log.info('Iteration: '.format(iteration))
            print('Iteration', iteration)
            # Remove all the NaNs in an image and calculate a median filtered image
            # each pixel takes the median of itself and the surrounding box_size x box_size box.
            nan_fixed_image = utils.replaceNaN(raw_image, mode='mean', boxsize=box_size)
            assert np.all(np.isfinite(nan_fixed_image))
            median_filter_image = spfilters.median_filter(nan_fixed_image, box_size, mode='mirror')
            standard_filter_image=spfilters.generic_filter(raw_image, calc_stdev, box_size, mode='mirror')

            threshold = median_filter_image + (nsigma_hot*standard_filter_image)

            # Any pixel that has a count level more than nSigma above the median should be flagged as hot:
            # True = bad pixel; False = good pixel.
            hot_mask = (median_filter_image > threshold) | initial_hot_mask

            # If no change between between this and the last iteration then stop iterating
            if np.all(hot_mask == initial_hot_mask): break

            # Otherwise update 'initial_hot_mask' and set all detected bad pixels to NaN for the next iteration
            initial_hot_mask = np.copy(hot_mask)
            raw_image[hot_mask] = np.nan

        # Finished with loop, make sure a pixel is not simultaneously hot and dead
        assert np.all(hot_mask & dead_mask == False)

        # Convert bools into 0s and 1s, use correct bad pix flags
        dead_mask_return=dead_mask*3
        hot_mask_return=hot_mask*1
        bad_mask=dead_mask_return+hot_mask_return

    return {'bad_mask': bad_mask, 'dead_mask': dead_mask, 'hot_mask': hot_mask,  'image': raw_image,
            'median_filter_image': median_filter_image, 'num_iter': iteration + 1}


def calc_stdev(x):
    xClean = x[~numpy.isnan(x)]
    n = len(xClean)
    if numpy.size(xClean) > 1 and xClean.min() != xClean.max():
        return scipy.stats.tstd(xClean)*stddev_bias_corr(n)
    #Otherwise...
    return numpy.nan

def stddev_bias_corr(n):
        if n == 1:
            corr = 1.0
        else:
            lut = [0.7978845608, 0.8862269255, 0.9213177319, 0.9399856030, 0.9515328619,
                   0.9593687891, 0.9650304561, 0.9693106998, 0.9726592741, 1.0]
            lut_ndx = max(min(n - 2, len(lut) - 1), 0)
            print(lut_ndx)

            corr = lut[lut_ndx]
            print(corr)

        return 1.0 / corr

def hpm_laplacian(image, box_size=5, nsigma_hot=4.0):
    """
    Required Input:
    :param image:           A 2D image array of photon counts.

    Other Input:
    :param box_size:           Scalar integer. Size box used for replacing the NaNs in the region surrounding each dead or NaN pixel.
    :param nsigma_hot:         Scalar float. If the flux ratio for a pixel is (nsigma_hot x expected error)
                                             above the max expected given the PSF FWHM, then flag it as hot.
    :return:
    A dictionary containing the result and various diagnostics. Keys are:

    'bad_mask': the main output. Contains a 2D array of integers of the same shape as the input image, where:
        0 = Good pixel
        1 = Hot pixel
        2 = Cold Pixel
        3 = Dead Pixel
    'dead_mask': 2D array of Bools of the same shape as the input image, where:
                True = Dead Pixel
                False = Not Dead Pixel
    'hot_mask': 2D array of Bools of the same shape as the input image, where:
                True = Hot Pixel
                False = Not Hot Pixel
    'image': 2D array containing the input image
    'laplacian_filter_image': The median-filtered image
    """

    raw_image = np.copy(image)

    # Assume everything with 0 counts is a dead pixel, turn dead pixel values into NaNs
    dead_mask = raw_image < 0.01
    raw_image[dead_mask] = np.nan

    # Initialise a mask for hot pixels (all False)
    hot_mask = np.zeros(shape=np.shape(raw_image), dtype=bool)

    nan_fixed_image = utils.replaceNaN(raw_image, mode='mean', boxsize=box_size)
    assert np.all(np.isfinite(nan_fixed_image))

    # In the case that *all* the pixels are dead, return a bad_mask where all the pixels are flagged as DEAD
    if np.sum(raw_image[np.where(np.isfinite(raw_image))]) <= 0:
        log.info('Entire image consists of dead pixels')
        bad_mask=dead_mask*3
        hot_mask=numpy.zeros_like(bad_mask, dtype=bool)
        dead_mask=numpy.ones_like(bad_mask, dtype=bool)
    else:
        laplacian_filter_image = spfilters.laplace(nan_fixed_image)
        threshold_laplace = -(np.std(laplacian_filter_image) + nsigma_hot * np.std(laplacian_filter_image))
        hot_pix=np.where(laplacian_filter_image<threshold_laplace)
        hot_pix_x=hot_pix[0]
        hot_pix_y=hot_pix[1]

        for i in np.arange(len(hot_pix_x)):
            pix_center=laplacian_filter_image[hot_pix_x[i],hot_pix_y[i]]
            pix_up = laplacian_filter_image[hot_pix_x[i], hot_pix_y[i]+1]
            pix_down = laplacian_filter_image[hot_pix_x[i], hot_pix_y[i]-1]
            pix_left = laplacian_filter_image[hot_pix_x[i]-1, hot_pix_y[i]]
            pix_right = laplacian_filter_image[hot_pix_x[i]+1, hot_pix_y[i]]
            if pix_up > 0 and pix_down > 0 and pix_left > 0  and pix_right > 0:
                hot_mask[hot_pix_x[i], hot_pix_y[i]]=True

        dead_mask_return=dead_mask*3
        hot_mask_return=hot_mask*1
        bad_mask=dead_mask_return+hot_mask_return

    return {'bad_mask': bad_mask, 'dead_mask': dead_mask, 'hot_mask': hot_mask,  'image': raw_image,
            'laplacian_filter_image': laplacian_filter_image}

def hpm_poisson_dist(obsfile):
    """
    :param obsfile:
    :return:
    """
    xpix = obsfile.nXPix
    ypix = obsfile.nYPix

    for iRow in range(xpix):
        for iCol in range(ypix):
            times = obsfile.getPixelPhotonList(iRow, iCol)
            times = times['Time']
            lc = getLightCurve(times, startTime=0, stopTime=600000000, effExpTime=1000000)
            lc2 = lc[2] / 1000000
            lc0 = lc[0]

            hist = histogramLC(lc0)
            lam = np.sum(hist[0] * hist[1]) / np.sum(hist[0])

            poissondist = poisson.pmf(hist[1], lam) * np.sum(hist[0])
            chisq = np.sum(((hist[0] - poissondist) ** 2.0) / poissondist)
            print(chisq)



def cps_cut_img (image, sigma=5, max_cut=2450, cold_mask=False):
    """
    NOTE:  This is a routine for masking hot pixels in .img files and the like, NOT robust
            OK to use for bad pixel masking for pretty-picture-generation or quicklook
    Finds the hot and dead pixels in a for a 2D input array.
    Input can be a stack, in that case, the median image will be generated
    First order:  masks everything with a count level higher than the max cut
                  If the pixel has counts less than 0.01, then the pixel is flagged as DEAD
    Second order: masks everything with a count level higher than the mean count rate + sigma*st. dev
    If cold_mask=True, third cut where everything with a count level less than mean count rate - sigma*stdev

    The HOT and DEAD masks are combined into a single BAD mask at the end

    Required Input:
    :param image:           A 2D array of photon counts: can be (nx, ny) or (ny, nx) --> true if made with loadStack

    Other Input:
    :param sigma:             Scalar float.  Cut pixels this number of standard deviations away from the mean
    :param max_cut:           Scalar integer.  Initial cut, 2450 as per the detector max
    :param cold_mask:         Boolean.  If True, *cold* pixels will also be masked

    :return:
    A dictionary containing the result and various diagnostics. Keys are:
    'bad_mask': the main output. Contains a 2D array of integers shaped as (nx, ny), where:
            0 = Good pixel
            1 = Hot pixel
            2 = Cold Pixel
            3 = Dead Pixel
    'image': 2D array containing the input image. Regardless of input array format, stack will be returned as
                                                  (nx, ny) just to standardize things
    """
    image_count=np.array(np.shape(image))
    if len(image_count) >= 2:
        raw_image = utils.medianStack(image)

    else:
        raw_image = np.array(image)

    #Check if input is (ny, nx) instead of (nx, ny), if so, rearrange it (for purposes of standardization)
    image_count_raw = np.array(np.shape(raw_image))
    if image_count_raw[0] > image_count_raw[1]:
        raw_image = raw_image.T

    # initial masking, flag dead pixels (counts < 0.01) and flag anything with cps > maxCut as hot
    bad_mask = np.zeros_like(raw_image)
    bad_mask[np.where(raw_image<=0.01)]=3
    bad_mask[np.where(raw_image>=max_cut)]=1

    #second round of masking, flag where cps > mean+sigma*std as hot
    with warnings.catch_warnings():
        # nan values will give an unnecessary RuntimeWarning
        warnings.simplefilter("ignore", category=RuntimeWarning)
        bad_mask[np.where(raw_image>=np.nanmedian(raw_image)+sigma*np.nanstd(raw_image))]=1

    #if coldCut is true, also mask cps < mean-sigma*std
    if cold_mask==True:
        with warnings.catch_warnings():
            # nan values will give an unnecessary RuntimeWarning
            warnings.simplefilter("ignore", category=RuntimeWarning)
            bad_mask[np.where(raw_image<=np.nanmedian(raw_image)-sigma*np.nanstd(raw_image))]=1

    return {'bad_mask': bad_mask, 'image': raw_image}


def cps_cut_stack(stack, len_stack, sigma=4, max_cut=2450, cold_mask=False, verbose=False):
    """
    Finds the hot and dead pixels in a stack of images
    Same procedure as cps_check_img except this procedure operates on a STACK of images (again, NOT robust!!)

    Required Input:
    :param stack:           A 3D array of (nx, ny, len_stack)
                            OR (len_stack, ny, nx)--> true if input was made using loadStack
    :param len_stack:       length of the stack of images

    Other Input:
    :param sigma:             Scalar float.  Cut pixels this number of standard deviations away from the mean
    :param max_cut:           Scalar integer.  Initial cut, 2450 as per the detector max
    :param cold_mask:         Boolean.  If True, *cold* pixels will also be masked

    :return:
    A dictionary containing the result and various diagnostics. Keys are:
    'bad_mask_stack': the main output. Contains a stack shaped as (nx, ny, len_stack) where:
            0 = Good pixel
            1 = Hot pixel
            2 = Cold Pixel
            3 = Dead Pixel
    'stack': 2D array containing the stack of images.  Regardless of input array format, stack will be returned as
                                                       (nx, ny, len_stack) just to standardize things
    """
    # rearrange to make sure that the format is (nx, ny, len_stack)
    stack = np.array(stack)
    image_count=np.array(np.shape(stack))
    index = np.where(image_count == len_stack)
    if index == 0:
        stack = stack.T

    bad_mask_stack = np.zeros_like(stack)

    for i in range(len_stack):
        raw_image = stack[:,:,i]

        # initial masking, flag dead pixels (counts < 0.01) and flag anything with cps > maxCut as hot
        bad_mask = np.zeros_like(raw_image)
        bad_mask[np.where(raw_image<=0.01)]=3
        bad_mask[np.where(raw_image>=max_cut)]=1

        #second round of masking, flag where cps > mean+sigma*std as hot
        with warnings.catch_warnings():
            # nan values will give an unnecessary RuntimeWarning
            warnings.simplefilter("ignore", category=RuntimeWarning)
            bad_mask[np.where(raw_image>=np.nanmean(raw_image)+sigma*np.nanstd(raw_image))]=1

        #if coldCut is true, also mask cps < mean-sigma*std
        if cold_mask==True:
            with warnings.catch_warnings():
                # nan values will give an unnecessary RuntimeWarning
                warnings.simplefilter("ignore", category=RuntimeWarning)
                bad_mask[np.where(raw_image<=np.nanmean(raw_image)-sigma*np.nanstd(raw_image))]=1

    bad_mask_stack[:,:,i] = bad_mask

    return {'bad_mask_stack': bad_mask_stack, 'stack': stack}

def getanumber2(method, extradiv=3, *args, **kwargs):
	number=method(*args, **kwargs)
	number=number/extradiv
	return(number)


def find_bad_pixels(obsfile, hpcutmethod, time_step=30, start_time=0, end_time= -1, **hpcutargs, **hpcutkwargs):
    """
    This routine is the main code entry point of the bad pixel masking code.
    Takes an obs. file as input and writes a 'bad pixel table' to that h5 file where each entry is an indicator of
    whether the pixel was good, dead, hot, or cold.  Defaults should be somewhat reasonable for a typical on-sky image.
    HPCut method is interchangeable with any of the methods listed here.

    The HOT and DEAD masks are combined into a single BAD mask at the end

    Required Input:
    :param obsfile:           user passes an obsfile instance here
    :param start_time         Scalar Integer.  Timestamp at which to begin bad pixel masking, default = 0
    :param end_time           Scalar Integer.  Timestamp at which to finish bad pixel masking, default = -1
                                               (run to the end of the file)
    :param time_step          Scalar Integer.  Number of seconds to do the bad pixel masking over (should be an integer
                                               number of steps through the obsfile), default = 30
    :param hpcutmethod        String.          Method to use to detect hot pixels.  Options are:
                                               hpm_median_movingbox
                                               hpm_flux_threshold
                                               hpm_laplacian
                                               cps_cut_img

    Other Input:
    Appropriate args and kwargs that go into the chosen hpcut function

    :return:
    Writes a 'bad pixel table' to an output h5 file titled 'badpixmask_--timestamp--.h5' where:
            0 = Good pixel
            1 = Hot pixel
            2 = Cold Pixel
            3 = Dead Pixel
    """
    #A few defaults that will be used in the absence of parameter file or arguments provided by the caller.
    if time_step is None: time_step = 30
    if start_time is None: start_time = 0
    if end_time is None: pass

    #Some arguments necessary to be passed into getPixelCountImage
    if self.info['isSpecCalibrated']:  applyWeight = True
    else:  applyWeight = False
    if method==cps_cut_img: scaleByEffInt = True
    else:  scaleByEffInt = False
    applyTPFWeight=False #Change once we write the noise calibration

    exp_time = obsfile.getFromHeader('exptime')
    if end_time < 0: end_time = exp_time
    step_starts = np.arange(start_time, end_time, time_step)  # Start time for each step (in seconds).
    step_ends = step_starts + time_step  # End time for each step
    nsteps = len(step_starts)
    assert (np.sum(step_ends > end_time) <= 1)  # Shouldn't be more than one end time that runs over
    step_ends[step_ends > end_time] = end_time  # Clip any time steps that run over the end of the requested time range.
    assert (np.all((step_starts >= start_time) & (step_starts <= end_time)))
    assert (np.all((step_starts >= start_time) & (step_ends <= end_time)))

    # Initialise stack of masks, one for each time step
    hot_masks = np.zeros([obsfile.xpix, obsfile.ypix, nsteps], dtype=np.int8)
    dead_masks = np.zeros([obsfile.xpix, obsfile.ypix, nsteps], dtype=np.int8)

    #Generate a stack of bad pixel mask, one for each time step
    for i, each_time in enumerate(step_starts):
        log.info('Processing time slice: '.format(str(each_time)) + ' - ' + str(each_time + time_step) + 's')
        raw_image_dict = obsfile.getPixelCountImage(firstSec=each_time, integrationTime=time_step, applyWeight=applyWeight,
                                                    applyTPFWeight=applyTPFWeight, scaleByEffInt=scaleByEffInt)
        bad_pixel_solution = hpcutmethod(image = raw_image_dict['image'], *hpcutargs, **hpcutkwargs)
        dead_masks[:,:,i] = bad_pixel_solution['dead_mask']
        hot_masks[:, :, i] = bad_pixel_solution['hot_mask']

    #Combine the bad pixel masks into a master mask
    dead_pixel_mask = (np.sum(dead_masks, axis=-1)/nsteps)*3
    hot_pixel_mask = np.sum(hot_masks, axis=-1)/nsteps
    bad_pixel_mask=dead_pixel_mask + hot_pixel_mask

    #Write it all out to the obs file

    obsfile.write_bad_pixels(obsfile, bad_pixel_mask)


def save_mask_array(mask= None, out_directory= None):
    """
     Write bad pixel mask to desired output directory

     :param mask:  Bad pixel array from cps_check_img
                   OR bad pixel stack from cps_check_stack
     :param out_directory:  Output directory that the bad pixel mask should be written

     :return:
     Saves .npz file in output directory titled 'badpixmask_--timestamp--.npz'

    """
    badpix_file_name = 'badpixmask_{}'.format(timestamp)
    mask_name = os.path.join(out_directory, badpix_file_name)

    np.savez(mask_name, mask=mask)

    return

def load_mask_array(file_path= None):
    """
    Load and return a mask

    :param file_path:  Path and name of the file containing the mask (np or npz file)

    :return:
    Mask or stack of masks
    """
    mask_file = np.load(file_path)
    mask = mask_file['mask']
    mask_file.close()
    return mask


def plot_mask_array(mask= None, image=None, title=None):
    """
    Plot a mask or masked image for viewing

    Required Input:
    :param mask:  Np array containing mask

    Other Input:
    :param title:  Title of the plot.  If no title is provided, the default title 'Bad Pixel Mask' will be used
    :param image:  Np array containing raw image.  If no image is provided, just the mask will be plotted

    :return:
    Display a plot of the mask or masked image for inspection
    """
    if not title:
        title='Bad Pixel Mask'

    #If image is provided, mask the bad pixels for plotting
    if image:
        image[np.where(mask > 0)]== np.NaN
        image_to_plot=image
    else:
        image_to_plot=mask
    try:
        plot_array(image_to_plot, title=title)
    except:
        plt.matshow(image_to_plot)
        plt.show()



if __name__ == "__main__":
    timestamp = int(datetime.utcnow().timestamp())
    # read in command line arguments
    parser = argparse.ArgumentParser(description='MKID Bad Pixel Masking Utility')
    parser.add_argument('--quiet', action='store_true', dest='quiet',
                        help='Disable logging')

    if not args.quiet:
        mkidcore.corelog.create_log('badpix', logfile='badpix_{}.log'.format(timestamp), console=False, propagate=False,
                                    fmt='%(levelname)s %(message)s', level=mkidcore.corelog.INFO)






