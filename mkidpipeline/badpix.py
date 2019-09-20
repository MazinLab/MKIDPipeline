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
                    c.  hpm_median_movingbox:  New method to find hot pixels using a median moving-box
                    d.  cps_cut_image:  Basic hot pixel search using a count-per-second threshold (not robust)

                  Bad Pixel Masking via Time Streams --> Best for uncalibrated obs data or calibration data
                    a.  hpm_poisson_dist:  Identifies pixels whose histogram lightcurves do not obey Poisson stats as
                                           potential HPs.  #TODO Develop this to iterate over all pixels

-------------
Functions for first-order HP masking, plotting, viewing (takes an image or numpy array as input):

hpm_cps_cut:    Creates a 2D bad pixel mask for a given time interval within a given
                exposure using a count-per-second threshold routine.  USE THIS FOR DATA VIEWING/QUICK CHECKS

save_mask_array:  Write bad pixel mask to desired output directory

load_mask_array:  Load and return a mask

plot_mask_array:  Plot a mask or masked image for viewing

-------------
Functions for rigorous HP masking of obs data (takes an image or numpy array as input):

hpm_flux_threshold: Creates a 2D bad pixel mask for a given time interval within a given
                    exposure using a robust PSF-comparison method.  USE THIS FOR DATA REDUCTION

hpm_median_movingbox:  Creates a 2D bad pixel mask for a given time interval within a given exposure using a
                       median-moving-box.  Is extremely variable to input parameters, requires a lot of find-tuning

hpm_laplacian:  Creates a 2D bad pixel mask for a given time interval within a given exposure using a Laplacian filter
                based on approximate second derivatives.
                Works all right on 51Eri Dither, got around 80% of HPs

hpm_poisson_dist:  Checks if photons arriving at pixels are obeying Poisson stats

-------------


"""

import os.path
from datetime import datetime
import warnings
import argparse
import numpy as np

import scipy.ndimage.filters as spfilters
from mkidpipeline.speckle.binned_rician import *
import scipy.stats

from mkidpipeline.hdf.photontable import ObsFile
from mkidpipeline.utils import utils
from mkidpipeline.utils.plottingTools import plot_array as pa
from mkidcore.corelog import getLogger
import mkidcore.corelog
import mkidcore.pixelflags as pixelflags


def _calc_stdev(x):
    return np.nanstd(x) * _stddev_bias_corr((~np.isnan(x)).sum())

def _stddev_bias_corr(n):
    if n == 1:
        corr = 1.0
    else:
        lut = [0.7978845608, 0.8862269255, 0.9213177319, 0.9399856030, 0.9515328619,
               0.9593687891, 0.9650304561, 0.9693106998, 0.9726592741, 1.0]
        lut_ndx = max(min(n - 2, len(lut) - 1), 0)
        corr = lut[lut_ndx]
    return 1.0 / corr


def hpm_flux_threshold(image, fwhm=4, box_size=5, nsigma_hot=4.0, max_iter=5, dead_threshold=0,
                       use_local_stdev=False, bkgd_percentile=50.0, dead_mask=None, min_background_sigma=.01):
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
    :param dead_threshold:     Scalar integer. If a dead_mask is not given a dead_mask is created with this value as the
                                               threshold. Defaults to 0.
    :param use_local_stdev:    Bool.  If True, use the local (robust) standard deviation within the
                                      moving box for the sigma value in the hot pixel thresholding
                                      instead of Poisson statistics. Mainly intended for situations where
                                      you know there is no astrophysical source in the image (e.g. flatfields,
                                      laser calibrations), where you can also set fwhm=np.inf
    :param bkgd_percentile:    Scalar Integer.  Percentile level (in %) in image to use as an estimate of the background.
                                                In an ideal world, this will be 50% (i.e., the median of the image).
                                                For raw images, however, there is often a gradient across the field,
                                                in which case it's sensible to use something lower than 50%.
    :param dead_mask:         Integer array. The input dead pixel mask.
    :param min_background_sigma: Scalar float. Minimum counts for the pixel to be flagged as DEAD

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
    'masked_image': The hot and dead pixel masked image
    'median_filter_image': The median-filtered image
    'max_ratio': 2D array - the maximum allowed ratio between pixel flux and the median-filtered image
    'difference_image': the difference between the input image and an image representing the max allowed flux in each pixel.
    'difference_image_error': the expected error in the difference calculation.
    'num_iter': number of iterations performed.
    """

    raw_image = np.copy(image)

    # Approximate peak/median ratio for an ideal (Gaussian) PSF sampled at
    # pixel locations corresponding to the median kernel used with the real data.
    gauss_array = utils.gaussian_psf(fwhm, box_size)
    max_ratio = np.max(gauss_array) / np.median(gauss_array)

    # Assume everything with 0 counts is a dead pixel, turn dead pixel values into NaNs
    if dead_mask is None:
        dead_mask = raw_image == dead_threshold
    raw_image[dead_mask] = np.nan

    # Initialise a mask for hot pixels (all False) for comparison on each iteration.
    initial_hot_mask = np.zeros_like(raw_image, dtype=bool)
    hot_mask = np.zeros_like(raw_image, dtype=bool)

    # Initialise some arrays with NaNs in case they don't get filled out during the iteration
    median_filter_image = np.full_like(raw_image, np.nan)
    difference_image = np.full_like(raw_image, np.nan)
    difference_image_error = np.full_like(raw_image, np.nan)

    # In the case that *all* the pixels
    # are dead, return a bad_mask where all the pixels are flagged as DEAD
    if raw_image[np.isfinite(raw_image)].sum() <= 0:
        getLogger(__name__).info('Entire image consists of dead pixels')
        bad_mask = dead_mask * pixelflags.badpixcal['dead']
        hot_mask = np.zeros_like(bad_mask, dtype=bool)
        iteration = -1
    else:
        for iteration in range(max_iter):
            getLogger(__name__).info('Doing iteration: {}'.format(iteration))
            # Remove all the NaNs in an image and calculate a median filtered image
            # each pixel takes the median of itself and the surrounding box_size x box_size box.
            nan_fixed_image = utils.replaceNaN(raw_image, mode='mean', boxsize=box_size)
            assert np.all(np.isfinite(nan_fixed_image))
            median_filter_image = spfilters.median_filter(nan_fixed_image, box_size, mode='mirror')

            overall_median = np.nanmedian(raw_image)
            overall_bkgd = np.percentile(raw_image[~np.isnan(raw_image)], bkgd_percentile)

            # Estimate the background std. dev.
            standard_filter_image = utils.nearestNrobustSigmaFilter(raw_image, n=box_size ** 2 - 1)
            overall_bkgd_sigma = max(min_background_sigma, np.nanmedian(standard_filter_image))
            standard_filter_image.clip(1, None, standard_filter_image)

            # Calculate difference between flux in each pixel and max_ratio * the median in the enclosing box.
            # Also calculate the error that would exist in a measurement of a pixel that *was* at the peak of a real PSF
            # Condition for flagging is:
            #        (flux - background)/(box median - background) > max_ratio.
            # Or:
            #        flux > max_ratio*median - background*(max_ratio-1)
            # If the threshold is *lower* than the background, then set it equal to the background level instead
            # (a pixel below the background level is unlikely to be hot!)
            getLogger(__name__).debug('overall_median: {}'.format(overall_median))
            getLogger(__name__).debug('overall_bkgd: {}'.format(overall_bkgd))
            getLogger(__name__).debug('overall_bkgd_sigma: {}'.format(overall_bkgd_sigma))
            getLogger(__name__).debug('max_ratio: {}'.format(max_ratio))
            threshold = np.maximum((max_ratio * median_filter_image - (max_ratio - 1) * overall_bkgd), overall_bkgd)
            difference_image = raw_image - threshold

            # Simple estimate, photon error in the max value allowed. Neglect errors in the median itself here.
            # Add in quadrature imaginary photon noise in the expected threshold level and background random noise
            if use_local_stdev is False:
                difference_image_error = np.sqrt(threshold + overall_bkgd_sigma ** 2)
            else:
                # use the local (robust) standard deviation within the moving box
                difference_image_error = standard_filter_image

            # Any pixel that has a peak/median ratio more than nSigma above the maximum ratio should be flagged as hot:
            # True = bad pixel; False = good pixel.
            hot_mask = (difference_image > (nsigma_hot * difference_image_error)) | initial_hot_mask

            # If no change between between this and the last iteration then stop iterating
            if np.all(hot_mask == initial_hot_mask):
                break

            # Otherwise update 'initial_hot_mask' and set all detected bad pixels to NaN for the next iteration
            initial_hot_mask = np.copy(hot_mask)
            raw_image[hot_mask] = np.nan

        # Finished with loop, make sure a pixel is not simultaneously hot and dead
        assert (~(hot_mask & dead_mask)).all()
        bad_mask = np.zeros_like(raw_image) \
            + dead_mask * pixelflags.badpixcal['dead'] \
            + hot_mask * pixelflags.badpixcal['hot']

    return {'hot_mask': hot_mask, 'masked_image': raw_image, 'image': image, 'bad_mask': bad_mask,
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
    sample sizes as per the function _stddev_bias_corr(n)

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
        getLogger(__name__).info('Entire image consists of dead pixels')
        bad_mask = dead_mask * 3
        hot_mask = np.zeros_like(bad_mask, dtype=bool)
        dead_mask = np.ones_like(bad_mask, dtype=bool)
    else:
        for iteration in range(max_iter):
            getLogger(__name__).info('Iteration: '.format(iteration))
            # Remove all the NaNs in an image and calculate a median filtered image
            # each pixel takes the median of itself and the surrounding box_size x box_size box.
            nan_fixed_image = utils.replaceNaN(raw_image, mode='mean', boxsize=box_size)
            assert np.all(np.isfinite(nan_fixed_image))
            median_filter_image = spfilters.median_filter(nan_fixed_image, box_size, mode='mirror')
            func = lambda x: np.nanstd(x) * _stddev_bias_corr((~np.isnan(x)).sum())
            standard_filter_image = spfilters.generic_filter(nan_fixed_image, func, box_size, mode='mirror')

            threshold = median_filter_image + (nsigma_hot * standard_filter_image)

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
        dead_mask_return = dead_mask * 3
        hot_mask_return = hot_mask * 1
        bad_mask = dead_mask_return + hot_mask_return

    return {'bad_mask': bad_mask, 'dead_mask': dead_mask, 'hot_mask': hot_mask, 'image': raw_image,
            'standard_filter_image': standard_filter_image, 'median_filter_image': median_filter_image,
            'num_iter': iteration + 1}


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
        getLogger(__name__).info('Entire image consists of dead pixels')
        bad_mask = dead_mask * 3
        hot_mask = np.zeros_like(bad_mask, dtype=bool)
        dead_mask = np.ones_like(bad_mask, dtype=bool)
    else:
        laplacian_filter_image = spfilters.laplace(nan_fixed_image)
        threshold_laplace = -(np.std(laplacian_filter_image) + nsigma_hot * np.std(laplacian_filter_image))
        hot_pix = np.where(laplacian_filter_image < threshold_laplace)
        hot_pix_x = hot_pix[0]
        hot_pix_y = hot_pix[1]

        for i in np.arange(len(hot_pix_x)):
            pix_center = laplacian_filter_image[hot_pix_x[i], hot_pix_y[i]]
            pix_up = laplacian_filter_image[hot_pix_x[i], hot_pix_y[i] + 1]
            pix_down = laplacian_filter_image[hot_pix_x[i], hot_pix_y[i] - 1]
            pix_left = laplacian_filter_image[hot_pix_x[i] - 1, hot_pix_y[i]]
            pix_right = laplacian_filter_image[hot_pix_x[i] + 1, hot_pix_y[i]]
            if pix_up > 0 and pix_down > 0 and pix_left > 0 and pix_right > 0:
                hot_mask[hot_pix_x[i], hot_pix_y[i]] = True

        dead_mask_return = dead_mask * 3
        hot_mask_return = hot_mask * 1
        bad_mask = dead_mask_return + hot_mask_return

    return {'bad_mask': bad_mask, 'dead_mask': dead_mask, 'hot_mask': hot_mask, 'image': raw_image,
            'laplacian_filter_image': laplacian_filter_image}


def hpm_poisson_dist(obsfile):
    """
    Required Input:
    :param obsfile:  The input obsfile
           xpix:     X-coordinate of the pixel whose lightcurve histogram we want to analyze
           ypix:     Y-coordinate of pixel

    :return:
    A dictionary containing the results of the poisson fit.  Keys are:
           poisson_dist:  The poisson distribution fit to the pixel histogram lightcurve
           chisq:          The chisquare of that fit
    """
    raise NotImplementedError

    stop_time = float(obsfile.getFromHeader('expTime'))
    times = obsfile.getPixelPhotonList(xpix, ypix)
    times = times['Time']
    lc = getLightCurve(times, startTime=0, stopTime=stop_time * 10e6, effExpTime=1000000)
    lc0 = lc[0]

    hist = histogramLC(lc0)
    lam = np.sum(hist[0] * hist[1]) / np.sum(hist[0])

    poisson_dist = scipy.stats.poisson.pmf(hist[1], lam) * np.sum(hist[0])
    chisq = np.sum(((hist[0] - poisson_dist) ** 2.0) / poisson_dist)
    return {'poisson_dist': poisson_dist, 'chisq': chisq}


def hpm_cps_cut(image, sigma=5, max_cut=2450, cold_mask=False):
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
    image_count = np.array(np.shape(image))
    if len(image_count) >= 3:
        raw_image = numpy.nanmedian(image, axis=0)
    else:
        raw_image = np.array(image)

    # Check if input is (ny, nx) instead of (nx, ny), if so, rearrange it (for purposes of standardization)
    if raw_image.shape[0] > raw_image.shape[1]:
        raw_image = raw_image.T

    # initial masking, flag dead pixels (counts < 0.01) and flag anything with cps > maxCut as hot
    hot_mask = np.zeros_like(raw_image)
    dead_mask = np.zeros_like(raw_image)
    hot_mask[raw_image >= max_cut] = pixelflags.badpixcal['hot']
    dead_mask[raw_image <= 0.01] = pixelflags.badpixcal['dead']

    # second round of masking, flag where cps > mean+sigma*std as hot
    with warnings.catch_warnings():
        # nan values will give an unnecessary RuntimeWarning
        warnings.simplefilter("ignore", category=RuntimeWarning)
        hot_mask[raw_image >= np.nanmedian(raw_image) + sigma * np.nanstd(raw_image)] = pixelflags.badpixcal['hot']

    # if coldCut is true, also mask cps < mean-sigma*std
    if cold_mask:
        with warnings.catch_warnings():
            # nan values will give an unnecessary RuntimeWarning
            warnings.simplefilter("ignore", category=RuntimeWarning)
            hot_mask[raw_image <= np.nanmedian(raw_image) - sigma * np.nanstd(raw_image)] = pixelflags.badpixcal['hot']

    bad_mask = dead_mask + hot_mask

    return {'bad_mask': bad_mask, 'dead_mask': dead_mask, 'hot_mask': hot_mask, 'image': raw_image}


def mask_hot_pixels(file, method='hpm_flux_threshold', step=30, startt=0, stopt=None, ncpu=1, **methodkw):
    """
    This routine is the main code entry point of the bad pixel masking code.
    Takes an obs. file as input and writes a 'bad pixel table' to that h5 file where each entry is an indicator of
    whether the pixel was good, dead, hot, or cold.  Defaults should be somewhat reasonable for a typical on-sky image.
    HPCut method is interchangeable with any of the methods listed here.

    The HOT and DEAD masks are combined into a single BAD mask at the end

    Required Input:
    :param obsfile:           user passes an obsfile instance here
    :param startt         Scalar Integer.  Timestamp at which to begin bad pixel masking, default = 0
    :param stopt           Scalar Integer.  Timestamp at which to finish bad pixel masking, default = -1
                                               (run to the end of the file)
    :param step          Scalar Integer.  Number of seconds to do the bad pixel masking over (should be an integer
                                               number of steps through the obsfile), default = 30
    :param hpcutmethod        String.          Method to use to detect hot pixels.  Options are:
                                               hpm_median_movingbox
                                               hpm_flux_threshold
                                               hpm_laplacian
                                               hpm_cps_cut

    Other Input:
    Appropriate args and kwargs that go into the chosen hpcut function

    :return:
    Writes a 'bad pixel table' to an output h5 file titled 'badpixmask_--timestamp--.h5' where:
            0 = Good pixel
            1 = Hot pixel
            2 = Cold Pixel
            3 = Dead Pixel
    """
    obsfile = ObsFile(file)
    if stopt is None:
        stopt = obsfile.getFromHeader('expTime')
    assert startt < stopt
    if step < stopt-startt:
        getLogger(__name__).warning(('Hot pixel step time longer than exposure time by {:.0f} s, using full '
                                     'exposure').format(stopt-startt-step))
        step = stopt-startt

    step_starts = np.arange(startt, stopt, step, dtype=int)  # Start time for each step (in seconds).
    step_ends = step_starts + int(step)  # End time for each step
    step_ends[step_ends > stopt] = int(stopt)  # Clip any time steps that run over the end of the requested time range.

    # Initialise stack of masks, one for each time step
    hot_masks = np.zeros([obsfile.nXPix, obsfile.nYPix, step_starts.size], dtype=bool)
    func = globals()[method]

    # Generate a stack of bad pixel mask, one for each time step
    for i, each_time in enumerate(step_starts):
        getLogger(__name__).info('Processing time slice: {} - {} s'.format(each_time, each_time + step))
        raw_image_dict = obsfile.getPixelCountImage(firstSec=each_time, integrationTime=step,
                                                    applyWeight=True, applyTPFWeight=True,
                                                    scaleByEffInt=method == 'hpm_cps_cut')
        bad_pixel_solution = func(raw_image_dict['image'], dead_mask=obsfile.pixelMask, **methodkw)
        hot_masks[:, :, i] = bad_pixel_solution['hot_mask']

    # Combine the bad pixel masks into a master mask
    obsfile.enablewrite()
    obsfile.applyHotPixelMask(np.any(hot_masks, axis=-1))
    obsfile.disablewrite()
