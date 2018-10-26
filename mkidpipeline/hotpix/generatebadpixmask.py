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

Main routines of interest are:

-------------
find_bad_pixels:  currently the main routine. Takes an obs. file (or input h5 file) as input and generates
                  a separate bad pixel h5 file named '--timestamp--_badpixmask.h5'

quick_check:    creates a 2D bad pixel mask for a given time interval within a given
                exposure using a count-per-second threshold routine.  USE THIS FOR DATA VIEWING/QUICK CHECKS

check_interval: creates a 2D bad pixel mask for a given time interval within a given
                exposure using a robust PSF-comparison method.  USE THIS FOR DATA REDUCTION

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

from mkidpipeline.hdf.darkObsFile import ObsFile
from mkidpipeline.utils import utils
import mkidcore.corelog as pipelinelog
from mkidcore.corelog import getLogger
from matplotlib.backends.backend_pdf import PdfPages


def check_interval(image=None, fwhm=2.5, box_size=5, nsigma_hot=4.0, max_iter=5,
                   use_local_stdev=False, bkgd_percentile=50.0):
    """
    Robust!  NOTE:  This is the routine that should be used for observational data in the complete pipeline.
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
    if np.sum(raw_image[np.where(np.isfinite(raw_image))]) <= 0:
        print('Entire image consists of dead pixels')
        bad_mask=dead_mask*3
        hot_mask=numpy.zeros_like(bad_mask, dtype=bool)
        dead_mask=numpy.ones_like(bad_mask, dtype=bool)
    else:
        for iteration in range(max_iter):
            print('Iteration: ', iteration)
            # Remove all the NaNs in an image and calculate a median filtered image
            # each pixel takes the median of itself and the surrounding box_size x box_size box.
            nan_fixed_image = utils.replaceNaN(raw_image, mode='mean', box_size=box_size)
            assert np.all(np.isfinite(nan_fixed_image))
            median_filter_image = spfilters.median_filter(nan_fixed_image, box_size, mode='mirror')

            overall_median = np.median(raw_image[~np.isnan(raw_image)])
            overall_bkgd = np.percentile(raw_image[~np.isnan(raw_image)], bkgd_percentile)

            # Estimate the background std. dev.
            standard_filter_image = utils.nearestNrobustSigmaFilter(raw_image, n=box_size ** 2 - 1)
            overall_bkgd_sigma = np.median(standard_filter_image[np.isfinite(standard_filter_image)])
            standard_filter_image[np.where(standard_filter_image < 1.)] = 1.
            if overall_bkgd_sigma < 0.01: overall_bkgd_sigma = 0.01  # Just so it's not 0

            # Calculate difference between flux in each pixel and max_ratio * the median in the enclosing box.
            # Also calculate the error that would exist in a measurement of a pixel that *was* at the peak of a real PSF
            # Condition for flagging is:
            #        (flux - background)/(box median - background) > max_ratio.
            # Or:
            #        flux > max_ratio*median - background*(max_ratio-1)
            # If the threshold is *lower* than the background, then set it equal to the background level instead
            # (a pixel below the background level is unlikely to be hot!)
            print('overall_median: ', overall_median)
            print('overall_bkgd: ', overall_bkgd)
            print('overall_bkgd_sigma: ', overall_bkgd_sigma)
            print('max_ratio: ', max_ratio)
            threshold = np.maximum((max_ratio * median_filter_image - (max_ratio - 1.) * overall_bkgd), overall_bkgd)
            difference_image = raw_image - threshold

            # Simple estimate, photon error in the max value allowed. Neglect errors in the median itself here.
            # Add in quadrature imaginary photon noise in the expected threshold level and background random noise
            if use_local_stdev is False:
                difference_image_error = np.sqrt(threshold + overall_bkgd_sigma ** 2)  # Note threshold = sqrt(threshold)**2
            else:
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

#TODO print-> logging


def quick_check_img (image=None, sigma=None, max_cut=2450, cold_mask=False):
    """
    NOTE:  This is a quick routine for masking hot pixels in .img files and the like, NOT robust
            OK to use for bad pixel masking for pretty-picture-generation or quicklook
    Finds the hot and dead pixels in a for a 2D input array.
    Input can be a stack, in that case, the median image will be generated
    First order:  masks everything with a count level higher than the max cut
                  If the pixel has counts less than 0.01, then the pixel is flagged as DEAD
    Second order: maks everythng with a count level higher than the mean count rate + sigma*st. dev
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
        bad_mask[np.where(raw_image>=np.nanmean(raw_image)+sigma*np.nanstd(raw_image))]=1

    #if coldCut is true, also mask cps < mean-sigma*std
    if cold_mask==True:
        with warnings.catch_warnings():
            # nan values will give an unnecessary RuntimeWarning
            warnings.simplefilter("ignore", category=RuntimeWarning)
            bad_mask[np.where(raw_image<=np.nanmean(raw_image)-sigma*np.nanstd(raw_image))]=1

    return {'bad_mask': bad_mask, 'image': raw_image}


def quick_check_stack(stack=None, len_stack=None, sigma=None, max_cut=2450, cold_mask=False):
    """
    Finds the hot and dead pixels in a stack of images
    Same procedure as quick_check except this procedure operates on a STACK of images (again, NOT robust!!)

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


def find_bad_pixels(input_filename=None, obsfile=None, time_step=1, start_time=0, end_time= -1, fwhm=2.5,
                    box_size=5, nsigma_hot=4.0, weighted=False, flux_weighted=False, max_iter=5,
                    use_local_stdev=False, use_raw_counts=True, bkgd_percentile=50.0):
    """
    This routine is the main code entry point of the bad pixel masking code.
    Takes an obs. file as input and writes a 'bad pixel table' to that h5 file where each entry is an indicator of
    whether the pixel was good, dead, hot, or cold.  Defaults should be somewhat reasonable for a typical on-sky image.

    The HOT and DEAD masks are combined into a single BAD mask at the end

    Required Input:
    :param input_filename:    string, pathname of input observation file.
    OR
    :param obsfile:           user can pass an obsfile instance here
    :param start_time         Scalar Integer.  Timestamp at which to begin bad pixel masking, default = 0
    :param end_time           Scalar Integer.  Timestamp at which to finish bad pixel masking, defailt = -1
                                               (run to the end of the file)
    :param time_step          Scalar Integer.  Number of seconds to do the bad pixel masking over (should be an integer
                                               number of steps through the obsfile

    Other Input:
    :param fwhm:               Scalar float. Estimated full-width-half-max of the PSF (in pixels).
    :param box_size:           Scalar integer. Size box used for calculating median flux in the region surrounding each pixel.
    :param nsigma_hot:         Scalar float. If the flux ratio for a pixel is (nsigma_hot x expected error)
                                             above the max expected given the PSF FWHM, then flag it as hot.
    :param weighted            Boolean, set to True to use flat cal weights (see obsFile.getPixelCountImage() )
    :param flux_weighted       Boolean, if True, flux cal weights are applied (also see obsfile.getPixelCountImage() )
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
    Writes a 'bad pixel table' to an output h5 file titled 'badpixmask_--timestamp--.h5' where:
            0 = Good pixel
            1 = Hot pixel
            2 = Cold Pixel
            3 = Dead Pixel
    """

    if obsfile is None:
        obsfile = ObsFile.ObsFile(input_filename)

    #A few defaults that will be used in the absence of parameter file or
    #arguments provided by the caller.
    if time_step is None: time_step = 1
    if start_time is None: start_time = 0
    if end_time is None: pass
    if max_iter is None: max_ter = 5
    if use_local_stdev is None: use_local_stdev = False

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
        print('Processing time slice: ', str(each_time) + ' - ' + str(each_time + time_step) + 's')
        raw_image_dict = obsfile.getPixelCountImage(firstSec=each_time, integrationTime=time_step, weighted=weighted,
                                                    fluxWeighted=flux_weighted, getRawCount=use_raw_counts)
        bad_pixel_solution = check_interval(image=raw_image_dict['image'], fwhm=fwhm, box_size=box_size, nsigma_hot=nsigma_hot,
                                            max_iter=max_iter, use_local_stdev=use_local_stdev, bkgd_percentile=bkgd_percentile)
        dead_masks[:,:,i] = bad_pixel_solution['dead_mask']
        hot_masks[:, :, i] = bad_pixel_solution['hot_mask']

    #Combine the bad pixel masks into a master mask
    dead_pixel_mask = (np.sum(dead_masks, axis=-1)/nsteps)*3
    hot_pixel_mask = np.sum(hot_masks, axis=-1)/nsteps
    bad_pixel_mask=dead_pixel_mask + hot_pixel_mask

    #Write it all out to the .h5 file

    write_bad_pixels(bad_pixel_mask, obsfile)


def write_bad_pixels(bad_pixel_mask, obsfile):
    """
    Write the output hot-pixel time masks table to an .h5 file. Called by
    find_bad_pixels().

    Required Input:
    :param bad_pixel_mask:    A 2D array of integers of the same shape as the input image, denoting locations
                              of bad pixels and the reason they are bad
    OR

    OUTPUTS:
    Writes a 'bad pixel table' to an output h5 file titled 'badpixmask_--timestamp--.h5'.

    """
    timestamp = datetime.utcnow().timestamp()
    badpix_file_name = 'badpixmask_{}.h5'.format(timestamp)

    try:
        badpix_file = tables.open_file(badpix_file_name, mode='w')
    except:
        print('Error: Couldn\'t create badpix file, ', badpix_file_name)
        return
    header = badpix_file.create_group(badpix_file.root, 'header', 'Bad Pixel Map information')
    tables.Array(header, 'beamMap', obj=obsfile.beamImage)
    tables.Array(header, 'xpix', obj=obsfile.xpix)
    tables.Array(header, 'ypix', obj=obsfile.ypix)

    badpixmask = badpix_file.create_group(badpix_file.root, 'badpixmap',
                                        'Bad Pixel Map')
    tables.Array(badpixmask, 'badpixmap', obj=bad_pixel_mask,
                 title='Bad Pixel Mask')

    badpix_file.flush()
    badpix_file.close()


if __name__ == "__main__":
    # read in command line arguments
    parser = argparse.ArgumentParser(description='MKID Bad Pixel Masking Utility')
    parser.add_argument('--quiet', action='store_true', dest='quiet',
                        help='Disable logging')

    if not args.quiet:
        pass






