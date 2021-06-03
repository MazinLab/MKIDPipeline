import warnings
import numpy as np
import scipy.ndimage.filters as spfilters
import time

from mkidcore.corelog import getLogger
from mkidpipeline.photontable import Photontable
from mkidpipeline.utils import fitting
from mkidpipeline.utils import smoothing
import mkidpipeline.config
from mkidcore.pixelflags import FlagSet


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


class StepConfig(mkidpipeline.config.BaseStepConfig):
    yaml_tag = u'!pixcal_cfg'
    REQUIRED_KEYS = (('method', 'median', 'method to use cpscut|laplacian|median|threshold'),
                     ('step', 30, 'Time interval for methods that need one'),
                     ('use_weight', True, 'Use photon weights'))


TEST_CFGS= (StepConfig(method='median', step=30), StepConfig(method='cpscut', step=30),
            StepConfig(method='laplacian', step=30), StepConfig(method='threshold', step=30))

FLAGS = FlagSet.define(('hot', 1, 'Hot pixel'),
                       ('cold', 2, 'Cold pixel'),
                       ('unstable', 3, 'Pixel is both hot at and at different times'))


# TODO make nsigma_hot and nsigma_cold user specified parameters in the config file
def threshold(image, fwhm=4, box_size=5, nsigma_hot=4.0, nsigma_cold=4.0, max_iter=10,
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
    gauss_array = fitting.gaussian_psf(fwhm, box_size)
    max_ratio = np.max(gauss_array) / np.median(gauss_array)

    # Assume everything with 0 counts is a dead pixel, turn dead pixel values into NaNs
    if dead_mask is None:
        dead_mask = np.ma.masked_where(raw_image == 0, raw_image).mask
    raw_image[dead_mask] = np.nan

    # Initialise a mask for hot pixels (all False) for comparison on each iteration.
    initial_hot_mask = np.zeros_like(raw_image, dtype=bool)
    initial_cold_mask = np.zeros_like(raw_image, dtype=bool)
    hot_mask = np.zeros_like(raw_image, dtype=bool)
    cold_mask = np.zeros_like(raw_image, dtype=bool)

    # Initialise some arrays with NaNs in case they don't get filled out during the iteration
    median_filter_image = np.full_like(raw_image, np.nan)
    difference_image = np.full_like(raw_image, np.nan)
    difference_image_error = np.full_like(raw_image, np.nan)

    # In the case that *all* the pixels
    # are dead, return a bad_mask where all the pixels are flagged as DEAD
    iteration = -1
    if raw_image[np.isfinite(raw_image)].sum() <= 0:
        getLogger(__name__).warning('Entire image consists of pixels with 0 counts')
        bad_mask = cold_mask * FLAGS.flags['cold'].bit
        hot_mask = np.zeros_like(bad_mask, dtype=bool)

    else:
        for iteration in range(max_iter):
            getLogger(__name__).info('Doing iteration: {}'.format(iteration))
            # Remove all the NaNs in an image and calculate a median filtered image
            # each pixel takes the median of itself and the surrounding box_size x box_size box.
            nan_fixed_image = smoothing.replace_nan(raw_image, mode='mean', box_size=box_size)
            assert np.all(np.isfinite(nan_fixed_image))
            median_filter_image = spfilters.median_filter(nan_fixed_image, box_size, mode='mirror')

            overall_median = np.nanmedian(raw_image)
            overall_bkgd = np.percentile(raw_image[~np.isnan(raw_image)], bkgd_percentile)

            # Estimate the background std. dev.
            standard_filter_image = smoothing.nearest_n_robust_sigma_filter(raw_image, n=box_size ** 2 - 1)
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
            cold_mask = (raw_image < (overall_bkgd/nsigma_cold)) | initial_cold_mask # TODO don't know if this is the best way to do this
            # If no change between between this and the last iteration then stop iterating
            if np.all(hot_mask == initial_hot_mask) and np.all(cold_mask == initial_cold_mask):
                break

            # Otherwise update 'initial_hot_mask' and 'initial_cold_mask' and set all detected bad pixels
            # to NaN for the next iteration
            initial_cold_mask = np.copy(cold_mask)
            initial_hot_mask = np.copy(hot_mask)
            raw_image[hot_mask] = np.nan
            raw_image[cold_mask] = np.nan

        # Finished with loop, make sure a pixel is not simultaneously hot and cold
        assert ~(hot_mask & cold_mask).any()

    return {'hot': hot_mask, 'cold': cold_mask, 'masked_image': raw_image, 'image': image,
            'median_filter_image': median_filter_image, 'max_ratio': max_ratio, 'difference_image': difference_image,
            'difference_image_error': difference_image_error, 'num_iter': iteration + 1}


def median(image, box_size=5, nsigma_hot=4.0, max_iter=5):
    """
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

    standard_filter_image = None
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
            nan_fixed_image = smoothing.replace_nan(raw_image, mode='mean', boxsize=box_size)
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
        assert ~(hot_mask & dead_mask).any()

    return {'cold': dead_mask, 'hot': hot_mask, 'image': raw_image,
            'standard_filter_image': standard_filter_image, 'median_filter_image': median_filter_image,
            'num_iter': iteration + 1}


def laplacian(image, box_size=5, nsigma_hot=4.0):
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

    nan_fixed_image = smoothing.replace_nan(raw_image, mode='mean', boxsize=box_size)
    assert np.all(np.isfinite(nan_fixed_image))

    laplacian_filter_image=None
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
            pix_up = laplacian_filter_image[hot_pix_x[i], hot_pix_y[i] + 1]
            pix_down = laplacian_filter_image[hot_pix_x[i], hot_pix_y[i] - 1]
            pix_left = laplacian_filter_image[hot_pix_x[i] - 1, hot_pix_y[i]]
            pix_right = laplacian_filter_image[hot_pix_x[i] + 1, hot_pix_y[i]]
            if pix_up > 0 and pix_down > 0 and pix_left > 0 and pix_right > 0:
                hot_mask[hot_pix_x[i], hot_pix_y[i]] = True

    return {'cold': dead_mask, 'hot': hot_mask, 'image': raw_image, 'laplacian_filter_image': laplacian_filter_image}

def _compute_mask(obs, method, step, startt, stopt, methodkw, weight):
    starts = np.arange(startt, stopt, step, dtype=int)  # Start time for each step (in seconds).
    step_ends = starts + int(step)  # End time for each step
    step_ends[step_ends > stopt] = int(stopt)  # Clip any time steps that run over the end of the requested time range.

    # Initialise stack of masks, one for each time step
    masks = np.zeros_like(obs.beamImage.shape+(starts.size, 2), dtype=bool)
    try:
        func = globals()[method]
    except KeyError:
        raise ValueError(f'"{method} is an unsupported pixel masking method')

    # Generate a stack of bad pixel mask, one for each time step
    for i, each_time in enumerate(starts):
        getLogger(__name__).info(f'Processing time slice: {each_time} - {each_time + step} s')
        img = obs.get_fits(start=each_time, duration=step, weight=weight, rate=False, cube_type='time',
                           bin_width=step)
        result = func(img['SCIENCE'].data, **methodkw)
        masks[:, :, i, 0] = result['hot']
        masks[:, :, i, 1] = result['cold']

    # check for any pixels that switched from one to the other
    mask = np.zeros_like(obs.beamImage.shape + (3,), dtype=bool)
    mask[:, :, :2] = masks.all(axis=2)
    mask[:, :, 2] = masks.any(axis=2) & ~mask.any(axis=2)

    meta = dict(method=method, step=step)  #TODO flesh this out with pixcal fits keys
    meta.update(methodkw)

    return mask, meta


def fetch(o, config=None):
    obs = Photontable(o.h5)
    if obs.query_header('pixcal'):
        getLogger(__name__).info('{} is already pixel calibrated'.format(o.h5))
        return None, None

    cfg = mkidpipeline.config.PipelineConfigFactory(step_defaults=dict(pixcal=StepConfig()), cfg=config, copy=True)
    startt, stopt = o.start, o.stop
    step = min(stopt-startt, cfg.pixcal.step)
    method = cfg.pixcal.method

    if cfg.pixcal.step > stopt-startt:
        getLogger(__name__).warning(f'Step time longer than data time by {step-(stopt-startt):.0f} s, '
                                    f'using full exposure.')

    # This is how method keywords are fetched is propagated
    exclude = [k[0] for k in StepConfig.REQUIRED_KEYS]
    methodkw = {k: mkidpipeline.config.config.pixcal.get(k) for k in mkidpipeline.config.config.pixcal.keys() if
                k not in exclude}

    return _compute_mask(obs, method, step, startt, stopt, methodkw, cfg.pixcal.use_weight)

def apply(o, config=None):
    mask, meta = fetch(o, config)
    if mask is None:
        return

    obs = Photontable(o.h5)
    tic = time.time()
    getLogger(__name__).info(f'Applying pixel mask to {o}')
    obs.enablewrite()
    f = obs.flags
    obs.flag(f.bitmask('pixcal.hot') * mask[:, :, 0])
    obs.flag(f.bitmask('pixcal.cold') * mask[:, :, 1])
    obs.flag(f.bitmask('pixcal.unstable') * mask[:, :, 2])
    obs.update_header('pixcal', True)
    for k, v in meta.items():
        obs.update_header(f'PIXCAL.{k}', v)
    obs.disablewrite()
    getLogger(__name__).info(f'Mask applied in {time.time() - tic:.3f}s')
