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
    REQUIRED_KEYS = (('method', 'median', 'method to use laplacian|median|threshold'),
                     ('step', 30, 'Time interval for methods that need one'),
                     ('use_weight', True, 'Use photon weights'))


TEST_CFGS= (StepConfig(method='median', step=30), StepConfig(method='cpscut', step=30),
            StepConfig(method='laplacian', step=30), StepConfig(method='threshold', step=30))

FLAGS = FlagSet.define(('hot', 1, 'Hot pixel'),
                       ('cold', 2, 'Cold pixel'),
                       ('unstable', 3, 'Pixel is both hot at and at different times'))

# TODO make n_sigma a user specified parameter in the config file
def threshold(image, fwhm=4, box_size=5, n_sigma=5.0, max_iter=5):
    """
    Compares the ratio of flux in each pixel to the median of the flux in an enclosing box. If the ratio is too high
     -- i.e. the flux is too tightly distributed compared to a Gaussian PSF of the expected FWHM -- then the pixel is
      flagged as HOT.
    :param image: 2D image array of photons (in counts)
    :param fwhm: estimated full-width-half-max of the PSF (in pixels)
    :param box_size: in pixels
    :param n_sigma: number of standard deviations above/below the expected value for which a pixel will be flagged as
     'hot'/'cold'
    :param max_iter: maximum number of iterations
    :return:
    A dictionary containing the result and various diagnostics. Keys are:
    'hot': boolean mask of hot pixels
    'cold': boolean mask of cold pixels
    'masked_image': The hot and dead pixel masked image
    'input_image': original input image
    'num_iter': number of iterations performed.
    """

    raw_image = np.copy(image)

    # Approximate peak/median ratio for an ideal (Gaussian) PSF sampled at
    # pixel locations corresponding to the median kernel used with the real data.
    gauss_array = fitting.gaussian_psf(fwhm, box_size)
    max_ratio = np.max(gauss_array) / np.median(gauss_array)

    # turn dead pixel values into NaNs
    raw_image[raw_image==0] = np.nan

    # Initialise a mask for hot pixels (all False) for comparison on each iteration.
    reference_hot_mask = np.zeros_like(raw_image, dtype=bool)
    reference_cold_mask = np.zeros_like(raw_image, dtype=bool)
    hot_mask = np.zeros_like(raw_image, dtype=bool)
    cold_mask = np.zeros_like(raw_image, dtype=bool)

    if raw_image[np.isfinite(raw_image)].sum() <= 0:
        getLogger(__name__).warning('Entire image consists of pixels with 0 counts')
        cold_mask = np.ones_like(raw_image, dtype=bool)
        iteration = -1
    else:
        for iteration in range(max_iter):
            getLogger(__name__).info('Performing iteration: {}'.format(iteration + 1))
            nan_fixed_image = smoothing.replace_nan(raw_image, mode='mean', box_size=box_size)
            assert np.all(np.isfinite(nan_fixed_image))
            median_filter_image = spfilters.median_filter(nan_fixed_image, box_size, mode='mirror')

            median_bkgd = np.nanmedian(raw_image)

            # Estimate the background std. dev.
            std_filter_image = smoothing.nearest_n_robust_sigma_filter(raw_image, n=box_size ** 2 - 1)

            # Calculate difference between flux in each pixel and max_ratio * the median in the enclosing box.
            # Also calculate the error that would exist in a measurement of a pixel that *was* at the peak of a real PSF
            # Condition for flagging is:
            #        (flux - background)/(box median - background) > max_ratio.
            # Or:
            #        flux > max_ratio*median - background*(max_ratio-1)
            # If the threshold is *lower* than the background, then set it equal to the background level instead
            # (a pixel below the background level is unlikely to be hot!)
            # TODO move above to better documentation location

            threshold = np.maximum((max_ratio * median_filter_image - (max_ratio - 1) * median_bkgd), median_bkgd)
            getLogger(__name__).debug('Hot Pixel Masking Parameters:'
                                      'hot flux threshold is {}'
                                      'median background is {}'.format(threshold, median_bkgd))
            hot_difference_image = raw_image - threshold
            cold_difference_image = raw_image - median_filter_image

            # Any pixel that has a peak/median ratio more than n_sigma above the maximum ratio should be flagged as hot
            # Any pixel that has a value less than n_sigma below the median should be flagged as cold
            hot_mask = (hot_difference_image > (n_sigma * std_filter_image)) | reference_hot_mask
            cold_mask = (cold_difference_image < -(n_sigma * std_filter_image)) | reference_cold_mask

            #If no change between between this and the last iteration then stop iterating
            if np.all(hot_mask == reference_hot_mask) and np.all(cold_mask == reference_cold_mask):
                break

            # Otherwise update 'reference_hot_mask' and 'reference_cold_mask' and set all detected bad pixels
            # to NaN for the next iteration
            reference_cold_mask = np.copy(cold_mask)
            reference_hot_mask = np.copy(hot_mask)
            raw_image[hot_mask] = np.nan
            raw_image[cold_mask] = np.nan
            if iteration == max_iter:
                getLogger(__name__).info('Reached max number of iterations ({}) - Increase max iterations to ensure'
                                         ' all hot pixels are masked or check your data for excessive'
                                         ' outliers'.format(max_iter))
        # make sure a pixel is not simultaneously hot and cold
        assert ~(hot_mask & cold_mask).any()
        getLogger(__name__).info('Masked {} hot pixels and {} cold pixels'.format(len(hot_mask[hot_mask!=False]),
                                                                                  len(cold_mask[cold_mask!=False])))
    return {'hot': hot_mask, 'cold': cold_mask, 'masked_image': raw_image, 'input_image': image,
            'num_iter': iteration + 1}

def median(image, box_size=5, n_sigma=4.0, max_iter=5):
    """
    Passes a box_size by box_size moving box over the entire array and checks if the pixel at the center of that window
    has counts higher than the median plus n_sigma times the standard deviation of the pixels in that window

    If the pixel has counts less than 0.01, then the pixel is flagged as DEAD

    Dead pixels are excluded from the standard deviation calculation and the standard deviation is corrected for small
    sample sizes as per the function _stddev_bias_corr(n)

    The HOT and DEAD masks are combined into a single BAD mask at the end

    Required Input:
    :param image:           A 2D image array of photon counts.

    Other Input:
    :param box_size:           Scalar integer. Size box used for calculating median counts in the region surrounding each pixel.
    :param n_sigma:         Scalar float. If the flux ratio for a pixel is n_sigma x standard deviation within the moving box
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
            nan_fixed_image = smoothing.replace_nan(raw_image, mode='mean', box_size=box_size)
            assert np.all(np.isfinite(nan_fixed_image))
            median_filter_image = spfilters.median_filter(nan_fixed_image, box_size, mode='mirror')
            func = lambda x: np.nanstd(x) * _stddev_bias_corr((~np.isnan(x)).sum())
            standard_filter_image = spfilters.generic_filter(nan_fixed_image, func, box_size, mode='mirror')

            threshold = median_filter_image + (n_sigma * standard_filter_image)

            # Any pixel that has a count level more than n_sigma above the median should be flagged as hot:
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


def laplacian(image, box_size=5, n_sigma=4.0):
    """
    Required Input:
    :param image:           A 2D image array of photon counts.

    Other Input:
    :param box_size:           Scalar integer. Size box used for replacing the NaNs in the region surrounding each dead or NaN pixel.
    :param n_sigma:         Scalar float. If the flux ratio for a pixel is (n_sigma x expected error)
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

    nan_fixed_image = smoothing.replace_nan(raw_image, mode='mean', box_size=box_size)
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
        threshold_laplace = -(np.std(laplacian_filter_image) + n_sigma * np.std(laplacian_filter_image))
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
    try:
        func = globals()[method]
    except KeyError:
        raise ValueError(f'"{method} is an unsupported pixel masking method')

    # Generate a stack of bad pixel mask, one for each time step
    img = obs.get_fits(start=startt, duration=stopt-startt, weight=weight, rate=False, cube_type='time', bin_width=step)
    masks = np.zeros(img['SCIENCE'].data.shape+(2,), dtype=bool)
    for i, (sl, each_time) in enumerate(zip(np.rollaxis(img['SCIENCE'].data, -1), img['CUBE_EDGES'].data.edges[:-1])):
        getLogger(__name__).info(f'Processing time slice: {each_time} - {each_time + step} s')
        result = func(sl, **methodkw)
        masks[:, :, i, 0] = result['hot']
        masks[:, :, i, 1] = result['cold']

    # check for any pixels that switched from one to the other
    mask = masks.all(axis=2)  # all hot, all cold
    mask = np.dstack([mask, (masks.any(axis=2) & ~mask).any(axis=2)])  # and with any hot, any cold

    meta = dict(method=method, step=step)  # TODO flesh this out with pixcal fits keys
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
        getLogger(__name__).warning(f'Step time longer than data time by {(step-(stopt-startt))*1000:.0f} ms, '
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
