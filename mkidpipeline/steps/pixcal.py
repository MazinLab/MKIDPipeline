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
    REQUIRED_KEYS = (('method', 'threshold', 'method to use laplacian|median|threshold'),
                     ('step', 30, 'Time interval for methods that need one'),
                     ('use_weight', True, 'Use photon weights'),
                     ('remake', False, 'Remake the calibration even if it exists'),
                     ('n_sigma', 5.0, 'number of standard deviations above/below the expected value for which a pixel'
                                      ' will be flagged as hot/cold'))


TEST_CFGS= (StepConfig(method='median', step=30), StepConfig(method='cpscut', step=30),
            StepConfig(method='laplacian', step=30), StepConfig(method='threshold', step=30))

FLAGS = FlagSet.define(('hot', 1, 'Hot pixel'),
                       ('cold', 2, 'Cold pixel'))

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
    dead_mask = raw_image == 0
    raw_image[dead_mask] = np.nan

    # Initialise a mask for hot pixels (all False) for comparison on each iteration.
    reference_hot_mask = np.zeros_like(raw_image, dtype=bool)
    reference_cold_mask = np.zeros_like(raw_image, dtype=bool)
    hot_mask = np.zeros_like(raw_image, dtype=bool)
    cold_mask = np.zeros_like(raw_image, dtype=bool)
    cold_mask[dead_mask] = True
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
                                      'median background is {}'.format(threshold[0][0], median_bkgd))
            hot_difference_image = raw_image - threshold
            cold_difference_image = raw_image - median_filter_image

            # Any pixel that has a peak/median ratio more than n_sigma above the maximum ratio should be flagged as hot
            # Any pixel that has a value less than n_sigma below the median should be flagged as cold
            hot_mask = (hot_difference_image > (n_sigma * std_filter_image)) | reference_hot_mask
            cold_mask = (cold_difference_image < -(n_sigma * std_filter_image)) | reference_cold_mask

            #If no change between between this and the last iteration then stop iterating
            if np.all(hot_mask == reference_hot_mask) and np.all(cold_mask == reference_cold_mask):
                break

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

def median(image, box_size=5, n_sigma=5.0, max_iter=5):
    """
    Passes a box_size by box_size moving box over the entire array and checks if the pixel at the center of that window
    has counts higher than the median plus n_sigma times the standard deviation of the pixels in that window

    :param image: 2D image array of photons (in counts)
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

    # Assume everything with 0 counts is a dead pixel, turn dead pixel values into NaNs
    dead_mask = raw_image == 0
    raw_image[dead_mask] = np.nan

    # Initialise a mask for hot pixels (all False) for comparison on each iteration.
    reference_hot_mask = np.zeros(shape=np.shape(raw_image), dtype=bool)
    hot_mask = np.zeros(shape=np.shape(raw_image), dtype=bool)
    reference_cold_mask = np.zeros(shape=np.shape(raw_image), dtype=bool)
    cold_mask = np.zeros(shape=np.shape(raw_image), dtype=bool)
    cold_mask[dead_mask] = True

    # Initialise some arrays with NaNs in case they don't get filled out during the iteration
    median_filter_image = np.zeros_like(raw_image)
    median_filter_image.fill(np.nan)
    iteration = -1

    if raw_image[np.isfinite(raw_image)].sum() <= 0:
        getLogger(__name__).warning('Entire image consists of pixels with 0 counts')
        cold_mask = np.ones_like(raw_image, dtype=bool)
        iteration = -1
    else:
        for iteration in range(max_iter):
            getLogger(__name__).info('Iteration: '.format(iteration))
            # Remove all the NaNs in an image and calculate a median filtered image
            # each pixel takes the median of itself and the surrounding box_size x box_size box.
            nan_fixed_image = smoothing.replace_nan(raw_image, mode='mean', box_size=box_size)
            assert np.all(np.isfinite(nan_fixed_image))
            median_filter_image = spfilters.median_filter(nan_fixed_image, box_size, mode='mirror')
            func = lambda x: np.nanstd(x) * _stddev_bias_corr((~np.isnan(x)).sum())
            std_filter_image = spfilters.generic_filter(nan_fixed_image, func, box_size, mode='mirror')

            hot_threshold = median_filter_image + (n_sigma * std_filter_image)
            cold_threshold = median_filter_image - (n_sigma * std_filter_image)

            hot_mask = (raw_image > hot_threshold) | reference_hot_mask
            cold_mask = (raw_image < cold_threshold) | reference_cold_mask

            # If no change between between this and the last iteration then stop iterating
            if np.all(hot_mask == reference_hot_mask) and np.all(cold_mask == reference_cold_mask): break

            # Otherwise update 'reference_hot_mask' and set all detected bad pixels to NaN for the next iteration
            reference_hot_mask = np.copy(hot_mask)
            raw_image[hot_mask] = np.nan

            reference_cold_mask = np.copy(cold_mask)
            raw_image[cold_mask] = np.nan

        # Make sure a pixel is not simultaneously hot and cold
        assert ~(hot_mask & cold_mask).any()
    getLogger(__name__).info('Masked {} hot pixels and {} cold pixels'.format(len(hot_mask[hot_mask != False]),
                                                                              len(cold_mask[cold_mask != False])))
    return {'hot': hot_mask, 'cold': cold_mask, 'masked_image': raw_image, 'input_image': image,
            'num_iter': iteration + 1}

def laplacian(image, box_size=5, n_sigma=5.0, max_iter=5):
    """
    :param image: 2D image array of photons (in counts)
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

    # Assume everything with 0 counts is a dead pixel, turn dead pixel values into NaNs
    dead_mask = raw_image == 0
    raw_image[dead_mask] = np.nan

    # Initialise a mask for hot pixels (all False)
    reference_hot_mask = np.zeros(shape=np.shape(raw_image), dtype=bool)
    hot_mask = np.zeros(shape=np.shape(raw_image), dtype=bool)
    reference_cold_mask = np.zeros(shape=np.shape(raw_image), dtype=bool)
    cold_mask = np.zeros(shape=np.shape(raw_image), dtype=bool)
    cold_mask[dead_mask] = True

    # In the case that *all* the pixels are dead, return a bad_mask where all the pixels are flagged as DEAD
    if raw_image[np.isfinite(raw_image)].sum() <= 0:
        getLogger(__name__).warning('Entire image consists of pixels with 0 counts')
        cold_mask = np.ones_like(raw_image, dtype=bool)
        iteration = -1
    else:
        for iteration in range(max_iter):
            getLogger(__name__).info('Iteration: '.format(iteration))
            # Remove all the NaNs in an image and calculate a median filtered image
            # each pixel takes the median of itself and the surrounding box_size x box_size box.
            nan_fixed_image = smoothing.replace_nan(raw_image, mode='mean', box_size=box_size)
            assert np.all(np.isfinite(nan_fixed_image))
            laplacian_filter_image = spfilters.laplace(nan_fixed_image)
            hot_threshold = (np.std(laplacian_filter_image) + n_sigma * np.std(laplacian_filter_image))
            cold_threshold = (np.std(laplacian_filter_image) - n_sigma * np.std(laplacian_filter_image))

            hot_mask = (laplacian_filter_image < hot_threshold) | reference_hot_mask
            cold_mask = (laplacian_filter_image > cold_threshold) | reference_cold_mask

            # If no change between between this and the last iteration then stop iterating
            if np.all(hot_mask == reference_hot_mask) and np.all(cold_mask == reference_cold_mask): break

            # Otherwise update 'reference_hot_mask' and set all detected bad pixels to NaN for the next iteration
            reference_hot_mask = np.copy(hot_mask)
            raw_image[hot_mask] = np.nan

            reference_cold_mask = np.copy(cold_mask)
            raw_image[cold_mask] = np.nan

        # Make sure a pixel is not simultaneously hot and cold
        assert ~(hot_mask & cold_mask).any()
    getLogger(__name__).info('Masked {} hot pixels and {} cold pixels'.format(len(hot_mask[hot_mask != False]),
                                                                              len(cold_mask[cold_mask != False])))
    return {'hot': hot_mask, 'cold': cold_mask, 'masked_image': raw_image, 'input_image': image,
            'num_iter': iteration + 1}

def _compute_mask(obs, method, step, startt, stopt, methodkw, weight, n_sigma):
    try:
        func = globals()[method]
    except KeyError:
        raise ValueError(f'"{method} is an unsupported pixel masking method')

    # Generate a stack of bad pixel mask, one for each time step
    img = obs.get_fits(start=startt, duration=stopt-startt, weight=weight, rate=False, cube_type='time', bin_width=step)
    masks = np.zeros(img['SCIENCE'].data.shape+(2,), dtype=bool)
    for i, (sl, each_time) in enumerate(zip(np.rollaxis(img['SCIENCE'].data, -1), img['CUBE_EDGES'].data.edges[:-1])):
        getLogger(__name__).info(f'Processing time slice: {each_time} - {each_time + step} s')
        result = func(sl, n_sigma=n_sigma, **methodkw)
        masks[:, :, i, 0] = result['hot']
        masks[:, :, i, 1] = result['cold']

    # check for any pixels that switched from one to the other
    mask = masks.all(axis=2)  # all hot, all cold
    mask = np.dstack([mask, (masks.any(axis=2) & ~mask).any(axis=2)])  # and with any hot, any cold

    meta = dict(method=method, step=step)  # TODO flesh this out with pixcal fits keys
    meta.update(methodkw)

    return mask, meta

def fetch(o, startt, stopt, config=None):
    obs = Photontable(o) if isinstance(o,str) else o

    cfg = mkidpipeline.config.PipelineConfigFactory(step_defaults=dict(pixcal=StepConfig()), cfg=config, copy=True)

    step = min(stopt-startt, cfg.pixcal.step)
    method = cfg.pixcal.method

    if cfg.pixcal.step > stopt-startt:
        getLogger(__name__).warning(f'Step time longer than data time by {(step-(stopt-startt))*1000:.0f} ms, '
                                    f'using full exposure.')

    # This is how method keywords are fetched is propagated
    exclude = [k[0] for k in StepConfig.REQUIRED_KEYS]
    methodkw = {k: mkidpipeline.config.config.pixcal.get(k) for k in mkidpipeline.config.config.pixcal.keys() if
                k not in exclude}

    return _compute_mask(obs, method, step, startt, stopt, methodkw, cfg.pixcal.use_weight, cfg.pixcal.n_sigma)

def apply(o, config=None):
    config = mkidpipeline.config.PipelineConfigFactory(step_defaults=dict(pixcal=StepConfig()), cfg=config, copy=True)
    if o.photontable.query_header('pixcal') and not config.pixcal.remake:
        getLogger(__name__).info('{} is already pixel calibrated'.format(o.h5))
        return

    obs = Photontable(o.h5)
    mask, meta = fetch(obs, o.start, o.stop, config=config)

    if mask is None:
        return

    tic = time.time()
    getLogger(__name__).info(f'Applying pixel mask to {o}')
    obs.enablewrite()
    f = obs.flags
    obs.flag(f.bitmask('pixcal.hot') * mask[:, :, 0])
    obs.flag(f.bitmask('pixcal.cold') * mask[:, :, 1])
    obs.update_header('pixcal', True)
    for k, v in meta.items():
        obs.update_header(f'PIXCAL.{k}', v)
    obs.disablewrite()
    getLogger(__name__).info(f'Mask applied in {time.time() - tic:.3f}s')