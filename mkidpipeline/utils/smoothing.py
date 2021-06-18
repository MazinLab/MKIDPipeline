import numpy as np
import ast
import matplotlib.pyplot as plt
import scipy.constants as con
from scipy.interpolate import griddata
import scipy.integrate
import astropy
import warnings


def smooth(x, window_len=11, window='hanning'):
    """
    From the scipy.org Cookbook (SignalSmooth)

    smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output_array signal.

    input:
           x: the input signal
           window_len: the dimension of the smoothing window; should be an odd integer
           window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                flat window will produce a moving average smoothing.
    output_array:
          the smoothed signal

    see also:
        np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve
        scipy.signal.lfilter

    TODO: the window parameter could be the window itself if an array instead of a string
    """
    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")

    if window_len < 3:
        return x

    if window not in ('flat', 'hanning', 'hamming', 'bartlett', 'blackman'):
        raise ValueError("Window is one of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[window_len - 1:0:-1], x, x[-1:-window_len:-1]]
    w = np.ones(window_len, 'd') if window == 'flat' else ast.literal_eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[int((window_len / 2) - 1):-int((window_len / 2))]


def gaussian_convolution(x, y, x_en_min=0.005, x_en_max=6.0, x_de=0.001, flux_units="lambda", r=8, nsig_gauss=1,
                         plots=False):
    """
    Seth 2-16-2015
    Given arrays of wavelengths and fluxes (x and y) convolves with gaussian of a given energy resolution (r)
    Input spectrum is converted into F_nu units, where a Gaussian of a given R has a constant width unlike in
    wavelength space, and regridded to an even energy gridding defined by x_en_min, x_en_max, and x_de
    INPUTS:
        x - wavelengths of data points in Angstroms or Hz
        y - fluxes of data points in F_lambda (ergs/s/cm^2/Angs) or F_nu (ergs/s/cm^2/Hz)
        x_en_min - minimum value of evenly spaced energy grid that spectrum will be interpolated on to
        x_en_max - maximum value of evenly spaced energy grid that spectrum will be interpolated on to
        x_de - energy spacing of evenly spaced energy grid that spectrum will be interpolated on to
        flux_units - "lambda" for default F_lambda, x must be in Angstroms. "nu" for F_nu, x must be in Hz.
        r - energy resolution of gaussian to be convolved with. R=8 at 405nm by default.
    output_arrayS:
        x_out - new x-values that convolution is calculated at (defined by x_en_min, x_en_max, x_de), returned in same units as original x
        y_out - fluxes calculated at new x-values are returned in same units as original y provided
    """
    # =======================  define some Constants     ============================
    c = con.c * 100  # cm/s
    h = con.h  # erg*s
    heV = h / con.e

    # ================  Convert to F_nu and put x-axis in frequency  ===================
    if flux_units == 'lambda':
        x_nu = heV * (c * 1.0E8) / x / heV
        y_nu = y * x ** 2 * 3.34E4  # convert Flambda to Fnu(Jy)
    elif flux_units == 'nu':
        x_nu = x
        y_nu = y
    else:
        raise ValueError("flux_units must be either 'nu' or 'lambda'")

    # ============  regrid to a constant energy spacing for convolution  ===============
    x_nu_grid = np.arange(x_en_min, x_en_max, x_de) / heV  # make new x-axis gridding in constant freq bins
    y_nu_grid = griddata(x_nu, y_nu, x_nu_grid, 'linear', fill_value=0)
    x_nu_grid = x_nu_grid[1:-1]  # remove weird effects with first and last values #TODO figure out why this is happening
    y_nu_grid = y_nu_grid[1:-1]
    if plots:
        plt.plot(x_nu_grid, y_nu_grid, label="Spectrum in energy space")

    # ======  define gaussian for convolution, on same gridding as spectral data  ======
    # WARNING: right now flux is NOT conserved
    offset = 0
    e0 = heV * c / (900 * 1E-7)  # 450rnm light is ~3eV
    de = e0 / r
    sig = de / heV / 2.355  # define sigma as FWHM converted to frequency
    # normalize the Gaussian
    amp = 1.0 / (np.sqrt(2 * np.pi) * sig)
    gauss_x = np.arange(-nsig_gauss * sig, nsig_gauss * sig, x_de / heV)
    gauss_y = amp * np.exp(-1.0 * (gauss_x - offset) ** 2 / (2.0 * (sig ** 2)))
    gauss_x = gauss_x[1:-1]
    gauss_y = gauss_y[1:-1]
    window_size = int(len(gauss_x) / 2)
    if plots:
        plt.plot(gauss_x, gauss_y * y_nu_grid.max(), label="Gaussian to be convolved")
        plt.legend()
        plt.show()

    # ====== Integrate curve to get total flux, required to ensure flux conservation later =======
    original_total_flux = scipy.integrate.simps(y_nu_grid[window_size:-window_size],
                                                x=x_nu_grid[window_size:-window_size])

    # ================================    convolve    ==================================
    conv_y = np.convolve(y_nu_grid, gauss_y, 'valid')
    if plots:
        plt.plot(x_nu_grid, conv_y, label="Convolved spectrum")
        plt.legend()
        plt.show()

    # ============ Conserve Flux ==============
    new_total_flux = scipy.integrate.simps(conv_y, x=x_nu_grid[window_size:-window_size])
    conv_y *= (original_total_flux / new_total_flux)

    # ==================   Convert back to wavelength space   ==========================
    if flux_units == 'lambda':
        x_out = c / x_nu_grid[window_size:-window_size] * 1E8
        y_out = conv_y / (x_out ** 2) * 3E-5  # convert Fnu(Jy) to Flambda
    else:
        x_out = x_nu_grid[window_size:-window_size]
        y_out = conv_y
    if plots:
        plt.plot(x_out[x_out < 25000], y_out[x_out < 25000], label="Convolved Spectrum")
        plt.plot(x, y, label="Original spectrum")
        plt.legend()
        plt.ylabel('F_%s' % flux_units)
        plt.show()

    return [x_out, y_out]


def median_filter_nan(input_array, size=5, *nkwarg, **kwarg):
    """
    NaN-handling version of the scipy median filter function
    (scipy.ndimage.filters.median_filter). Any NaN values in the input array are
    simply ignored in calculating medians. Useful e.g. for filtering 'salt and pepper
    noise' (e.g. hot/dead pixels) from an image to make things clearer visually.
    (but note that quantitative applications are probably limited.)

    Works as a simple wrapper for scipy.ndimage.filters.generic-filter, to which
    calling arguments are passed.

    Note: mode='reflect' looks like it would repeat the edge row/column in the
    'reflection'; 'mirror' does not, and may make more sense for our applications.

    Arguments/return values are same as for scipy median_filter.
    INPUTS:
        input_array : array-like, input array to filter (can be n-dimensional)
        size : scalar or tuple, optional, size of edge(s) of n-dimensional moving box. If
                scalar, then same value is used for all dimensions.
    output_arrayS:
        NaN-resistant median filtered version of input_array.

    For other parameters see documentation for scipy.ndimage.filters.median_filter.

    e.g.:

        filteredImage = median_filter_nan(imageArray,size=3)

    -- returns median boxcar filtered image with a moving box size 3x3 pixels.

    JvE 12/28/12
    """
    return scipy.ndimage.filters.generic_filter(input_array, np.nanmedian, size, *nkwarg, **kwarg)


def mean_filter_nan(input_array, size=3, *nkwarg, **kwarg):
    """
    Basically a box-car smoothing filter. Same as median_filter_nan, but calculates a mean instead.
    Any NaN values in the input array are ignored in calculating means.
    See median_filter_nan for details.
    JvE 1/4/13
    """
    return scipy.ndimage.filters.generic_filter(input_array, np.nanmean, size, *nkwarg, **kwarg)


def find_nearest_finite(im, i, j, n=10):
    """
    JvE 2/25/2014
    Find the indices of the nearest n finite-valued (i.e. non-nan, non-infinity)
    elements to a given location within a 2D array. Pretty easily extendable
    to n dimensions in theory, but would probably mean slowing it down somewhat.

    The element i,j itself is *not* returned.

    If n is greater than the number of finite valued elements available,
    it will return the indices of only the valued elements that exist.
    If there are no finite valued elements, it will return a tuple of
    empty arrays.

    No guarantees about the order
    in which elements are returned that are equidistant from i,j.



    INPUTS:
        im - a 2D numerical array
        i,j - position to search around (i=row, j=column)
        n - find the nearest n finite-valued elements

    output_arrayS:
        (#Returns an boolean array matching the shape of im, with 'True' where the nearest
        #n finite values are, and False everywhere else. Seems to be outdated - see below. 06/11/2014,
        JvE. Should probably check to be sure there wasn't some mistake. ).

        Returns a tuple of index arrays (row_array, col_array), similar to results
        returned by the np 'where' function.
    """

    im_shape = np.shape(im)
    assert len(im_shape) == 2
    n_rows, n_cols = im_shape
    ii2, jj2 = np.atleast_2d(np.arange(-i, n_rows - i, dtype=float), np.arange(-j, n_cols - j, dtype=float))
    dist_sq = ii2.T ** 2 + jj2 ** 2
    good = np.isfinite(im)
    good[i, j] = False  # Get rid of element i,j itself.
    n_good = np.sum(good)
    dist_sq[~good] = np.nan  # Get rid of non-finite valued elements
    # Find indices of the nearest finite values, and unravel the flattened results back into 2D arrays
    # Should ignore NaN values automatically
    nearest = np.unravel_index((np.argsort(dist_sq, axis=None))[0:min(n, n_good)], im_shape)

    # Below version is maybe slightly quicker, but at this stage doesn't give quite the same results -- not worth the trouble
    # to figure out right now. Should ignore NaN values automatically
    # nearest = np.unravel_index((np.argpartition(distsq,min(n,ngood)-1,axis=None))[0:min(n,ngood)], imShape)
    return nearest


def nearest_n_robust_sigma_filter(input_array, n=24):
    """
    JvE 4/8/2014
    Similar to nearest_n_std_filter, but estimate the standard deviation using the
    median absolute deviation instead, scaled to match 1-sigma (for a normal
    distribution - see http://en.wikipedia.org/wiki/Robust_measures_of_scale).
    Should be more robust to outliers than regular standard deviation.

    INPUTS:
        input_array - 2D input array of values.
        n - number of nearest finite neighbours to sample for calculating std. dev. around each pixel.

    output_arrayS:
        A 2D array of standard deviations with the same shape as input_array
    """

    output_array = np.zeros_like(input_array)
    output_array.fill(np.nan)
    n_row, n_col = np.shape(input_array)
    for row in np.arange(n_row):
        for col in np.arange(n_col):
            vals = input_array[find_nearest_finite(input_array, row, col, n=n)]
            # MAD seems to give best compromise between speed and reasonable results.
            # Biweight midvariance is good, somewhat slower.
            # output_array[row,col] = np.diff(np.percentile(vals,[15.87,84.13]))/2.
            output_array[row, col] = astropy.stats.median_absolute_deviation(vals) * 1.4826
            # output_array[row,col] = astropy.stats.biweight_midvariance(vals)
    return output_array


def nearest_n_med_filter(input_array, n=24):
    """
    JvE 2/25/2014
    Same idea as nearest_n_std_filter, but returns medians instead of std. deviations.

    INPUTS:
        input_array - 2D input array of values.
        n - number of nearest finite neighbours to sample for calculating median around each pixel.

    output_arrayS:
        A 2D array of medians with the same shape as input_array
    """

    output_array = np.zeros_like(input_array)
    output_array.fill(np.nan)
    n_row, n_col = np.shape(input_array)
    for row in np.arange(n_row):
        for col in np.arange(n_col):
            output_array[row, col] = np.median(input_array[find_nearest_finite(input_array, row, col, n=n)])
    return output_array


def nearest_n_robust_mean_filter(input_array, n=24, n_sigma_clip=3.):
    """
    Matt 7/18/2014
    Same idea as nearest_n_std_filter, but returns sigma clipped mean instead of std. deviations.

    INPUTS:
        input_array - 2D input array of values.
        n - number of nearest finite neighbours to sample for calculating median around each pixel.

    output_arrayS:
        A 2D array of medians with the same shape as input_array
    """

    output_array = np.zeros_like(input_array)
    output_array.fill(np.nan)
    n_row, n_col = np.shape(input_array)
    for row in np.arange(n_row):
        for col in np.arange(n_col):
            output_array[row, col] = np.ma.mean(
                astropy.stats.sigma_clip(input_array[find_nearest_finite(input_array, row, col, n=n)],
                                         sig=n_sigma_clip, iters=None))
    return output_array


def replace_nan(input_array, mode='mean', box_size=3, iterate=True):
    """
    Replace all NaN values in an array with the mean (or median)
    of the surrounding pixels. Should work for any number of dimensions,
    but only fully tested for 2D arrays at the moment.

    INPUTS:
        input_array - input array
        mode - 'mean', 'median', or 'nearestNmedian', to replace with the mean or median of
                the neighbouring pixels. In the first two cases, calculates on the basis of
                a surrounding box of side 'box_size'. In the latter, calculates median on the
                basis of the nearest N='box_size' non-NaN pixels (this is probably a lot slower
                than the first two methods).
        box_size - scalar integer, length of edge of box surrounding bad pixels from which to
                  calculate the mean or median; or in the case that mode='nearestNmedian', the
                  number of nearest non-NaN pixels from which to calculate the median.
        iterate - If iterate is set to True then iterate until there are no NaN values left.
                  (To deal with cases where there are many adjacent NaN's, where some NaN
                  elements may not have any valid neighbours to calculate a mean/median.
                  Such elements will remain NaN if only a single pass is done.) In principle,
                  should be redundant if mode='nearestNmedian', as far as I can think right now

    output_arrayS:
        Returns 'input_array' with NaN values replaced.

    TO DO: currently spits out multiple 'invalid value encoutered' warnings if
           NaNs are not all removed on the first pass. These can safely be ignored.
           Will implement some warning catching to suppress them.
    JvE 1/4/2013
    """

    output_array = np.copy(input_array)
    while np.isnan(output_array).sum() and not np.isnan(output_array).all():

        # Calculate interpolates at *all* locations (because it's easier...)
        if mode == 'mean':
            interpolates = mean_filter_nan(output_array, size=box_size, mode='mirror')
        elif mode == 'median':
            interpolates = median_filter_nan(output_array, size=box_size, mode='mirror')
        elif mode == 'nearest_n_median':
            interpolates = nearest_n_med_filter(output_array, n=box_size)
        else:
            raise ValueError('Invalid mode selection - should be one of "mean", "median", or "nearest_n_median"')

        # Then substitute those values in wherever there are NaN values.
        output_array[np.isnan(output_array)] = interpolates[np.isnan(output_array)]
        if not iterate:
            break

    return output_array


def nearest_n_std_filter(input_array, n=24):
    """
    JvE 2/25/2014
    Return an array of the same shape as the (2D) input_array, with output_array values at each element
    corresponding to the standard deviation of the nearest n finite values in input_array.

    INPUTS:
        input_array - 2D input array of values.
        n - number of nearest finite neighbours to sample for calculating std. dev. around each pixel.

    output_arrayS:
        A 2D array of standard deviations with the same shape as input_array
    """

    output_array = np.zeros_like(input_array)
    output_array.fill(np.nan)
    n_row, n_col = np.shape(input_array)
    for row in np.arange(n_row):
        for col in np.arange(n_col):
            output_array[row, col] = np.std(input_array[find_nearest_finite(input_array, row, col, n=n)])
    return output_array
