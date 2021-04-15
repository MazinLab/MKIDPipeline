import numpy as np
import ast
import matplotlib.pyplot as plt
import scipy.constants as con
from scipy.interpolate import griddata
import scipy.integrate

def smooth(x, window_len=11, window='hanning'):
    """
    From the scipy.org Cookbook (SignalSmooth)

    smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.

    input:
           x: the input signal
           window_len: the dimension of the smoothing window; should be an odd integer
           window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
                flat window will produce a moving average smoothing.
    output:
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
    w = np.ones(window_len, 'd') if window == 'flat' else  ast.literal_eval('np.' + window + '(window_len)')
    y = np.convolve(w / w.sum(), s, mode='valid')
    return y[int((window_len / 2) - 1):-int((window_len / 2))]

def gaussianConvolution(x, y, xEnMin=0.005, xEnMax=6.0, xdE=0.001, fluxUnits="lambda", r=8, nsig_gauss=1, plots=False):
    """
    Seth 2-16-2015
    Given arrays of wavelengths and fluxes (x and y) convolves with gaussian of a given energy resolution (r)
    Input spectrum is converted into F_nu units, where a Gaussian of a given R has a constant width unlike in
    wavelength space, and regridded to an even energy gridding defined by xEnMin, xEnMax, and xdE
    INPUTS:
        x - wavelengths of data points in Angstroms or Hz
        y - fluxes of data points in F_lambda (ergs/s/cm^2/Angs) or F_nu (ergs/s/cm^2/Hz)
        xEnMin - minimum value of evenly spaced energy grid that spectrum will be interpolated on to
        xEnMax - maximum value of evenly spaced energy grid that spectrum will be interpolated on to
        xdE - energy spacing of evenly spaced energy grid that spectrum will be interpolated on to
        fluxUnits - "lambda" for default F_lambda, x must be in Angstroms. "nu" for F_nu, x must be in Hz.
        r - energy resolution of gaussian to be convolved with. R=8 at 405nm by default.
    OUTPUTS:
        xOut - new x-values that convolution is calculated at (defined by xEnMin, xEnMax, xdE), returned in same units as original x
        yOut - fluxes calculated at new x-values are returned in same units as original y provided
    """
    # =======================  Define some Constants     ============================
    c = con.c * 100  # cm/s
    h = con.h  # erg*s
    k = 1.3806488E-16  # erg/K
    heV = h / con.e

    # ================  Convert to F_nu and put x-axis in frequency  ===================
    if fluxUnits == 'lambda':
        xEn = heV * (c * 1.0E8) / x
        xNu = xEn / heV
        yNu = y * x ** 2 * 3.34E4  # convert Flambda to Fnu(Jy)
    elif fluxUnits == 'nu':
        xNu = x
        xEn = xNu * heV
        yNu = y
    else:
        raise ValueError("fluxUnits must be either 'nu' or 'lambda'")

    # ============  regrid to a constant energy spacing for convolution  ===============
    xNuGrid = np.arange(xEnMin, xEnMax, xdE) / heV  # make new x-axis gridding in constant freq bins
    yNuGrid = griddata(xNu, yNu, xNuGrid, 'linear', fill_value=0)
    xNuGrid = xNuGrid[1:-1]  # remove weird effects with first and last values #TODO figure out why this is happening
    yNuGrid = yNuGrid[1:-1]
    if plots == True:
        plt.plot(xNuGrid, yNuGrid, label="Spectrum in energy space")

    # ======  define gaussian for convolution, on same gridding as spectral data  ======
    # WARNING: right now flux is NOT conserved
    offset = 0
    E0 = heV * c / (900 * 1E-7)  # 450rnm light is ~3eV
    dE = E0 / r
    sig = dE / heV / 2.355  # define sigma as FWHM converted to frequency
    # normalize the Gaussian
    amp = 1.0 / (np.sqrt(2 * np.pi) * sig)
    gaussX = np.arange(-nsig_gauss * sig, nsig_gauss * sig, xdE / heV)
    gaussY = amp * np.exp(-1.0 * (gaussX - offset) ** 2 / (2.0 * (sig ** 2)))
    gaussX = gaussX[1:-1]
    gaussY = gaussY[1:-1]
    window_size = int(len(gaussX) / 2)
    if plots:
        plt.plot(gaussX, gaussY * yNuGrid.max(), label="Gaussian to be convolved")
        plt.legend()
        plt.show()

    # ====== Integrate curve to get total flux, required to ensure flux conservation later =======
    originalTotalFlux = scipy.integrate.simps(yNuGrid[window_size:-window_size], x=xNuGrid[window_size:-window_size])

    # ================================    convolve    ==================================
    convY = np.convolve(yNuGrid, gaussY, 'valid')
    if plots:
        plt.plot(xNuGrid, convY, label="Convolved spectrum")
        plt.legend()
        plt.show()

    # ============ Conserve Flux ==============
    newTotalFlux = scipy.integrate.simps(convY, x=xNuGrid[window_size:-window_size])
    convY *= (originalTotalFlux / newTotalFlux)

    # ==================   Convert back to wavelength space   ==========================
    if fluxUnits == 'lambda':
        xOut = c / xNuGrid[window_size:-window_size] * 1E8
        yOut = convY / (xOut ** 2) * 3E-5  # convert Fnu(Jy) to Flambda
    else:
        xOut = xNuGrid[window_size:-window_size]
        yOut = convY
    if plots:
        plt.plot(xOut[xOut < 25000], yOut[xOut < 25000], label="Convolved Spectrum")
        plt.plot(x, y, label="Original spectrum")
        plt.legend()
        plt.ylabel('F_%s' % fluxUnits)
        plt.show()

    return [xOut, yOut]


def median_filter_nan(inputarray, size=5, *nkwarg, **kwarg):
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
        inputarray : array-like, input array to filter (can be n-dimensional)
        size : scalar or tuple, optional, size of edge(s) of n-dimensional moving box. If 
                scalar, then same value is used for all dimensions.
    OUTPUTS:
        NaN-resistant median filtered version of inputarray.

    For other parameters see documentation for scipy.ndimage.filters.median_filter.

    e.g.:

        filteredImage = median_filter_nan(imageArray,size=3)

    -- returns median boxcar filtered image with a moving box size 3x3 pixels.

    JvE 12/28/12
    """
    return scipy.ndimage.filters.generic_filter(inputarray, lambda x: np.median(x[~np.isnan(x)]), size,
                                                *nkwarg, **kwarg)



def mean_filter_nan(inputarray, size=3, *nkwarg, **kwarg):
    """
    Basically a box-car smoothing filter. Same as median_filter_nan, but calculates a mean instead.
    Any NaN values in the input array are ignored in calculating means.
    See median_filter_nan for details.
    JvE 1/4/13
    """
    return scipy.ndimage.filters.generic_filter(inputarray, lambda x: np.mean(x[~np.isnan(x)]), size,
                                                *nkwarg, **kwarg)


def findNearestFinite(im, i, j, n=10):
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

    OUTPUTS:
        (#Returns an boolean array matching the shape of im, with 'True' where the nearest
        #n finite values are, and False everywhere else. Seems to be outdated - see below. 06/11/2014,
        JvE. Should probably check to be sure there wasn't some mistake. ).

        Returns a tuple of index arrays (row_array, col_array), similar to results
        returned by the numpy 'where' function.
    """

    imShape = numpy.shape(im)
    assert len(imShape) == 2
    nRows, nCols = imShape
    ii2, jj2 = numpy.atleast_2d(numpy.arange(-i, nRows - i, dtype=float), numpy.arange(-j, nCols - j, dtype=float))
    distsq = ii2.T ** 2 + jj2 ** 2
    good = numpy.isfinite(im)
    good[i, j] = False  # Get rid of element i,j itself.
    ngood = numpy.sum(good)
    distsq[~good] = numpy.nan  # Get rid of non-finite valued elements
    # Find indices of the nearest finite values, and unravel the flattened results back into 2D arrays
    # Should ignore NaN values automatically
    nearest = numpy.unravel_index((numpy.argsort(distsq, axis=None))[0:min(n, ngood)], imShape)

    # Below version is maybe slightly quicker, but at this stage doesn't give quite the same results -- not worth the trouble
    # to figure out right now. Should ignore NaN values automatically
    # nearest = numpy.unravel_index((numpy.argpartition(distsq,min(n,ngood)-1,axis=None))[0:min(n,ngood)], imShape)
    return nearest

def nearestNrobustSigmaFilter(inputArray, n=24):
    """
    JvE 4/8/2014
    Similar to nearestNstdDevFilter, but estimate the standard deviation using the
    median absolute deviation instead, scaled to match 1-sigma (for a normal
    distribution - see http://en.wikipedia.org/wiki/Robust_measures_of_scale).
    Should be more robust to outliers than regular standard deviation.

    INPUTS:
        inputArray - 2D input array of values.
        n - number of nearest finite neighbours to sample for calculating std. dev. around each pixel.

    OUTPUTS:
        A 2D array of standard deviations with the same shape as inputArray
    """

    outputArray = numpy.zeros_like(inputArray)
    outputArray.fill(numpy.nan)
    nRow, nCol = numpy.shape(inputArray)
    for iRow in numpy.arange(nRow):
        for iCol in numpy.arange(nCol):
            vals = inputArray[findNearestFinite(inputArray, iRow, iCol, n=n)]
            # MAD seems to give best compromise between speed and reasonable results.
            # Biweight midvariance is good, somewhat slower.
            # outputArray[iRow,iCol] = numpy.diff(numpy.percentile(vals,[15.87,84.13]))/2.
            outputArray[iRow, iCol] = astropy.stats.median_absolute_deviation(vals) * 1.4826
            # outputArray[iRow,iCol] = astropy.stats.biweight_midvariance(vals)
    return outputArray


def nearestNmedFilter(inputArray, n=24):
    """
    JvE 2/25/2014
    Same idea as nearestNstdDevFilter, but returns medians instead of std. deviations.

    INPUTS:
        inputArray - 2D input array of values.
        n - number of nearest finite neighbours to sample for calculating median around each pixel.

    OUTPUTS:
        A 2D array of medians with the same shape as inputArray
    """

    outputArray = numpy.zeros_like(inputArray)
    outputArray.fill(numpy.nan)
    nRow, nCol = numpy.shape(inputArray)
    for iRow in numpy.arange(nRow):
        for iCol in numpy.arange(nCol):
            outputArray[iRow, iCol] = numpy.median(inputArray[findNearestFinite(inputArray, iRow, iCol, n=n)])
    return outputArray


def nearestNRobustMeanFilter(inputArray, n=24, nSigmaClip=3., iters=None):
    """
    Matt 7/18/2014
    Same idea as nearestNstdDevFilter, but returns sigma clipped mean instead of std. deviations.

    INPUTS:
        inputArray - 2D input array of values.
        n - number of nearest finite neighbours to sample for calculating median around each pixel.

    OUTPUTS:
        A 2D array of medians with the same shape as inputArray
    """

    outputArray = numpy.zeros_like(inputArray)
    outputArray.fill(numpy.nan)
    nRow, nCol = numpy.shape(inputArray)
    for iRow in numpy.arange(nRow):
        for iCol in numpy.arange(nCol):
            outputArray[iRow, iCol] = numpy.ma.mean(
                astropy.stats.sigma_clip(inputArray[findNearestFinite(inputArray, iRow, iCol, n=n)], sig=nSigmaClip,
                                         iters=None))
    return outputArray


def replace_nan(inputarray, mode='mean', boxsize=3, iterate=True):
    """
    Replace all NaN values in an array with the mean (or median)
    of the surrounding pixels. Should work for any number of dimensions,
    but only fully tested for 2D arrays at the moment.

    INPUTS:
        inputarray - input array
        mode - 'mean', 'median', or 'nearestNmedian', to replace with the mean or median of
                the neighbouring pixels. In the first two cases, calculates on the basis of
                a surrounding box of side 'boxsize'. In the latter, calculates median on the
                basis of the nearest N='boxsize' non-NaN pixels (this is probably a lot slower
                than the first two methods).
        boxsize - scalar integer, length of edge of box surrounding bad pixels from which to
                  calculate the mean or median; or in the case that mode='nearestNmedian', the
                  number of nearest non-NaN pixels from which to calculate the median.
        iterate - If iterate is set to True then iterate until there are no NaN values left.
                  (To deal with cases where there are many adjacent NaN's, where some NaN
                  elements may not have any valid neighbours to calculate a mean/median.
                  Such elements will remain NaN if only a single pass is done.) In principle,
                  should be redundant if mode='nearestNmedian', as far as I can think right now

    OUTPUTS:
        Returns 'inputarray' with NaN values replaced.

    TO DO: currently spits out multiple 'invalid value encoutered' warnings if
           NaNs are not all removed on the first pass. These can safely be ignored.
           Will implement some warning catching to suppress them.
    JvE 1/4/2013
    """

    outputarray = numpy.copy(inputarray)
    while numpy.isnan(outputarray).sum() and not numpy.isnan(outputarray).all():

        # Calculate interpolates at *all* locations (because it's easier...)
        if mode == 'mean':
            interpolates = mean_filter_nan(outputarray, size=boxsize, mode='mirror')
        elif mode == 'median':
            interpolates = median_filter_nan(outputarray, size=boxsize, mode='mirror')
        elif mode == 'nearestNmedian':
            interpolates = nearestNmedFilter(outputarray, n=boxsize)
        else:
            raise ValueError('Invalid mode selection - should be one of "mean", "median", or "nearestNmedian"')

        # Then substitute those values in wherever there are NaN values.
        outputarray[numpy.isnan(outputarray)] = interpolates[numpy.isnan(outputarray)]
        if not iterate:
            break

    return outputarray


def nearestNstdDevFilter(inputArray, n=24):
    """
    JvE 2/25/2014
    Return an array of the same shape as the (2D) inputArray, with output values at each element
    corresponding to the standard deviation of the nearest n finite values in inputArray.

    INPUTS:
        inputArray - 2D input array of values.
        n - number of nearest finite neighbours to sample for calculating std. dev. around each pixel.

    OUTPUTS:
        A 2D array of standard deviations with the same shape as inputArray
    """

    outputArray = numpy.zeros_like(inputArray)
    outputArray.fill(numpy.nan)
    nRow, nCol = numpy.shape(inputArray)
    for iRow in numpy.arange(nRow):
        for iCol in numpy.arange(nCol):
            outputArray[iRow, iCol] = numpy.std(inputArray[findNearestFinite(inputArray, iRow, iCol, n=n)])
    return outputArray
