import glob
import inspect
import math
import os
import sys

import matplotlib as mpl
import matplotlib.pylab as plt
import numpy
import scipy
import scipy.ndimage
import scipy.stats
from scipy.interpolate import griddata

import tables
import numpy as np
from numpy import linalg

try:
    from skimage.transform import rotate as imrotate
except ImportError:
    from scipy.misc import imrotate

import astropy.stats
from astropy import wcs
from astropy.io import fits
from astropy.coordinates import Angle
from mkidcore.corelog import getLogger
from matplotlib.colors import LogNorm


def confirm(prompt, default=True):
    """
    Displays a prompt, accepts a yes or no answer, and returns a boolean
    default is the response returned if no response is given
    if an ill-formed response is given, the prompt is given again
    """
    opt = '[y]|n' if default else 'y|[n]'
    rdict = {'y': True, 'n': False, '': default}
    good = False
    while not good:
        try:
            response = input(f'{prompt} {opt}:').lower()
            try:
                response = rdict[response[0] if response else '']
                good = True
            except KeyError:
                pass
        except:
            pass
        if not good:
            print('Unrecognized response. Try again.')
    return response


def ds9Array(xyarray, colormap='B', normMin=None, normMax=None, sigma=None, scale=None, frame=None):
    """
    Display a 2D array as an image in DS9 if available. Similar to 'plotArray()'
    
    xyarray is the array to plot

    colormap - string, takes any value in the DS9 'color' menu.

    normMin minimum used for normalizing color values

    normMax maximum used for normalizing color values

    sigma calculate normMin and normMax as this number of sigmas away
    from the mean of positive values

    scale - string, can take any value allowed by ds9 xpa interface.
    Allowed values include:
        linear|log|pow|sqrt|squared|asinh|sinh|histequ
        mode minmax|<value>|zscale|zmax
        limits <minvalue> <maxvalue>
    e.g.: scale linear
        scale log 100
        scale datasec yes
        scale histequ
        scale limits 1 100
        scale mode zscale
        scale mode 99.5 
        ...etc.
    For more info see:
        http://hea-www.harvard.edu/saord/ds9/ref/xpa.html#scale

    ## Not yet implemented: pixelsToMark a list of pixels to mark in this image

    ## Not yet implemented: pixelMarkColor is the color to fill in marked pixels
    
    frame - to specify which DS9 frame number the array should be displayed in.
             Default is None. 
    
    """
    if sigma != None:
        # Chris S. does not know what accumulatePositive is supposed to do
        # so he changed the next two lines.
        # meanVal = numpy.mean(accumulatePositive(xyarray))
        # stdVal = numpy.std(accumulatePositive(xyarray))
        meanVal = numpy.mean(xyarray)
        stdVal = numpy.std(xyarray)
        normMin = meanVal - sigma * stdVal
        normMax = meanVal + sigma * stdVal

    d = ds9.ds9()  # Open a ds9 instance
    if type(frame) is int:
        d.set('frame ' + str(frame))

    d.set_np2arr(xyarray)
    # d.view(xyarray, frame=frame)
    d.set('zoom to fit')
    d.set('cmap ' + colormap)
    if normMin is not None and normMax is not None:
        d.set('scale ' + str(normMin) + ' ' + str(normMax))
    if scale is not None:
        d.set('scale ' + scale)

    # plt.matshow(xyarray, cmap=colormap, origin='lower',norm=norm, fignum=False)

    # for ptm in pixelsToMark:
    #    box = mpl.patches.Rectangle((ptm[0]-0.5,ptm[1]-0.5),\
    #                                    1,1,color=pixelMarkColor)
    #    #box = mpl.patches.Rectangle((1.5,2.5),1,1,color=pixelMarkColor)
    #    fig.axes[0].add_patch(box)


def gaussian_psf(fwhm, boxsize, oversample=50):
    """
    Returns a simulated Gaussian PSF: an array containing a 2D Gaussian function
    of width fwhm (in pixels), binned down to the requested box size. 
    JvE 12/28/12
    
    INPUTS:
        fwhm - full-width half-max of the Gaussian in pixels
        boxsize - size of (square) output array
        oversample (optional) - factor by which the raw (unbinned) model Gaussian
                                oversamples the final requested boxsize.
    
    OUTPUTS:
        2D boxsize x boxsize array containing the binned Gaussian PSF
    
    (Verified against IDL astro library daoerf routine)
        
    """
    fineboxsize = boxsize * oversample

    xcoord = ycoord = numpy.arange(-(fineboxsize - 1.) / 2., (fineboxsize - 1.) / 2. + 1.)
    xx, yy = numpy.meshgrid(xcoord, ycoord)
    xsigma = ysigma = fwhm / (2. * math.sqrt(2. * math.log(2.))) * oversample
    zx = (xx ** 2 / (2 * xsigma ** 2))
    zy = (yy ** 2 / (2 * ysigma ** 2))
    fineSampledGaussian = numpy.exp(-(zx + zy))

    # Bin down to the required output boxsize:
    binnedGaussian = rebin_2d(fineSampledGaussian, boxsize, boxsize)

    return binnedGaussian


def fitRigidRotation(x, y, ra, dec, x0=0, y0=0):
    """
    calculate the rigid rotation from row,col positions to ra,dec positions

    return dictionary of theta,tx,ty, such that

    ra  = c*dx - s*dx + dra
    dec = s*dy + c*dy + ddec
    
    with c = scale*cos(theta) and s = scale*sin(theta)
         dx = x-x0 and dy = y-y0

    ra,dec are input in decimal degrees

    if chatter is True print some things to stdout

    The scale and rotation of the transform are recovered from the cd matrix;
      rm = w.wcs.cd
      wScale = math.sqrt(rm[0,0]**2+rm[0,1]**2) # degrees per pixel
      wTheta = math.atan2(rm[1,0],rm[0,0])      # radians


    """
    assert (len(x) == len(y) == len(ra) == len(dec)), "all inputs must be same length"
    assert (len(x) > 1), "need at least two points"

    dx = x - x0
    dy = y - y0
    a = numpy.zeros((2 * len(x), 4))
    b = numpy.zeros(2 * len(x))
    for i in range(len(x)):
        a[2 * i, 0] = -dy[i]
        a[2 * i, 1] = dx[i]
        a[2 * i, 2] = 1
        b[2 * i] = ra[i]

        a[2 * i + 1, 0] = dx[i]
        a[2 * i + 1, 1] = dy[i]
        a[2 * i + 1, 3] = 1
        b[2 * i + 1] = dec[i]
    answer, residuals, rank, s = linalg.lstsq(a, b)

    # put the fit parameters into the WCS structure
    sst = answer[0]  # scaled sin theta
    sct = answer[1]  # scaled cos theta
    dra = answer[2]
    ddec = answer[3]
    scale = math.sqrt(sst ** 2 + sct ** 2)
    theta = math.degrees(math.atan2(sst, sct))
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [x0, y0]  # reference pixel position
    w.wcs.crval = [dra, ddec]  # reference sky position
    w.wcs.cd = [[sct, -sst], [sst, sct]]  # scaled rotation matrix
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def mean_filter_nan(inputarray, size=3, *nkwarg, **kwarg):
    """
    Basically a box-car smoothing filter. Same as median_filter_nan, but calculates a mean instead.
    Any NaN values in the input array are ignored in calculating means.
    See median_filter_nan for details.
    JvE 1/4/13
    """
    return scipy.ndimage.filters.generic_filter(inputarray, lambda x: numpy.mean(x[~numpy.isnan(x)]), size,
                                                *nkwarg, **kwarg)


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
    return scipy.ndimage.filters.generic_filter(inputarray, lambda x: numpy.median(x[~numpy.isnan(x)]), size,
                                                *nkwarg, **kwarg)


def nan_stddev(x):
    """
    NaN resistant standard deviation - basically scipy.stats.tstd, but
    with NaN rejection, and returning NaN if there aren't enough non-NaN
    input values, instead of just crashing. Used by stddev_filter_nan.
    INPUTS:
        x - array of input values
    OUTPUTS:
        The standard deviation....
    """
    xClean = x[~numpy.isnan(x)]
    if numpy.size(xClean) > 1 and xClean.min() != xClean.max():
        return scipy.stats.tstd(xClean)
    # Otherwise...
    return numpy.nan


def plotArray(xyarray, colormap=mpl.cm.gnuplot2,
              normMin=None, normMax=None, showMe=True,
              cbar=False, cbarticks=None, cbarlabels=None,
              plotFileName='arrayPlot.png',
              plotTitle='', sigma=None,
              pixelsToMark=[], pixelMarkColor='red',
              fignum=1, pclip=None):
    """
    Plots the 2D array to screen or if showMe is set to False, to
    file.  If normMin and normMax are None, the norm is just set to
    the full range of the array.

    xyarray is the array to plot

    colormap translates from a number in the range [0,1] to an rgb color,
    an existing matplotlib.cm value, or create your own

    normMin minimum used for normalizing color values

    normMax maximum used for normalizing color values

    showMe=True to show interactively; false makes a plot

    cbar to specify whether to add a colorbar
    
    cbarticks to specify whether to add ticks to the colorbar

    cbarlabels lables to put on the colorbar

    plotFileName where to write the file

    plotTitle put on the top of the plot

    sigma calculate normMin and normMax as this number of sigmas away
    from the mean of positive values

    pixelsToMark a list of pixels to mark in this image

    pixelMarkColor is the color to fill in marked pixels
    
    fignum - to specify which window the figure should be plotted in.
             Default is 1. If None, automatically selects a new figure number.
            Added 2013/7/19 2013, JvE
    
    pclip - set to percentile level (in percent) for setting the upper and
            lower colour stretch limits (overrides sigma).
    
    """
    if sigma != None:
        # Chris S. does not know what accumulatePositive is supposed to do
        # so he changed the next two lines.
        # meanVal = numpy.mean(accumulatePositive(xyarray))
        # stdVal = numpy.std(accumulatePositive(xyarray))
        meanVal = numpy.nanmean(xyarray)
        stdVal = numpy.nanstd(xyarray)
        normMin = meanVal - sigma * stdVal
        normMax = meanVal + sigma * stdVal
    if pclip != None:
        normMin = numpy.percentile(xyarray[numpy.isfinite(xyarray)], pclip)
        normMax = numpy.percentile(xyarray[numpy.isfinite(xyarray)], 100. - pclip)
    if normMin == None:
        normMin = xyarray.min()
    if normMax == None:
        normMax = xyarray.max()
    norm = mpl.colors.Normalize(vmin=normMin, vmax=normMax)

    figWidthPt = 550.0
    inchesPerPt = 1.0 / 72.27  # Convert pt to inch
    figWidth = figWidthPt * inchesPerPt  # width in inches
    figHeight = figWidth * 1.0  # height in inches
    figSize = [figWidth, figHeight]
    params = {'backend': 'ps',
              'axes.labelsize': 10,
              'axes.titlesize': 12,
              'text.fontsize': 10,
              'legend.fontsize': 10,
              'xtick.labelsize': 10,
              'ytick.labelsize': 10,
              'figure.figsize': figSize}

    fig = plt.figure(fignum)  ##JvE - Changed fignum=1 to allow caller parameter
    plt.clf()
    plt.rcParams.update(params)
    plt.matshow(xyarray, cmap=colormap, origin='lower', norm=norm, fignum=False)

    for ptm in pixelsToMark:
        box = mpl.patches.Rectangle((ptm[0] - 0.5, ptm[1] - 0.5), \
                                    1, 1, color=pixelMarkColor)
        # box = mpl.patches.Rectangle((1.5,2.5),1,1,color=pixelMarkColor)
        fig.axes[0].add_patch(box)

    if cbar:
        if cbarticks == None:
            cbar = plt.colorbar(shrink=0.8)
        else:
            cbar = plt.colorbar(ticks=cbarticks, shrink=0.8)
        if cbarlabels != None:
            cbar.ax.set_yticklabels(cbarlabels)

    plt.ylabel('Row Number')
    plt.xlabel('Column Number')
    plt.title(plotTitle)

    if showMe == False:
        plt.savefig(plotFileName)
    else:
        plt.show()




def rebin_2d(a, ysize, xsize):
    """
    Rebin an array to a SMALLER array. Rescales the values such that each element
    in the output array is the mean of the elememts which it encloses in the input
    array (i.e., not the total). Similar to the IDL rebin function.
    Dimensions of binned array must be an integer factor of the input array.
    Adapted from SciPy cookbook - see http://www.scipy.org/Cookbook/Rebinning
    JvE 12/28/12

    INPUTS:
        a - array to be rebinned
        ysize - new ysize (must be integer factor of y-size of input array)
        xsize - new xsize (ditto for x-size of input array)

    OUTPUTS:
        Returns the original array rebinned to the new dimensions requested.        
    """

    yfactor, xfactor = numpy.asarray(a.shape) / numpy.array([ysize, xsize])
    return a.reshape(ysize, int(yfactor), xsize, int(xfactor), ).mean(1).mean(2)


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


def stddev_filter_nan(inputarray, size=5, *nkwarg, **kwarg):
    """
    Calculated a moving standard deviation across a 2D (image) array. The standard
    deviation is calculated for a box of side 'size', centered at each pixel (element)
    in the array. NaN values are ignored, and the center pixel at which the box is located
    is masked out, so that only the surrounding pixels are included in calculating the
    std. dev. Thus each element in the array can later be compared against
    this std. dev. map in order to effectively find outliers.
    
    Works as a simple wrapper for scipy.ndimage.filters.generic-filter, to which
    calling arguments are passed.
    
    Arguments/return values are same as for scipy median_filter.
    INPUTS:
        inputarray : array-like, input array to filter (can be n-dimensional)
        size : scalar or tuple, optional, size of edge(s) of n-dimensional moving box. If 
                scalar, then same value is used for all dimensions.
    OUTPUTS:
        NaN-resistant std. dev. filtered version of inputarray.
    
    """

    # Can set 'footprint' as follows to remove the central array element:
    # footprint = numpy.ones((size,size))
    # footprint[size/2,size/2] = 0

    return scipy.ndimage.filters.generic_filter(inputarray, nan_stddev, size=size, *nkwarg, **kwarg)


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
    nearest = numpy.unravel_index( (numpy.argsort(distsq, axis=None))[0:min(n, ngood)], imShape )

    # Below version is maybe slightly quicker, but at this stage doesn't give quite the same results -- not worth the trouble
    # to figure out right now. Should ignore NaN values automatically
    # nearest = numpy.unravel_index((numpy.argpartition(distsq,min(n,ngood)-1,axis=None))[0:min(n,ngood)], imShape)
    return nearest


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


def countsToApparentMag(cps, filterName='V', telescope=None):
    """
    routine to convert counts measured in a given filter to an apparent magnitude
    input: cps = counts/s to be converted to magnitude. Can accept np array of numbers.
           filterName = filter to have magnitude calculated for
           telescope = name of telescope used. This is necessary to determine the collecting
                       area since the counts need to be in units of counts/s/m^2 for conversion to mags
                       If telescope = None: assumes data was already given in correct units
    output: apparent magnitude. Returns same format as input (either single value or np array)
    """
    Jansky2Counts = 1.51E7
    dLambdaOverLambda = {'U': 0.15, 'B': 0.22, 'V': 0.16, 'R': 0.23, 'I': 0.19, 'g': 0.14, 'r': 0.14, 'i': 0.16,
                         'z': 0.13}
    f0 = {'U': 1810., 'B': 4260., 'V': 3640., 'R': 3080., 'I': 2550., 'g': 3730., 'r': 4490., 'i': 4760., 'z': 4810.}

    if filterName not in list(f0.keys()):
        raise ValueError("Not a valid filter. Please select from 'U','B','V','R','I','g','r','i','z'")

    if telescope in ['Palomar', 'PAL', 'palomar', 'pal', 'Hale', 'hale']:
        telArea = 17.8421  # m^2 for Hale 200" primary
    elif telescope in ['Lick', 'LICK', 'Shane']:
        raise ValueError("LICK NOT IMPLEMENTED")
    elif telescope == None:
        print(
            "WARNING: no telescope provided for conversion to apparent mag. Assuming data is in units of counts/s/m^2")
        telArea = 1.0
    else:
        raise ValueError("No suitable argument provided for telescope name. Use None if data in counts/s/m^2 already.")

    cpsPerArea = cps / telArea
    mag = -2.5 * numpy.log10(cpsPerArea / (f0[filterName] * Jansky2Counts * dLambdaOverLambda[filterName]))
    return mag


def get_device_orientation(coords, fits_filename='Theta1 Orionis B_mean.fits', separation=0.938, pa=253):
    """
    Given the position angle and offset of secondary calculate its RA and dec then
    continually update the FITS with different rotation matricies to tune for device orientation

    Default pa and offset for Trap come from https://arxiv.org/pdf/1308.4155.pdf figures 7 and 11

    B1 vs B2B3 barycenter separation is 0.938 and the position angle is 253 degrees

    :param coords:
    :param fits_filename:
    :param separation:
    :param pa:
    :return:
    """

    angle_from_east = 270 - pa

    companion_ra_arcsec = np.cos(np.deg2rad(angle_from_east)) * separation
    companion_ra_offset = (companion_ra_arcsec * u.arcsec).to(u.deg).value
    companion_ra = coords.ra.deg + companion_ra_offset

    companion_dec_arcsec = np.sin(np.deg2rad(angle_from_east)) * separation
    companion_dec_offset = (companion_dec_arcsec * u.arcsec).to(u.deg).value
    # minus sign here since reference object is below central star
    companion_dec = coords.dec.deg - companion_dec_offset

    getLogger(__name__).info('Target RA {} and dec {}'.format(Angle(companion_ra * u.deg).hms,
                                                              Angle(companion_dec * u.deg).dms))

    update = True
    device_orientation = 0
    hdu1 = fits.open(fits_filename)[1]

    field = hdu1.data
    while update:

        getLogger(__name__).info('Close this figure')
        ax1 = plt.subplot(111, projection=wcs.WCS(hdu1.header))
        ax1.imshow(field, norm=LogNorm(), origin='lower', vmin=1)
        plt.show()

        user_input = input(' *** INPUT REQUIRED *** \nEnter new angle (deg) or F to end: ')
        if user_input == 'F':
            update = False
        else:
            device_orientation += float(user_input)

        getLogger(__name__).warning('Using untested migration from scipy.misc.imrotate to skimage.transform.rotate. '
                                    'Verify results and remove this log message.')
        field = imrotate(hdu1.data, device_orientation, interp='bilinear')

    getLogger(__name__).info('Using position angle {} deg for device'.format(device_orientation))

    return np.deg2rad(device_orientation)

