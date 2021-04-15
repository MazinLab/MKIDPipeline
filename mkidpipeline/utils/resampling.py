import numpy as np
import scipy

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

    yfactor, xfactor = np.asarray(a.shape) / np.array([ysize, xsize])
    return a.reshape(ysize, int(yfactor), xsize, int(xfactor), ).mean(1).mean(2)

def rebin(x, y, binedges):
    """
    Seth Meeker 1-29-2013
    Given arrays of wavelengths and fluxes (x and y) rebins to specified bin size by taking average value of input data within each bin
    use: rebinnedData = rebin(x,y,binedges)
    binedges typically can be imported from a FlatCal after being applied in an Photontable
    returns rebinned data as 2D array:
        rebinned[:,0] = centers of wvl bins
        rebinned[:,1] = average of y-values per new bins
    """
    # must be passed binedges array since spectra will not be binned with evenly sized bins
    start = binedges[0]
    stop = binedges[-1]
    # calculate how many new bins we will have
    nbins = len(binedges) - 1
    # create output arrays
    rebinned = np.zeros((nbins, 2), dtype=float)
    for i in range(nbins):
        rebinned[i, 0] = binedges[i] + (binedges[i + 1] - binedges[i]) / 2.0
    n = 0
    binsize = binedges[n + 1] - binedges[n]
    while start + (binsize / 2.0) < stop:
        rebinned[n, 0] = (start + (binsize / 2.0))
        ind = np.where((x > start) & (x < start + binsize))
        rebinned[n, 1] = (scipy.integrate.trapz(y[ind], x=x[ind])) / binsize
        start += binsize
        n += 1
        try:
            binsize = binedges[n + 1] - binedges[n]
        except IndexError:
            break
    return rebinned