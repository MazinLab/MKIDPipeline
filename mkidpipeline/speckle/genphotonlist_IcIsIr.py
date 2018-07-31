#!/usr/bin/env python

from __future__ import print_function
import numpy as np
from scipy import special, interpolate
import mkidpipeline.speckle.photonstats_utils as utils

def MRicdf(Ic, Is, interpmethod='cubic'):

    """
    Compute an interpolation function to give the inverse CDF of the
    modified Rician with a given Ic and Is.  

    Arguments:
    Ic: float, parameter for M-R
    Is: float > 0, parameter for M-R

    Optional argument:
    interpmethod: keyword passed as 'kind' to interpolate.interp1d

    Returns: 
    interpolation function f for the inverse CDF of the M-R

    """

    if Is <= 0 or Ic < 0:
        raise ValueError("Cannot compute modified Rician CDF with Is<=0 or Ic<0.")
    
    # Compute mean and variance of modified Rician, compute CDF by
    # going 15 sigma to either side (but starting no lower than zero).
    # Use 1000 points, or about 30 points/sigma.

    mu = Ic + Is
    sig = np.sqrt(Is**2 + 2*Ic*Is)
    I1 = max(0, mu - 15*sig)
    I2 = mu + 15*sig
    I = np.linspace(I1, I2, 1000)

    # Grid spacing.  Set I to be offset by dI/2 to give the
    # trapezoidal rule by direct summation.
    
    dI = I[1] - I[0]
    I += dI/2

    # Modified Rician PDF, and CDF by direct summation of intensities
    # centered on the bins to be integrated.  Enforce normalization at
    # the end since our integration scheme is off by a part in 1e-6 or
    # something.

    #p_I = 1./Is*np.exp(-(Ic + I)/Is)*special.iv(0, 2*np.sqrt(I*Ic)/Is)
    p_I = 1./Is*np.exp((2*np.sqrt(I*Ic) - (Ic + I))/Is)*special.ive(0, 2*np.sqrt(I*Ic)/Is)

    cdf = np.cumsum(p_I)*dI
    cdf /= cdf[-1]

    # The integral is defined with respect to the bin edges.
    
    I = np.asarray([0] + list(I + dI/2))
    cdf = np.asarray([0] + list(cdf))

    # The interpolation scheme doesn't want duplicate values.  Pick
    # the unique ones, and then return a function to compute the
    # inverse of the CDF.
    
    i = np.unique(cdf, return_index=True)[1]
    return interpolate.interp1d(cdf[i], I[i], kind=interpmethod)

def corrsequence(Ttot, tau):

    """
    Generate a sequence of correlated Gaussian noise, correlation time
    tau.  Algorithm is recursive and from Markus Deserno.  The
    recursion is implemented as an explicit for loop but has a lower
    computational cost than converting uniform random variables to
    modified-Rician random variables.
    
    Arguments:
    Ttot: int, the total integration time in microseconds.
    tau: float, the correlation time in microseconds.

    Returns:
    t: a list of integers np.arange(0, Ttot)
    r: a correlated Gaussian random variable, zero mean and unit variance, array of length Ttot

    """
    
    t = np.arange(Ttot)
    g = np.random.normal(0, 1, Ttot)
    r = np.zeros(g.shape)
    f = np.exp(-1./tau)
    sqrt1mf2 = np.sqrt(1 - f**2)
    r = utils.recursion(r, g, f, sqrt1mf2, g.shape[0])
    
    return t, r


def genphotonlist(Ic, Is, Ir, Ttot, tau, deadtime=0, interpmethod='cubic',
                  taufac=500, return_IDs=False):

    """
    Generate a photon list from an input Ic, Is with an arbitrary
    photon rate.  All times are measured in seconds or inverse
    seconds; the returned list of times is in microseconds.

    Arguments:
    Ic: float, units 1/seconds
    Is: float, units 1/seconds
    Ir: float, units 1/seconds
    Ttot: int, total exposure time in seconds
    tau: float, correlation time in seconds

    Optional arguments:
    interpmethod: argument 'kind' to interpolate.interp1d 
    taufac: float, discretize intensity with bin width tau/taufac.  Doing so speeds up the code immensely.  Default 500 (intensity errors ~1e-3)
    return_IDs: return an array giving the distribution (MR or constant) that produced each photon?  Default False.

    Returns:
    t, 1D array of photon arrival times

    Optional additional return:
    p, 1D array, 0 if photon came from Ic/Is MR, 1 if photon came from Ir

    """

    # Generate a correlated Gaussian sequence, corrlation time tau.
    # Then transform this to a random variable uniformly distributed
    # between 0 and 1, and finally back to a modified Rician random
    # variable.  This method ensures that: (1) the output is M-R
    # distributed, and (2) it is exponentially correlated.  Finally,
    # return a list of photons determined by the probability of each
    # unit of time giving a detected photon.

    # Number of microseconds per bin in which we discretize intensity
    
    N = max(int(tau*1e6/taufac), 1)

    if Is > 1e-8*Ic:

        t, normal = corrsequence(int(Ttot*1e6/N), tau*1e6/N)
        uniform = 0.5*(special.erf(normal/np.sqrt(2)) + 1)
        t *= N
        f = MRicdf(Ic, Is, interpmethod=interpmethod)
        I = f(uniform)/1e6

    elif Is >= 0:
        N = max(N, 1000)
        t = np.arange(int(Ttot*1e6))
        I = Ic/1e6*np.ones(t.shape)
    else:
        raise ValueError("Cannot generate a photon list with Is<0.")

    # Number of photons from each distribution in each time bin
    
    n1 = np.random.poisson(I*N)
    n2 = np.random.poisson(np.ones(t.shape)*Ir/1e6*N)

    # Go ahead and make the list with repeated times
    
    tlist = t[np.where(n1 > 0)]
    tlist_r = t[np.where(n2 > 0)]

    for i in range(1, max(np.amax(n1), np.amax(n2)) + 1):
        tlist = np.concatenate((tlist, t[np.where(n1 > i)]))
        tlist_r = np.concatenate((tlist_r, t[np.where(n2 > i)]))

    tlist_tot = np.concatenate((tlist, tlist_r))*1.

    # Add a random number to give the exact arrival time within the bin

    tlist_tot += N*np.random.rand(len(tlist_tot))

    # Cython is much, much faster given that this has to be an
    # explicit for loop; without Cython (even with numba) this step
    # would dominate the run time.  Returns indices of the times we
    # keep.

    indx = np.argsort(tlist_tot)
    keep = utils.removedeadtime(tlist_tot[indx], deadtime)

    # plist tells us which distribution (MR or constant) actually
    # produced a given photon; return this if desired.
        
    if return_IDs:
        plist1 = np.zeros(tlist.shape).astype(int)
        plist2 = np.ones(tlist_r.shape).astype(int)
        plist_tot = np.concatenate((plist1, plist2))
        ikeep = np.where(keep)
        return [tlist_tot[indx][ikeep], plist_tot[indx][ikeep]]
    
    return tlist_tot[indx][np.where(keep)]
    
if __name__ == "__main__":

    # Demonstration: Ic=1000/s, Is=300/s, Ir=500/s, 5s integration,
    # decorrelation time 0.1s.  Returns list of ~9000 times.

    Ic, Is, Ir, Ttot, tau = [1000, 300, 500, 5, 0.1]
    t, p = genphotonlist(Ic, Is, Ir, Ttot, tau, deadtime=10, return_IDs=True)
    
