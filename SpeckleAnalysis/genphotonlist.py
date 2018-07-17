#!/usr/bin/env python

import numpy as np
from scipy import special, interpolate
from numba import jit
from astropy.io import fits


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

@jit
def _recursion(r, g, f, sqrt1mf2, n):
    """
    Implement the recursive step for generating correlated noise using
    a numba speedup.  Algorithm is from Markus Deserno.

    """
    r[0] = g[0]
    for i in range(1, n):
        r[i] = r[i - 1]*f + g[i]*sqrt1mf2
    return r

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
    r = _recursion(r, g, f, sqrt1mf2, g.shape[0])
    
    return t, r

@jit
def removedeadtime(t, deadtime):

    """
    Remove photons within the pixel's "dead time".  We will check each
    photon in order, and keep it if, stepping backwards, we do not
    find another one within deadtime that was recorded as valid.  By
    going in order, we correctly handle, e.g., the case of a photon 
    arriving at t=0, another at t=deadtime-1, and a third at t=deadtime+1.

    Inputs: 
    t: one-dimensional array of photon arrival times
    deadtime: int or float, the inter-photon "dead time"

    Returns:
    A 1-D array of the valid/recorded photon arrival times
    """
    
    keep = np.ones(t.shape).astype(int)
    for i in range(t.shape[0]):
        for j in range(i - 1, -1, -1):
            if t[i] - t[j] > deadtime:
                break
            elif keep[j]:
                keep[i] = 0
                break
    print("Removed "+str(len(t)-np.sum(keep))+" photons due to deadTime")
    return t[np.where(keep)]
    

def genphotonlist(Ic, Is, Ttot, tau, deadtime=0, interpmethod='cubic',
                  lookuptable=True):

    """
    Generate a photon list from an input Ic, Is with an arbitrary
    photon rate.  All times are measured in seconds or inverse
    seconds; the returned list of times is in microseconds.

    Arguments:
    Ic: float, units 1/seconds
    Is: float, units 1/seconds
    Ttot: int, total exposure time in seconds
    tau: float, correlation time in seconds

    Optional arguments:
    interpmethod: argument 'kind' to interpolate.interp1d 
    lookuptable: use a lookuptable to invert the modified Rician CDF?  Default True

    Returns:
    1D array of photon arrival times

    """

    # Generate a correlated Gaussian sequence, corrlation time tau.
    # Then transform this to a random variable uniformly distributed
    # between 0 and 1, and finally back to a modified Rician random
    # variable.  This method ensures that: (1) the output is M-R
    # distributed, and (2) it is exponentially correlated.  Finally,
    # return a list of photons determined by the probability of each
    # unit of time giving a detected photon.

    if Is > 1e-8*Ic:
        t, normal = corrsequence(int(Ttot*1e6), tau*1e6)
        uniform = 0.5*(special.erf(normal/np.sqrt(2)) + 1)
        f = MRicdf(Ic, Is, interpmethod=interpmethod)

        if lookuptable:
            n = 100000
            table = np.linspace(0, 1, n + 1)
            vals = f(table)
            I = vals[(n*uniform + 0.5).astype(int)]/1e6
        else:
            I = f(uniform)/1e6

    elif Is >= 0:
        t = np.arange(int(Ttot*1e6))
        I = Ic/1e6*np.ones(t.shape)
    else:
        raise ValueError("Cannot generate a photon list with Is<0.")
        
    
    tlist = t[np.where(np.random.rand(len(t)) < I)]
    #return tlist
    return removedeadtime(tlist, deadtime)


if __name__ == "__main__":

    # Demonstration: Ic=1000/s, Is=300/s, 5s integration,
    # decorrelation time 0.1s.  Returns list of ~6500 times.

    Ic, Is, Ttot, tau = [1000, 300, 5, 0.1]
    t = genphotonlist(Ic, Is, Ttot, tau, deadtime=10)
