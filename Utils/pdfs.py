'''
Author: Seth Meeker        Date: Feb 11, 2017


Define some probability density functions for fitting speckle histograms

'''

import sys, os, time, struct
import numpy as np

from scipy.signal import convolve
from scipy.interpolate import griddata
from scipy.misc import factorial
from scipy.optimize.minpack import curve_fit
from scipy import ndimage
from scipy import signal
from scipy import special
from scipy.stats import rv_continuous

from statsmodels.tsa.stattools import acovf
from statsmodels.stats.diagnostic import *

import astropy
import cPickle as pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from functools import partial
from arrayPopup import plotArray


def modifiedRician(I, Ic, Is):
    '''
    MR pdf(I) = 1/Is * exp(-(I+Ic)/Is) * I0(2*sqrt(I*Ic)/Is)
    mean = Ic + Is
    variance = Is^2 + 2*Ic*Is
    '''
    mr = 1.0/Is * np.exp(-1.0*(I+Ic)/Is)* special.iv(0,2.0*np.sqrt(I*Ic)/Is)
    return mr

def poisson(I,mu):
    #poissonian pdf(I) = e^-mu * mu^I / I!
    pois = np.exp(-1.0*mu) * np.power(mu,I)/factorial(I)
    return pois

def gaussian(I,mu,sig):
    #gaussian pdf(I) = e^(-(x-mu)^2/(2(sig^2)*1/(sqrt(2*pi)*sig)
    gaus = np.exp(-1.0*np.power((I-mu),2)/(2.0*np.power(sig,2))) * 1/(sig*np.sqrt(2*np.pi))
    return gaus

def fitMR(x, y, guessIc, guessIs):
    '''
    Given a histogram of intensity values (x = I bin centers, y = N(I))
    and a guess for Ic and Is, returns fit values for Ic and Is.
    '''
    mr_guess = [guessIc,guessIs]
    mrf = lambda fx, Ic, Is: modifiedRician(fx, Ic, Is)
    params, cov = curve_fit(mrf, x, y, p0=mr_guess, maxfev=2000)
    return params[0], params[1] #params = [fitIc, fitIs]

def fitPoisson(x,y,guessLambda):
    '''
    Given a histogram of intensity values (x = I bin centers, y = N(I))
    and a guess for expectation value, returns fit values for lambda.
    '''
    p_guess = [guessLambda]
    pf = lambda fx, lam: poisson(fx, lam)
    params, cov = curve_fit(pf, x, y, p0=p_guess, maxfev=2000)
    return params[0] #params = [lambda]

def fitGaussian(x,y,guessMu,guessSigma):
    '''
    Given a histogram of intensity values (x = I bin centers, y = N(I))
    and a guess for mu and sigma, returns fits for mu and sigma
    '''
    g_guess = [guessMu,guessSigma]
    gf = lambda fx, mu sigma: gaussian(fx, mu, sigma)
    params, cov = curve_fit(gf, x, y, p0=g_guess, maxfev=2000)
    return params[0], params[1] #params = [mu, sigma]


class mr_gen(rv_continuous):
    '''
    Modified Rician distribution for drawing random variates
    Define distribution with mr = mr_gen(). Class already knows (Ic, Is) are shape of PDF for rvs.
    Get random variates with randomSamples = mr.rvs(Ic=Ic, Is=Is, size=N)
    '''
    def _pdf(self, x, Ic, Is):
        return modifiedRician(x,Ic,Is)
    def _stats(self, Ic, Is):
        return [Ic+Is, np.power(Is,2)+2*Ic*Is, np.nan, np.nan]


if __name__ == '__main__':
    x = np.arange(200)/100.
    mr = modifiedRician(x,0.5,0.1)
    p = poisson(x,1.0)
    g = gaussian(x,1.0,0.3)
    c = np.convolve(mr,g,'same')
    plt.plot(x,mr,label="MR")
    plt.plot(x,p,label="Poisson")
    plt.plot(x,g,label="Gaussian")
    plt.plot(x,c/np.max(c),label="MR x G")
    plt.legend()
    plt.show()


