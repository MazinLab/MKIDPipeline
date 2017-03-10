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

from statsmodels.tsa.stattools import acovf
from statsmodels.stats.diagnostic import *

import astropy
import cPickle as pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from functools import partial
import mpfit
import parsePacketDump2
from parsePacketDump2 import parsePacketData
from arrayPopup import plotArray
from readDict import readDict
from img2fitsExample import writeFits
import hotpix.hotPixels as hp
import headers.TimeMask as tm
from readFITStest import readFITS

def modifiedRician(I, Ic, Is):
    #MR pdf(I) = 1/Is * exp(-(I+Ic)/Is) * I0(2*sqrt(I*Ic)/Is)
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


