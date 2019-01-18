'''All credit for this function goes to Eelco Hoogendoom at
stackoverflow.com/questions/21100716/fast-arbitrary-distribution-random-sampling'''

import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import scipy.special as special

class Distribution(object):
    """
    draws samples from a one dimensional probability distribution,
    by means of inversion of a discrete inverstion of a cumulative density function

    the pdf can be sorted first to prevent numerical error in the cumulative sum
    this is set as default; for big density functions with high contrast,
    it is absolutely necessary, and for small density functions,
    the overhead is minimal

    a call to this distibution object returns indices into density array
    """
    def __init__(self, pdf, sort = True, interpolation = True, transform = lambda x: x):
        self.shape          = pdf.shape
        self.pdf            = pdf.ravel()
        self.sort           = sort
        self.interpolation  = interpolation
        self.transform      = transform

        #a pdf can not be negative
        assert(np.all(pdf>=0))

        #sort the pdf by magnitude
        if self.sort:
            self.sortindex = np.argsort(self.pdf, axis=None)
            self.pdf = self.pdf[self.sortindex]
        #construct the cumulative distribution function
        self.cdf = np.cumsum(self.pdf)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def sum(self):
        """cached sum of all pdf values; the pdf need not sum to one, and is imlpicitly normalized"""
        return self.cdf[-1]

    def __call__(self, N):
        """draw """
        #pick numbers which are uniformly random over the cumulative distribution function
        # print N, self.ndim, self.sum
        choice = np.random.uniform(high = self.sum, size = N)
        #find the indices corresponding to this point on the CDF
        index = np.searchsorted(self.cdf, choice)
        #if necessary, map the indices back to their original ordering
        if self.sort:
            index = self.sortindex[index]
        #map back to multi-dimensional indexing
        index = np.unravel_index(index, self.shape)
        index = np.vstack(index)

        #is this a discrete or piecewise continuous distribution?
        if self.interpolation:
            index = np.float_(index)
            # print index[0][:50], np.random.uniform(size=index.shape[1])[:50], (index[0] + np.random.uniform(size=index.shape[1]))[:50], index.shape, type(index[0])
            index[0] += np.random.uniform(size=index.shape[1])
            # print index[0][:50]

        return self.transform(index)

def poisson(lamda, k):
    pdf = (lamda ** k * np.exp(-lamda)) / scipy.misc.factorial(k)
    return pdf

def bessel(k):
    '''modified zero order besset=l'''
    pdf = scipy.special.i0(k)
    return pdf

def MR(I,Ic,Is):
    '''modified rician distribution'''
    # print np.exp(-1.0*(I+Ic)/Is)[::10], special.iv(0,2.0*np.sqrt(I*Ic)/Is)[::10] 
    # print Ic, Is
    pdf = 1.0/Is * np.exp(-1.0*(I+Ic)/Is)* special.iv(0,2.0*np.sqrt(I*Ic)/Is)
    #pdf = (1./Is) * np.exp(-(I+Ic)/Is) * bessel(2*np.sqrt(I*Ic)/Is)
    pdf = pdf/sum(pdf)
    return pdf

def gaussian(mu, sig, x):
    pdf = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    return pdf

def gaussian2(x, sig,mu):
    pdf = np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))
    pdf = pdf/sum(pdf)
    return pdf

def lognorm(x,mu, sigma):
    pdf = (np.exp(-(np.log10(x - mu))**2 / (2 * sigma**2))/ (x * sigma))
    pdf = pdf/sum(pdf)
    return pdf

if __name__=='__main__':
    res_elements = 1000
    I = np.linspace(0.1, 200, res_elements)

    ratios = [0,0.5,1,2,7]
    for ratio in ratios:
        Is = 10
        pdf = MR(Is*ratio,Is,I)
        pdf = pdf/ np.sum(pdf)
        # plt.plot(pdf)


        dist = Distribution(pdf)#, transform=lambda i: i - res_elements/2)
        TOAs = dist(100)
        print(TOAs, np.shape(TOAs))
        hist, bins = np.histogram(TOAs, bins ='auto')
        plt.plot(bins[:-1], hist)

    plt.show()
