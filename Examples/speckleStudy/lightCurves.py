'''
Author Seth Meeker 06-11-2017

Tools to handle lightcurves for speckle statistics studies

'''

import numpy as np
import matplotlib.pylab as plt
import pdfs


def lightCurve(times, shortExposure):
    '''
    Given an array of timestamps, bins them into a lightcurve.
    Returns time (spaced as shortExposure bins) and counts per exposure.
    '''

    #Nothing here yet

    return


def simulatedLightcurve(pdf = None, shortExposure = 0.01, totalIntegration = 300, mean = 15, ratio = 5, sigma = 5, lowerBound=0):
    ''' 
    Given a desired PDF, draws samples to create a simulated lightcurve
    with total duration of <totalIntegration> seconds and single exposures of
    <shortExposure> seconds, intensity mean set by <mean>.

    Lower bound for MR needs to be >=0 (distribution undefined below I=0)

    MR defined by mean (Ic+Is) and ratio (Ic/Is)
    Poisson defined by mean (lambda)
    Gaussian defined by mean (mu) and std dev (sigma)

    Returns 2-column array with time and random intensity, each with length N=totalInt/shortExp
    '''

    n = int(totalIntegration/shortExposure)
    t = np.arange(n)*shortExposure

    if pdf in ('MR','mr','modified rician','Modified Rician','ModRic','modric'):
        Is = mean/(ratio+1)
        Ic = mean - Is
        sigma = np.nan
        mr = pdfs.mr_gen(a=lowerBound) # a is lower bound of distribution that rvs allowed to come from!!!!
        print "Drawing %i values from MR PDF..."%n   
        rvs = mr.rvs(Ic=Ic, Is=Is, size=n)
    
    elif pdf in ('poisson','Poisson','Pois','pois','psn'):
        ratio = np.nan
        sigma = np.nan
        print "Drawing %i values from Poisson PDF..."%n
        rvs = np.random.poisson(lam=mean,size=n)

    elif pdf in ('Gaussian','gaussian','gauss','Gauss','gsn'):
        ratio = np.nan
        print "Drawing %i values from Gaussian PDF..."%n
        rvs = np.random.normal(loc=mean, scale=sigma, size=n)

    else:
        print "Only MR, Poisson, and Gaussian PDFs currently implemented"
        rvs = np.zeros(len(t))

    return {"time":t, "intensity":rvs, "pdf":pdf, "mean":mean, "ratio":ratio, "sigma":sigma}


def histogramLC(I, N=35, span=[0,35], norm=True, centers = False):
    '''
    given a set of lightcurve intensities, returns histogram
    normalized such that integral of histogram = 1 (so, gives PDF) if norm=True
    gives bin centers if centers == True, otherwise returns bin edges
    '''
    intHist, binEdges = np.histogram(I,bins=N,range=span)
    if norm==True:
        intHist = np.array(intHist)/float(len(I))

    if centers==True:
        bws = np.diff(binEdges)
        cents = binEdges[:-1]+bws[0]/2.0
        bins = cents
    else:
        bins = binEdges

    return intHist, bins


def plotACF(lcTime,lcInt):
    '''
    calculate correlation curve of lc
    return correlation, ljunbBox statistics, and pvalue
    '''
    #calculate auto-corr function: Pearson correlation of lc w/itself shifted by various lags (tau)
    corr,ljb,pvalue = acf(lcInt,unbiased=False,qstat=True,nlags=len(lcTime))
    #plot correlation as function of lag time
    plt.plot(lcTime,corr)
    plt.xlabel(r"$\tau(s)$",fontsize=14)
    plt.ylabel(r"$R(\tau)$",fontsize=14)
    plt.title(r"Autocorrelation $R(\tau)$",fontsize=14)
    plt.show()

    return corr, ljb, pvalue


def plotPSD(lcInt, shortExp):
    '''
    plot power spectral density of lc
    return frequencies and powers from periodogram
    '''
    freq = 1.0/shortExp
    f, p = periodogram(lcInt,fs = 1./shortExp)

    plt.plot(f,p/np.max(p))
    plt.xlabel(r"Frequency (Hz)",fontsize=14)
    plt.ylabel(r"Normalized PSD",fontsize=14)
    plt.title(r"Lightcurve Power Spectrum",fontsize=14)
    plt.show()

    return f,p


if __name__ == '__main__':
    
    Ic = 12.
    Is = 3.
    plt.rc('font',family='serif')

    # Intensities to plot PDF over
    I = np.arange(350)/10.

    #make simPDF a parameter that can be changed with command line argument
    simPDF = "MR"

    if simPDF == 'MR':
        simLabel = "Simulated MR PDF: Ic=%i, Is=%i"%(Ic, Is)
        # MR distribution used to make simulated LC
        simMR = pdfs.modifiedRician(I,Ic,Is) #just for plotting purposes
    elif simPDF == 'Poisson':
        simMR = pdfs.poisson(I,Ic+Is)
        simLabel = "Simulated Poisson PDF: Lambda=%2.2f"%(Ic+Is)    
    

    # Lightcurve intensities drawn from the PDF
    lcDict = simulatedLightcurve(pdf=simPDF,totalIntegration=30, shortExposure=0.01, mean=Ic+Is,ratio=Ic/Is)

    plt.plot(lcDict['time'],lcDict['intensity'], label=simLabel)
    plt.xlabel("Time (s)",fontsize=14)
    plt.ylabel("Counts",fontsize=14)
    plt.legend()
    plt.show()

    # take histogram of LC intensities
    hist, bins = histogramLC(lcDict['intensity'], norm=True, centers=True)
    guessIc = np.mean(lcDict['intensity'])*0.7
    guessIs = np.mean(lcDict['intensity'])*0.3

    # fit a MR to the histogram of the lightcurve intensities
    fitIc, fitIs = pdfs.fitMR(bins, hist, guessIc, guessIs)
    fitMR = pdfs.modifiedRician(I, fitIc, fitIs)

    # fit a poisson to the histogram to show it doesn't do as well
    guessLam = np.mean(lcDict['intensity'])
    fitLam = pdfs.fitPoisson(bins,hist,guessLam)
    fitPoisson = pdfs.poisson(I,fitLam)


    plt.plot(I,simMR,label=simLabel)
    plt.step(bins,hist,color='grey',label="Histogram of LC Intensities",where='mid')
    plt.plot(I,fitMR,label="MR fit to histogram: Ic=%2.2f, Is=%2.2f"%(fitIc, fitIs))
    plt.plot(I,fitPoisson,label="Poisson fit to histogram: Lambda=%2.2f"%(fitLam))
    plt.legend()
    plt.xlabel("Intensity",fontsize=14)
    plt.ylabel("Frequency",fontsize=14)
    plt.show()

