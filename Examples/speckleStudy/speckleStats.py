'''
Author: Seth Meeker        Date: Feb 11, 2017


load params from speckleStatsPlot.cfg

'''

import sys, os, time, struct
import numpy as np

from scipy import ndimage
from scipy import signal
from scipy import stats
from scipy.integrate import simps
from scipy.optimize.minpack import curve_fit
from statsmodels.tsa.stattools import acovf,acf
from statsmodels.stats.diagnostic import *

import astropy
import cPickle as pickle
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from functools import partial
import mpfit

#from binFile import *
from binFileC import *

from arrayPopup import plotArray
from readDict import readDict
from img2fitsExample import writeFits
import hotpix.hotPixels as hp
import headers.TimeMask as tm
from readFITStest import readFITS
import pdfs

def binLightCurve(start, stop, times, intTime=0.01):
    histIntTime = float(intTime)
    endTime=int(stop)
    startTime=int(start)
    duration = endTime-startTime
    print histIntTime
    print "N_bins = %f"%((endTime-startTime)/histIntTime)
    histBinEdges = np.arange(startTime,endTime,histIntTime)
    hists = []
    if histBinEdges[-1]+histIntTime == endTime:
        histBinEdges = np.append(histBinEdges,endTime)

    hist,_ = np.histogram(times,bins=histBinEdges)

    binWidths = np.diff(histBinEdges)
    lightCurve = 1.*hist#/binWidths

    return histBinEdges, lightCurve

def plotHist(ax,histBinEdges,hist,**kwargs):
    ax.plot(histBinEdges,np.append(hist,hist[-1]),drawstyle='steps-post',**kwargs)


configFileName = 'speckleStats_Propus_a.cfg'
configData = readDict()
configData.read_from_file(configFileName)

# Extract parameters from config file
timeSpan = np.array(configData['timeSpan'], dtype=int)
darkSpan = np.array(configData['darkSpan'], dtype=int)
target = str(configData['target'])
date = str(configData['date'])
intTime = float(configData['intTime'])
specklePix = np.array(configData['specklePixels'],dtype=int)
companionPix = np.array(configData['companionPixels'],dtype=int)
binDir = str(configData['binDir'])
outputDir = str(configData['outputDir'])

binPath = os.path.join(binDir,date)
dataDir = binPath
print "Loading data from .bin files"

#load dark frames
#print "Loading dark frame"
#darkStack = loadStack(dataDir, darkSpan[0], darkSpan[1],useImg = useImg, nCols=numCols, nRows=numRows)
#dark = medianStack(darkStack)
#plotArray(dark,title='Dark',origin='upper')

print "Loading photons from times: %i to %i..."%(timeSpan[0],timeSpan[1])
timestampList = np.arange(timeSpan[0],timeSpan[1]+1)

sTime = time.time()
photonTstamps,photonPhases,photonBases,photonXs,photonYs,photonPixelIDs = parseBinFiles(dataDir,timestampList)
eTime = time.time()
print "Loaded %i photons"%len(photonTstamps)
print "Took %f minutes"%((eTime-sTime)/60.)

nRows=125
nCols=80

tauArray = np.full((nRows,nCols),np.nan,dtype=float)

for p in np.arange(len(specklePix)):
        col= specklePix[p][0]
        row= specklePix[p][1]

    #col= companionPix[3][0]
    #row= companionPix[3][1]

#for row in np.arange(nRows):
#    for col in np.arange(nCols):

        print "\n-----------------------------\n"
        print "Pixel (%i, %i)"%(row, col)

        try:
            selPixelId = photonPixelIDs[np.where((photonXs==col) & (photonYs==row))][0]
            msTimes = photonTstamps[np.where(photonPixelIDs==selPixelId)]

            times = 1.0E-3*(msTimes-msTimes[0])+timeSpan[0]

            phases = photonPhases[np.where(photonPixelIDs==selPixelId)]
            #print times
            #print phases

# run loop over variety of short integrations (t) (~5ms to 100? ms) to check impact t has on acvs
#for n in np.arange(1):
            newInt = intTime#*(n+1)

            #MOVE PLOTTING TO PDF OUTPUT SO DOESNT STOP PROGRAM
            #setup figure for lightcurve plots
            f1, ax1 = plt.subplots()
            ax1.set_title("Pixel (%i, %i) Normalized Lightcurves, integration = %3.2f ms"%(col,row,1000*newInt))
            x_formatter = matplotlib.ticker.ScalarFormatter(useOffset=False)
            x_formatter.set_scientific(False)
            ax1.xaxis.set_major_formatter(x_formatter)
            ax1.set_xlabel('time (s)')
            ax1.set_ylabel('I/<I>')

            #setup figure for acvs plots
            f2, ax2 = plt.subplots()
            ax2.set_title("Pixel (%i, %i) ACVS, integration = %3.2f ms"%(col,row,1000*newInt))
            #ax2.set_xlim([0,1])
            #ax2.set_ylim([0,1])
            ax2.set_xlabel('lag (Tau)')
            ax2.set_ylabel('ACVS')

            #setup figure for acvs plots
            f3, ax3 = plt.subplots()
            ax3.set_title("Pixel (%i, %i) ACF, integration = %3.2f ms"%(col,row,1000*newInt))
            ax3.set_xlabel('lag (Tau)')
            ax3.set_ylabel('Autocorrelation = (ACVS/ACVS[0]')
            ax3.set_xlim([0,1])
            ax3.set_ylim([0,1])
            

        # run a loop over variety of total time included in lightcurve (T) to check impact T has on acvs
        #for T in np.arange(10,timeSpan[1]-timeSpan[0]+1,2):
        #for T in np.arange(30,31):
            T = timeSpan[1]-timeSpan[0]
            print "T = %i s, t = %3.2f ms"%(T,1000.*newInt)

        #try:
            bins, lc = binLightCurve(timeSpan[0],timeSpan[0]+T, times, newInt)
            lcAve = np.mean(lc)
            print "<I> = %4.2f"%lcAve
            lcVar = np.power(np.std(lc),2)
            print "sigma(I)^2 = %3.3f"%lcVar
            lcNorm = lc/lcAve
        
            #plot timestream with T total seconds binned into t=(n+1)*intTime integration time bins
            ax1.plot(bins,np.append(lcNorm,lcNorm[-1]),drawstyle='steps-post', label="%i s"%(T))

            print "Calculating auto-covariance sequence..."
            #acvs = acovf(lc,unbiased=True,demean=True) #* 1.0/(lcVar-lcAve)
            acvs = acovf(lc,unbiased=False,demean=False)

            #corr,ljb,pvalue = acf(lc,unbiased=True,qstat=True,nlags = T/newInt)
            corr,ljb,pvalue = acf(lc,unbiased=False,qstat=True,nlags = T/newInt)
            
            standalone_ljb, standalone_pvalue = acorr_ljungbox(lc)
        
            print "Min(p-value) of acf Ljung-Box test = %f"%np.min(pvalue)
            try:
                print "Min(p) of acf LB at index %i of %i"%(np.where(pvalue==np.min(pvalue))[0],len(pvalue))
                mostCorrLag = np.where(pvalue==np.min(pvalue))[0] * newInt*1000
                print "Min(p) of acf LB at lag = %4.3f ms"%mostCorrLag

            except TypeError:
                print "Min(p) of acf LB at index %i of %i"%(np.where(pvalue==np.min(pvalue))[0][0],len(pvalue))
                mostCorrLag = np.where(pvalue==np.min(pvalue))[0][0] * newInt*1000
                print "Min(p) of acf LB at lag = %4.3f ms"%mostCorrLag

            print "Min(p-value) of standalone Ljung-Box test = %f"%np.min(standalone_pvalue)
        
            try:
                print "Min(p) of standalone LB at index %i of %i"%(np.where(standalone_pvalue==np.min(standalone_pvalue))[0],len(standalone_pvalue))
            except TypeError:
                print "Min(p) of standalone LB at index %i of %i"%(np.where(standalone_pvalue==np.min(standalone_pvalue))[0][0],len(standalone_pvalue))

            ax2.plot(np.arange(len(acvs))*newInt,acvs,label="%i s, p-value=%2.5f at lag = %4.3f ms"%(T,np.min(pvalue),mostCorrLag))

            #check calculation of tau two ways.
            #first, integral of acvs/sigma^2 (where sigma is acvs[0]).
            #second, should be equivalent to integral of corr as corr = acvs/sigma^2.
            #approx integral with Simpson's integration which takes y-data and dx.
            #NOT EQUIVALENT.
            #since ACVS includes very long lags (up to full timestream of 30s or whatever is used)
            #these have huge amount of noise and few samples compared against corr, which
            #has a defined number of lags, defined here to truncate at 1 second.
            tau_acvs = simps(acvs[::100],dx=newInt) #bad!!!! Do not use!!!!!
            tau_corr = simps(corr[::100],dx=newInt)

            print "ACVS Tau_C = %4.3f"%tau_acvs
            print "Corr Tau_C = %4.3f"%tau_corr

            ax3.plot(np.arange(len(corr))*newInt,corr,color='black',label="%i s, EW Tau_C at %4.3f ms"%(T,1000*tau_corr))
            ax3.axvline(x=tau_corr,ymin=0,ymax=1,color='grey',linestyle = "-.")

            if lcAve <=1.0:
                print "Cold/dead pixel, setting Tau to NAN"
                tauArray[row,col]=np.nan
            else:
                tauArray[row,col]=tau_corr
            #add another check here for hot pixel masks to force those to NAN as well

        except:
            print "Failed to get tau_corr, setting Tau to NAN"
            tauArray[row,col]=np.nan
       
        
        
        ax1.legend(loc=4)
        ax2.legend()
        ax3.legend(loc=0)
        plt.show()
plotArray(tauArray*1000,title='Tau_C [ms]',origin='upper',vmin=0)


'''

    #take light curve and bin into histogram of intensities
    intHistBinEdges = None
    nBinsPerUnitInt = 0.5
    nBins = 30
    pixCutOffIntensities=[]
    listToHist = lc
    intRange = np.array([np.min(listToHist),np.max(listToHist)])
    intBins = (intRange[1]-intRange[0])//nBinsPerUnitInt
    
    intHist,intHistBinEdges = np.histogram(listToHist,bins=nBins,range=intRange)
    intHist = np.array(intHist)/float(len(lc))
    print "Total probability (sum of intensity bins/total # lc points) = ",sum(intHist)

    binWidths = np.diff(intHistBinEdges)
    I = intHistBinEdges[:-1]+binWidths[0]/2.0
    y = intHist

    #try fitting Intensity histogram with various functions
    #### MR ####
    guess_Ic, guess_Is = np.mean(I), np.mean(I)/3.0 #Ic, Is
    mr_guess = [guess_Ic, guess_Is]

    mr = lambda fx, Ic, Is: pdfs.modifiedRician(fx,Ic,Is)
    params, cov = curve_fit(mr, I, y, p0=mr_guess, maxfev=2000)
    fitIc, fitIs = params
    print "MR guess: Ic = %f, Is = %f"%(guess_Ic, guess_Is)
    print "Results of MR fit: Ic = %f, Is = %f"%(fitIc,fitIs)

    #perform chisq fit
    mr_chisq, mr_p = stats.chisquare(y*float(len(lc)), float(len(lc))*mr(I,fitIc,fitIs),ddof=len(mr_guess))
    #mr_chisq = np.sum(np.power(mr(I,fitIc,fitIs)-y,2)/mr(I,fitIc,fitIs))
    #mr_p = 0.5
    #mr_redchisq = mr_chisq/(nBins-2.0)
    print "Chi^2 test of MR: X^2 = %f, p = %f"%(mr_chisq, mr_p)

    #perform kolmogorov-smirnov test
    #mr_kstest, mr_ksp = stats.kstest(lc, lambda x: pdfs.modifiedRician(x,fitIc,fitIs),N=len(lc))
    #print "KS test of MR: KS = %f, p = %f"%(mr_kstest, mr_ksp)
    
    print "\n----------------------------\n"

    #### Gaussian ####
    guess_mu, guess_sig = np.mean(I), np.std(I)
    g_guess = [guess_mu,guess_sig]
    g = lambda fx, mu, sig: pdfs.gaussian(fx, mu, sig)
    params, cov = curve_fit(g, I, y, p0=g_guess, maxfev=2000)
    fitmu, fitsig = params
    print "Gaus guess: mu = %f, sig = %f"%(guess_mu, guess_sig)
    print "Results of G fit: mu = %f, sig = %f"%(fitmu,fitsig)

    g_chisq, g_p = stats.chisquare(y*float(len(lc)), float(len(lc))*g(I,fitmu,fitsig),ddof=len(g_guess))
    #g_chisq = np.sum(np.power(g(I,fitmu,fitsig)-y,2)/g(I,fitmu,fitsig))
    #g_p = 0.5
    print "Chi^2 test of G: X^2 = %f, p = %f"%(g_chisq,g_p)

    #perform kolmogorov-smirnov test
    #g_kstest, g_ksp = stats.kstest(lc, lambda x: pdfs.gaussian(x,fitmu,fitsig),N=len(lc))
    #print "KS test of Gauss: KS = %f, p = %f"%(g_kstest, g_ksp)
    
    print "\n----------------------------\n"

    f3, ax3 = plt.subplots()
    ax3.plot(intHistBinEdges/lcAve,np.append(intHist,intHist[-1]),drawstyle='steps-post',label = "Data, Nsamp = %i"%len(lc))
    ax3.plot(I/lcAve,mr(I,fitIc, fitIs),label = "MR, Ic/Is = %1.3f, chi^2/dof = %1.5f"%(fitIc/fitIs,mr_chisq/(nBins-len(mr_guess))) )
    ax3.plot(I/lcAve,g(I,fitmu, fitsig),label = "G, mu = %1.3f, sig = %1.3f, chi^2/dof = %1.5f"%(fitmu, fitsig,g_chisq/(nBins-len(g_guess))))
    ax3.set_title("Pixel (%i, %i) Intensity Dist., integration = %2.1f ms, nBins = %i"%(col,row,1000*newInt,nBins))
    ax3.set_ylabel('Probability')
    ax3.set_xlabel('I/<I>')
    plt.legend()
    plt.show()
'''



