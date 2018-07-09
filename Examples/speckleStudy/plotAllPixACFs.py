'''
Author Seth Meeker 08-17-2017

Code for plotting the LC, ACF, and MR dist for all pixels in a data set.
Need to start exploring the spatial behavior of these correlations

'''

#import stuff
from darkObsFile import darkObsFile
from P3Utils.FileName import FileName
from arrayPopup import plotArray
import numpy as np
import lightCurves as lc
import matplotlib.pylab as plt
plt.rc('font', family='serif')
import pdfs
from scipy import stats
from scipy.signal import periodogram
from statsmodels.tsa.stattools import acovf,acf,ccf
import HotPix.darkHotPixMask as dhpm

object = '69h'
totalInt = 80 #for 69 Her data. Seems to be a move around 80s in.
#totalInt = 118 #for piHer data. Dither move shortly after.

### WL and PiHer data used for WL vs. on-sky correlation comparisons
#load up timed packet images
#this WL obs file has a move at 18 seconds. Just cut it out, stationary after.
if object == 'wl':
    #WL timed packet image
    target = r'WL'
    #load obs file for wl
    obs = darkObsFile(FileName(run='PAL2017a',date='20170410',tstamp='1491869200').obs())
    wlTPI = obs.getTimedPacketImage(firstSec=18, integrationTime=118)
    bpm = dhpm.loadMask('/mnt/data0/CalibrationFiles/darkHotPixMasks/20170410/1491870115.npz')
    basePath = '/mnt/data0/ProcessedData/seth/speckleAnalysis/WL_stats/allPlots_20170817/10ms/'
    TPI = wlTPI

elif object == 'ph':
    #pi Her timed packet image
    target = r'$\pi$ Her'
    #load obs file for Pi Her
    obs = darkObsFile(FileName(run='PAL2017a',date='20170409',tstamp='1491827562').obs())
    phTPI = obs.getTimedPacketImage(firstSec=0, integrationTime=totalInt)
    bpm = dhpm.loadMask('/mnt/data0/CalibrationFiles/darkHotPixMasks/20170409/1491826154.npz')
    basePath = '/mnt/data0/ProcessedData/seth/speckleAnalysis/piHer_stats/allPlots_20170817/30ms/'
    TPI = phTPI

elif object == '69h':
    # ssd target 69 Herculis timed packet image
    target = r'SAO65921'
    #69 Her data
    obs = darkObsFile(FileName(run='PAL2017a',date='20170409',tstamp='1491823630').obs())
    saoTPI = obs.getTimedPacketImage(firstSec=0, integrationTime=totalInt)
    #bad pix mask for SAO65921 (69 Her)
    bpm = dhpm.loadMask('/mnt/data0/CalibrationFiles/darkHotPixMasks/20170409/1491819780.npz')
    basePath = '/mnt/data0/ProcessedData/seth/speckleAnalysis/sao65921_stats/allPlots_20170817/30ms/'
    TPI = saoTPI

## BE SURE TO CHANGE BASE PATH TO RIGHT FOLDER TO MATCH SHORT INTEGRATION TIME
shortT = 0.03
show=False

for i in range(0,80): #80
    for j in range(0,125): #125
        print i,j
        #check if pixel is not in bad pix mask
        if len(TPI['timestamps'][i][j])>0 and bpm[j][i]==0:
            #grab timestamps for desired pixel
            TS = TPI['timestamps'][i][j]
            if object == 'wl': TS-=18
            #bin TS into lightcurve
            LC = lc.lightCurve(TS,shortExposure=shortT,start=0,stop=totalInt)
            Time = LC['time']
            Int = LC['intensity']
            #plot two lightcurves, one full time, one zoomed in to 300ms
            plt.plot(Time,Int,label=r"t=%i ms"%(shortT*1000))
            plt.xlabel(r"Time(s)",fontsize=14)
            plt.ylabel(r"Counts",fontsize=14)
            plt.title(r"%s pixel (%i,%i), t=%i ms"%(target,i,j,shortT*1000),fontsize=14)
            plt.legend()
            if show: plt.show()
            #save full LC
            plt.savefig(basePath+'%i_%i_LC_%ims_long.png'%(i,j,shortT*1000))
            plt.clf()
            #plot short LC
            plt.plot(Time,Int,label=r"t=%i ms"%(shortT*1000))
            plt.xlabel(r"Time(s)",fontsize=14)
            plt.ylabel(r"Counts",fontsize=14)
            plt.title(r"%s pixel (%i,%i), t=%i ms"%(target,i,j,shortT*1000),fontsize=14)
            plt.legend()
            plt.xlim(0,0.300*(shortT/0.01))
            if show: plt.show()
            #save short LC
            plt.savefig(basePath+'%i_%i_LC_%ims_%ims.png'%(i,j,shortT*1000,1000*0.300*(shortT/0.01)))
            plt.clf()
            #Plot distribution and fits
            #check the histogram of that lc, fit with MR or Poisson
            histS, binsS = lc.histogramLC(Int,centers=True,N=30*(shortT/0.01),span=[0,30*(shortT/0.01)])
            guessIc = np.mean(Int)*0.7
            guessIs = np.mean(Int)*0.3
            guessLam = np.mean(Int)
            fitIc, fitIs = pdfs.fitMR(binsS,histS,guessIc, guessIs)
            fitMR = pdfs.modifiedRician(binsS,fitIc,fitIs)
            fitLam = pdfs.fitPoisson(binsS,histS,guessLam)
            fitPoisson = pdfs.poisson(binsS,fitLam)
            plt.step(binsS,histS,color="grey",label=r"Histogram of intensities",where="mid")
            plt.plot(binsS,fitMR,color="black",linestyle="-.",label=r"MR fit to histogram: Ic=%2.2f, Is=%2.2f"%(fitIc, fitIs))
            plt.plot(binsS,fitPoisson,linestyle="--",color="black",label=r"Poisson fit to histogram: Lambda=%2.2f"%(fitLam))
            plt.legend()
            plt.xlabel(r"Intensity",fontsize=14)
            plt.ylabel(r"Probability",fontsize=14)
            plt.title(r"Intensity Distribution, %s pixel (%i,%i), t=%i ms"%(target,i,j,shortT*1000),fontsize=14)
            if show: plt.show()
            plt.savefig(basePath+'%i_%i_%ims_Dist.png'%(i,j,shortT*1000))
            plt.clf()
            #Plot Correlation function
            #calculate auto-corr function: Pearson correlation of lc w/itself shifted by various lags (tau)
            acfTime = Time-Time[0] #move ACF time to 0 lag (Time is centered on LC bin centers)
            corr,ljb,pvalue = acf(Int,unbiased=True,qstat=True,nlags=len(acfTime))
            #plot correlation as function of lag time, FULL ACF
            plt.plot(acfTime,corr)
            plt.xlabel(r"$\tau(s)$",fontsize=14)
            plt.ylabel(r"$R(\tau)$",fontsize=14)
            plt.title(r"Autocorrelation, %s pixel (%i,%i), t=%i ms"%(target,i,j,shortT*1000),fontsize=14)
            if show: plt.show()
            plt.savefig(basePath+'%i_%i_%ims_ACF_full.png'%(i,j,shortT*1000))
            plt.clf()
            #plot correlation as function of lag time, ZOOM 1 ACF
            plt.plot(acfTime,corr)
            plt.xlabel(r"$\tau(s)$",fontsize=14)
            plt.ylabel(r"$R(\tau)$",fontsize=14)
            plt.xlim(0,20)
            plt.ylim(-0.05,min([0.2*(shortT/0.01),1.0]))
            plt.title(r"Autocorrelation, %s pixel (%i,%i), t=%i ms"%(target,i,j,shortT*1000),fontsize=14)
            if show: plt.show()
            plt.savefig(basePath+'%i_%i_%ims_ACF_20s.png'%(i,j,shortT*1000))
            plt.clf()
            #plot correlation as function of lag time, ZOOM 2 ACF
            plt.plot(acfTime,corr)
            plt.xlabel(r"$\tau(s)$",fontsize=14)
            plt.ylabel(r"$R(\tau)$",fontsize=14)
            plt.xlim(0,0.300*(shortT/0.01))
            plt.ylim(-0.05,min([0.1*(shortT/0.01),1.0]))
            plt.title(r"Autocorrelation, %s pixel (%i,%i), t=%i ms"%(target,i,j,shortT*1000),fontsize=14)
            if show: plt.show()
            plt.savefig(basePath+'%i_%i_%ims_ACF_%ims.png'%(i,j,shortT*1000,1000*0.3*(shortT/0.01)))
            plt.clf()
            #Save col, row, time, lc, acfTime, acf to NPZ file
            np.savez(basePath+'%i_%i_%ims.npz'%(i,j,shortT*1000),col=j,row=i,t=shortT, lcTime=Time,lcInt=Int,acfTime=acfTime,acf = corr,Ic=fitIc,Is=fitIs)
        else:
            print "Skipping bad pixel (%i,%i)"%(i,j)
            pass


