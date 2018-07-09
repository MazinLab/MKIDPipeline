'''
Author Seth Meeker 06-25-2017

Tools to handle lightcurves for speckle statistics studies

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
 

### WL and PiHer data used for WL vs. on-sky correlation comparisons
#load obs file for wl
wlobs = darkObsFile(FileName(run='PAL2017a',date='20170410',tstamp='1491869200').obs())
 
#load up timed packet images
#this WL obs file has a move at 18 seconds. Just cut it out, stationary after.
#wlTPI = wlobs.getTimedPacketImage(firstSec=18, integrationTime=118)

#Concatenate photon time stamp lists together for 3 selected pixels
wlA = wlTPI['timestamps'][24][76]
wlB = wlTPI['timestamps'][24][77]
wlD = wlTPI['timestamps'][25][76]
wlTS = wlA
wlTS = np.append(wlTS,wlB)
wlTS = np.append(wlTS,wlD)

# 69 her companion location timestamps
saoA = saoTPI['timestamps'][46][98]
saoB = saoTPI['timestamps'][47][98]
saoC = saoTPI['timestamps'][46][99]
saoTSC = saoA
#saoTSC = np.append(saoTSC,saoB)
#saoTSC = np.append(saoTSC,saoC)

# 69 her speckle location timestamps
saoD = saoTPI['timestamps'][30][54]
saoE = saoTPI['timestamps'][30][55]
saoF = saoTPI['timestamps'][31][54]
saoG = saoTPI['timestamps'][31][53]
saoTSS = saoD
saoTSS = np.append(saoTSS,saoE)
saoTSS = np.append(saoTSS,saoF)
saoTSS = np.append(saoTSS,saoG)

# make 5ms light curves
wlLC = lc.lightCurve(wlTS,shortExposure=.01)
wlTime = wlLC['time']-18
wlInt = wlLC['intensity']

wlCorr,wlLJB,wlPvalue = acf(wlInt,unbiased=True,qstat=True,nlags=len(wlTime))

#plot correlation as function of lag time
plt.plot(wlTime-wlTime[0],wlCorr)
plt.xlabel(r"$\tau(s)$",fontsize=14)
plt.ylabel(r"$R(\tau)$",fontsize=14)
plt.title(r"WL Autocorrelation $R(\tau)$",fontsize=14)
plt.show()

plt.plot(wlTime,wlInt)
plt.show()



#load obs file for Pi Her
#phobs = darkObsFile(FileName(run='PAL2017a',date='20170409',tstamp='1491827562').obs())
 
#phTPI = phobs.getTimedPacketImage(firstSec=0, integrationTime=118)

#Concatenate photon time stamp lists together for 3 selected pixels
phA = phTPI['timestamps'][49][98]
phB = phTPI['timestamps'][50][98]
phC = phTPI['timestamps'][49][99]

#phA = phTPI['timestamps'][24][88]
#phB = phTPI['timestamps'][23][88]
#phC = phTPI['timestamps'][24][87]

#phTS = phA
#phTS = np.append(phTS,phB)
#phTS = np.append(phTS,phC)

phTS = phA
phTS = np.append(phTS,phB)
phTS = np.append(phTS,phC)

#try with WL times

intTimes = []
ratios = []
wilks = []
for n in np.arange(1,200):

    shortT = n*0.002
    print shortT
    #nFitTo = 20./shortT #which index in the corr func do we want to fit out to (1s/short int time)
    #print nFitTo

    phLC = lc.lightCurve(saoTSC,shortExposure=shortT)
    phTime = phLC['time']
    phInt = phLC['intensity']
    Int = phInt

    #calculate auto-corr function: Pearson correlation of lc w/itself shifted by various lags (tau)

    #phCorr,phLJB,phPvalue = acf(phInt,unbiased=True,qstat=True,nlags=len(phTime))
    #x= phTime-phTime[0]
    #x = phTime[1:nFitTo]-phTime[0]

    #check the histogram of that lc, fit with MR or Poisson

    #good binning for 69Her companion and pi Her data
    #histS, binsS = lc.histogramLC(Int,centers=True,N=10+60*(shortT/0.01)-(5*(n-1)),span=[5*(n-1),10+60*(shortT/0.01)])

    #guess at good binning for 69 Her data
    histS, binsS = lc.histogramLC(Int,centers=True,N=40+18*(shortT/0.01),span=[0,40+18*(shortT/0.01)])
    
    #low = n*200.
    #high = n*450.
    #histS, binsS = lc.histogramLC(Int,centers=True,N=(high-low)/5.,span=[low,high])
    guessIc = np.mean(Int)*0.7
    guessIs = np.mean(Int)*0.3
    guessLam = np.mean(Int)
    fitIc, fitIs = pdfs.fitMR(binsS,histS,guessIc, guessIs)
    fitMR = pdfs.modifiedRician(binsS,fitIc,fitIs)
    fitLam = pdfs.fitPoisson(binsS,histS,guessLam)
    fitPoisson = pdfs.poisson(binsS,fitLam)
    fitMu,fitSig =pdfs.fitGaussian(binsS,histS,guessLam,np.std(Int))
    fitGaussian = pdfs.gaussian(binsS,fitMu,fitSig)

    sww, swp = stats.shapiro(phInt)
    print stats.shapiro(phInt)
    print "Shapiro Wilks W,p = %3.3f, %3.3f"%(sww,swp)
    print "Ic, Is = %3.3f, %3.3f"%(fitIc, fitIs)
    print "Ic/Is = %3.3f"%(fitIc/fitIs)

    plt.step(binsS,histS,color="grey",label=r"Histogram of intensities",where="mid")
    plt.plot(binsS,fitMR,color="black",linestyle="-.",label=r"MR fit to histogram: Ic=%2.2f, Is=%2.2f"%(fitIc, fitIs))
    #plt.plot(binsS,fitPoisson,linestyle="--",color="black",label=r"Poisson fit to histogram: $\lambda$=%2.2f"%(fitLam))
    plt.plot(binsS,fitGaussian,linestyle=":",color="black",label=r"Gaussian fit to histogram: $\mu$=%2.2f, $\sigma$=%2.2f"%(fitMu,fitSig))
    plt.legend()
    plt.xlabel(r"Intensity",fontsize=14)
    plt.ylabel(r"Probability",fontsize=14)
    plt.title(r"Intensity Distribution, %s pixel (%i,%i), t=%i ms"%(target,i,j,shortT*1000),fontsize=14)
    #plt.show()
    plt.clf()
    #plt.savefig(basePath+'%i_%i_%ims_Dist.png'%(i,j,shortT*1000))
    wilks.append(sww)
    intTimes.append(shortT)
    ratios.append(fitIc/fitIs)



#Plot stuff!

plt.plot(phIntTimes,phWilks,linewidth=3,linestyle=':',label=r'$\pi$ Her Speckle')
#plt.plot(wlIntTimes,wlWilks,linewidth=3,linestyle='-.',label=r'WL Speckle')
#plt.plot(intTimes,wilks,linewidth=3,linestyle='--',label=r'SAO 65921 Speckle')
#plt.plot(saoCIntTimes,saoCWilks,color='black',linewidth=3,label=r'SAO 65921 Companion')
plt.legend(loc=4)
plt.xlabel(r'Lightcurve Short Exposure Time (s)')
plt.ylabel(r'Shapiro-Wilk Test (W)')
plt.show()




