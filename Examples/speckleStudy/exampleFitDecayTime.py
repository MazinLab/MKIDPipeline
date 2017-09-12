'''
Author Seth Meeker 06-25-2017

Tools to handle lightcurves for speckle statistics studies

'''

#import stuff
from darkObsFile import darkObsFile
from Utils.FileName import FileName
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

phTS = phA
phTS = np.append(phTS,phB)
phTS = np.append(phTS,phC)

shortT = 0.002
#nFitTo = 7./shortT #which index in the corr func do we want to fit out to (1s/short int time)
nFitTo = 50
print nFitTo

phLC = lc.lightCurve(phTS,shortExposure=shortT)
phTime = phLC['time']#-18 # for WL subtract 18
phInt = phLC['intensity']

#calculate auto-corr function: Pearson correlation of lc w/itself shifted by various lags (tau)

phCorr,phLJB,phPvalue = acf(phInt,unbiased=True,qstat=True,nlags=len(phTime))
#x= phTime-phTime[0]
x = phTime[1:nFitTo]-phTime[0]

#Trying to fit exponential
guessLam=0
guessTau=0.3
guessf0=0.1
lam,tau,f0 = pdfs.fitExponential(phTime[1:nFitTo]-phTime[0],phCorr[1:nFitTo],guessLam,guessTau, guessf0)
fity = pdfs.exponential(x,lam,tau,f0)
print lam
print tau
print f0

### Trying to fit lorentzian, not working very well.
#guessGam1 = 0.04
#guessGam2 = 0.8
#guessX1 = guessX2 = 0
#g1,x1,g2,x2 = pdfs.fitDoubleLorentzian(phTime[0:nFitTo]-phTime[0],phCorr[0:nFitTo],guessGam1,guessX1, guessGam2,guessX2)
#g1,g2 = pdfs.fitDoubleLorentzian(phTime[0:nFitTo]-phTime[0],phCorr[0:nFitTo],guessGam1,guessGam2)
#lor1 = pdfs.lorentzian(x,g1,x1)
#lor2 = pdfs.lorentzian(x,g2,x2)
#fity = lor1+lor2
#print phTime[nFitTo]
#print g1
#print x1
#print g2
#print x2

#plot correlation as function of lag time
plt.plot(phTime-phTime[0],phCorr,marker='o',color='black',alpha=0.7)
#plt.plot(x,fity,color='green',linewidth=2,label=r'$\tau_{1}$ = %f3.1 ms; $\tau_{2}$ = %f3.1 ms'%(g1*1000.,g2*1000.))
plt.plot(x,fity,color='green',linewidth=2,label=r'$\tau$ = %3.2f ms'%(1000*tau))
plt.xlabel(r"$\tau(s)$",fontsize=14)
plt.ylabel(r"$R(\tau)$",fontsize=14)
plt.title(r"$\pi$ Her Autocorrelation $R(\tau)$",fontsize=14)
#plt.title(r"WL Autocorrelation $R(\tau)$",fontsize=14)
plt.legend()
plt.show()

plt.plot(phTime,phInt,label=r"%2.0f ms Exposures"%(shortT*1000))
plt.xlabel(r'Time (s)')
plt.legend()
plt.ylabel(r'Counts')
plt.show()


