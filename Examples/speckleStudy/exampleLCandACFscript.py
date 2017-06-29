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
import pdfs
from scipy import stats
from scipy.signal import periodogram
from statsmodels.tsa.stattools import acovf,acf,ccf
 

#load obs file for wl
obs = darkObsFile(FileName(run='PAL2017a',date='20170410',tstamp='1491869200').obs())
 
#load obs file for Pi Her
phobs = darkObsFile(FileName(run='PAL2017a',date='20170409',tstamp='1491827562').obs())
 
#load up timed packet images
#this WL obs file has a move at 18 seconds. Just cut it out, stationary after.
wlTPI = obs.getTimedPacketImage(firstSec=18, integrationTime=118)
phTPI = phobs.getTimedPacketImage(firstSec=0, integrationTime=118)

#hd91782 data
hdobs = darkObsFile(FileName(run='PAL2017a',date='20170410',tstamp='1491896381').obs())
hdTPI = hdobs.getTimedPacketImage(firstSec=0, integrationTime=-1)

#69 Her data
saoObs = darkObsFile(FileName(run='PAL2017a',date='20170409',tstamp='1491823630').obs())

#wvl cal
#cfn = FileName(run='PAL2017a',date='20170410',tstamp='1491894285').calSoln()
cfn = FileName(run='PAL2017a',date='20170409',tstamp='1491793638').calSoln()
saoObs.loadWvlCalFile(cfn)

#load bad pix mask for pi Her
#bpm = dhpm.loadMask('/mnt/data0/CalibrationFiles/darkHotPixMasks/20170409/1491826154.npz')
#load bad pix mask for WL
#bpm = dhpm.loadMask('/mnt/data0/CalibrationFiles/darkHotPixMasks/20170410/1491870115.npz')
#load bad pix mask for HD91782 data
#bpm = dhpm.loadMask('/mnt/data0/CalibrationFiles/darkHotPixMasks/20170410/1491894755.npz')
#bad pix mask for SAO65921 (69 Her)
bpm = dhpm.loadMask('/mnt/data0/CalibrationFiles/darkHotPixMasks/20170409/1491819780.npz')

hdobs.plotPixelSpectra(34, 91, firstSec=0, integrationTime= -1,
                         weighted=False, fluxWeighted=False, wvlStart=8000, wvlStop=14000,
                         energyBinWidth=0.07,verbose=False)

cAdict = saoObs.getPixelWvlList(46,98)
cBdict = saoObs.getPixelWvlList(47,98)
cCdict = saoObs.getPixelWvlList(46,99)
#cDdict = hdobs.getPixelWvlList(32,59)
#cEdict = hdobs.getPixelWvlList(32,60)
#cFdict = hdobs.getPixelWvlList(34,58)
#cGdict = hdobs.getPixelWvlList(34,59)
#cHdict = hdobs.getPixelWvlList(34,60)
#cIdict = hdobs.getPixelWvlList(33,57)

cTS = cAdict['timestamps']
cTS = np.append(cTS,cBdict['timestamps'])
cTS = np.append(cTS,cCdict['timestamps'])
#cTS = np.append(cTS,cDdict['timestamps'])
#cTS = np.append(cTS,cEdict['timestamps'])
#cTS = np.append(cTS,cFdict['timestamps'])
#cTS = np.append(cTS,cGdict['timestamps'])
#cTS = np.append(cTS,cHdict['timestamps'])
#cTS = np.append(cTS,cIdict['timestamps'])

sAdict = saoObs.getPixelWvlList(25,94)
sBdict = saoObs.getPixelWvlList(30,54)
sCdict = saoObs.getPixelWvlList(30,55)
sDdict = saoObs.getPixelWvlList(31,54)
sEdict = saoObs.getPixelWvlList(31,53)
#sFdict = saoObs.getPixelWvlList(34,90)
#sGdict = hdobs.getPixelWvlList(33,91)
#sHdict = hdobs.getPixelWvlList(34,91)
#sIdict = hdobs.getPixelWvlList(33,92)

sTS = sAdict['timestamps']
sTS = np.append(sTS,sBdict['timestamps'])
sTS = np.append(sTS,sCdict['timestamps'])
sTS = np.append(sTS,sDdict['timestamps'])
sTS = np.append(sTS,sEdict['timestamps'])
#sTS = np.append(sTS,sFdict['timestamps'])
#sTS = np.append(sTS,sGdict['timestamps'])
#sTS = np.append(sTS,sHdict['timestamps'])
#sTS = np.append(sTS,sIdict['timestamps'])


for i in range(14,50):
    for j in range(40,125):
        print i,j
        basePath = '/mnt/data0/ProcessedData/seth/speckleAnalysis/HD_stats/allPlots/'
        #make a lightcurve out of one of the good WL pixels (as determined in quicklook)
        if len(hdTPI['timestamps'][i][j])>0 and bpm[j][i]==0:
lcS = lc.lightCurve(cTS,shortExposure=0.005,start=0,stop=90)
timeS = lcS['time']
intS = lcS['intensity']
plt.plot(timeS,intS,label=r"$t=$5 ms")
plt.xlabel(r"Time(s)",fontsize=14)
plt.ylabel(r"Counts",fontsize=14)
plt.title(r"SAO65921 Location (A) Lightcurve",fontsize=14)
plt.legend()
plt.show()
            #plt.savefig(basePath+'%i_%i_LC.png'%(i,j))
            #plt.clf()
            #Plot distribution and fits
            #check the histogram of that lc, fit with MR or Poisson
histS, binsS = lc.histogramLC(intS,centers=True,N=40,span=[0,40])
guessIc = np.mean(intS)*0.7
guessIs = np.mean(intS)*0.3
guessLam = np.mean(intS)
fitIc, fitIs = pdfs.fitMR(binsS,histS,guessIc, guessIs)
fitMR = pdfs.modifiedRician(binsS,fitIc,fitIs)
fitLam = pdfs.fitPoisson(binsS,histS,guessLam)
fitPoisson = pdfs.poisson(binsS,fitLam)
plt.step(binsS,histS,color="grey",label=r"Histogram of (A) intensities",where="mid")
plt.plot(binsS,fitMR,color="black",linestyle="-.",label=r"MR fit to histogram: Ic=%2.2f, Is=%2.2f"%(fitIc, fitIs))
plt.plot(binsS,fitPoisson,linestyle="--",color="black",label=r"Poisson fit to histogram: Lambda=%2.2f"%(fitLam))
plt.legend()
plt.xlabel(r"Intensity",fontsize=14)
plt.ylabel(r"Probability",fontsize=14)
plt.title(r"SAO65921 Location (A) Intensity Distribution",fontsize=14)
plt.show()
            plt.savefig(basePath+'%i_%i_Dist.png'%(i,j))
            plt.clf()
            #Plot Correlation function
#calculate auto-corr function: Pearson correlation of lc w/itself shifted by various lags (tau)
corr,ljb,pvalue = acf(intS,unbiased=False,qstat=True,nlags=len(timeS))
#plot correlation as function of lag time
plt.plot(timeS,corr)
plt.xlabel(r"$\tau(s)$",fontsize=14)
plt.ylabel(r"$R(\tau)$",fontsize=14)
plt.title(r"HD91782 Location (B) Autocorrelation $R(\tau)$",fontsize=14)
plt.show()
            plt.savefig(basePath+'%i_%i_ACF.png'%(i,j))
            plt.clf()
        else:
            pass

