'''
Author Seth Meeker 08-29-2017

Short script to make image stack out of pi Her 5ms frames, look for TT jitter

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
#import skimage for in-painting interp
from skimage import restoration
 


#load bad pix mask for pi Her
bpm = dhpm.loadMask('/mnt/data0/CalibrationFiles/darkHotPixMasks/20170409/1491826154.npz')
#load obs file for Pi Her
phobs = darkObsFile(FileName(run='PAL2017a',date='20170409',tstamp='1491827562').obs())
 
phTPI = phobs.getTimedPacketImage(firstSec=0, integrationTime=118)


shortT = 0.01
show=False
totalT = 1.000
nFrames = int(totalT/shortT)

stack = []

for n in range(nFrames):
    im = np.empty((80,40),dtype=float)
    im.fill(np.nan)
    for i in np.arange(20,60): #80
        for j in np.arange(40,120): #125
            if bpm[j,i]!=1:
                print i,j
                ts = phTPI['timestamps'][i][j]
                nphots = len(np.where(np.logical_and(ts>n*shortT, ts<(n+1)*shortT))[0])
                print nphots
                im[j-40,i-20] = float(nphots) 
            else:
                print "Bad pixel"
                pass
    stack.append(im)
    
#save fig 
for fn in np.arange(len(stack)):
    frame = np.array(stack[fn])
    #frame[np.where(frame==0)]=np.nan
    #mask = 1.-np.isfinite(frame)
    #interpImage = restoration.inpaint.inpaint_biharmonic(frame,mask,False)
    #plt.matshow(interpImage,vmin=0,vmax=40)
    plt.matshow(frame,vmin=0,vmax=40)
    plt.colorbar()
    plt.title(r't = %3.0f ms'%(1000*fn*shortT))
    plt.savefig('/mnt/data0/ProcessedData/seth/speckleAnalysis/piHer_stats/shortExpStack/10ms/phStack_%i.png'%fn)
    plt.clf()



