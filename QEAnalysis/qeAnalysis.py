import numpy as np
import matplotlib.pylab as plt
#plt.rc('text', usetex=True)
plt.rc('font', family='serif')
from loadStack import loadIMGStack
import HotPix.darkHotPixMask as dhpm
from Utils.utils import medianStack
from arrayPopup import plotArray

# Error from Zemax calc of illumination uniformity in beam: 5% (+/- 2.5% over mean)
# Can we use spread in QE across array as proxy for this?

# Error in DARKNESS data can be calculated from stdev of data at each wvl (and dark stdev)
# Error in PD data can be determined from stdev of each dark measurement

# PD path
#pdPath = '/mnt/data0/ScienceData/LabData/20170712/QE_test_20170712_50nmSteps.txt'

areaPD = 0.00155179165 #m^2
nPDraw = np.array([2.898,3.920,6.557,9.919,13.167,15.451,31.061,34.071,44.933,43.431,40.892,42.396,25.825])
nPDraw*=1.E7
nPDraw/=areaPD #pd data now in photons per second per m^2


# QE start timestamp: 1499913093
# QE stop timestamp: 1499914932

# Use quicklook to determine light and dark spans
# Data was 800 to 1400 with 50 nm steps
wvls = np.arange(800,1401,50)

dark = [[1499913093,1499913216],[1499913240,1499913330],[1499913353,1499913438],[1499913462,1499913551],[1499913575,1499913665],
        [1499913688,1499913778],[1499913798,1499914037],[1499914061,1499914154],[1499914178,1499914270],[1499914294,1499914386],
        [1499914410,1499914503],[1499914526,1499914619],[1499914642,1499914735]]

light= [[1499913218,1499913238],[1499913331,1499913351],[1499913440,1499913459],[1499913553,1499913573],[1499913667,1499913686],
        [1499913780,1499913795],[1499914039,1499914059],[1499914156,1499914175],[1499914272,1499914292],[1499914389,1499914408],
        [1499914505,1499914524],[1499914621,1499914640],[1499914737,1499914756]]

areaMKID = 2.25E-8 #m^2

medQEArray = []
stdevQEArray = []

# Use Nick's optimal filter fits to flag useable pixels
# In future data sets we'll need to come up with a repeatable version of this cut
# because we might not always have optimal filter data with this level of analysis
# to use like this. Really just need a way to remove bad pixels other than low cps.
# In this 04/2017 data we removed any pixel that Nick couldn't fit with a 
# Gaussian+exponential for a given laser wavelength.
rdata = np.load('qe_curves/wavecal_fits.npz')
rWvls = np.array([808,920,980,1120,1310])
rmasks = np.array([rdata['good_fits_808nm'],rdata['good_fits_920nm'],rdata['good_fits_980nm'],rdata['good_fits_1120nm'],rdata['good_fits_1310nm']])


dataDir = '/mnt/data0/ScienceDataIMGs/LabData/20170712/'
for i in range(len(wvls)):
    #find nearest "good pix" mask from Nick's R data
    wvlDiffs = np.abs(rWvls-wvls[i])
    nearestWvl = rWvls[np.where(wvlDiffs==np.min(wvlDiffs))][0]
    print nearestWvl
    nearestMask = rmasks[np.where(rWvls==nearestWvl)][0].transpose()

    if i==0: plotArray(nearestMask)
    stack = loadIMGStack(dataDir, light[i][0], light[i][1], nCols=80, nRows=125)
    darkstack = loadIMGStack(dataDir, dark[i][0], dark[i][1], nCols=80, nRows=125)

    bpm = dhpm.makeMask(run='LabData', date='20170712', startTimeStamp=dark[i][0], 
                        stopTimeStamp=dark[i][1], maxCut=2400, coldCut=True,verbose=False)

    medLight = medianStack(stack)
    
    print np.shape(medLight)
    print np.shape(nearestMask)
    medLight[np.where(bpm==1)]=np.nan
    medLight[np.where(nearestMask==False)]=np.nan

    medDark = medianStack(darkstack)
    medDark[np.where(bpm==1)]=np.nan
    medDark[np.where(nearestMask==False)]=np.nan
    
    #Lazy cold cut
    coldcut = 5

    medSub = medLight-medDark
    medSub[np.where(medSub<=coldcut)]=np.nan

    cpsPerArea = medSub/areaMKID
    QE = cpsPerArea/nPDraw[i]

    print QE


    if i==0: plotArray(medSub)
    if i==0:
        hist,bins = np.histogram(QE,bins=40,range=[0,0.2])
        plt.step(bins[:-1],hist,label=wvls[i])
        plt.xlabel(r'QE fraction')
        plt.ylabel(r'N resonators')
        plt.legend()
        plt.show()
    

    medQE = np.nanmedian(QE.flatten())
    stdevQE = np.nanstd(QE.flatten())

    medQEArray.append(medQE)
    stdevQEArray.append(stdevQE)

#print wvls
#print medQEArray
#print stdevQEArray

### Build theoretical QE curves
#Fused Silica transmission, assume 93% across band
fSiTrans = 0.93

#Inductor fill factor, i.e. amount lost to inductor leg gaps
inductorFF = 0.9

#MLA fill factor from manufacturer is 93%
mlaFF = 0.93

#MLA transmission from data sheet ~98%
mlaTrans = 0.98

#load up IR blocker filter
ibData = np.loadtxt('qe_curves/IR_blocker.txt',delimiter=',')
ibWvls = ibData[:,0]
ibTrans = ibData[:,1]

absData = np.load("./qe_curves/PtSi_Abs.npz")
#flip absorption data arrays for interpolate to work
absWvlsRev = absData['wvls']
absWvls = absWvlsRev[::-1]
absAbsOGRev = absData['absorption']*100.
absAbsOG = absAbsOGRev[::-1]
#interpolate absorption data to IR blocker data points
absAbs = np.interp(ibWvls,absWvls,absAbsOG)

print absWvls
print absAbsOG
print absAbs

mlaGood = 1.0
mlaAve = 0.8

theorTrans = (ibTrans/100.)*(ibTrans/100.)*(absAbs/100.)*mlaTrans*mlaFF*inductorFF*fSiTrans

#plt.plot(ibWvls,absAbs,linestyle=':')
#plt.plot(ibWvls,100.*(theorTrans*mlaGood),linestyle=':',linewidth=2, color='black',label=r'Theoretical; MLA = 100%')
plt.plot(ibWvls,100.*(theorTrans*mlaAve),linestyle='-.',linewidth=2, color='black',label=r'Theoretical')
plt.plot(wvls,np.array(medQEArray)*100,linewidth=3,color='black',label=r'Measured')
y2 = np.array(medQEArray)*100+np.array(stdevQEArray)*100
y1 = np.array(medQEArray)*100-np.array(stdevQEArray)*100

plt.fill_between(wvls, y1, y2, where=y2 >= y1, color='green',facecolor='green', interpolate=True,alpha=0.1)

plt.xlabel(r'Wavelength (nm)')
plt.ylabel(r'QE (%)')
plt.legend()
plt.xlim([799,1401])
plt.ylim([0,25])
plt.show()







