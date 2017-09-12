'''
Author Seth Meeker
Date 2017-07-03

Short script to open DARKNESS wave cal soln files and generate histograms of all pixels' R from each laser line
'''

import numpy as np
from tables import *
from darkObsFile import darkObsFile
from Utils.FileName import FileName
import matplotlib.pylab as plt
plt.rc('font',family='serif')


#load obs file and cal file
obs = darkObsFile(FileName(run='PAL2017a',date='20170409',tstamp='1491793638').obs())
cfn = FileName(run='PAL2017a',date='20170410',tstamp='1491870376').calSoln()
obs.loadWvlCalFile(cfn)

#808 nm sigmas, measured in eV
sigmas = obs.wvlErrorTable[np.where(obs.wvlFlagTable==0)]

sigmas808 = sigmas[:,0]
sigmas808 = sigmas808[np.where(sigmas808!=0)]
sigmas808 = sigmas808[np.where(sigmas808!=1)]

sigmas1120 = sigmas[:,3]
sigmas1120 = sigmas1120[np.where(sigmas1120!=0)]
sigmas1120 = sigmas1120[np.where(sigmas1120!=1)]

# h * c in nm*eV units
hc = 1239.84 #nm*eV

#808 nm in eV
Ec808 = 1.5345 #eV
E1= Ec808-sigmas808/2.
E2 = Ec808+sigmas808/2.

Rs808 = (1./Ec808)/(1./E1 -  1./E2)
med808 = np.nanmedian(Rs808)
print Rs808

#1120 nm in eV
Ec1120 = 1.1070 #eV
E1= Ec1120-sigmas1120/2.
E2 = Ec1120+sigmas1120/2.

Rs1120 = (1./Ec1120)/(1./E1 -  1./E2)
med1120 = np.nanmedian(Rs1120)
print Rs1120


hist808, binEdges = np.histogram(Rs808,bins=30,range=(0,15))
hist1120, binEdges1120 = np.histogram(Rs1120,bins=30,range=(0,15))

bws = np.diff(binEdges)
cents = binEdges[:-1]+bws[0]/2.0
bins = cents

fig,ax1 = plt.subplots()
ax1.step(bins,hist808,color="blue",linewidth=2,label=r"808 nm, Median R = %2.2f"%med808,where="mid")
ax1.step(bins,hist1120,color="red",linewidth=2,label=r"1120 nm, Median R = %2.2f"%med1120,where="mid")

ax1.axvline(x=med808,ymin=0,ymax=1000,linestyle='--',color='blue',linewidth=2)
ax1.axvline(x=med1120,ymin=0,ymax=1000,linestyle='--',color='red',linewidth=2)

ax1.set_xlabel(r'Energy Resolution (R=$\lambda$/$\Delta$$\lambda$)',fontsize=14)
ax1.set_ylabel(r'N',fontsize=14)
plt.legend()
plt.show()
