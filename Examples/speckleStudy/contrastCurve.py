
import numpy as np
import matplotlib.pylab as plt
plt.rc('font',family='serif')
import pdfs
from scipy import stats
from scipy.signal import periodogram
from statsmodels.tsa.stattools import acovf,acf,ccf
import vip
from readFITStest import readFITS

#load up pi Her psf and speckles
psf = readFITS('/mnt/data0/ProcessedData/seth/imageStacks/PAL2017a/piHer_psf_smoothed.fits')
speckles = readFITS('/mnt/data0/ProcessedData/seth/imageStacks/PAL2017a/piHer_coron_smoothed.fits')

#normalization to piHer unocculted psf, obtained with VIP aperture phot on center of psf manually
norm = 743716

lod = 8 #8 pixels in these upsampled images = one lambda/d

nlod = 12 #how many lambda/D do we want to calculate out to
sep = np.arange(nlod)+1
sepAS = sep*0.025*2

print sep
print sepAS

#pixel coords of center of images
centerx=360
centery=450

psfMeans = []
psfStds = []
psfSNRs = []

spMeans = []
spStds = []
spSNRs = []


for i in np.arange(nlod)+1:
    psf_an = vip.phot.snr_ss(psf,(centerx+i*lod,centery),fwhm=lod,plot=False,seth_hack=True)
    psfMeans.append(psf_an[3])
    psfStds.append(psf_an[4])
    psfSNRs.append(psf_an[5])

    sp_an = vip.phot.snr_ss(speckles,(centerx+i*lod,centery),fwhm=lod,plot=False,seth_hack=True)
    spMeans.append(sp_an[3])
    spStds.append(sp_an[4])
    spSNRs.append(sp_an[5])

spMeans = np.array(spMeans)
spStds = np.array(spStds)
spSNRs = np.array(spSNRs)
psfMeans = np.array(psfMeans)
psfStds = np.array(psfStds)
psfSNRs = np.array(psfSNRs)

fig,ax1 = plt.subplots()
#ax1.errorbar(sep,psfMeans/norm,yerr=psfStds/norm,linewidth=2,label=r'Mean Unocculted PSF Contrast')
#ax1.errorbar(sep,spMeans/norm,yerr=spStds/norm,linestyle='-.',linewidth=2,label=r'Mean Coronagraphic Raw Contrast')

ax1.errorbar(sep,psfMeans/norm+5*psfStds/norm,linewidth=2,label=r'5-$\sigma$ Unocculted PSF Contrast')
ax1.errorbar(sep,spMeans/norm+5*spStds/norm,linestyle='-.',linewidth=2,label=r'5-$\sigma$ Coronagraphic Raw Contrast')

ax1.axvline(x=3.3,ymin=1e-4,ymax=1,linestyle='--',color='black',linewidth=2,label = 'FPM Radius')
ax1.set_xlabel(r'Separation ($\lambda$/D)',fontsize=14)
ax1.set_ylabel(r'Contrast',fontsize=14)
#ax1.set_xlim(1,12)
#ax1.set_ylim(1e-4,1)
ax1.set_yscale('log')

ax2 = ax1.twiny()
ax2.plot(sepAS,psfMeans/norm,alpha=0)
ax2.set_xlabel(r'Separation (as)',fontsize=14)

ax1.legend()

plt.show()



