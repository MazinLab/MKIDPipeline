"""
Author: Alex Walter
Date: Feb 22, 2018

This code is for analyzing the photon arrival time statistics in a bin-free way
to find a maximum likelihood estimate of Ic, Is. 
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


from DarknessPipeline.RawDataProcessing.darkObsFile import ObsFile



class IcIsEstimator:

    def __init__(self, filename, verbose=False):
        """
        
        """
        if verbose: print(filename)
        self.fileName=filename
        self.obs = ObsFile(filename)

    def plotImage(self,**kwargs):
        im = self.obs.getPixelCountImage(**kwargs)
        
        #im['image'][31,79]+=10**4
        plt.figure()
        plt.imshow((im['image']/im['effIntTimes']).T)
        #plt.show()
    
    def plotPixelLightCurve(self,xCoord,yCoord,cadence=1,**kwargs):
        
        lightCurve = self.obs.getPixelLightCurve(xCoord,yCoord,cadence=cadence,**kwargs)
        t=np.arange(0,len(lightCurve))*cadence
        plt.figure()
        plt.plot(t,lightCurve)
        plt.xlabel('Time since start of file (s)')
        plt.ylabel('Counts')
        plt.title(self.fileName+' - pixel x,y = '+str(xCoord)+','+str(yCoord))
        #plt.show()
    
    def histPixelLightCurve(self,xCoord,yCoord,bins=None,cadence=1,**kwargs):
        lightCurve = self.obs.getPixelLightCurve(xCoord,yCoord,cadence=cadence,**kwargs)
        lightCurve/=cadence
        plt.figure()
        plt.hist(lightCurve, bins=bins,histtype='step')
        plt.xlabel('Count rate (/s)')
        plt.ylabel('#')
        plt.title(self.fileName+' - pixel x,y = '+str(xCoord)+','+str(yCoord))
        #plt.show()
    
    def histPixelSpectrum(self,xCoord,yCoord, bins=None, **kwargs):
        photonList = self.obs.getPixelPhotonList(xCoord, yCoord,**kwargs)
        wvls = photonList['Wavelength']
        plt.figure()
        plt.hist(wvls, bins=bins,histtype='step')
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('#Photons')
        plt.title(self.fileName+' - pixel x,y = '+str(xCoord)+','+str(yCoord))
    
    def likelihood(self, dt, Ic, Is):
        lnL = -1.*dt*Ic/(1.+dt*Is) + np.log(Ic + Is + dt*Is**2.) - 3.*np.log(1.+dt*Is)
        return np.sum(lnL)
        
    def plotLogLPixel(self, xCoord, yCoord, Ic_list, Is_list, **kwargs):
        photonList = self.obs.getPixelPhotonList(xCoord, yCoord,**kwargs)
        times=photonList['Time']
        print("#photons: "+str(len(times)))
        dt = times[1:] - times[:-1]
        #plt.plot(dt)
        #plt.show()
        dt=dt[np.where(dt<1.e6)]/10.**6
        print(np.mean(dt))
        im = np.zeros((len(Ic_list),len(Is_list)))
        x,y,z=[],[],[]
        for i, Ic in enumerate(Ic_list):
            for j, Is in enumerate(Is_list):
                lnL = self.likelihood(dt, Ic, Is)
                im[i,j] = lnL
                #x.append(Ic)
                #y.append(Is)
                #z.append(lnL)
        
        #im = np.expm1(im)
        Ic_ind, Is_ind=np.unravel_index(im.argmax(), im.shape)
        print(str(Ic_ind)+', '+str(Is_ind))
        print("Ic="+str(Ic_list[Ic_ind])+", Is="+str(Is_list[Is_ind]))
        print(im[Ic_ind, Is_ind])
        #plt.figure()
        #plt.imshow(im.T, origin='lower')
        #xi=np.linspace(Ic_list[0], Ic_list[-1], 1000)
        #yi=np.linspace(Is_list[0], Is_list[-1], 1000)
        #Z = matplotlib.mlab.griddata(x, y, z, xi, yi, interp='linear')
        #print(Z)
        #print(np.reshape(z,)
        #X,Y=np.meshgrid(x,y)
        plt.figure()
        l_90 = np.percentile(im, 90)
        l_max=np.amax(im)
        l_min=np.amin(im)
        levels=np.linspace(l_90,l_max,int(len(im.flatten())*.1))
        #levels=np.append(l_min, levels)
        
        #levels=np.arange(np.amin(im), np.amax(im), (np.amax(im)-np.amin(im))/len(im.flatten()))
        #levels=np.append(levels,[levels[-1]+levels[-1]-levels[-2]])
        
        #levels=np.logspace(np.log10(np.amin(im)),np.log10(np.amax(im)),num=50, endpoint=True)
        
        plt.contourf(Ic_list, Is_list,im.T,levels=levels,extend='min')
        #plt.contourf(Ic_list, Is_list,im.T,vmin=l_90)
        plt.plot(Ic_list[Ic_ind],Is_list[Is_ind],"xr")
        
        
        
        


if __name__ == "__main__":
    # apply wavecal
    a=['/home/abwalter/peg32/'+f for f in os.listdir('/home/abwalter/peg32') if f.endswith('.h5')]
    #for f in a:
    #    print(f)
    #    obs = ObsFile(f,'write')
    #    obs.applyWaveCal('/home/abwalter/peg32/wavecalsoln/calsol_1519382454.h5')


    fn = '/home/abwalter/peg32/1507175503.h5'
    est = IcIsEstimator(fn,True)
    #est.plotImage(wvlRange=[100, 1300])
    #est.plotPixelLightCurve(32,79,wvlRange=[100, 1100],cadence=.01)
    #est.histPixelLightCurve(32,79,cadence=.01,wvlRange=[100, 1100],bins=100)
    #est.plotPixelLightCurve(50,71,wvlRange=[100, 1100],cadence=.01)
    #est.histPixelLightCurve(50,71,cadence=.01,wvlRange=[100, 1100],bins=100)
    #est.histPixelSpectrum(50,71,bins=20)
    
    Ic_list = np.arange(240,241.,.01)
    Is_list = np.arange(1.5,3.,.01)
    est.plotLogLPixel(30,81, Ic_list, Is_list, wvlStart=100, wvlStop=900)
    
    
    plt.show()









