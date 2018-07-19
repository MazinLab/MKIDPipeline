#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 11:24:00 2018

Author: Clint Bockstiegel
Date: June 21, 2018
Last Updated: June 21, 2018

This code contains functions for analyzing photon arrival times using BINNING. 

For example usage, see if __name__ == "__main__": 
"""

import numpy as np
from scipy.optimize import minimize
from statsmodels.base.model import GenericLikelihoodModel
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit

from mkidpipeline.hdf.darkObsFile import ObsFile
from mkidpipeline.speckle.genphotonlist import genphotonlist
from mpmath import mp, hyp1f1
from scipy import special



def blurredMR(n,Ic,Is):
    """
    Calculates the probability of getting a bin with n counts given Ic & Is. 
    n, Ic, Is must have the same units. 
    
    INPUTS:
        n - array of the number of counts you want to know the probability of encountering. numpy array, can have length = 1. Units are the same as Ic & Is
        Ic - the constant part of the speckle pattern [counts/time]. User needs to keep track of the bin size. 
        Is - the random part of the speckle pattern [units] - same as Ic
    OUTPUTS:
        p - array of probabilities
        
    EXAMPLE:
        n = np.arange(8)
        Ic,Is = 4.,6.
        p = blurredMR(n,Ic,Is)
        plt.plot(n,p)
        #plot the probability distribution of the blurredMR vs n
        
        n = 5
        p = blurredMR(n,Ic,Is)
        #returns the probability of getting a bin with n counts
    
    """
    
    n = n.astype(int)
    p = np.zeros(len(n))
    for ii in range(len(n)):
        p[ii] = 1/(Is + 1)*(1 + 1/Is)**(-n[ii])*np.exp(-Ic/Is)*hyp1f1(float(n[ii]) + 1,1,Ic/(Is**2 + Is))
    return p



def modifiedRician(I, Ic, Is):
    '''
    MR pdf(I) = 1/Is * exp(-(I+Ic)/Is) * I0(2*sqrt(I*Ic)/Is)
    mean = Ic + Is
    variance = Is^2 + 2*Ic*Is
    '''
    mr = 1.0/Is * np.exp(-1.0*(I+Ic)/Is)* special.iv(0,2.0*np.sqrt(I*Ic)/Is)
    return mr



def getLightCurve(photonTimeStamps,startTime =0,stopTime =10,effExpTime=.01):
    """
    Takes a 1d array of arrival times and bins it up with the given effective exposure 
    time to make a light curve.
    
    INPUTS:
        photonTimeStamps - 1d numpy array with units of microseconds
    OUTPUTS:
        lightCurveIntensityCounts - array with units of counts. Float.
        lightCurveIntensity - array with units of counts/sec. Float.
        lightCurveTimes - array with times corresponding to the bin centers of the light curve. Float.
    """
    histBinEdges = np.arange(startTime,stopTime,effExpTime)
    
    hist,_ = np.histogram(photonTimeStamps/10**6,bins=histBinEdges) #if histBinEdges has N elements, hist has N-1
    lightCurveIntensityCounts = 1.*hist  #units are photon counts
    lightCurveIntensity = 1.*hist/effExpTime  #units are counts/sec
    lightCurveTimes = histBinEdges[:-1] + 1.0*effExpTime/2
    
    return lightCurveIntensityCounts, lightCurveIntensity, lightCurveTimes
    # [lightCurveIntensityCounts] = counts
    # [lightCurveIntensity] = counts/sec
    
    
    
    

def histogramLC(lightCurve):
    """
    makes a histogram of the light curve intensities
    
    INPUTS:
        lightCurve - 1d array specifying number of photons in each bin
    OUTPUTS:
        intensityHist - 1d array containing the histogram
        bins - 1d array specifying the bins (0 photon, 1 photon, etc.)
    
    
    """
    #Nbins=30  #smallest number of bins to show
    
    Nbins = int(np.amax(lightCurve))
    
    if Nbins==0:
        intensityHist = np.zeros(30)
        bins = np.arange(30)
        #print('LightCurve was zero for entire time-series.')
        return intensityHist, bins
    
    #count the number of times each count rate occurs in the timestream
    intensityHist, _ = np.histogram(lightCurve,bins=Nbins,range=[0,Nbins])
    
    intensityHist = intensityHist/float(len(lightCurve))      
    bins = np.arange(Nbins)
    
    return intensityHist, bins



def muVar_to_IcIs(mu,var,effExpTime):
    """
    given a mean count rate mu and variance of the count rate of a light curve,
    calculate Ic and Is.
    INPUTS:
        mu - the mean count rate. [counts/bin]
        var - the variance of the count rate [counts/bin]
    OUTPUTs:
        Ic - counts/sec
        Is - counts/sec
    """
    
    if not np.isnan(np.sqrt(mu**2 - var + mu)):
        Ic = np.sqrt(mu**2 - var + mu)
        Is = mu - Ic
    else:
        return

        
    Ic/=effExpTime
    Is/=effExpTime    
    
    return Ic,Is



def fitBlurredMR(bins,intensityHist,effExpTime, **kwargs): 
    """
    fit a blurred modified rician to a histogram of intensities
    INPUTS:
        bins - 1d array specifying the bins (0 photons, 1 photon, etc.)
        intensityHist - 1d array containing the histogram
        effExpTime - effective exposure time, aka bin size. [seconds]
        **kwargs - keywords for fitBlurredMR, including
            Ic_guess
            Is_guess
    OUTPUTS:
        Ic - counts/sec
        Is - counts/sec
        pcov - covariance matrix returned by curve_fit
    """
    try: Ic_guess = kwargs['Ic_guess']
    except: p0 = [1,1]
    else:
        Is_guess = kwargs['Is_guess']
        p0 = [Ic_guess, Is_guess]
    
    sigma = np.sqrt(intensityHist)
    sigma[np.where(sigma==0)] = 1
#    print('\np0 is ',p0)
#    print('Ic_guess, Is_guess are: ',Ic_guess,Is_guess,'\n\n')
    try:
        popt,pcov = curve_fit(blurredMR,bins,intensityHist,p0=p0,sigma=sigma,bounds=(0,np.inf))
        #units of pcov are I^2
        
        Ic = popt[0]
        Is = popt[1]
    except RuntimeError:
        Ic, Is = 1,0.1
        print('WARNING: curve_fit failed :(')
    else:
        Ic/=effExpTime
        Is/=effExpTime
    return Ic,Is,pcov



def binMRlogL(n, Ic, Is):
    '''
    Given a light curve, calculate the Log likelihood that
    its intensity distribution follows a blurred modified Rician with Ic, Is. 
    
    INPUTS:
        n: 1d array containing the (binned) intensity as a function of time, i.e. a lightcurve [counts/sec]. Bin size must be fixed. 
        Ic: Coherent portion of MR [1/second]
        Is: Speckle portion of MR [1/second]
    OUTPUTS:
        [float] the Log likelihood. 
    
    '''
    lnL = np.zeros(len(n)) 
    tmp = np.zeros(len(n))
    for ii in range(len(n)): #hyp1f1 can't do numpy arrays because of its data type, which is mpf
        tmp[ii] = float(hyp1f1(n[ii] + 1,1,Ic/(Is**2 + Is)))
    lnL = np.log(1./(Is+1)) - n*np.log(1+1./Is) - Ic/Is + np.log(tmp)
        
    return np.sum(lnL)



def plotLogLMap(n, Ic_list, Is_list, effExpTime):
    """
    plots a map of the MR log likelihood function over the range of Ic, Is
    
    INPUTS:
        n - light curve [counts]
        Ic_list - list of Ic values [photons/second]
        Is_list - list
    OUTPUTS:
        
    """
    
    Ic_list_countsperbin = Ic_list*effExpTime  #convert from cps to counts/bin
    Is_list_countsperbin = Is_list*effExpTime
    
    im = np.zeros((len(Ic_list),len(Is_list)))
    
    for i, Ic in enumerate(Ic_list_countsperbin): #calculate maximum likelihood for a grid of 
        for j, Is in enumerate(Is_list_countsperbin):   #Ic,Is values using counts/bin
            print('Ic,Is = ',Ic/effExpTime,Is/effExpTime)
            lnL = binMRlogL(n, Ic, Is)
            im[i,j] = lnL
            
    Ic_ind, Is_ind=np.unravel_index(im.argmax(), im.shape)
    print('Max at ('+str(Ic_ind)+', '+str(Is_ind)+')')
    print("Ic="+str(Ic_list[Ic_ind])+", Is="+str(Is_list[Is_ind]))
    print(im[Ic_ind, Is_ind])

    
#    l_90 = np.percentile(im, 90)
#    l_max=np.amax(im)
#    l_min=np.amin(im)
#    levels=np.linspace(l_90,l_max,int(len(im.flatten())*.1))
    
    plt.figure()
#    plt.contourf(Ic_list, Is_list,im.T,levels=levels,extend='min')
  
    
    X, Y = np.meshgrid(Ic_list, Is_list)
    sigmaLevels = np.array([8.36, 4.78, 2.1])
    levels = np.amax(im) - sigmaLevels

    MYSTYLE = {'contour.negative_linestyle':'solid'}
    oldstyle = {key:matplotlib.rcParams[key] for key in MYSTYLE}
    matplotlib.rcParams.update(MYSTYLE)

#    plt.contourf(X,Y,im.T)
    plt.imshow(im.T,extent = [np.amin(Ic_list), np.amax(Ic_list), np.amin(Is_list), np.amax(Is_list)],aspect='auto',origin = 'lower')
    plt.contour(X,Y,im.T,colors='black',levels = levels)
    plt.plot(Ic_list[Ic_ind],Is_list[Is_ind],"xr")
    plt.xlabel('Ic [/s]')
    plt.ylabel('Is [/s]')
    plt.title('Map of log likelihood')
    
    matplotlib.rcParams.update(oldstyle)

    return X,Y,im






if __name__ == "__main__":

    if 0:
        print("Generating photon list...",end="", flush=True)
        Ic, Is, Ttot, tau = [300., 30., 300., .1] # [Ic, Is] = cps, [Ttot, tau] = sec
        ts = genphotonlist(Ic, Is, Ttot, tau)
        print("[Done]\n")
    
        print("=====================================")
    
    
    if 0:
        """
        Make a plot showing a histogram fit.
        """
        effExpTime = 0.01 #second

        
        lightCurveIntensityCounts, lightCurveIntensity, lightCurveTimes = getLightCurve(ts,ts[0]/1e6,ts[-1]/1e6,effExpTime)
        if 0:
            plt.figure(1)
            plt.plot(lightCurveTimes,lightCurveIntensityCounts)
            plt.xlabel('time [s]')
            plt.ylabel('counts')
            plt.title('light curve')
            plt.show()
        
        intensityHist, bins = histogramLC(lightCurveIntensityCounts)
        plt.figure(2)
        plt.bar(bins,intensityHist)
        plt.xlabel('intensity [counts/bin]')
        plt.ylabel('frequency')
        plt.title('intensity histogram. Effective exposure time = {:.4f}'.format(effExpTime))
        
        
        Ic_est,Is_est,covMatrix = fitBlurredMR(bins,intensityHist,effExpTime)
        
        print('\nIc and Is from curve_fit are: ',Ic_est,'  ',Is_est, ' counts/sec')
        
        #calculate the error on Ic and Is using the covariance matrix returned by curve_fit
        perr = np.sqrt(np.diag(covMatrix))/effExpTime #1 sigma errors, assuming the likelihood function is gaussian.
        
        
        mu = np.mean(lightCurveIntensityCounts)
        var = np.var(lightCurveIntensityCounts)  
    

        plt.plot(bins,blurredMR(np.arange(len(bins)),Ic_est*effExpTime,Is_est*effExpTime),'.-k',label = r'blurred MR from curve_fit. Ic,Is = {:.2f} $\pm$ {:.2f}, {:.2f} $\pm$ {:.2f}'.format(Ic_est,perr[0],Is_est,perr[1]))
        plt.plot(bins,blurredMR(np.arange(len(bins)),Ic*effExpTime,Is*effExpTime),'.-b',label = 'blurred MR from actual Ic and Is. Ic,Is = {:.2f}, {:.2f}'.format(Ic,Is))
#        plt.plot(bins,modifiedRician(np.arange(len(bins)),Ic_est*effExpTime,Is_est*effExpTime),'.-r',label = 'pure MR  Ic,Is = {:.2f}, {:.2f}'.format(Ic,Is))
        
        
        
        ''''
        try:
            IIc = np.sqrt(mu**2 - var + mu)  #intensity per bin. 
        except:
            pass
        else:
            IIs = mu - IIc  #intensity per bin
            
            IIc/=effExpTime  #change to intensity per second
            IIs/=effExpTime  #change to intensity per second
            
            plt.plot(bins,blurredMR(np.arange(len(bins)),IIc*effExpTime,IIs*effExpTime),'.-c',label = 'blurredMR from mean and variance Ic,Is = {:.2f}, {:.2f}'.format(IIc,IIs))
            
        '''
        
        try:
            IIc,IIs = muVar_to_IcIs(mu,var,effExpTime)
        except:
            print('unable to get Ic and Is from mean and variance.')
        else:
            plt.plot(bins,blurredMR(np.arange(len(bins)),IIc*effExpTime,IIs*effExpTime),'.-c',label = 'blurredMR from mean and variance Ic,Is = {:.2f}, {:.2f}'.format(IIc,IIs))
            
    
        plt.legend()
        
        
        
    if 0:
        """
        Plot Ic,Is vs effExpTime
        """
    
        effExpTimeArray = np.arange(.05,.2,.05) #second   (.005,.5,.005)
        
        IcArray = np.array([])
        IsArray = np.array([])
        covMatrixArray = np.array([])
        IIcArray = np.array([])
        IIsArray = np.array([])
        NdownSampleArray = np.array([])
        
        for effExpTime in effExpTimeArray:
            lightCurveIntensityCounts, lightCurveIntensity, lightCurveTimes = getLightCurve(ts,ts[0]/1e6,ts[-1]/1e6,effExpTime)
            

            if 0:
                '''
                Down-sample the light curve so that we do a fit of the distribution after removing
                some bins. Use the decorrelation time to determine how often to select a bin, then
                discard the bins in between for that particular fit of Ic & Is. 
                
                For example, if the decorrelation time is 8*effExpTime, do the fit
                with bins 0,8,16,24,... and ignore bins 1-7, 9-15, 17-23, ...
                Then do a fit with bins 1,9,17,25 while ignoring bins 2-8, 10-16, ...
                '''
                NdownSample = 1*np.ceil(tau/effExpTime).astype(int) 
                NdownSampleArray = np.append(NdownSampleArray,NdownSample)
                lightCurveIntensityCounts = lightCurveIntensityCounts[::NdownSample]
                lightCurveIntensity = lightCurveIntensity[::NdownSample]
                lightCurveTimes = lightCurveTimes[::NdownSample]
            
            intensityHist, bins = histogramLC(lightCurveIntensityCounts)
            
            
            mu = np.mean(lightCurveIntensityCounts)
            var = np.var(lightCurveIntensityCounts)
            
            '''
            if not np.isnan(np.sqrt(mu**2 - var + mu)):
                IIc = np.sqrt(mu**2 - var + mu)
                IIs = mu - IIc
            else:
                IIc = mu/2
                IIs = mu - IIc
            '''
            
            
            try:
                IIc,IIs = muVar_to_IcIs(mu,var,effExpTime)
            except:
                IIc = mu/2       #just create a reasonable seed for curve_fit
                IIs = mu - IIc
                IIc/=effExpTime
                IIs/=effExpTime
                
                
                
            IIcArray= np.append(IIcArray,IIc)
            IIsArray= np.append(IIsArray,IIs)
            '''
            old code:
            try:
                IIc = np.sqrt(mu**2 - var + mu)
            except:
                IIc = 0
            else:
                IIs = mu - IIc
                IIcArray= np.append(IIcArray,IIc/effExpTime)
                IIsArray= np.append(IIsArray,IIs/effExpTime)
            
            '''

            Ic_est,Is_est,covMatrix = fitBlurredMR(bins,intensityHist,effExpTime,Ic_guess = IIc,Is_guess = IIs)
            
            IcArray = np.append(IcArray,Ic_est)
            IsArray = np.append(IsArray,Is_est)
            covMatrixArray = np.append(covMatrixArray,covMatrix)
            


  
        covMatrixArray = covMatrixArray.reshape(len(effExpTimeArray),2,2)
        #calculate the error on Ic and Is using the covariance matrix returned by curve_fit
        Icerr = np.sqrt(covMatrixArray[:,0,0])/effExpTimeArray #1 sigma errors, assuming the likelihood function is gaussian.
        Iserr = np.sqrt(covMatrixArray[:,1,1])/effExpTimeArray
        
        if len(np.where(Icerr>2*IcArray)[0])>0 or len(np.where(Iserr>2*IsArray)[0])>0:
            print('\n\nSome points have HUGE error bars. Setting the error bars to zero so the plot is still usable. ')
            temp = np.where(Icerr>2*IcArray)[0]
            print('\nThe indices in Icerr with values greater than 2*Ic are: \n',temp)
            Icerr[temp]=0  #sometimes the error bars are huge and screw up the plot.
            temp = np.where(Iserr>2*IsArray)[0]
            print('\nThe indices in Iserr with values greater than 2*Is are: \n',temp)
            Iserr[temp]=0  #Just set the error to zero, so it's obvious something is wrong.
        

        
        
        plt.figure(3)
        plt.plot(effExpTimeArray,IcArray,'b.-',label = 'fit from binned light curve')
        plt.plot(effExpTimeArray,IIcArray,'k.-',label = 'from mean and variance')
        plt.plot(effExpTimeArray,Ic*np.ones(len(effExpTimeArray)),'r')
        plt.fill_between(effExpTimeArray,IcArray+Icerr,IcArray-Icerr,alpha = 0.4)
        plt.axvline(x=tau,color='g',ls='--')
        plt.xlabel('effective exposure time [sec]')
        plt.ylabel('counts/sec')
        plt.title('fitted Ic')
        plt.legend()
        
        plt.figure(4)
        plt.plot(effExpTimeArray,IsArray,'b.-',label = 'fit from binned light curve')
        plt.plot(effExpTimeArray,IIsArray,'k.-',label = 'from mean and variance')
        plt.plot(effExpTimeArray,Is*np.ones(len(effExpTimeArray)),'r')
        plt.fill_between(effExpTimeArray,IsArray+Iserr,IsArray-Iserr,alpha = 0.4)
        plt.axvline(x=tau,color='g',ls='--')
        plt.xlabel('effective exposure time [sec]')
        plt.ylabel('counts/sec')
        plt.title('fitted Is')
        plt.legend()
        
        
        '''
        Plot the down-sampling parameter
        
        plt.figure(5)
        plt.plot(effExpTimeArray,NdownSampleArray-1,'b.-')
        plt.axvline(x=tau,color='g',ls='--')
        plt.xlabel('effective exposure time [sec]')
        plt.ylabel('# of bins discarded per effExpTime')
        plt.title('down-sample parameter')        
        '''
        
        
        
    if 0:
        '''
        Plot the log likelihood vs Is while keeping Ic constant. 
        '''
        effExpTime = 0.01 #second
        lightCurveIntensityCounts, lightCurveIntensity, lightCurveTimes = getLightCurve(ts,ts[0]/1e6,ts[-1]/1e6,effExpTime)
        
        Ic = 100.
        Is = np.linspace(30,300,11)
        logLArray = np.zeros(len(Is))
        for ii in range(len(Is)):    
            logLArray[ii] = binMRlogL(lightCurveIntensityCounts,Ic,Is[ii])
            
        plt.plot(Is,logLArray,'b.-')
        plt.xlabel('Is')
        plt.ylabel('log L')
        plt.title('Ic = {}'.format(Ic))
        
        
        
    if 0:
        x = np.arange(30)
        Ic = 3.
        Is = .3
        y = blurredMR(x,Ic,Is)
        plt.plot(x,y,'b.-')
        
        tmp = np.amax(y)
        x = np.arange(300)
        Ic = 30.
        Is = 3.
        y = blurredMR(x,Ic,Is)
        plt.plot(x/10,y/np.amax(y)*tmp,'r.-')
        print('sum is: ', np.sum(y))
            
            
        
    if 1:
        
        lightCurveIntensityCounts, lightCurveIntensity, lightCurveTimes = getLightCurve(ts,ts[0]/1e6,ts[-1]/1e6,effExpTime)
        
        
        print("Mapping...")
        Ic_list=np.linspace(285,315,2)  #linspace(start,stop,number of steps)
        Is_list=np.linspace(25,35,2)
        X,Y,im = plotLogLMap(lightCurveIntensityCounts, Ic_list, Is_list, effExpTime)
        
        """
        Save the logL plot data in a pickle file:
        
        with open('junk.pkl','wb') as f:
            pickle.dump([ts,Ic_list,Is_list,X,Y,im],f)
            f.close()
            
            
        
        Open the LogL plot data in another python session:
            
        with open('junk.pkl','rb') as f:
            ts,Ic_list,Is_list,X,Y,im = pickle.load(f)
            f.close() 
            
        plt.contourf(X,Y,im.T)
        plt.xlabel('Ic [/s]')
        plt.ylabel('Is [/s]')
        plt.title('Map of log likelihood')
        plt.show()
            
            
        """
    
    
    
    
    
    
    
    
