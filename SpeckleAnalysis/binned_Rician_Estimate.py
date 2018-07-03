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
from scipy.optimize import curve_fit

from DarknessPipeline.RawDataProcessing.darkObsFile import ObsFile
from DarknessPipeline.SpeckleAnalysis.genphotonlist import genphotonlist
from mpmath import mp, hyp1f1
from scipy import special



def blurredMR(x,Ic,Is):
    p = np.zeros(len(x))
    for ii in x:
        p[ii] = 1/(Is + 1)*(1 + 1/Is)**(-ii)*np.exp(-Ic/Is)*hyp1f1(float(x[ii]) + 1,1,Ic/(Is**2 + Is))
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
        lightCurveIntensityCounts - array with units of counts
        lightCurveIntensity - array with units of counts/sec
        lightCurveTimes - array with times corresponding to the bin centers of the light curve
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
        bins - 1d array specifying the bins (0 photons, 1 photon, etc.)
    
    
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




def fitBlurredMR(bins,intensityHist,effExpTime): 
    """
    fit a blurred modified rician to a histogram of intensities
    INPUTS:
        bins - 1d array specifying the bins (0 photons, 1 photon, etc.)
        intensityHist - 1d array containing the histogram
    OUTPUTS:
        Ic
        Is
    """
    sigma = np.sqrt(intensityHist)
    sigma[np.where(sigma==0)] = 1
    try:
        popt,pcov = curve_fit(blurredMR,bins,intensityHist,p0=[1,1],sigma=sigma,bounds=(0,np.inf))
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






if __name__ == "__main__":

    if 0:
        print("Generating photon list...",end="", flush=True)
        Ic, Is, Ttot, tau = [300., 30., 300., .1]
        ts = genphotonlist(Ic, Is, Ttot, tau)
        print("[Done]\n")
    
        print("=====================================")
    
    
    if 1:
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
        plt.xlabel('intensity [counts]')
        plt.ylabel('frequency')
        plt.title('intensity histogram. Effective exposure time = {:.4f}'.format(effExpTime))
        
        
        Ic_est,Is_est,covMatrix = fitBlurredMR(bins,intensityHist,effExpTime)
        
        print('\nIc and Is from curve_fit are: ',Ic_est,'  ',Is_est)
        
        #calculate the error on Ic and Is using the covariance matrix returned by curve_fit
        perr = np.sqrt(np.diag(covMatrix))/effExpTime #1 sigma errors, assuming the likelihood function is gaussian.
        
        
        mu = np.mean(lightCurveIntensityCounts)
        var = np.var(lightCurveIntensityCounts)  
    

        plt.plot(bins,blurredMR(np.arange(len(bins)),Ic_est*effExpTime,Is_est*effExpTime),'.-k',label = r'blurred MR from curve_fit. Ic,Is = {:.2f} $\pm$ {:.2f}, {:.2f} $\pm$ {:.2f}'.format(Ic_est,perr[0],Is_est,perr[1]))
        plt.plot(bins,blurredMR(np.arange(len(bins)),Ic*effExpTime,Is*effExpTime),'.-b',label = 'blurred MR from actual Ic and Is. Ic,Is = {:.2f}, {:.2f}'.format(Ic,Is))
#        plt.plot(bins,modifiedRician(np.arange(len(bins)),Ic_est*effExpTime,Is_est*effExpTime),'.-r',label = 'pure MR  Ic,Is = {:.2f}, {:.2f}'.format(Ic,Is))
        
        try:
            IIc = np.sqrt(mu**2 - var + mu)
        except:
            pass
        else:
            IIs = mu - IIc
            plt.plot(bins,blurredMR(np.arange(len(bins)),IIc,IIs),'.-c',label = 'blurredMR from mean and variance Ic,Is = {:.2f}, {:.2f}'.format(IIc/effExpTime,IIs/effExpTime))
            
    
        plt.legend()
        
        
        
    if 0:
        """
        Plot Ic,Is vs effExpTime
        """
    
        effExpTimeArray = np.arange(.005,.2,.005) #second
        
        IcArray = np.array([])
        IsArray = np.array([])
        covMatrixArray = np.array([])
        IIcArray = np.array([])
        IIsArray = np.array([])
        
        for effExpTime in effExpTimeArray:
            lightCurveIntensityCounts, lightCurveIntensity, lightCurveTimes = getLightCurve(ts,ts[0]/1e6,ts[-1]/1e6,effExpTime)
            intensityHist, bins = histogramLC(lightCurveIntensityCounts)
            
            Ic_est,Is_est,covMatrix = fitBlurredMR(bins,intensityHist,effExpTime)
            
            IcArray = np.append(IcArray,Ic_est)
            IsArray = np.append(IsArray,Is_est)
            covMatrixArray = np.append(covMatrixArray,covMatrix)
            
            mu = np.mean(lightCurveIntensityCounts)
            var = np.var(lightCurveIntensityCounts)
            try:
                IIc = np.sqrt(mu**2 - var + mu)
            except:
                IIc = 0
            else:
                IIs = mu - IIc
                IIcArray= np.append(IIcArray,IIc/effExpTime)
                IIsArray= np.append(IIsArray,IIs/effExpTime)

  
        covMatrixArray = covMatrixArray.reshape(len(effExpTimeArray),2,2)
        #calculate the error on Ic and Is using the covariance matrix returned by curve_fit
        Icerr = np.sqrt(covMatrixArray[:,0,0])/effExpTimeArray #1 sigma errors, assuming the likelihood function is gaussian.
        Iserr = np.sqrt(covMatrixArray[:,1,1])/effExpTimeArray
        

        
        
        plt.figure(3)
        plt.plot(effExpTimeArray,IcArray,'b.-',label = 'fit from binned light curve')
        plt.plot(effExpTimeArray,IIcArray,'k.-',label = 'from mean and variance')
        plt.plot(effExpTimeArray,Ic*np.ones(len(effExpTimeArray)),'r')
        plt.fill_between(effExpTimeArray,IcArray+Icerr,IcArray-Icerr,alpha = 0.4)
        plt.xlabel('effective exposure time [sec]')
        plt.ylabel('counts/sec')
        plt.title('fitted Ic')
        plt.legend()
        
        plt.figure(4)
        plt.plot(effExpTimeArray,IsArray,'b.-',label = 'fit from binned light curve')
        plt.plot(effExpTimeArray,IIsArray,'k.-',label = 'from mean and variance')
        plt.plot(effExpTimeArray,Is*np.ones(len(effExpTimeArray)),'r')
        plt.fill_between(effExpTimeArray,IsArray+Iserr,IsArray-Iserr,alpha = 0.4)
        plt.xlabel('effective exposure time [sec]')
        plt.ylabel('counts/sec')
        plt.title('fitted Is')
        plt.legend()
            
    
    
    
    
    
    
    
    
