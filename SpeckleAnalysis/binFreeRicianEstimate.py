"""
Author: Alex Walter
Date: Feb 22, 2018
Last Updated: May 8, 2018

This code is for analyzing the photon arrival time statistics in a bin-free way
to find a maximum likelihood estimate of Ic, Is. 

For example usage, see if __name__ == "__main__": 
"""

import numpy as np
from scipy.optimize import minimize
from statsmodels.base.model import GenericLikelihoodModel
import matplotlib.pyplot as plt

from DarknessPipeline.RawDataProcessing.darkObsFile import ObsFile
from DarknessPipeline.SpeckleAnalysis.genphotonlist import genphotonlist


class MR_SpeckleModel(GenericLikelihoodModel):
    def __init__(self, ts):
        """
        Class for fitting maximum likelihood
        
        INPUTS:
            ts - Photon arrival times [us]
        """
        dt = ts[1:] - ts[:-1]
        dt=dt[np.where(dt<1.e6)]/10.**6
        endog=np.asarray(dt, dtype=[('dt','float64')])
        exog=np.ones(len(endog),dtype={'names':['Ic','Is'],'formats':['float64','float64']})
        score=lambda params: MRlogL_Jacobian(dt, params[0], params[1])
        hess=lambda params: MRlogL_Hessian(dt, params[0], params[1])
        loglike=lambda params: MRlogL(dt, params[0], params[1])
        super(MR_SpeckleModel, self).__init__(endog,exog=exog, loglike=loglike, score=score, hessian=hess)
    
    def fit(self, start_params=[1.,1.], method='powell', **kwargs):
        """
        Find maximum log likelihood
        
        INPUTS:
            start_params - Initial guess for [Ic, Is] floats
            method - Optimization method
            **kwargs - keywords for LikelihoodModel.fit(). 
                       See www.statsmodels.org/stable/dev/generated/statsmodels.base.model.LikelihoodModel.fit.html
        OUTPUTS:
            GenericLikelihoodModelResults Object
            See www.statsmodels.org/stable/dev/generated/statsmodels.base.model.GenericLikelihoodModelResults.html
        """
        return super(MR_SpeckleModel, self).fit(start_params,method=method,**kwargs)  

def MRlogL_Jacobian(dt, Ic, Is):
    """
    Calculates the Jacobian of the MR log likelihood function.
    Also called a Gradient or Fisher's Score Function. 
    It's just the vector of first partial derivatives.
    
    INPUTS:
        dt: list of inter-arrival times [seconds]
        Ic: Coherent portion of MR [1/second]
        Is: Speckle portion of MR [1/second]
    OUTPUTS:
        jacobian vector [dlnL/dIc, dlnL/dIs] at Ic, Is
    """
    jac_Ic = -1./(1./dt+Is) + 1./(Ic+Is+dt*Is**2.)
    jac_Is = dt**2.*Ic/(1.+dt*Is)**2. - 3.*dt/(1.+dt*Is) + (1.+2.*dt*Is)/(Ic+Is+dt*Is**2.)
    return np.asarray([np.sum(jac_Ic), np.sum(jac_Is)])

def MRlogL_Hessian(dt, Ic, Is):
    """
    Calculates the Hessian matrix of the MR log likelihood function.
    It's just the matrix of second partial derivitives.
    It's the negative of Fisher's Observed Information Matrix
    
    INPUTS:
        dt: list of inter-arrival times [seconds]
        Ic: Coherent portion of MR [1/second]
        Is: Speckle portion of MR [1/second]
    OUTPUTS:
        Hessian matrix [[d2lnL/dIc2, d2lnL/dIcdIs], [d2lnL/dIsdIc, d2lnL/dIsdIs]] at Ic, Is
    """
    h_IcIc = np.sum(-1./(Ic+Is+dt*Is**2.)**2.)
    h_IsIs = np.sum(-2.*dt**3.*Ic/(1.+dt*Is)**3. + 3.*dt**2./(1.+dt*Is)**2. + (4.*dt*Ic -1.)/(Ic+Is+dt*Is**2.)**2. - 2.*dt/(Ic+Is+dt*Is**2.))
    h_IcIs = np.sum(dt**2./(1.+dt*Is)**2. - (1.+2.*dt*Is)/(Ic+Is+dt*Is**2.)**2.)
    return np.asarray([[h_IcIc, h_IcIs],[h_IcIs, h_IsIs]])

def MRlogL_opgCov(dt, Ic, Is):
    """
    Calculates the Outer Product Gradient estimate of the asymptotic covariance matrix 
    evaluated at the Maximum Likelihood Estimates for Ic, Is
    
    It's the invert of the outer product of the gradients
    
    INPUTS:
        dt: list of inter-arrival times [seconds]
        Ic: The maximum likelihood estimate of Ic [1/second]
        Is: 
    OUTPUTS:
        covariance matrix for mle Ic, Is from opg method
        [[cov(Ic,Ic), cov(Ic,Is)], [cov(Is,Ic), cov(Is,Is)]]
    """
    grad_Ic = -1./(1./dt+Is) + 1./(Ic+Is+dt*Is**2.)
    grad_Is = dt**2.*Ic/(1.+dt*Is)**2. - 3.*dt/(1.+dt*Is) + (1.+2.*dt*Is)/(Ic+Is+dt*Is**2.)
    
    #n=1.0*len(dt)
    grad_Ic2 = np.sum(grad_Ic**2.)
    grad_Is2 = np.sum(grad_Is**2.)
    grad_IcIs = np.sum(grad_Ic*grad_Is)
    
    return np.asarray([[grad_Is2, -1.*grad_IcIs],[-1.*grad_IcIs, grad_Ic2]])/(grad_Ic2*grad_Is2 - grad_IcIs**2.)

def MRlogL_hessianCov(dt, Ic, Is):
    """
    Calculates the Hessian estimate of the asymptotic covariance matrix 
    evaluated at the Maximum Likelihood Estimates for Ic, Is
    
    It's the invert of the observed Information matrix
    
    INPUTS:
        dt: list of inter-arrival times [seconds]
        Ic: The maximum likelihood estimate of Ic [1/second]
        Is: 
    OUTPUTS:
        covariance matrix for mle Ic, Is from Hessian method
        [[cov(Ic,Ic), cov(Ic,Is)], [cov(Is,Ic), cov(Is,Is)]]
    """
    h=MRlogL_Hessian(dt,Ic,Is)
    #n=1.0*len(dt)
    
    return -1./(h[0][0]*h[1][1] - h[0][1]**2.)*np.asarray([[h[1][1], -1.*h[0][1]],[-1.*h[1][0],h[0][0]]])

def MRlogL_sandwichCov(dt, Ic, Is):
    """
    Estimates the asymptotic covariance matrix with the sandwich method
    evaluated at the Maximum Likelihood Estimates for Ic, Is
    
    It's Cov_hessian * Cov_OPG^-1 * Cov_hessian
    
    INPUTS:
        dt: list of inter-arrival times [seconds]
        Ic: The maximum likelihood estimate of Ic [1/second]
        Is: 
    OUTPUTS:
        covariance matrix for mle Ic, Is from sandwich method
        [[cov(Ic,Ic), cov(Ic,Is)], [cov(Is,Ic), cov(Is,Is)]]
    """
    h_cov = MRlogL_hessianCov(dt, Ic, Is)
    
    grad_Ic = -1./(1./dt+Is) + 1./(Ic+Is+dt*Is**2.)
    grad_Is = dt**2.*Ic/(1.+dt*Is)**2. - 3.*dt/(1.+dt*Is) + (1.+2.*dt*Is)/(Ic+Is+dt*Is**2.)
    
    #n=1.0*len(dt)
    grad_Ic2 = np.sum(grad_Ic**2.)
    grad_Is2 = np.sum(grad_Is**2.)
    grad_IcIs = np.sum(grad_Ic*grad_Is)
    opg_cov_inv = np.asarray([[grad_Ic2, grad_IcIs], [grad_IcIs, grad_Is2]])
    
    return np.matmul(np.matmul(h_cov, opg_cov_inv),h_cov)

def MRlogL(dt, Ic, Is):
    """
    Given an array of photon interarrival times, calculate the Log likelihood that
    those photons follow a modified Rician Intensity probability distribution with Ic, Is. 
    
    INPUTS:
        dt: list of inter-arrival times [seconds]
        Ic: Coherent portion of MR [1/second]
        Is: Speckle portion of MR [1/second]
    OUTPUTS:
        [float] the Log likelihood. 
    """
    lnL = -1.*dt*Ic/(1.+dt*Is) + np.log(Ic + Is + dt*Is**2.) - 3.*np.log(1.+dt*Is)
    return np.sum(lnL)

def maxMRlogL(ts, Ic_guess=1., Is_guess=1., method='Powell'):
    """
    Get maximum likelihood estimate of Ic, Is given dt
    
    It scipy.optimize.minimize to minimizes the negative log likelihood
    
    INPUTS:
        ts - list of photon arrival times [us]
        Ic_guess - guess for Ic [photons/second]
        Is_guess - 
        method - the optimization method (see scipy.optimize.minimize)
    OUTPUTS:
        res - OptimizeResult Object
              see https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html
        
    """
    dt = ts[1:] - ts[:-1]
    dt=dt[np.where(dt<1.e6)]/10.**6
    nLogL = lambda p: -1.*MRlogL(dt, p[0], p[1])
    nScore=lambda params: -1.*MRlogL_Jacobian(dt, params[0], params[1])
    nHess=lambda params: -1.*MRlogL_Hessian(dt, params[0], params[1])
    res = minimize(nLogL, [Ic_guess, Is_guess], method=method,jac=nScore, hess=nHess)
    return res

def getPixelPhotonList(filename, xCoord, yCoord,**kwargs):
    """
    Gets the list of photon arrival times from a H5 file
    
    INPUTS:
        filename - Name of H5 data file
        xCoord - 
        yCoord - 
        **kwargs - keywords for Obsfile getpixelphotonlist()
    OUTPUTS:
        ts - timestamps in us of photon arrival times
    """
    obs = ObsFile(filename)
    photonList = obs.getPixelPhotonList(xCoord, yCoord,**kwargs)
    times=photonList['Time']
    print("#photons: "+str(len(times)))
    del obs #make sure to close files nicely
    return times


def plotLogLMap(ts, Ic_list, Is_list):
    """
    plots a map of the MR log likelihood function over the range of Ic, Is
    
    INPUTS:
        ts - list of photon arrival times [us]
        Ic_list - list of Ic values [photons/second]
        Is_list - list
    OUTPUTS:
        2d array of logL values at the given Ic, Is coorinates
    """
    dt = ts[1:] - ts[:-1]
    dt=dt[np.where(dt<1.e6)]/10.**6     #remove negative values and convert into seconds
    print('mean dt[s]: '+str(np.mean(dt)))
    im = np.zeros((len(Ic_list),len(Is_list)))
    for i, Ic in enumerate(Ic_list):
        for j, Is in enumerate(Is_list):
            lnL = MRlogL(dt, Ic, Is)
            im[i,j] = lnL


    Ic_ind, Is_ind=np.unravel_index(im.argmax(), im.shape)
    print('Max at ('+str(Ic_ind)+', '+str(Is_ind)+')')
    print("Ic="+str(Ic_list[Ic_ind])+", Is="+str(Is_list[Is_ind]))
    print(im[Ic_ind, Is_ind])
    
    l_90 = np.percentile(im, 90)
    l_max=np.amax(im)
    l_min=np.amin(im)
    levels=np.linspace(l_90,l_max,int(len(im.flatten())*.1))
    
    plt.figure()
    plt.contourf(Ic_list, Is_list,im.T,levels=levels,extend='min')
    plt.plot(Ic_list[Ic_ind],Is_list[Is_ind],"xr")
    plt.xlabel('Ic [/s]')
    plt.ylabel('Is [/s]')
    plt.title('Map of log likelihood')



if __name__ == "__main__":
    #fn = '/home/abwalter/peg32/1507175503.h5'
    
    '''
    Ic=[]
    Ic_err=[]
    Is=[]
    Is_err=[]
    
    x=range(100)
    Ic_val, Is_val, Ttot, tau = [1000., 300., 5., .1]
    for i in x:
        ts = genphotonlist(Ic_val, Is_val, Ttot, tau)
        dt = ts[1:] - ts[:-1]
        dt=dt[np.where(dt<1.e6)]/10.**6
        
        m = MR_SpeckleModel(dt)
        res=m.fit([1., 1.])
        Ic.append(res.params[0])
        Is.append(res.params[1])
        Ic_err.append(res.bse[0])
        Is_err.append(res.bse[1])
    
    
    
    plt.errorbar(x, Ic, yerr=Ic_err)
    plt.errorbar(x, Is, yerr=Is_err)
    plt.show()
    '''

    
    print("Generating photon list...",end="", flush=True)
    Ic, Is, Ttot, tau = [300., 30., 300., .1]
    ts = genphotonlist(Ic, Is, Ttot, tau)
    print("[Done]\n")

    print("=====================================")
    print("Optimizing with Scipy...")
    res=maxMRlogL(ts)
    print(res.message)
    print("Number of Iterations: "+str(res.nit))
    print("Max LogLikelihood: "+str(-res.fun))         # We minimized negative log likelihood
    print("[Ic, Is]: "+str(res.x))
    print("Estimating Cov Matrix...")
    dt = ts[1:] - ts[:-1]
    dt=dt[np.where(dt<1.e6)]/10.**6
    Ic_mle, Is_mle = res.x
    print(np.sqrt(MRlogL_opgCov(dt, Ic_mle, Is_mle)))
    print(np.sqrt(MRlogL_hessianCov(dt, Ic_mle, Is_mle)))
    print(np.sqrt(MRlogL_sandwichCov(dt, Ic_mle, Is_mle)))
    print("=====================================\n")
    

    print("Optimizing with StatModels...")
    m = MR_SpeckleModel(ts)
    res=m.fit()
    print(res.summary())
    #print(res.params)
    #print(res.bse)
    
    print("\n#photons: "+str(len(ts)))
    print('photons/s: '+str(len(ts)/Ttot))
    print('MLE (Ic+Is): '+str(np.sum(res.params)))

    #print("Mapping...")
    #Ic_list=np.arange(80.,140.,.5)
    #Is_list=np.arange(4.,20.,0.5)
    #plotLogLMap(ts, Ic_list, Is_list)
    #plt.show()



