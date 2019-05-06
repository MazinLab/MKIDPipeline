"""
Author: Alex Walter
Date: Feb 22, 2018
Last Updated: Sept 19, 2018

This code is for analyzing the photon arrival time statistics in a bin-free way
to find a maximum likelihood estimate of Ic, Is in the presence of an incoherent background source Ir.

For example usage, see if __name__ == "__main__": 
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
from functools import partial
from scipy import integrate
from scipy.optimize import minimize
from statsmodels.base.model import GenericLikelihoodModel

from mkidpipeline.hdf.photontable import ObsFile
from mkidpipeline.speckle.genphotonlist_IcIsIr import genphotonlist




def plotROC(Ic, Is, Ir, Ttot, tau=0.1, deadtime=10):
    print("[Ic, Is, Ir, Ttot, tau, deadTime]: "+str([Ic, Is, Ir, Ttot, tau, deadtime]))
    ts_total, ts_star, pInds = genphotonlist(Ic, Is, Ir, Ttot, tau, deadtime, return_IDs=True)
    dt_total = (ts_total[1:] - ts_total[:-1])*10.**-6.
    dt_star = (ts_star[1:] - ts_star[:-1])*10.**-6.
    deadtime*=10.**-6.

    res = optimize_IcIsIr(dt_total, guessParams=[Ic,Is,Ir], deadtime=deadtime)
    print(res.summary())
    a=MRlogL(res.params, dt_total, deadtime)

    I1_list = np.arange(-3.*res.bse[0], 3.*res.bse[0], res.bse[0]/10.) + res.params[0]
    I2_list = np.arange(-3.*res.bse[1], 3.*res.bse[1], res.bse[1]/10.) + res.params[1]
    Ir_list = np.arange(-res.params[2], 5.*res.bse[2], res.bse[2]/10.) + res.params[2]
    I1_list=np.asarray(I1_list)[np.asarray(I1_list)>=0]
    I2_list=np.asarray(I2_list)[np.asarray(I2_list)>=0]
    Ir_list=np.asarray(Ir_list)[np.asarray(Ir_list)>=0]

    print("Gathering lnL data for all photons...")
    lnLMap=np.zeros([len(I1_list),len(I2_list),len(Ir_list)])
    for i, I_i in enumerate(I1_list):
        for j, I_j in enumerate(I2_list):
            for k, I_k in enumerate(Ir_list):
                lnLMap[i,j,k]=MRlogL([I_i,I_j,I_k], dt_total, deadtime)
    lnLMap-=np.amax(lnLMap)
    lnLMap=np.exp(lnLMap)
    cumPDF_total = integrate.trapz(lnLMap,I1_list,axis=0)
    cumPDF_total = integrate.trapz(cumPDF_total,I2_list,axis=0)
    probDens_total=np.copy(cumPDF_total)
    cumPDF_total = integrate.cumtrapz(cumPDF_total,Ir_list)
    cumPDF_total/=cumPDF_total[-1]

    print("Gathering lnL data for only star photons...")
    lnLMap_star=np.zeros([len(I1_list),len(I2_list),len(Ir_list)])
    for i, I_i in enumerate(I1_list):
        for j, I_j in enumerate(I2_list):
            for k, I_k in enumerate(Ir_list):
                lnLMap_star[i,j,k]=MRlogL([I_i,I_j,I_k], dt_star, deadtime)
    lnLMap_star-=np.amax(lnLMap_star)
    lnLMap_star=np.exp(lnLMap_star)
    cumPDF_star = integrate.trapz(lnLMap_star,I1_list,axis=0)
    cumPDF_star = integrate.trapz(cumPDF_star,I2_list,axis=0)
    probDens_star=np.copy(cumPDF_star)
    cumPDF_star = integrate.cumtrapz(cumPDF_star,Ir_list)
    cumPDF_star/=cumPDF_star[-1]


    plt.figure()
    plt.plot(Ir_list, probDens_total, label='total')
    plt.plot(Ir_list, probDens_star, label='star only')
    plt.title("prob density")
    plt.ylabel("Prob")
    plt.xlabel("Ir")
    plt.legend()

    plt.figure()
    plt.plot(Ir_list[1:], cumPDF_total, label='total')
    plt.plot(Ir_list[1:], cumPDF_star, label='star only')
    plt.title("Cumulative prob dist")
    plt.ylabel("Prob")
    plt.xlabel("Ir")
    plt.legend()

    plt.figure()
    plt.plot(1.-cumPDF_star, 1.-cumPDF_total)
    plt.plot([0,1],[0,1],'k--')
    plt.title("ROC curve: Ic, Is, Ir, Ttot, tau, = "+str([Ic, Is, Ir, Ttot, tau]))
    plt.ylabel("True Pos Rate")
    plt.xlabel("False Pos Rate")
    plt.legend()

    plt.show()


def optimize_IcIsIr2(dt, guessParams=[-1,-1,-1], deadtime=1.e-5, method='Newton-CG', prior=[np.nan]*3, prior_sig=[np.nan]*3, **kwargs):
    """
    Uses scipy.optimize.minimize
    """
    if prior is None: prior=[np.nan]*3
    prior=np.append(prior, [np.nan]*(3-len(prior)))
    if prior_sig is None: prior_sig=[np.nan]*3
    prior_sig=np.append(prior_sig, [np.nan]*(3-len(prior_sig)))

    #Provide a reasonable guess everywhere that guessParams<0
    #If a prior is given then make that the guess
    #equally distributes the average flux amongst params<0
    #ie. guess=[-1, 30, -1] and I_avg=330 then guess-->[150,30,150]. 
    guessParams=np.asarray(guessParams)
    assert len(guessParams)==3, "Must provide a guess for I1, I2, Ir. Choose -1 for automatic guess."
    
    if np.any(prior==None): prior[prior==None]=np.nan
    guessParams[np.isfinite(prior)]=prior[np.isfinite(prior)]
    if np.any(guessParams<0):
        I_avg=(len(dt))/np.sum(dt)
        I_guess = (I_avg-np.sum(guessParams[guessParams>=0])) /np.sum(guessParams<0)
        guessParams[guessParams<0]=max(I_guess,0)    

    loglike = lambda p: _posterior(p, dt=dt, deadtime=deadtime, prior=prior, prior_sig=prior_sig) 
    score = lambda p: _posterior_jacobian(p, dt=dt, deadtime=deadtime, prior=prior, prior_sig=prior_sig)
    hess = lambda p: _posterior_hessian(p, dt=dt, deadtime=deadtime, prior=prior, prior_sig=prior_sig)

    res = minimize(loglike, guessParams, method='trust-constr',bounds=[[0,np.inf]]*3,jac=score, hess=hess, **kwargs)
    return res

def optimize_IcIsIr(dt, guessParams=[-1,-1,-1], deadtime=1.e-5, method='ncg', prior=[np.nan]*3, prior_sig=[np.nan]*3, **kwargs):
    """
    This function optimizes the loglikelihood for the bin-free SSD analysis.
    It returns the model fit for the most likely I1, I2, Ir.


    INPUTS:
        dt - Float, [seconds], list of photon inter-arrival times
        params - Floats, [1/seconds], list of initial guess for I1, I2, Ir
                                    if the guess is <0 then equally distribute the total flux amongst all params with a guess <0
        deadtime - float, [seconds], MKID deadtime after photon event
        method - optimization method. 'nm'=Nelder-Mead, 'ncg'=Newton-conjugate gradient, etc...
        paramsFixed - booleans, fix param_i during optimization to the initial guess

        **kwargs - additional kwargs to GenericLikelihoodModel.fit()
                   www.statsmodels.org/stable/dev/generated/statsmodels.base.model.LikelihoodModel.fit.html

    OUTPUTS:
        GenericLikelihoodModelResults Object
        See www.statsmodels.org/stable/dev/generated/statsmodels.base.model.GenericLikelihoodModelResults.html
    """

    if prior is None: prior=[np.nan]*3
    prior=np.append(prior, [np.nan]*(3-len(prior)))
    if prior_sig is None: prior_sig=[np.nan]*3
    prior_sig=np.append(prior_sig, [np.nan]*(3-len(prior_sig)))

    #Provide a reasonable guess everywhere that guessParams<0
    #If a prior is given then make that the guess
    #equally distributes the average flux amongst params<0
    #ie. guess=[-1, 30, -1] and I_avg=330 then guess-->[150,30,150]. 
    guessParams=np.asarray(guessParams)
    assert len(guessParams)==3, "Must provide a guess for I1, I2, Ir. Choose -1 for automatic guess."
    
    if np.any(prior==None): prior[prior==None]=np.nan
    guessParams[np.isfinite(prior)]=prior[np.isfinite(prior)]
    if np.any(guessParams<0):
        I_avg=(len(dt))/np.sum(dt)
        I_guess = (I_avg-np.sum(guessParams[guessParams>=0])) /np.sum(guessParams<0)
        guessParams[guessParams<0]=max(I_guess,0)

    #Define some functions
    loglike = partial(posterior, dt=dt, deadtime=deadtime, prior=prior, prior_sig=prior_sig)
    score = partial(posterior_jacobian, dt=dt, deadtime=deadtime, prior=prior, prior_sig=prior_sig)
    hess = partial(posterior_hessian, dt=dt, deadtime=deadtime, prior=prior, prior_sig=prior_sig)

    #Setup model
    endog=np.asarray(dt, dtype=[('dt','float64')])
    names = np.asarray(['Ic','Is','Ir'])
    exog=np.ones(len(endog),dtype={'names':names,'formats':['float64']*len(names)})
    model = GenericLikelihoodModel(endog,exog=exog, loglike=loglike, score=score, hessian=hess)
    try: kwargs['disp']
    except KeyError: kwargs['disp']=False   #change default disp kwarg to false

    #fit model
    return model.fit(guessParams,method=method,**kwargs)
    


def MRlogL(params, dt, deadtime=1.e-5):
    """
    Given an array of photon interarrival times, calculate the Log likelihood that
    those photons follow a modified Rician Intensity probability distribution with Ic, Is.

    INPUTS:
        params: 2 or 3 element list of Ic, Is, Ir
            Ic - Coherent portion of MR [1/second]
            Is - Speckle portion of MR [1/second]
            (optional) Ir - Incoherent background [1/second]
        dt: list of inter-arrival times [seconds]
        deadtime: MKID deadtime [seconds]
    OUTPUTS:
        [float] the Log likelihood.
    """
    Ic=params[0]
    Is=params[1]
    try: Ir=params[2]
    except IndexError: Ir=0

    # Stellar Intensity should be strictly positive, and each Ic, Is, Ir should be nonnegative.
    if Ic < 0 or Is <= 0 or Ir < 0:
        return -1e100

    nslice = 20
    u = 1./(1 + dt*Is)
    u2 = u*u
    u3 = u2*u
    u4 = u2*u2
    u5 = u4*u
    N = len(u)

    umax = 1./(1 + deadtime*Is)
    arg_log = (Ic**2)*u5 + (4*Ic*Is)*u4 + (2*Is**2 + 2*Ir*Ic)*u3
    if Ir > 0:
        arg_log += (2*Ir*Is)*u2 + (Ir**2)*u

    ###################################################################
    # Reshape the array and multiply along the short axis to reduce
    # the number of calls to np.log.
    ###################################################################

    _n = (len(u)//nslice + 1)*nslice
    _arg_log = np.ones(_n)
    _arg_log[:len(u)] = arg_log
    _arg_log = np.prod(np.reshape(_arg_log, (nslice, -1)), axis=0)
    lnL = np.sum(np.log(_arg_log))

    lnL += -Ir*np.sum(dt) + Ic/Is*np.sum(u) - N*Ic/Is
    lnL -= N*(umax - 1)*(Ir + Ic*umax)/(Is*umax)
    lnL -= N*np.log(Ic*umax**3 + Is*umax**2 + Ir*umax)

    if np.isfinite(lnL):
        return lnL
    else:
        return -1e100


def _MRlogL(params, dt, deadtime=1.e-5):
    return -MRlogL(params, dt, deadtime=deadtime)


def posterior(params, dt, deadtime=1.e-5, prior=None, prior_sig=None):
    """
    Given an array of photon interarrival times, calculate the posterior using the
    Log likelihood that those photons follow a modified Rician Intensity probability 
    distribution with [Ic, Is, Ip]=params
    
    If given, assume a guassian prior on the parameters with the value and 1 sigma errors given.
    The log prior probability is .5*((I - I_prior)/sigma_I_prior)**2

    INPUTS:
        params: 2 or 3 element list of Ic, Is, Ir
            Ic - Coherent portion of MR [1/second]
            Is - Speckle portion of MR [1/second]
            (optional) Ir - Incoherent background [1/second]
        dt: list of inter-arrival times [seconds]
        deadtime: MKID deadtime [seconds]
        prior: 1, 2, or 3 element list of Ic, Is, Ir priors
               If None, or [None, None, None] then ignore that prior
        prior_sig: 1 sigma errors on priors
               same shape as prior

    OUTPUTS:
        [float] the posterior probability.
    """

    posterior = MRlogL(params, dt, deadtime)
    if posterior<=-1e100 or prior is None or np.all(prior==None) or np.all(~np.isfinite(prior)):
        return posterior

    prior=np.atleast_1d(prior)
    prior_sig = np.atleast_1d(prior_sig)
    assert len(prior) == len(prior_sig)
    for i in range(len(prior)):
        try:
            tmp=-0.5*((params[i] - prior[i])/prior_sig[i])**2.
            if np.isfinite(tmp): posterior+=tmp
        except: pass

    return posterior



def _posterior(params, dt, deadtime=1.e-5, prior=None, prior_sig=None):
    return -posterior(params, dt, deadtime=deadtime, prior=prior, prior_sig=prior_sig)


def MRlogL_Jacobian(params, dt, deadtime=1.e-5):
    """
    Calculates the Jacobian of the MR log likelihood function.
    Also called a Gradient or Fisher's Score Function. 
    It's just the vector of first partial derivatives.
    
    INPUTS:
        params: 2 or 3 element list of Ic, Is, Ir
            Ic - Coherent portion of MR [1/second]
            Is - Speckle portion of MR [1/second]
            (optional) Ir - Incoherent background [1/second]
        dt: list of inter-arrival times [seconds]
        deadtime: MKID deadtime [seconds]
    OUTPUTS:
        jacobian vector [dlnL/dIc, dlnL/dIs, dlnL/dIr] at Ic, Is, Ir
        if Ir not given in params then force Ir=0 and dlnL/dIr not calculated
    """
    
    Ic=params[0]
    Is=params[1]
    try: Ir=params[2]
    except IndexError: Ir=0.

    if np.sum(np.asarray(params)<0)>0: 
        print("Negatives in Jac")
        #We'll find the slope nearby and then later force it positive
    if Ir<0: Ir=0.
    if Ic<0: Ic=1.e-100
    if Is<=0: Is=1.e-100

    ###################################################################
    # Pre-compute combinations of variables
    ###################################################################

    u = 1./(1 + dt*Is)
    umax = 1./(1 + deadtime*Is)
    u2 = u*u
    u3 = u2*u
    u4 = u2*u2
    sum_u = np.sum(u)
    N = len(u)
    denom_inv = 1/(Ic**2*u4 + 4*Ic*Is*u3 + (2*Is**2 + 2*Ir*Ic)*u2 + 2*Ir*Is*u + Ir**2)

    d_Ic = 2*np.sum((Ic*u4 + 2*Is*u3 + Ir*u2)*denom_inv)
    d_Ic += (sum_u - N)/Is
    d_Ic -= N*umax**2/(Ic*umax**2 + Is*umax + Ir)
    d_Ic += N*(1 - umax)/Is

    num = (4*Ic**2/Is)*u4*u + (12*Ic - 4*Ic**2/Is)*u4
    num += (4*Is - 8*Ic + 4*Ir*Ic/Is)*u3 + (2*Ir - 4*Ir*Ic/Is)*u2
    d_Is = np.sum(num*denom_inv)
    d_Is += Ic/Is**2*(np.sum(u2) - 2*sum_u + N)
    d_Is += (sum_u - N)/Is
    d_Is -= N*Ic*((umax - 1)/Is)**2
    d_Is -= N*(umax - 1)/Is
    d_Is -= N*umax**2*(2*Ic*(umax - 1)/Is + 1)/(Ic*umax**2 + Is*umax + Ir)

    if len(params)==3:
        d_Ir = 2*np.sum((Ic*u2 + Is*u + Ir)*denom_inv)
        d_Ir -= np.sum(dt)
        d_Ir += N*deadtime
        d_Ir -= N/(Ic*umax**2 + Is*umax + Ir)

        if params[0]<0: d_Ic = np.abs(d_Ic) #force positive to guide back towards Ic>0
        if params[1]<0: d_Is = np.abs(d_Is)
        if params[2]<0: d_Ir = np.abs(d_Ir)

        return np.asarray([d_Ic, d_Is, d_Ir])

    if params[0]<0: d_Ic = np.abs(d_Ic) #force positive to guide back towards Ic>0
    if params[1]<0: d_Is = np.abs(d_Is)
    return np.asarray([d_Ic, d_Is])


def _MRlogL_Jacobian(params, dt, deadtime=1.e-5):
    return -MRlogL_Jacobian(params, dt, deadtime=deadtime)


def posterior_jacobian(params, dt, deadtime=1.e-5, prior=None, prior_sig=None):
    """
    Calculates the Jacobian of the MR log likelihood function.
    Also called a Gradient or Fisher's Score Function.
    It's just the vector of first partial derivatives.

    INPUTS:
        params: 2 or 3 element list of Ic, Is, Ir
            Ic - Coherent portion of MR [1/second]
            Is - Speckle portion of MR [1/second]
            (optional) Ir - Incoherent background [1/second]
        dt: list of inter-arrival times [seconds]
        deadtime: MKID deadtime [seconds]
        prior: 1, 2, or 3 element list of Ic, Is, Ir priors. [1/second]
               If None, or [None, None, None] then ignore that prior
        prior_sig: 1 sigma errors on priors. [1/second]
               same shape as prior
    OUTPUTS:
        jacobian vector [dlnL/dIc, dlnL/dIs, dlnL/dIr] at Ic, Is, Ir
        if Ir not given in params then force Ir=0 and dlnL/dIr not calculated
    """

    jac = MRlogL_Jacobian(params, dt, deadtime=deadtime)
    if prior is None or np.all(prior==None) or np.all(~np.isfinite(prior)):
        return jac

    prior=np.atleast_1d(prior)
    prior_sig = np.atleast_1d(prior_sig)
    assert len(prior) == len(prior_sig)
    for i in range(len(prior)):
        try:
            tmp=-(params[i] - prior[i])/prior_sig[i]**2.
            if np.isfinite(tmp): jac[i]+=tmp
        except: pass


    return jac



def _posterior_jacobian(params, dt, deadtime=1.e-5, prior=None, prior_sig=None):
    return -posterior_jacobian(params, dt, deadtime=deadtime, prior=prior, prior_sig=prior_sig)


def MRlogL_Hessian(params, dt, deadtime=1.e-5):
    """
    Calculates the Hessian matrix of the MR log likelihood function.
    It's just the matrix of second partial derivitives.
    It's the negative of Fisher's Observed Information Matrix
    
    INPUTS:
        params: 2 or 3 element list of Ic, Is, Ir
            Ic - Coherent portion of MR [1/second]
            Is - Speckle portion of MR [1/second]
            (optional) Ir - Incoherent background [1/second]
        dt: list of inter-arrival times [seconds]
        deadtime: MKID deadtime [seconds]
    OUTPUTS:
        Hessian matrix [[d2lnL/dIc2, d2lnL/dIcdIs], [d2lnL/dIsdIc, d2lnL/dIsdIs]] at Ic, Is
    """
    Ic=params[0]
    Is=params[1]
    try: Ir=params[2]
    except IndexError: Ir=0.

    if np.sum(params<0)>0: 
        print("Negatives in Hess")
        #We'll find the curvature nearby
    if Ir<0: Ir=0.
    if Ic<0: Ic=1.e-100
    if Is<=0: Is=1.e-100

    ###################################################################
    # Pre-compute combinations of variables
    ###################################################################

    u = 1./(1 + dt*Is)
    umax = 1./(1 + deadtime*Is)
    u2 = u*u
    u3 = u2*u
    u4 = u2*u2
    u5 = u4*u
    u_m_1 = u - 1.
    u_m_1_sq = u_m_1*u_m_1
    N = len(u)
    
    denom_inv = 1./((Ic**2)*u4 + (4*Ic*Is)*u3 + (2*Is**2 + 2*Ir*Ic)*u2 + (2*Ir*Is)*u + Ir**2)
    denom2_inv = denom_inv*denom_inv

    num_Ic = Ic*u4 + (2*Is)*u3 + Ir*u2
    d_IcIc = -4*np.sum(num_Ic*num_Ic*denom2_inv)
    d_IcIc += 2*np.sum(u4*denom_inv)
    d_IcIc += N*(umax**2/(Ic*umax**2 + Is*umax + Ir))**2

    num_Is = (4*Ic**2/Is)*u5 + (12*Ic - 4*Ic**2/Is)*u4
    num_Is += (4*Is - 8*Ic + 4*Ir*Ic/Is)*u3 + (2*Ir - 4*Ir*Ic/Is)*u2

    argsum = (u_m_1*((2.*Ic/Is)*u2 + 3.*u + Ir/Is) + u)*denom_inv
    argsum -= ((Ic/2.)*u2 + Is*u + Ir/2.)*num_Is*denom2_inv
    d_IcIs = 4*np.sum(u2*argsum)
    d_IcIs  += np.sum(u_m_1_sq)/Is**2

    d_IcIs -= N*((umax - 1)/Is)**2
    d_IcIs += N*umax**4*(2*Ic*(umax - 1)/Is + 1)/(Ic*umax**2 + Is*umax + Ir)**2
    d_IcIs -= 2*N*umax**2*(umax - 1)/Is/(Ic*umax**2 + Is*umax + Ir)

    argsum = u_m_1_sq*((5*Ic**2/Is**2)*u2 + (6*Ic/Is)*u + 3*Ic*Ir/Is**2)
    argsum += u_m_1*((6*Ic/Is)*u2 + 3*u + Ir/Is) + u
    d_IsIs = 4*np.sum(u2*argsum*denom_inv)

    d_IsIs -= np.sum(num_Is*num_Is*denom2_inv)
    d_IsIs += 2*Ic/Is**3*np.sum(u_m_1_sq*u_m_1)
    d_IsIs += np.sum(u_m_1_sq)/Is**2
    d_IsIs -= 2*N*Ic*((umax - 1)/Is)**3
    d_IsIs -= N*((umax - 1)/Is)**2
    d_IsIs += N*(umax**2*(2*Ic*(umax - 1)/Is + 1)/(Ic*umax**2 + Is*umax + Ir))**2
    d_IsIs -= N*(6*Ic*umax**2*(umax - 1)**2/Is**2 + 2*umax**2*(umax - 1)/Is)/(Ic*umax**2 + Is*umax + Ir)



    if len(params)==3:

        num_Ir = Ic*u2 + Is*u + Ir
        d_IrIr = -4*np.sum(num_Ir*num_Ir*denom2_inv)
        d_IrIr += 2*np.sum(denom_inv)
        d_IrIr += N/(Ic*umax**2 + Is*umax + Ir)**2

        d_IcIr = -4*np.sum(num_Ir*num_Ic*denom2_inv)
        d_IcIr += 2*np.sum(u2*denom_inv)
        d_IcIr += N*(umax/(Ic*umax**2 + Is*umax + Ir))**2

        d_IrIs = np.sum(((4*Ic/Is)*u3 + (2 - 4*Ic/Is)*u2)*denom_inv)
        d_IrIs -= 2*np.sum((Ic*u2 + Is*u + Ir)*num_Is*denom2_inv)
        d_IrIs += N*umax**2*(2*Ic*(umax - 1)/Is + 1)/(Ic*umax**2 + Is*umax + Ir)**2

        return np.asarray([[d_IcIc, d_IcIs, d_IcIr],
                       [d_IcIs, d_IsIs, d_IrIs],
                       [d_IcIr, d_IrIs, d_IrIr]])

    return np.asarray([[d_IcIc, d_IcIs],[d_IcIs, d_IsIs]])


def _MRlogL_Hessian(params, dt, deadtime=1.e-5):
    return -MRlogL_Hessian(params, dt, deadtime=deadtime)


def posterior_hessian(params, dt, deadtime=1.e-5, prior=None, prior_sig=None):
    """
    Calculates the Hessian matrix of the MR log likelihood function.
    It's just the matrix of second partial derivitives.
    It's the negative of Fisher's Observed Information Matrix

    INPUTS:
        params: 2 or 3 element list of Ic, Is, Ir
            Ic - Coherent portion of MR [1/second]
            Is - Speckle portion of MR [1/second]
            (optional) Ir - Incoherent background [1/second]
        dt: list of inter-arrival times [seconds]
        deadtime: MKID deadtime [seconds]
        prior: 1, 2, or 3 element list of Ic, Is, Ir priors. [1/second]
               If None, or [None, None, None] then ignore that prior
        prior_sig: 1 sigma errors on priors. [1/second]
               same shape as prior
    OUTPUTS:
        Hessian matrix [[d2lnL/dIc2, d2lnL/dIcdIs], [d2lnL/dIsdIc, d2lnL/dIsdIs]] at Ic, Is
    """
    hess = MRlogL_Hessian(params, dt, deadtime=deadtime)
    if prior is None or np.all(prior==None) or np.all(~np.isfinite(prior)):
        return hess

    prior=np.atleast_1d(prior)
    prior_sig = np.atleast_1d(prior_sig)
    assert len(prior) == len(prior_sig)
    for i in range(len(prior)):
        try:
            tmp=-1./prior_sig[i]**2.
            if np.isfinite(tmp): hess[i,i]+=tmp
        except: pass

    return hess


def _posterior_hessian(params, dt, deadtime=1.e-5, prior=None, prior_sig=None):
    return -posterior_hessian(params, dt, deadtime=deadtime, prior=prior, prior_sig=prior_sig)


def MRlogL_opgCov(Ic, Is, Ir, dt, deadtime=0):
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
    raise NotImplementedError
    grad_Ic = -1./(1./dt+Is) + 1./(Ic+Is+dt*Is**2.)
    grad_Is = dt**2.*Ic/(1.+dt*Is)**2. - 3.*dt/(1.+dt*Is) + (1.+2.*dt*Is)/(Ic+Is+dt*Is**2.)
    
    #n=1.0*len(dt)
    grad_Ic2 = np.sum(grad_Ic**2.)
    grad_Is2 = np.sum(grad_Is**2.)
    grad_IcIs = np.sum(grad_Ic*grad_Is)
    
    return np.asarray([[grad_Is2, -1.*grad_IcIs],[-1.*grad_IcIs, grad_Ic2]])/(grad_Ic2*grad_Is2 - grad_IcIs**2.)

def MRlogL_hessianCov(Ic, Is, Ir, dt, deadtime=0):
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
    h=MRlogL_Hessian(Ic,Is,Ir, dt, deadtime)
    return np.linalg.inv(-1.*h)
    #return -1./(h[0][0]*h[1][1] - h[0][1]**2.)*np.asarray([[h[1][1], -1.*h[0][1]],[-1.*h[1][0],h[0][0]]])

def MRlogL_sandwichCov(Ic, Is, Ir, dt, deadtime=0):
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
    h_cov = MRlogL_hessianCov(Ic, Is, Ir, dt, deadtime)
    opg_cov = MRlogL_opgCov(Ic, Is, Ir, dt, deadtime)
    opg_cov_inv = np.linalg.inv(opg_cov)
    
    return np.matmul(np.matmul(h_cov, opg_cov_inv),h_cov)





def maxMRlogL(ts, Ic_guess=1., Is_guess=1., method='Powell'):
    """
    Get maximum likelihood estimate of Ic, Is given dt
    
    It uses scipy.optimize.minimize to minimizes the negative log likelihood
    
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
    #res = minimize(nLogL, [Ic_guess, Is_guess], method=method)
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




if __name__ == "__main__":

    
    print("Getting photon list: ")
    Ic, Is, Ir, Ttot, tau, deadTime = [30., 300.,0., 30., .1, 10.]
    print("[Ic, Is, Ir, Ttot, tau, deadTime]: "+str([Ic, Is, Ir, Ttot, tau, deadTime]))
    print("\t...",end="", flush=True)
    ts = genphotonlist(Ic, Is, Ir, Ttot, tau, deadTime)
    print("Done.\n")
    
    #dt = ts[1:] - ts[:-1]
    #dt=dt[np.where(dt<1.e6)]/10.**6     #remove negative values and convert into seconds
    
    
    
    #fn = '/home/abwalter/peg32/1507175503.h5'
    #print("From: ",fn)
    #print("\t...",end="", flush=True)
    #ts=getPixelPhotonList(filename=fn, xCoord=30, yCoord=81,wvlStart=100, wvlStop=900)
    #Ttot=3.
    #print("Done.\n")

    print("=====================================")
    print("Optimizing with Scipy...")
    res=maxMRlogL(ts)
    print(res.message)
    print("Number of Iterations: "+str(res.nit))
    print("Max LogLikelihood: "+str(-res.fun))         # We minimized negative log likelihood
    print("[Ic, Is]: "+str(res.x))
    #raise IOError
    print("Estimating Cov Matrix...")
    dt = ts[1:] - ts[:-1]
    dt=dt[np.where(dt<1.e6)]/10.**6
    Ic_mle, Is_mle = res.x
    print(np.sqrt(MRlogL_opgCov(dt, Ic_mle, Is_mle)))
    print(np.sqrt(MRlogL_hessianCov(dt, Ic_mle, Is_mle)))
    print(np.sqrt(MRlogL_sandwichCov(dt, Ic_mle, Is_mle)))
    print("=====================================\n")
    

    print("Optimizing with StatModels...")
    m = MR_SpeckleModel(ts, deadtime=deadTime, inttime=Ttot)
    res=m.fit()
    print(res.summary())
    #print(res.params)
    #print(res.bse)
    
    print("\n#photons: "+str(len(ts)))
    countRate = len(ts)/Ttot
    print('photons/s: '+str(countRate))
    print('photons/s (deadtime corrected): '+str(countRate/(1.-countRate*deadTime*10.**-6.)))
    print('MLE Ic+Is: '+str(np.sum(res.params)))


    I = 1./dt
    plt.hist(I, 10000, range=(0,25000))
    #plt.show()
    u_I = np.average(I,weights=dt)
    #var_I = np.var(I)
    var_I=np.average((I-u_I)**2., weights=dt)
    print(u_I)
    print(var_I)
    print(Is+Ic)
    print(Is**2.+2.*Ic*Is+Ic+Is)
    Ic_stat= np.sqrt(u_I**2 - var_I + u_I)
    Is_stat = u_I - Ic_stat
    print('Stat Ic, Is: '+str(Ic_stat)+', '+str(Is_stat))

    print("Mapping...")
    Ic_list=np.arange(0.,200.,1.)
    Is_list=np.arange(100.,400.,1.)
    plotLogLMap(ts, Ic_list, Is_list,deadtime=deadTime, inttime=Ttot)
    plt.show()


