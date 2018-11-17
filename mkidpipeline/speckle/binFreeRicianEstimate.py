"""
Author: Alex Walter
Date: Feb 22, 2018
Last Updated: Sept 19, 2018

This code is for analyzing the photon arrival time statistics in a bin-free way
to find a maximum likelihood estimate of Ic, Is in the presence of an incoherent background source Ir.

For example usage, see if __name__ == "__main__": 
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize
from statsmodels.base.model import GenericLikelihoodModel

from mkidpipeline.hdf.darkObsFile import ObsFile
from mkidpipeline.speckle.genphotonlist_IcIsIr import genphotonlist


class MR_SpeckleModel(GenericLikelihoodModel):
    def __init__(self, ts, deadtime=10.):
        """
        Class for fitting maximum likelihood
        
        INPUTS:
            ts - Photon arrival times [us]
            deadtime - MKID deadtime after photon event [us]
        """
        dt = ts[1:] - ts[:-1]
        dt=dt[np.where(dt<1.e6)]/10.**6
        endog=np.asarray(dt, dtype=[('dt','float64')])
        exog=np.ones(len(endog),dtype={'names':['Ic','Is','Ir'],'formats':['float64','float64','float64']})
        score=lambda params: MRlogL_Jacobian(*params, dt=dt, deadtime=deadtime)
        hess=lambda params: MRlogL_Hessian(*params, dt=dt, deadtime=deadtime)
        loglike=lambda params: MRlogL(*params, dt=dt, deadtime=deadtime)
        #hess=lambda params: MRlogL_Hessian(dt, params[0], params[1])
        #loglike=lambda params: MRlogL(dt, *params, deadtime=deadtime, inttime=inttime)
        super(MR_SpeckleModel, self).__init__(endog,exog=exog, loglike=loglike, score=score, hessian=hess)
        #super(MR_SpeckleModel, self).__init__(endog,exog=exog, loglike=loglike)
    
    def fit(self, start_params=[1.,1.,1.], method='nm', **kwargs):
        """
        Find maximum log likelihood
        
        INPUTS:
            start_params - Initial guess for [Ic, Is, Ir] floats
            method - Optimization method
            **kwargs - keywords for LikelihoodModel.fit(). 
                       See www.statsmodels.org/stable/dev/generated/statsmodels.base.model.LikelihoodModel.fit.html
        OUTPUTS:
            GenericLikelihoodModelResults Object
            See www.statsmodels.org/stable/dev/generated/statsmodels.base.model.GenericLikelihoodModelResults.html
        """
        return super(MR_SpeckleModel, self).fit(start_params,method=method,**kwargs)  

def MRlogL_Jacobian(Ic, Is, Ir, dt, deadtime=0):
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

    d_Ir = 2*np.sum((Ic*u2 + Is*u + Ir)*denom_inv)
    d_Ir -= np.sum(dt)
    d_Ir += N*deadtime
    d_Ir -= N/(Ic*umax**2 + Is*umax + Ir)

    num = (4*Ic**2/Is)*u4*u + (12*Ic - 4*Ic**2/Is)*u4
    num += (4*Is - 8*Ic + 4*Ir*Ic/Is)*u3 + (2*Ir - 4*Ir*Ic/Is)*u2

    d_Is = np.sum(num*denom_inv)
    d_Is += Ic/Is**2*(np.sum(u2) - 2*sum_u + N)
    d_Is += (sum_u - N)/Is
    d_Is -= N*Ic*((umax - 1)/Is)**2
    d_Is -= N*(umax - 1)/Is
    d_Is -= N*umax**2*(2*Ic*(umax - 1)/Is + 1)/(Ic*umax**2 + Is*umax + Ir)

    return np.asarray([d_Ic, d_Is, d_Ir])

def MRlogL_Hessian(Ic, Is, Ir, dt, deadtime=0):
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

    num_Ir = Ic*u2 + Is*u + Ir
    d_IrIr = -4*np.sum(num_Ir*num_Ir*denom2_inv)
    d_IrIr += 2*np.sum(denom_inv)
    d_IrIr += N/(Ic*umax**2 + Is*umax + Ir)**2

    d_IcIr = -4*np.sum(num_Ir*num_Ic*denom2_inv)
    d_IcIr += 2*np.sum(u2*denom_inv)
    d_IcIr += N*(umax/(Ic*umax**2 + Is*umax + Ir))**2

    num_Is = (4*Ic**2/Is)*u5 + (12*Ic - 4*Ic**2/Is)*u4
    num_Is += (4*Is - 8*Ic + 4*Ir*Ic/Is)*u3 + (2*Ir - 4*Ir*Ic/Is)*u2

    argsum = (u_m_1*((2.*Ic/Is)*u2 + 3.*u + Ir/Is) + u)*denom_inv
    argsum -= ((Ic/2.)*u2 + Is*u + Ir/2.)*num_Is*denom2_inv
    d_IcIs = 4*np.sum(u2*argsum)
    d_IcIs  += np.sum(u_m_1_sq)/Is**2

    d_IcIs -= N*((umax - 1)/Is)**2
    d_IcIs += N*umax**4*(2*Ic*(umax - 1)/Is + 1)/(Ic*umax**2 + Is*umax + Ir)**2
    d_IcIs -= 2*N*umax**2*(umax - 1)/Is/(Ic*umax**2 + Is*umax + Ir)

    d_IrIs = np.sum(((4*Ic/Is)*u3 + (2 - 4*Ic/Is)*u2)*denom_inv)
    d_IrIs -= 2*np.sum((Ic*u2 + Is*u + Ir)*num_Is*denom2_inv)
    d_IrIs += N*umax**2*(2*Ic*(umax - 1)/Is + 1)/(Ic*umax**2 + Is*umax + Ir)**2

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

    return np.asarray([[d_IcIc, d_IcIs, d_IcIr],
                       [d_IcIs, d_IsIs, d_IrIs],
                       [d_IcIr, d_IrIs, d_IrIr]])


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


def MRlogL(Ic, Is, Ir, dt, deadtime=0.):
    """
    Given an array of photon interarrival times, calculate the Log likelihood that
    those photons follow a modified Rician Intensity probability distribution with Ic, Is. 
    
    INPUTS:
        dt: list of inter-arrival times [seconds]
        Ic: Coherent portion of MR [1/second]
        Is: Speckle portion of MR [1/second]
        Ir: Background companion [1/second]
        deadtime: MKID deadtime [microseconds]
    OUTPUTS:
        [float] the Log likelihood. 
    """
    # Intensity should be strictly positive, and each Ic, Is, Ir should be nonnegative.
    if Ic < 0 or Is < 0 or Ir < 0 or Ic + Is + Ir == 0:
        return -1e100 
    deadtime*=10.**-6.  #convert to seconds
    Ir=0.
    
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


def plotLogLMap(ts, Ic_list, Is_list, deadtime=10, inttime=0):
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
            lnL = MRlogL(dt, Ic, Is,deadtime=deadtime, inttime=inttime)
            im[i,j] = lnL


    Ic_ind, Is_ind=np.unravel_index(im.argmax(), im.shape)
    print('Max at ('+str(Ic_ind)+', '+str(Is_ind)+')')
    print("Ic="+str(Ic_list[Ic_ind])+", Is="+str(Is_list[Is_ind]))
    print(im[Ic_ind, Is_ind])
    
    l_90 = np.percentile(im, 68)
    l_max=np.amax(im)
    l_min=np.amin(im)
    levels=np.linspace(l_90,l_max,int(len(im.flatten())*.1))
    
    plt.figure()
    plt.contourf(Ic_list, Is_list,im.T,levels=levels,extend='min')

    plt.plot(Ic_list[Ic_ind],Is_list[Is_ind],"xr")
    x_lim = plt.gca().get_xlim()
    y_lim = plt.gca().get_ylim()
    plt.plot(np.asarray(x_lim), -1.*np.asarray(x_lim)+Ic_list[Ic_ind]+Is_list[Is_ind],'r--')
    
    rad = np.sqrt(Ic_list[Ic_ind]+Is_list[Is_ind])
    circ=plt.Circle((Ic_list[Ic_ind],Is_list[Is_ind]), rad, color='r', fill=False)
    plt.gca().add_artist(circ)
    
    plt.xlabel('Ic [/s]')
    plt.ylabel('Is [/s]')
    plt.title('Map of log likelihood')



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


