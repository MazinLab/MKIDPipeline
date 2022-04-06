"""
Author: Alex Walter
Date: Feb 22, 2018
Last Updated: Sept 19, 2018

This code is for analyzing the photon arrival time statistics in a bin-free way
to find a maximum likelihood estimate of Ic, Is in the presence of an incoherent background source Ir.

For example usage, see if __name__ == "__main__":
"""
import numpy as np
from scipy.optimize import minimize
from mkidpipeline.photontable import Photontable

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
    Ic = params[0]
    Is = params[1]
    try: Ir = params[2]
    except IndexError: Ir = 0

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

def blurredMR(n, Ic, Is):
    """
    Depricated.

    Calculates the probability of getting a bin with n counts given Ic & Is.
    n, Ic, Is must have the same units.

    Does the same thing as binMR_like, but slower.

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
    for ii in range(len(n)):  # TODO: change hyp1f1 to laguerre polynomial. It's way faster.
        p[ii] = 1 / (Is + 1) * (1 + 1 / Is) ** (-n[ii]) * np.exp(-Ic / Is) * hyp1f1(float(n[ii]) + 1, 1,
                                                                                    Ic / (Is ** 2 + Is))
    return p

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
    dt = dt[np.where(dt < 1.e6)]/10.**6
    nLogL = lambda p: -1.*MRlogL([p[0], p[1]], dt)
    # nScore=lambda params: -1.*MRlogL_Jacobian(dt, params[0], params[1])
    # nHess=lambda params: -1.*MRlogL_Hessian(dt, params[0], params[1])
    # res = minimize(nLogL, [Ic_guess, Is_guess], method=method, jac=nScore, hess=nHess)
    res = minimize(nLogL, [Ic_guess, Is_guess], method=method)
    return res


def getPixelPhotonList(filename, xCoord, yCoord, **kwargs):
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
    obs = Photontable(filename)
    times = obs.query(pixel=(xCoord, yCoord), column='time', **kwargs)
    print("#photons: "+str(len(times)))
    del obs  # make sure to close files nicely
    return times

def getLightCurve(photonTimeStamps, startTime=None, stopTime=None, effExpTime=.01):
    """
    Takes a 1d array of arrival times and bins it up with the given effective exposure
    time to make a light curve.

    INPUTS:
        photonTimeStamps - 1d numpy array with units of seconds
        startTime -     ignore the photonTimeStamps before startTime. [seconds]
        stopTime -      ignore the photonTimeStamps after stopTime. [seconds]
        effExpTime -    bin size of the light curver. [seconds]
    OUTPUTS:
        lightCurveIntensityCounts - array with units of counts/bin. Float.
        lightCurveIntensity - array with units of counts/sec. Float.
        lightCurveTimes - array with times corresponding to the bin
                            centers of the light curve. Float.
    """
    if startTime is None:
        startTime = photonTimeStamps[0]
    if stopTime is None:
        stopTime = photonTimeStamps[-1]
    histBinEdges = np.arange(startTime, stopTime, effExpTime)

    hist, _ = np.histogram(photonTimeStamps, bins=histBinEdges)  # if histBinEdges has N elements, hist has N-1
    lightCurveIntensityCounts = hist  # units are photon counts
    lightCurveIntensity = 1. * hist / effExpTime  # units are counts/sec
    lightCurveTimes = histBinEdges[:-1] + 1.0 * effExpTime / 2

    return lightCurveIntensityCounts, lightCurveIntensity, lightCurveTimes
    # [lightCurveIntensityCounts] = counts
    # [lightCurveIntensity] = counts/sec


def histogramLC(lightCurve):
    """
    makes a histogram of the light curve intensities

    INPUTS:
        lightCurve - 1d array specifying number of photons in each bin
    OUTPUTS:
        intensityHist - 1d array containing the histogram. It's normalized, so the area under the curve is 1.
        bins - 1d array specifying the bins (0 photon, 1 photon, etc.)
    """
    # Nbins=30  #smallest number of bins to show

    Nbins = int(np.amax(lightCurve))

    if Nbins == 0:
        intensityHist = np.zeros(30)
        bins = np.arange(30)
        # print('LightCurve was zero for entire time-series.')
        return intensityHist, bins

    # count the number of times each count rate occurs in the timestream
    intensityHist, _ = np.histogram(lightCurve, bins=Nbins, range=[0, Nbins])

    intensityHist = intensityHist / float(len(lightCurve))
    bins = np.arange(Nbins)

    return intensityHist, bins