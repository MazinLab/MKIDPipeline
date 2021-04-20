"""
Author: Seth Meeker        Date: Feb 11, 2017


Define some probability density functions for fitting speckle histograms

"""
import matplotlib.pyplot as plt
import numpy as np
from mpmath import hyp1f1
from scipy import special
from scipy.optimize.minpack import curve_fit
from scipy.special import factorial
from scipy.stats import rv_continuous


def modifiedRician(I, Ic, Is):
    """
    MR pdf(I) = 1/Is * exp(-(I+Ic)/Is) * I0(2*sqrt(I*Ic)/Is)
    mean = Ic + Is
    variance = Is^2 + 2*Ic*Is
    """
    mr = 1.0 / Is * np.exp(-1.0 * (I + Ic) / Is) * special.iv(0, 2.0 * np.sqrt(I * Ic) / Is)
    return mr


def poisson(I, mu):
    # poissonian pdf(I) = e^-mu * mu^I / I!
    pois = np.exp(-1.0 * mu) * np.power(mu, I) / factorial(I)
    return pois


def gaussian(I, mu, sig):
    # gaussian pdf(I) = e^(-(x-mu)^2/(2(sig^2)*1/(sqrt(2*pi)*sig)
    gaus = np.exp(-1.0 * np.power((I - mu), 2) / (2.0 * np.power(sig, 2))) * 1 / (sig * np.sqrt(2 * np.pi))
    return gaus


def exponential(x, lam, tau, f0):
    expon = lam * np.exp(-x / tau) + f0
    return expon


def lorentzian(x, gamma, x0):
    loren = (1. / (np.pi * gamma)) * (gamma * gamma / (np.power((x - x0), 2) + gamma * gamma))
    # loren = (1./(np.pi * gamma))*(gamma*gamma/(np.power((x),2)+gamma*gamma))
    return loren


def fitLorentzian(x, y, guessGam, guessX0):
    """
    Given a histogram of intensity values (x = I bin centers, y = N(I))
    and a guess for Gamma and x0, returns fit values for Gamma and x0.
    """
    lor_guess = [guessGam, guessX0]
    lf = lambda fx, gam, x0: lorentzian(fx, gam, x0)
    params, cov = curve_fit(lf, x, y, p0=lor_guess, maxfev=2000)
    return params[0], params[1]  # params = [gamma, x0]


def fitDoubleLorentzian(x, y, guessGam1, guessX1, guessGam2, guessX2):
    dlor_guess = [guessGam1, guessX1, guessGam2, guessX2]
    dlf = lambda fx, gam1, x1, gam2, x2: lorentzian(fx, gam1, x1) + lorentzian(fx, gam2, x2)
    params, cov = curve_fit(dlf, x, y, p0=dlor_guess, maxfev=2000)
    return params[0], params[1], params[2], params[3]  # params = [gamma1, x1, gamma2, x2]


def fitMR(x, y, guessIc, guessIs):
    """
    Given a histogram of intensity values (x = I bin centers, y = N(I))
    and a guess for Ic and Is, returns fit values for Ic and Is.
    """
    mr_guess = [guessIc, guessIs]
    mrf = lambda fx, Ic, Is: modifiedRician(fx, Ic, Is)
    params, cov = curve_fit(mrf, x, y, p0=mr_guess, maxfev=2000)
    return params[0], params[1]  # params = [fitIc, fitIs]


def fitPoisson(x, y, guessLambda):
    """
    Given a histogram of intensity values (x = I bin centers, y = N(I))
    and a guess for expectation value, returns fit values for lambda.
    """
    p_guess = [guessLambda]
    pf = lambda fx, lam: poisson(fx, lam)
    params, cov = curve_fit(pf, x, y, p0=p_guess, maxfev=2000)
    return params[0]  # params = [lambda]


def fitGaussian(x, y, guessMu, guessSigma):
    """
    Given a histogram of intensity values (x = I bin centers, y = N(I))
    and a guess for mu and sigma, returns fits for mu and sigma
    """
    g_guess = [guessMu, guessSigma]
    gf = lambda fx, mu, sigma: gaussian(fx, mu, sigma)
    params, cov = curve_fit(gf, x, y, p0=g_guess, maxfev=2000)
    return params[0], params[1]  # params = [mu, sigma]


def fitExponential(x, y, guessLam, guessTau, guessf0):
    """
    Given a histogram of intensity values (x = I bin centers, y = N(I))
    and a guess for mu and sigma, returns fits for mu and sigma
    """
    e_guess = [guessLam, guessTau, guessf0]
    ef = lambda fx, lam, tau, f0: exponential(fx, lam, tau, f0)
    params, cov = curve_fit(ef, x, y, p0=e_guess, maxfev=2000)
    return params[0], params[1], params[2]  # params = [lambda, tau, f0]


def blurredMR1(x, Ic, Is):
    w = modifiedRician(x, Ic, Is)  # these are the weights
    f = np.zeros(len(x))  # initialize empty array that we will fill with numbers later

    pmatrix = np.zeros((len(x), len(x)))
    #    plt.figure(2)
    for mu in range(len(x)):
        # make a matrix where the rows are weighted poissons
        pmatrix[mu, :] = poisson(x, mu) * w[mu]
    #        plt.plot(pmatrix[mu,:] * w[mu])
    f = np.sum(pmatrix, 0)  # the first element of f is the sum of the elements of the first column of pmatrix
    #    plt.figure(1)
    return f


def blurredMR(x, Ic, Is):
    p = np.zeros(len(x))
    for ii in x:
        p[ii] = 1 / (Is + 1) * (1 + 1 / Is) ** (-ii) * np.exp(-Ic / Is) * hyp1f1(float(x[ii]) + 1, 1,
                                                                                 Ic / (Is ** 2 + Is))
    return p


class mr_gen(rv_continuous):
    """
    Modified Rician distribution for drawing random variates
    Define distribution with mr = mr_gen(). Class already knows (Ic, Is) are shape of PDF for rvs.
    Get random variates with randomSamples = mr.rvs(Ic=Ic, Is=Is, size=N)
    """

    def _pdf(self, x, Ic, Is):
        return modifiedRician(x, Ic, Is)

    def _stats(self, Ic, Is):
        return [Ic + Is, np.power(Is, 2) + 2 * Ic * Is, np.nan, np.nan]
