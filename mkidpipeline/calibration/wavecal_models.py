import numpy as np
import lmfit as lm
import scipy.optimize as opt


def find_amplitudes(x, y, model_functions, args, variance=None):
    if variance is None:
        variance = np.ones(y.shape)
    # make coefficient matrix
    a = np.vstack([model(x, *args[index]) / variance
                   for index, model in enumerate(model_functions)])
    # rescale y
    b = y / variance

    # solve for amplitudes enforcing positive values
    amplitudes, _ = opt.nnls(a, b)

    return amplitudes


def gaussian(x, center, sigma):
    """Gaussian function with unit amplitude"""
    return np.exp(-((x - center) / sigma)**2 / 2)


def exponential(x, decay):
    """Exponential function with unit amplitude"""
    return np.exp(-x / decay)


class PartialLinearModel(object):
    """Base model class for fitting a linear combination of multiple functions to data.
    The linearity of the coefficients multiplying each function is used to reduce the
    dimension of the solution space by the number of functions. This reduction leads to
    more robust weighted, nonlinear least squares optimization.

    This class must be subclassed to run with the reduced_fit_function and
    full_fit_function methods patched. The following is an example outline for a two
    component model:

    @staticmethod
    def full_fit_function(x, arg0, arg1, arg2, arg3):
        return arg0 * function1(x, arg1) + arg2 * function2(x, arg3)

    @staticmethod
    def reduced_fit_function(x, y, arg1, arg3, variance=None, return_amplitudes=False):
        amplitudes = find_amplitudes(x, y, [function1, function2], [[arg1], [arg3]],
                                     variance=variance)
        if return_amplitudes:
            amplitudes = lm.Parameters()
            amplitudes.add('arg0', amplitudes[0])
            amplitudes.add('arg2', amplitudes[1])
            return amplitudes
        return full_fit_function(x, amplitudes[0], arg1, amplitudes[1], arg3)

    Use 'signal_center' as the fit function argument for the center of the distribution
    used to fit the phase to energy calibration.

    The guess method should be overwritten to provide a reasonable guess for the fitting
    routine based on self.x, self.y and self.variance. An optional 'index' keyword
    argument is provided for specifying more than one guess.

    The has_good_solution() method should also be overwritten to provide a more detailed
    check of the solution which is model dependent.

    self.x, self.y and self.variance must be defined by the user before fitting.
    """
    def __init__(self):
        self._reduced_model = lm.Model(self.reduced_fit_function,
                                       independent_vars=['x', 'y', 'variance'])
        self._full_model = lm.Model(self.full_fit_function, independent_vars=['x'])
        self.fit_result = None
        self.x = None
        self.y = None
        self.variance = None
        self.variance = None
        self.best_fit_result = None
        self.used_last_fit = None

    def fit(self, guess):
        keep = (self.y != 0)
        x = self.x[keep]
        y = self.y[keep]
        if self.variance is None:
            variance = self.variance[keep]
        else:
            variance = self.variance
        try:
            # solve the reduced weighted least squares optimization
            if variance is None:
                fit_result = self._reduced_model.fit(y, params=guess, x=x, y=y,
                                                     scale_covar=True)
            else:
                fit_result = self._reduced_model.fit(y, params=guess, x=x, y=y,
                                                     weights=1 / variance,
                                                     scale_covar=False)
            # find the linear amplitude coefficients
            amplitudes = self._reduced_model.eval(self.fit_result.params, x=x,
                                                  y=y, variance=variance,
                                                  return_amplitudes=True)
            # create a new guess with the least squares solution
            guess = fit_result.params.copy()
            guess.add_many(*[amplitudes[key] for key in amplitudes.keys()])
        except NotImplementedError:
            # skip if reduced_fit_function is not implemented
            pass

        # solve the full system to get the full covariance matrix
        if variance is None:
            self.fit_result = ModelResult(self._full_model, guess, scale_covar=True)
            self.fit_result.fit(y, x=x)
        else:
            self.fit_result = ModelResult(self._full_model, guess, scale_covar=False)
            self.fit_result.fit(y, x=x, weights=1 / variance)

        # save the data in best_fit_result if it is better than previous fits
        self.used_last_fit = (self.best_fit_result is None or
                              self.fit_result.chisq < self.best_fit_result.chisq)
        if self.used_last_fit:
            self.best_fit_result = self.fit_result

    @property
    def signal_center(self):
        if self.best_fit_result is None:
            message = "A fit for this model has not been computed yet."
            raise RuntimeError(message)
        return self.best_fit_result.params['signal_center'].value

    @property
    def signal_center_standard_error(self):
        if not self.best_fit_result.errorbars:
            message = ("The best fit for this model isn't good enough to have computed"
                       "errors on its parameters")
            raise RuntimeError(message)
        return self.best_fit_result.params['signal_center'].stderr

    def plot(self):
        pass

    def has_good_solution(self):
        return self.best_fit_result.success

    def guess(self, index=0):
        raise NotImplementedError


class GaussianExponential(PartialLinearModel):
    """Gaussian signal plus exponential background"""
    @staticmethod
    def full_fit_function(x, signal_amplitude, signal_center, signal_sigma,
                          background_amplitude, background_decay):
        result = (signal_amplitude * gaussian(x, signal_center, signal_sigma),
                  background_amplitude * exponential(x, background_decay))
        return result

    @staticmethod
    def reduced_fit_function(x, y, signal_center, signal_sigma, background_decay,
                             variance=None, return_amplitudes=False):
        model_functions = [gaussian, exponential]
        args = [[signal_center, signal_sigma], [background_decay]]
        amplitudes = find_amplitudes(x, y, model_functions, args, variance=variance)
        if return_amplitudes:
            parameters = lm.Parameters()
            parameters.add('signal_amplitude', value=amplitudes[0])
            parameters.add('background_amplitude', value=amplitudes[1])
            return parameters

        result = self.full_fit_function(x, amplitudes[0], signal_center, signal_sigma,
                                        amplitudes[1], background_decay)
        return result

    def has_good_solution(self):
        return self.best_fit_result.success

    def guess(self, index=0):
        raise NotImplementedError


class GaussianGaussian(PartialLinearModel):
    """Gaussian signal plus gaussian background"""
    @staticmethod
    def full_fit_function(x, signal_amplitude, signal_center, signal_sigma,
                          background_amplitude, background_center, background_sigma):
        result = (signal_amplitude * gaussian(x, signal_center, signal_sigma),
                  background_amplitude * gaussian(x, background_center, background_sigma))
        return result

    @staticmethod
    def reduced_fit_function(x, y, signal_center, signal_sigma, background_center,
                             background_sigma, variance=None, return_amplitudes=False):
        model_functions = [gaussian, gaussian]
        args = [[signal_center, signal_sigma], [background_center, background_sigma]]
        amplitudes = find_amplitudes(x, y, model_functions, args, variance=variance)
        if return_amplitudes:
            parameters = lm.Parameters()
            parameters.add('signal_amplitude', value=amplitudes[0])
            parameters.add('background_amplitude', value=amplitudes[1])
            return parameters

        result = self.full_fit_function(x, amplitudes[0], signal_center, signal_sigma,
                                        amplitudes[1], background_center,
                                        background_sigma)
        return result

    def has_good_solution(self):
        return self.best_fit_result.success

    def guess(self, index=0):
        raise NotImplementedError


class XErrorsModel(object):
    """Base class for fitting a function to data with errors in the x variable by
    assuming that the x error is small enough such that the function can be approximated
    as linear near the error point.

    This class must be subclassed to run with the fit_function and dfdx methods patched.
    The following is an example outline:

    @staticmethod
    def fit_function(x, parameters):
        p = parameters.valuesdict()
        return p['c2'] * x**2 + p['c1'] * x + p['c0']

    @staticmethod
    def dfdx(x, parameters):
        p = parameters.valuesdict()
        return 2 * p['c2'] * x + p['c1']

    The guess method should be overwritten to provide a reasonable guess for the fitting
    routine based on self.x, self.y, and self.variance.

    The has_good_solution() method should also be overwritten to provide a more detailed
    check of the solution which is model dependent.

    self.x, self.y, and self.variance must be defined by the user before fitting."""
    def __init__(self):
        self.fit_result = None
        self.x = None
        self.y = None
        self.variance = None

    def fit(self, guess):
        scale = True
        variance = self.variance
        if self.variance is None:
            variance = np.ones(y.shape)
            scale = False
        arguments = (self.x, self.y, variance, self.fit_function, self.dfdx)
        self.fit_result = lm.minimize(self.chi_squared, guess, args=arguments,
                                      scale_covar=scale)

    @staticmethod
    def chi_squared(parameters, x, y, variance, f, dfdx):
        return (f(x, parameters) - y) / (dfdx(x, parameters)**2 * variance)

    def plot(self):
        pass

    def has_good_solution(self):
        return self.fit_result.success

    def guess(self):
        raise NotImplementedError


class Quadratic(XErrorsModel):
    @staticmethod
    def fit_function(x, parameters):
        p = parameters.valuesdict()
        return p['c2'] * x**2 + p['c1'] * x + p['c0']

    @staticmethod
    def dfdx(x, parameters):
        p = parameters.valuesdict()
        return 2 * p['c2'] * x + p['c1']

    def has_good_solution(self):
        return self.fit_result.success

    def guess(self):
        poly = np.polyfit(x, y, 2)
        parameters = lm.Parameters()
        parameters.add('c0', value=poly[2])
        parameters.add('c1', value=poly[1])
        parameters.add('c2', value=poly[0])
        return parameters


class Linear(XErrorsModel):
    @staticmethod
    def fit_function(x, parameters):
        p = parameters.valuesdict()
        return p['c1'] * x + p['c0']

    @staticmethod
    def dfdx(_, parameters):
        p = parameters.valuesdict()
        return p['c1']

    def has_good_solution(self):
        return self.fit_result.success

    def guess(self):
        poly = np.polyfit(x, y, 1)
        parameters = lm.Parameters()
        parameters.add('c0', value=poly[1])
        parameters.add('c1', value=poly[0])
        return parameters
