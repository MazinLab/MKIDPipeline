import os
import sys
import copy
import pickle
import inspect
import astropy
import warnings
import numpy as np
import lmfit as lm
from cycler import cycler
import scipy.optimize as opt
from scipy.stats import chi2
from scipy.signal import find_peaks
if sys.version_info.major == 3:
    from inspect import signature
else:
    from funcsigs import signature
from matplotlib import pyplot as plt
from scipy.special import erfc, erfcx

import mkidcore.corelog as pipelinelog

PLANK_CONSTANT_EVS = astropy.constants.h.to('eV s').value
SPEED_OF_LIGHT_NMS = astropy.constants.c.to('nm/s').value

log = pipelinelog.getLogger('mkidpipeline.calibration.wavecal_models', setup=False)

pixel_flags = {
    "good histogram": 0,
    "photon data": 1,
    "hot pixel": 2,
    "time cut": 3,
    "positive cut": 4,
    "few bins": 5,
    "histogram convergence": 6,
    "histogram validation": 7,
    "good calibration": 10,
    "few histograms": 11,
    "not monotonic": 12,
    "calibration convergence": 13,
    "calibration validation": 14
}

flag_definitions = {
    pixel_flags["good histogram"]: "histogram fit - converged and validated",
    pixel_flags["photon data"]: "histogram not fit - not enough data points",
    pixel_flags["hot pixel"]: "histogram not fit - too much data (hot pixel)",
    pixel_flags["time cut"]: "histogram not fit - not enough data left after arrival time cut",
    pixel_flags["positive cut"]: "histogram not fit - not enough data left after negative phase only cut",
    pixel_flags["few bins"]: "histogram not fit - not enough histogram bins to fit the model",
    pixel_flags["histogram convergence"]: "histogram not fit - best fit did not converge",
    pixel_flags["histogram validation"]: "histogram not fit - best fit converged but failed validation",
    pixel_flags["good calibration"]: "energy fit - converged and validated",
    pixel_flags["few histograms"]: "energy not fit - not enough data points",
    pixel_flags["not monotonic"]: "energy not fit - data not monotonic enough",
    pixel_flags["calibration convergence"]: "energy not fit - best fit did not converge",
    pixel_flags["calibration validation"]: "energy not fit - best fit converged but failed validation"
}


def port_model_result(model, parameters, fit_result):
    model_result = copy.deepcopy(fit_result)
    model_result.model = model
    model_result.params = parameters
    model_result.nvarys = len(signature(model_result.model.func).parameters) - 1
    return model_result


def switch_centers(partial_linear_model):
    old_parameters = partial_linear_model.best_fit_result.params
    new_parameters = partial_linear_model.best_fit_result.params.copy()
    new_parameters['signal_center'] = old_parameters['background_center']
    new_parameters['signal_sigma'] = old_parameters['background_sigma']
    new_parameters['signal_amplitude'] = old_parameters['background_amplitude']
    new_parameters['background_center'] = old_parameters['signal_center']
    new_parameters['background_sigma'] = old_parameters['signal_sigma']
    new_parameters['background_amplitude'] = old_parameters['signal_amplitude']
    partial_linear_model.best_fit_result.params = new_parameters
    partial_linear_model.phm = new_parameters['positive_half_max'].value
    partial_linear_model.nhm = new_parameters['negative_half_max'].value
    success = True if new_parameters['signal_sigma'] < new_parameters['background_center'] else False
    return success


def add_fwhm(guess):
    # TODO: add gaussian independent fwhm calculation
    if "positive_half_max" not in guess.keys():
        guess.add("positive_half_max", expr="signal_center + 2.355 * signal_sigma / 2")
    if "negative_half_max" not in guess.keys():
        guess.add("negative_half_max", expr="signal_center - 2.355 * signal_sigma / 2")

    return guess


def find_amplitudes(x, y, model_functions, args, variance=None):
    if variance is None:
        variance = np.ones(y.shape)
    # make coefficient matrix
    a = np.vstack([model(x, *args[index]) / np.sqrt(variance)
                   for index, model in enumerate(model_functions)]).T
    # rescale y
    b = y / np.sqrt(variance)

    # solve for amplitudes enforcing positive values
    amplitudes, _ = opt.nnls(a, b)
    amplitudes[amplitudes > 1e100] = 1e100  # prevent overflows

    return amplitudes


def skewed_gaussian(x, center, sigma, gamma):
    """
    Return an exponentially modified Gaussian distribution.
    Use gamma = np.inf to return the usual Gaussian distribution

    https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution
    """
    try:
        result = np.zeros(x.shape)
    except AttributeError:
        x = np.array(x)
        result = np.zeros(x.shape)

    z = (sigma * gamma - (x - center) / sigma) / np.sqrt(2)
    logic1 = (z < 0)
    logic2 = np.logical_and(z >= 0, z < 6.71e7)
    logic3 = (z >= 6.71e7)

    if logic1.any():
        arg1 = (sigma * gamma) ** 2 / 2 - (x[logic1] - center) * gamma
        arg2 = z[logic1]
        a = np.sqrt(2 * np.pi) * np.abs(sigma) * gamma / 2
        result[logic1] = a * np.exp(arg1) * erfc(arg2)

    if logic2.any():
        arg1 = -0.5 * ((x[logic2] - center) / sigma) ** 2
        arg2 = z[logic2]
        a = np.sqrt(2 * np.pi) * np.abs(sigma) * gamma / 2
        result[logic2] = a * np.exp(arg1) * erfcx(arg2)

    if logic3.any():
        arg1 = -0.5 * ((x[logic3] - center) / sigma) ** 2
        a = (1 / (1 + (x[logic3] - center) / (sigma ** 2 * gamma)))
        result[logic3] = a * np.exp(arg1)

    return result


def gaussian(x, center, sigma):
    """Gaussian function with unit amplitude"""
    return np.exp(-((x - center) / sigma)**2 / 2)


def exponential(x, rate):
    """Exponential function with unit amplitude"""
    return np.exp(rate * x)


def plot_text(axes, flag, color):
    if flag is None:
        return
    x_limits = axes.get_xlim()
    y_limits = axes.get_ylim()
    dx, dy = np.diff(x_limits), np.diff(y_limits)
    axes.text(x_limits[0] + 0.01 * dx, y_limits[1] - 0.01 * dy,
              flag_definitions[flag], color=color, ha='left', va='top')


class PartialLinearModel(object):
    """Base model class for fitting a linear combination of multiple functions to data.
    The linearity of the coefficients multiplying each function is used to reduce the
    dimension of the solution space by the number of functions. This reduction leads to
    more robust weighted, nonlinear least squares optimization.

    This class must be subclassed to run with the __init__, reduced_fit_function and
    full_fit_function methods patched. The following is an example outline for a two
    component model:

    class MyClass(PartialLinearModel):
        @staticmethod
        def full_fit_function(x, arg0, arg1, arg2, arg3):
            return arg0 * function1(x, arg1) + arg2 * function2(x, arg3)

        @staticmethod
        def reduced_fit_function(x, y, arg1, arg3, variance=None,
                                 return_amplitudes=False):
            amplitudes = find_amplitudes(x, y, [function1, function2], [[arg1], [arg3]],
                                         variance=variance)
            if return_amplitudes:
                amplitudes = lm.Parameters()
                amplitudes.add('arg0', amplitudes[0])
                amplitudes.add('arg2', amplitudes[1])
                return amplitudes
            result = (amplitudes[0] * function1(x, arg1) +
                      amplitudes[1] * function2(x, arg3))
            return result

    Use 'signal_center' as the fit function argument for the center of the distribution
    used to fit the phase to energy calibration.

    The guess method should be overwritten to provide a reasonable guess for the fitting
    routine based on self.x, self.y and self.variance. An optional 'index' keyword
    argument is provided for specifying more than one guess.

    The has_good_solution() method can be overwritten to provide a more detailed
    check of the solution which is model dependent.

    self.x, self.y and self.variance must be defined by the user before fitting.
    """
    def __init__(self, pixel=None, res_id=None):
        self.pixel = pixel
        self.res_id = res_id
        self._reduced_model = lm.Model(self.reduced_fit_function,
                                       independent_vars=['x', 'y', 'variance'])
        self._full_model = lm.Model(self.full_fit_function, independent_vars=['x'])
        self._cycler = cycler('color', ['orange', 'purple', 'yellow', 'black'])
        self.x = None
        self.y = None
        self.variance = None
        self.best_fit_result = None
        self.best_fit_result_good = None
        self.flag = None  # flag for wavecal computation condition
        self.phm = None  # positive half width half max
        self.nhm = None  # negative half width half max
        self.max_parameters = 10
        if len(signature(self._full_model.func).parameters) - 1 > self.max_parameters:
            message = "no more than {} parameters are allowed in the full_fit_function"
            raise SyntaxError(message.format(self.max_parameters))

    def __getstate__(self):
        b = self.best_fit_result
        r = (b.aic, b.success, b.params.dumps(), b.chisqr, b.errorbars, b.residual, b.nvarys) if b is not None else None
        state = {'pixel': self.pixel, 'res_id': self.res_id, 'x': self.x, 'y': self.y, 'variance': self.variance,
                 'best_fit_result': r, 'best_fit_result_good': self.best_fit_result_good, 'flag': self.flag,
                 'phm': self.phm, 'nhm': self.nhm}
        return state

    def __setstate__(self, state):
        self.__init__(state['pixel'], state['res_id'])
        self.x = state['x']
        self.y = state['y']
        self.variance = state['variance']
        if state['best_fit_result'] is None:
            self.best_fit_result = None
        else:
            r = lm.model.ModelResult(self._full_model, lm.Parameters())
            (r.aic, r.success, params, r.chisqr, r.errorbars, r.residual, r.nvarys) = state['best_fit_result']
            r.params = lm.Parameters().loads(params)
            self.best_fit_result = r
        self.best_fit_result_good = state['best_fit_result_good']
        self.flag = state['flag']
        self.phm = state['phm']
        self.nhm = state['nhm']

    def fit(self, guess):
        self._check_data()
        guess = add_fwhm(guess)
        good_fit = True
        keep = (self.y != 0)
        x = self.x[keep]
        y = self.y[keep]
        variance = self.variance
        fit_result = None
        if variance is not None:
            variance = variance[keep]
        try:
            with warnings.catch_warnings():
                # suppress warning when fits sample bad parts of the parameter space
                # warnings.simplefilter("ignore", category=RuntimeWarning)
                # solve the reduced weighted least squares optimization
                if variance is None:
                    fit_result = self._reduced_model.fit(y, params=guess, x=x, y=y, variance=[], scale_covar=True,
                                                         nan_policy='propagate')
                else:
                    fit_result = self._reduced_model.fit(y, params=guess, x=x, y=y, variance=variance,
                                                         weights=1 / np.sqrt(variance), scale_covar=False,
                                                         nan_policy='propagate')
            # find the linear amplitude coefficients
            amplitudes = self._reduced_model.eval(fit_result.params, x=x, y=y, variance=variance,
                                                  return_amplitudes=True)
            # set the minimum amplitude to be 0 for the full fit
            for amplitude in amplitudes.values():
                amplitude.set(min=0)
            # create a new guess with the least squares solution
            guess = fit_result.params.copy()
            guess.add_many(*[amplitudes[key] for key in amplitudes.keys()])
            # check if the model converged by temporarily swapping in the fit_result
            old_best_fit = copy.deepcopy(self.best_fit_result)
            self.best_fit_result = port_model_result(self._full_model, guess, fit_result)
            good_fit = self.has_good_solution()
            # replace if we will be refitting or if both the old fit exists and if the
            # old fit has a better chi2
            if not good_fit:
                fit_result = self.best_fit_result
            if good_fit or (old_best_fit is not None and
                            old_best_fit.chisqr < self.best_fit_result.chisqr):
                self.best_fit_result = old_best_fit

        except (NotImplementedError, AttributeError) as error:
            # skip if reduced_fit_function is not implemented
            log.warning(error)
            pass

        # solve the full system to get the full covariance matrix
        if good_fit:
            with warnings.catch_warnings():
                # suppress warning when fits sample bad parts of the parameter space
                warnings.simplefilter("ignore", category=RuntimeWarning)
                if variance is None:
                    fit_result = self._full_model.fit(y, params=guess, x=x, scale_covar=True, nan_policy='propagate')
                else:
                    fit_result = self._full_model.fit(y, params=guess, x=x, weights=1 / np.sqrt(variance),
                                                      scale_covar=False, nan_policy='propagate')

        # save the data in best_fit_result if it is better than previous fits
        used_last_fit = good_fit and (self.best_fit_result is None or fit_result.chisqr < self.best_fit_result.chisqr)
        if used_last_fit:
            # patch init_params and init_guess since they get over loaded during first fit
            fit_result.init_params = guess
            fit_result.init_values = fit_result.model._make_all_args(guess)
            self.best_fit_result = fit_result
            self.phm = fit_result.params['positive_half_max'].value
            self.nhm = fit_result.params['negative_half_max'].value
            self.best_fit_result_good = None
            self.best_fit_result_good = self.has_good_solution()

    def histogram_function(self, x):
        self._check_fit()
        return self._full_model.eval(self.best_fit_result.params, x=x)

    @property
    def signal_center(self):
        try:
            self._check_fit()
            return self.best_fit_result.params['signal_center']
        except RuntimeError:
            # catch for when there is no good fit so we can at least return a parameter
            # object with the same name
            return lm.Parameter('signal_center')

    @property
    def signal_sigma(self):
        try:
            self._check_fit()
            return self.best_fit_result.params['signal_sigma']
        except RuntimeError:
            # catch for when there is no good fit so we can at least return a parameter
            # object with the same name
            return lm.Parameter('signal_sigma')

    @property
    def signal_center_standard_error(self):
        self._check_fit()
        if not self.best_fit_result.errorbars:
            message = ("The best fit for this model isn't good enough to have computed "
                       "errors on its parameters")
            raise RuntimeError(message)
        return self.best_fit_result.params['signal_center'].stderr

    def plot(self, axes=None, legend=True, title=True, x_label=True, y_label=True, text=True):
        # set up plot basics
        if axes is None:
            if legend and self.best_fit_result is not None:
                size = (8.4, 4.8)
            else:
                size = (6.4, 4.8)
            fig, axes = plt.subplots(figsize=size)
        if self.has_good_solution():
            color = "green"
            label = "fit accepted"
        else:
            color = "red"
            label = "fit rejected"
        if x_label:
            axes.set_xlabel('phase [degrees]')
        if title:
            t = "Model '{}'" + os.linesep + "Pixel ({}, {}) : ResID {}"
            axes.set_title(t.format(type(self).__name__, self.pixel[0], self.pixel[1],  self.res_id))

        # no data
        if self.x is None or len(self.x) == 1:
            if text:
                plot_text(axes, self.flag, color)
            if y_label:
                axes.set_ylabel('counts')
            return axes

        # plot data
        cycle = self._cycler()
        difference = np.diff(self.x)
        widths = np.hstack([difference[0], difference])
        axes.bar(self.x, self.y, widths)
        if y_label:
            axes.set_ylabel('counts per {:.1f} degrees'.format(np.mean(widths)))
        # no fit
        if self.best_fit_result is None:
            if text:
                plot_text(axes, self.flag, color)
            return axes
        # plot fit
        xx = np.linspace(self.x.min(), self.x.max(), 1000)
        yy = self.best_fit_result.eval(x=xx)
        axes.plot(xx, yy, color=color, label=label, zorder=3)
        all_parameters = self._full_model.make_params()
        reduced_parameters = self._reduced_model.make_params()
        amplitude_names = []
        for parameter_name in all_parameters.keys():
            if parameter_name not in reduced_parameters:
                amplitude_names.append(parameter_name)
        for amplitude_name in amplitude_names:
            fit_parameters = self.best_fit_result.params.copy()
            for parameter_name in amplitude_names:
                if parameter_name != amplitude_name:
                    fit_parameters[parameter_name].value = 0
            axes.plot(xx, self.best_fit_result.eval(fit_parameters, x=xx),
                      color=next(cycle)['color'], linestyle='--', label=amplitude_name)

        # make legend
        if legend:
            axes.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            plt.tight_layout()

        # add text
        if text:
            plot_text(axes, self.flag, color)

        return axes

    def has_good_solution(self):
        # no fit
        if self.best_fit_result is None:
            return False

        # solver failed
        success = self.best_fit_result.success
        if not success:
            return success

        # bad fit to data
        p = self.best_fit_result.params
        # TODO: chi2 > 200 is a bit ridiculous. Need better models to use the p_value
        # p_value = chi2.sf(*self.chi2())
        chi_squared, df = self.chi2()
        high_chi2 = chi_squared / df > 200
        no_errors = not self.best_fit_result.errorbars
        max_phase = np.min([-10., np.max(self.x) * 1.7])
        min_phase = np.min(self.x)
        out_of_bounds_peak = (p['signal_center'] > max_phase or
                              p['signal_center'] < min_phase)
        large_sigma = 2. * p['signal_sigma'] > np.max(self.x) - np.min(self.x)
        small_sigma = p['signal_sigma'] < 2
        success = not (high_chi2 or no_errors or out_of_bounds_peak or large_sigma or
                       small_sigma)
        return success

    def guess(self, index=0):
        raise NotImplementedError

    def copy(self):
        return copy.deepcopy(self)

    def chi2(self):
        """chi squared +/- 2 sigma from the signal_peak and degrees of freedom"""
        self._check_fit()
        p = self.best_fit_result.params
        center = p['signal_center'].value
        sigma = p['signal_sigma'].value
        left = center - 2 * sigma
        right = center + 2 * sigma
        x = self.x[self.y != 0]
        logic = np.logical_and(x > left, x < right)
        chi_squared = np.sum(self.best_fit_result.residual[logic]**2)
        df = np.max([1., float(np.sum(logic) - self.best_fit_result.nvarys)])
        return chi_squared, df

    def _check_fit(self):
        if self.best_fit_result is None:
            raise RuntimeError("A fit for this model has not been computed")

    def _check_data(self):
        if self.x is None or self.y is None:
            raise RuntimeError("Data for this model has not been computed yet")


class GaussianAndExponential(PartialLinearModel):
    """Gaussian signal plus exponential background"""
    @staticmethod
    def full_fit_function(x, signal_amplitude, signal_center, signal_sigma,
                          trigger_amplitude, trigger_tail):
        result = (signal_amplitude * gaussian(x, signal_center, signal_sigma) +
                  trigger_amplitude * exponential(x, trigger_tail))
        return result

    @staticmethod
    def reduced_fit_function(x, y, signal_center, signal_sigma, trigger_tail,
                             variance=None, return_amplitudes=False):
        model_functions = [gaussian, exponential]
        args = [[signal_center, signal_sigma], [trigger_tail]]
        amplitudes = find_amplitudes(x, y, model_functions, args, variance=variance)
        if return_amplitudes:
            parameters = lm.Parameters()
            parameters.add('signal_amplitude', value=amplitudes[0])
            parameters.add('trigger_amplitude', value=amplitudes[1])
            return parameters

        result = (amplitudes[0] * gaussian(x, signal_center, signal_sigma) +
                  amplitudes[1] * exponential(x, trigger_tail))
        return result

    def has_good_solution(self):
        if self.best_fit_result_good is not None:
            return self.best_fit_result_good
        success = super(GaussianAndExponential, self).has_good_solution()
        if not success:
            return success
        p = self.best_fit_result.params

        g = p['signal_amplitude'].value * gaussian(p['signal_center'].value,
                                                   p['signal_center'].value,
                                                   p['signal_sigma'].value)
        e = p['trigger_amplitude'].value * exponential(p['signal_center'].value,
                                                       p['trigger_tail'].value)
        swamped_peak = g < 2. * e
        small_amplitude = g < self.y.sum() / len(self.x) / 10.
        success = not (swamped_peak or small_amplitude)
        return success

    def guess(self, index=0):
        parameters = lm.Parameters()
        if index == 0:
            peaks, _ = find_peaks(self.y, height=self.y.sum() / len(self.x) / 2)
            if len(peaks) > 0:
                signal_center = self.x[peaks[0]]
            else:
                phase_smoothed = np.convolve(self.y, np.ones(10) / 10.0, mode='same')
                signal_center = self.x[np.argmax(phase_smoothed)]

            amplitude = 90
            r = 4  # typical R at amplitude degrees
            r_max = 200  # no R bigger than this at amplitude degrees
            signal_sigma = amplitude / (2.355 * r)
            sigma_min = amplitude / (2.355 * r_max)

            trigger_tail = 0.2

        elif index == 1:
            signal_center = (np.max(self.x) + np.min(self.x)) / 2
            signal_sigma = (np.max(self.x) - np.min(self.x)) / 10
            sigma_min = signal_sigma / 100
            trigger_tail = 0.2

        else:
            signal_center = -100
            signal_sigma = 10
            sigma_min = 0.1
            trigger_tail = 0.1
        parameters.add('signal_center', value=signal_center, min=np.min(self.x),
                       max=np.max(self.x))
        parameters.add('signal_sigma', value=signal_sigma, min=sigma_min, max=np.inf)
        parameters.add('trigger_tail', value=trigger_tail, min=0, max=np.inf)
        return parameters


class GaussianAndGaussian(PartialLinearModel):
    """Gaussian signal plus gaussian background"""
    @staticmethod
    def full_fit_function(x, signal_amplitude, signal_center, signal_sigma,
                          background_amplitude, background_center, background_sigma):
        result = (signal_amplitude * gaussian(x, signal_center, signal_sigma) +
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

        result = (amplitudes[0] * gaussian(x, signal_center, signal_sigma) +
                  amplitudes[1] * gaussian(x, background_center, background_sigma))
        return result

    def has_good_solution(self):
        if self.best_fit_result_good is not None:
            return self.best_fit_result_good
        success = super(GaussianAndGaussian, self).has_good_solution()
        if not success:
            return success
        p = self.best_fit_result.params
        # switch center and background if fit backwards
        if p['signal_center'] > p['background_center']:
            success = switch_centers(self)
            if not success:
                return success

        g = p['signal_amplitude'].value * gaussian(p['signal_center'].value,
                                                   p['signal_center'].value,
                                                   p['signal_sigma'].value)
        g2 = p['background_amplitude'].value * gaussian(p['signal_center'].value,
                                                        p['background_center'].value,
                                                        p['background_sigma'].value)
        swamped_peak = g < 2. * g2
        large_background_sigma = p['background_sigma'] > 2 * p['signal_sigma']
        small_amplitude = g < self.y.sum() / len(self.x) / 10.
        success = not (swamped_peak or large_background_sigma or small_amplitude)

        return success

    def guess(self, index=0):
        parameters = lm.Parameters()
        if index == 0:
            peaks, _ = find_peaks(self.y, height=self.y.sum() / len(self.x) / 2)
            if len(peaks) > 0:
                signal_center = self.x[peaks[0]]
            else:
                phase_smoothed = np.convolve(self.y, np.ones(10) / 10.0, mode='same')
                signal_center = self.x[np.argmax(phase_smoothed)]

            amplitude = 90
            r = 4  # typical R at amplitude degrees
            r_max = 200  # no R bigger than this at amplitude degrees
            signal_sigma = amplitude / (2.355 * r)
            sigma_min = amplitude / (2.355 * r_max)

            amplitude = 20
            r = 5
            r_max = 10
            if len(peaks) > 1:
                background_center = self.x[peaks[1]]
            else:
                background_center = np.min([-30, np.max(self.x)])
            background_sigma = amplitude / (2.355 * r)
            background_sigma_min = amplitude / (2.355 * r_max)

        elif index == 1:
            signal_center = (np.max(self.x) + np.min(self.x)) / 2
            signal_sigma = (np.max(self.x) - np.min(self.x)) / 10
            sigma_min = signal_sigma / 100
            background_center = np.min([2 * signal_center / 3, np.max(self.x)])
            background_sigma = 2 * signal_sigma
            background_sigma_min = background_sigma / 100

        else:
            signal_center = np.min([-100, np.max(self.x)])
            signal_sigma = 10
            sigma_min = 0.1
            background_center = np.min([-40, np.max(self.x)])
            background_sigma = 20
            background_sigma_min = 2
        parameters.add('signal_center', value=signal_center, min=np.min(self.x),
                       max=np.max(self.x))
        parameters.add('signal_sigma', value=signal_sigma, min=sigma_min, max=np.inf)
        parameters.add('background_center', value=background_center, min=-np.inf, max=0)
        parameters.add('background_sigma', value=background_sigma,
                       min=background_sigma_min, max=np.inf)

        return parameters


class GaussianAndGaussianExponential(PartialLinearModel):
    """Gaussian signal plus gaussian background"""
    @staticmethod
    def full_fit_function(x, signal_amplitude, signal_center, signal_sigma,
                          background_amplitude, background_center, background_sigma,
                          trigger_amplitude, trigger_tail):
        result = (signal_amplitude * gaussian(x, signal_center, signal_sigma) +
                  background_amplitude * gaussian(x, background_center,
                                                  background_sigma) +
                  trigger_amplitude * exponential(x, trigger_tail))
        return result

    @staticmethod
    def reduced_fit_function(x, y, signal_center, signal_sigma, background_center,
                             background_sigma, trigger_tail, variance=None,
                             return_amplitudes=False):
        model_functions = [gaussian, gaussian, exponential]
        args = [[signal_center, signal_sigma], [background_center, background_sigma],
                [trigger_tail]]
        amplitudes = find_amplitudes(x, y, model_functions, args, variance=variance)
        if return_amplitudes:
            parameters = lm.Parameters()
            parameters.add('signal_amplitude', value=amplitudes[0])
            parameters.add('background_amplitude', value=amplitudes[1])
            parameters.add('trigger_amplitude', value=amplitudes[2])
            return parameters

        result = (amplitudes[0] * gaussian(x, signal_center, signal_sigma) +
                  amplitudes[1] * gaussian(x, background_center, background_sigma) +
                  amplitudes[2] * exponential(x, trigger_tail))
        return result

    def has_good_solution(self):
        if self.best_fit_result_good is not None:
            return self.best_fit_result_good
        success = super(GaussianAndGaussianExponential, self).has_good_solution()
        if not success:
            return success
        p = self.best_fit_result.params
        # switch center and background if fit backwards
        if p['signal_center'] > p['background_center']:
            success = switch_centers(self)
            if not success:
                return success

        g = p['signal_amplitude'].value * gaussian(p['signal_center'].value,
                                                   p['signal_center'].value,
                                                   p['signal_sigma'].value)
        e = p['trigger_amplitude'].value * exponential(p['signal_center'].value,
                                                       p['trigger_tail'].value)
        g2 = p['background_amplitude'].value * gaussian(p['signal_center'].value,
                                                        p['background_center'].value,
                                                        p['background_sigma'].value)
        swamped_peak = g < 2. * (g2 + e)
        large_background_sigma = p['background_sigma'] > 2 * p['signal_sigma']
        small_amplitude = g < self.y.sum() / len(self.x) / 10.
        success = not (swamped_peak or large_background_sigma or small_amplitude)

        return success

    def guess(self, index=0):
        parameters = lm.Parameters()
        if index == 0:
            peaks, _ = find_peaks(self.y, height=self.y.sum() / len(self.x) / 2)
            if len(peaks) > 0:
                signal_center = self.x[peaks[0]]
            else:
                phase_smoothed = np.convolve(self.y, np.ones(10) / 10.0, mode='same')
                signal_center = self.x[np.argmax(phase_smoothed)]

            amplitude = 90
            r = 4  # typical R at amplitude degrees
            r_max = 200  # no R bigger than this at amplitude degrees
            signal_sigma = amplitude / (2.355 * r)
            sigma_min = amplitude / (2.355 * r_max)

            amplitude = 20
            r = 5
            r_max = 10
            if len(peaks) > 1:
                background_center = self.x[peaks[1]]
            else:
                background_center = np.min([-30, np.max(self.x)])
            background_sigma = amplitude / (2.355 * r)
            background_sigma_min = amplitude / (2.355 * r_max)

            trigger_tail = 0.2

        elif index == 1:
            signal_center = (np.max(self.x) + np.min(self.x)) / 2
            signal_sigma = (np.max(self.x) - np.min(self.x)) / 10
            sigma_min = signal_sigma / 100
            background_center = np.min([2 * signal_center / 3, np.max(self.x)])
            background_sigma = 2 * signal_sigma
            background_sigma_min = background_sigma / 100
            trigger_tail = 0.2

        else:
            signal_center = np.min([-100, np.max(self.x)])
            signal_sigma = 10
            sigma_min = 0.1
            background_center = np.min([-40, np.max(self.x)])
            background_sigma = 20
            background_sigma_min = 2
            trigger_tail = 0.1
        parameters.add('signal_center', value=signal_center, min=np.min(self.x),
                       max=np.max(self.x))
        parameters.add('signal_sigma', value=signal_sigma, min=sigma_min, max=np.inf)
        parameters.add('background_center', value=background_center, min=np.min(self.x),
                       max=0)
        parameters.add('background_sigma', value=background_sigma,
                       min=background_sigma_min, max=np.inf)
        parameters.add('trigger_tail', value=trigger_tail, min=0, max=np.inf)
        return parameters


class SkewedGaussianAndGaussianExponential(PartialLinearModel):
    """Gaussian signal plus gaussian background. Do not use. Doesn't compute energy
     resolution correctly"""
    @staticmethod
    def full_fit_function(x, signal_amplitude, signal_center, signal_sigma, signal_gamma,
                          background_amplitude, background_center, background_sigma,
                          trigger_amplitude, trigger_tail):
        result = (signal_amplitude * skewed_gaussian(x, signal_center, signal_sigma,
                                                     signal_gamma) +
                  background_amplitude * gaussian(x, background_center,
                                                  background_sigma) +
                  trigger_amplitude * exponential(x, trigger_tail))
        return result

    @staticmethod
    def reduced_fit_function(x, y, signal_center, signal_sigma, signal_gamma,
                             background_center, background_sigma, trigger_tail,
                             variance=None, return_amplitudes=False):
        model_functions = [skewed_gaussian, gaussian, exponential]
        args = [[signal_center, signal_sigma, signal_gamma],
                [background_center, background_sigma], [trigger_tail]]
        amplitudes = find_amplitudes(x, y, model_functions, args, variance=variance)
        if return_amplitudes:
            parameters = lm.Parameters()
            parameters.add('signal_amplitude', value=amplitudes[0])
            parameters.add('background_amplitude', value=amplitudes[1])
            parameters.add('trigger_amplitude', value=amplitudes[2])
            return parameters

        result = (amplitudes[0] * gaussian(x, signal_center, signal_sigma) +
                  amplitudes[1] * gaussian(x, background_center, background_sigma) +
                  amplitudes[2] * exponential(x, trigger_tail))
        return result

    def has_good_solution(self):
        if self.best_fit_result_good is not None:
            return self.best_fit_result_good
        success = super(SkewedGaussianAndGaussianExponential, self).has_good_solution()
        if not success:
            return success
        p = self.best_fit_result.params
        # switch center and background if fit backwards
        if p['signal_center'] > p['background_center']:
            success = switch_centers(self)
            if not success:
                return success

        g = p['signal_amplitude'].value * gaussian(p['signal_center'].value,
                                                   p['signal_center'].value,
                                                   p['signal_sigma'].value)
        e = p['trigger_amplitude'].value * exponential(p['signal_center'].value,
                                                       p['trigger_tail'].value)
        g2 = p['background_amplitude'].value * gaussian(p['signal_center'].value,
                                                        p['background_center'].value,
                                                        p['background_sigma'].value)
        swamped_peak = g < 2. * (g2 + e)
        large_background_sigma = p['background_sigma'] > 2 * p['signal_sigma']
        small_amplitude = g < self.y.sum() / len(self.x) / 10.
        success = not (swamped_peak or large_background_sigma or small_amplitude)

        return success

    def guess(self, index=0):
        parameters = lm.Parameters()
        if index == 0:
            peaks, _ = find_peaks(self.y, height=self.y.sum() / len(self.x) / 2)
            if len(peaks) > 0:
                signal_center = self.x[peaks[0]]
            else:
                phase_smoothed = np.convolve(self.y, np.ones(10) / 10.0, mode='same')
                signal_center = self.x[np.argmax(phase_smoothed)]

            amplitude = 90
            r = 4  # typical R at amplitude degrees
            r_max = 200  # no R bigger than this at amplitude degrees
            signal_sigma = amplitude / (2.355 * r)
            sigma_min = amplitude / (2.355 * r_max)
            signal_gamma = 10

            amplitude = 20
            r = 5
            r_max = 10
            if len(peaks) > 1:
                background_center = self.x[peaks[1]]
            else:
                background_center = np.min([-30, np.max(self.x)])
            background_sigma = amplitude / (2.355 * r)
            background_sigma_min = amplitude / (2.355 * r_max)

            trigger_tail = 0.2

        elif index == 1:
            signal_center = (np.max(self.x) + np.min(self.x)) / 2
            signal_sigma = (np.max(self.x) - np.min(self.x)) / 10
            sigma_min = signal_sigma / 100
            signal_gamma = 10
            background_center = np.min([2 * signal_center / 3, np.max(self.x)])
            background_sigma = 2 * signal_sigma
            background_sigma_min = background_sigma / 100
            trigger_tail = 0.2

        else:
            signal_center = np.min([-100, np.max(self.x)])
            signal_sigma = 10
            sigma_min = 0.1
            signal_gamma = 5
            background_center = np.min([-30, np.max(self.x)])
            background_sigma = 20
            background_sigma_min = 2
            trigger_tail = 0.05
        parameters.add('signal_center', value=signal_center, min=np.min(self.x),
                       max=np.max(self.x))
        parameters.add('signal_sigma', value=signal_sigma, min=sigma_min, max=np.inf)
        parameters.add('signal_gamma', value=signal_gamma, min=0)
        parameters.add('background_center', value=background_center, min=np.min(self.x),
                       max=0)
        parameters.add('background_sigma', value=background_sigma,
                       min=background_sigma_min, max=np.inf)
        parameters.add('trigger_tail', value=trigger_tail, min=0, max=np.inf)
        return parameters


class XErrorsModel(object):
    """Base class for fitting a function to data with errors in the x variable by
    assuming that the x error is small enough such that the function can be approximated
    as linear near the error point.

    This class must be subclassed to run with the fit_function and dfdx methods patched.
    The following is an example outline:

    @staticmethod
    def fit_function(x, p):
        return p['c2'].value * x**2 + p['c1'] .value* x + p['c0'].value

    @staticmethod
    def dfdx(x, p):
        return 2 * p['c2'].value * x + p['c1'].value

    The guess method should be overwritten to provide a reasonable guess for the fitting
    routine based on self.x, self.y, and self.variance.

    The has_good_solution() method should also be overwritten to provide a more detailed
    check of the solution which is model dependent.

    self.x, self.y, and self.variance must be defined by the user before fitting."""
    def __init__(self, pixel=None, res_id=None):
        self.pixel = pixel
        self.res_id = res_id
        self.x = None
        self.y = None
        self.variance = None
        self.best_fit_result = None
        self.best_fit_result_good = None
        self.flag = None  # flag for wavecal computation condition
        self.max_x = None
        self.min_x = None

    def __getstate__(self):
        b = self.best_fit_result
        r = (b.aic, b.success, b.params.dumps(), b.chisqr, b.errorbars, b.residual, b.nvarys) if b is not None else None
        state = {'pixel': self.pixel, 'res_id': self.res_id, 'x': self.x, 'y': self.y, 'variance': self.variance,
                 'best_fit_result': r, 'best_fit_result_good': self.best_fit_result_good, 'flag': self.flag,
                 'max_x': self.max_x, 'min_x': self.min_x}
        return state

    def __setstate__(self, state):
        self.__init__(state['pixel'], state['res_id'])
        self.x = state['x']
        self.y = state['y']
        self.variance = state['variance']
        if state['best_fit_result'] is None:
            self.best_fit_result = None
        else:
            r = lm.minimizer.MinimizerResult()
            (r.aic, r.success, params, r.chisqr, r.errorbars, r.residual, r.nvarys) = state['best_fit_result']
            r.params = lm.Parameters().loads(params)
            self.best_fit_result = r
        self.best_fit_result_good = state['best_fit_result_good']
        self.flag = state['flag']
        self.max_x = state['max_x']
        self.min_x = state['min_x']

    def fit(self, guess):
        self._check_data()
        scale = True
        variance = self.variance
        if self.variance is None:
            variance = np.ones(y.shape)
            scale = False
        arguments = (self.x, self.y, variance, self.fit_function, self.dfdx)
        fit_result = lm.minimize(self.chi_squared, guess, args=arguments, scale_covar=scale)

        # save the data in best_fit_result if it is better than previous fits
        used_last_fit = (self.best_fit_result is None or fit_result.chisqr < self.best_fit_result.chisqr)
        if used_last_fit:
            self.best_fit_result = fit_result
            self.best_fit_result_good = None
            self.best_fit_result_good = self.has_good_solution()

    def calibration_function(self, x):
        self._check_fit()
        return self.fit_function(x, self.best_fit_result.params)

    def wavelength_function(self, x):
        self._check_fit()
        return PLANK_CONSTANT_EVS * SPEED_OF_LIGHT_NMS / self.fit_function(x, self.best_fit_result.params)

    @staticmethod
    def chi_squared(parameters, x, y, variance, f, dfdx):
        return (f(x, parameters) - y) / (dfdx(x, parameters) * np.sqrt(variance))

    def plot(self, axes=None, legend=True, title=True, x_label=True, y_label=True, text=True):
        # set up plot basics
        if axes is None:
            fig, axes = plt.subplots()
        if self.has_good_solution():
            color = "green"
            label = "fit accepted"
        else:
            color = "red"
            label = "fit rejected"
        if x_label:
            axes.set_xlabel('phase [degrees]')
        if y_label:
            axes.set_ylabel('energy [eV]')
        if title:
            t = "Model '{}'" + os.linesep + "Pixel ({}, {}) : ResID {}"
            axes.set_title(t.format(type(self).__name__, self.pixel[0], self.pixel[1], self.res_id))
        # no data
        if self.x is None or self.y is None:
            if text:
                plot_text(axes, self.flag, color)
            return axes

        # plot data
        axes.errorbar(self.x, self.y, xerr=np.sqrt(self.variance), linestyle='--',
                      marker='o', markersize=5, markeredgecolor='black',
                      markeredgewidth=0.5, ecolor='black', capsize=3, elinewidth=0.5)
        # no fit
        if self.best_fit_result is None:
            if text:
                plot_text(axes, self.flag, color)
            return axes
        # plot fit
        x_limit = [1.05 * min(self.x - np.sqrt(self.variance)), 0.95 * max(self.x + np.sqrt(self.variance))]
        axes.set_xlim(x_limit)
        xx = np.linspace(x_limit[0], x_limit[1], 1000)
        yy = self.fit_function(xx, self.best_fit_result.params)
        axes.plot(xx, self.fit_function(xx, self.best_fit_result.params), color=color, label=label)

        y_limit = [0.95 * min(yy), max(yy) * 1.05]
        axes.set_ylim(y_limit)

        if text:
            plot_text(axes, self.flag, color)
        if legend:
            axes.legend(loc="lower left")

        return axes

    def has_good_solution(self):
        if self.best_fit_result is None:
            return False
        return self.best_fit_result.success

    def guess(self):
        raise NotImplementedError

    def copy(self):
        return copy.deepcopy(self)

    def _check_data(self):
        if self.x is None or self.y is None:
            raise RuntimeError("Data for this model has not been computed yet")

    def _check_fit(self):
        if self.best_fit_result is None:
            raise RuntimeError("No fit has been computed for this model")


class Quadratic(XErrorsModel):
    @staticmethod
    def fit_function(x, p):
        return p['c2'].value * x**2 + p['c1'].value * x + p['c0'].value

    @staticmethod
    def dfdx(x, p):
        return 2 * p['c2'].value * x + p['c1'].value

    def has_good_solution(self):
        if self.best_fit_result_good is not None:
            return self.best_fit_result_good
        success = super(Quadratic, self).has_good_solution()
        if not success:
            return success

        p = self.best_fit_result.params
        vertex = - p['c1'].value / (2. * p['c2'].value)
        min_slope = 2. * p['c2'].value * self.min_x + p['c1'].value
        max_slope = 2. * p['c2'].value * self.max_x + p['c1'].value
        min_value = self.fit_function(self.min_x, self.best_fit_result.params)
        max_value = self.fit_function(self.max_x, self.best_fit_result.params)

        vertex_in_data = self.min_x < vertex < self.max_x
        positive_slope = min_slope > 0 or max_slope > 0
        negative_data = min_value < 0 or max_value < 0

        success = not (vertex_in_data or positive_slope or negative_data)

        return success

    def guess(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.RankWarning)
            poly = np.polyfit(self.x, self.y, 2)
        parameters = lm.Parameters()
        parameters.add('c0', value=poly[2])
        parameters.add('c1', value=poly[1])
        parameters.add('c2', value=poly[0])
        return parameters


class Linear(XErrorsModel):
    @staticmethod
    def fit_function(x, p):
        return p['c1'].value * x + p['c0'].value

    @staticmethod
    def dfdx(_, p):
        return p['c1'].value

    def has_good_solution(self):
        if self.best_fit_result_good is not None:
            return self.best_fit_result_good
        success = super(Linear, self).has_good_solution()
        if not success:
            return success
        p = self.best_fit_result.params
        positive_slope = p['c1'] > 0
        negative_energy = (self.max_x > -p['c0'].value / p['c1'].value and
                           not positive_slope)
        success = not (positive_slope or negative_energy)
        return success

    def guess(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", np.RankWarning)
            poly = np.polyfit(self.x, self.y, 1)
        parameters = lm.Parameters()
        parameters.add('c0', value=poly[1])
        parameters.add('c1', value=poly[0])
        return parameters
