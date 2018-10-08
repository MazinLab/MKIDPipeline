import os
import copy
import inspect
import numpy as np
import lmfit as lm
from cycler import cycler
import scipy.optimize as opt
from scipy.stats import chi2
from matplotlib import pyplot as plt
from scipy.special import erfc, erfcx

from mkidcore.pixelflags import waveCal as flag_dict


def switch_centers(partial_linear_model):
    new_guess = partial_linear_model.guess()
    parameters = partial_linear_model.best_fit_result.params
    new_guess['signal_center'] = parameters['background_center']
    new_guess['signal_sigma'] = parameters['background_sigma']
    new_guess['background_center'] = parameters['signal_center']
    new_guess['background_sigma'] = parameters['signal_sigma']
    partial_linear_model.fit(new_guess)
    p = partial_linear_model.best_fit_result.params.valuesdict()
    success = p['signal_center'] < p['background_center']
    return success


def find_amplitudes(x, y, model_functions, args, variance=None):
    if variance is None or not np.any(variance):
        variance = np.ones(y.shape)
    # make coefficient matrix
    a = np.vstack([model(x, *args[index]) / np.sqrt(variance)
                   for index, model in enumerate(model_functions)]).T
    # rescale y
    b = y / np.sqrt(variance)

    # solve for amplitudes enforcing positive values
    amplitudes, _ = opt.nnls(a, b)

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


def plot_text(axis, flag, color):
    x_limits = axis.get_xlim()
    y_limits = axis.get_ylim()
    dx, dy = np.diff(x_limits), np.diff(y_limits)
    axis.text(x_limits[0] + 0.01 * dx, y_limits[1] - 0.01 * dy,
              flag_dict[flag], color=color, ha='left', va='top')


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
        self.fit_result = None
        self.initial_guess = None
        self.x = None
        self.y = None
        self.variance = None
        self.best_fit_result = None
        self.best_fit_result_guess = None
        self.used_last_fit = None
        self.flag = None

    def fit(self, guess):
        self._check_data()
        self.initial_guess = guess.copy()
        keep = (self.y != 0)
        x = self.x[keep]
        y = self.y[keep]
        variance = self.variance
        if variance is not None:
            variance = variance[keep]

        try:
            # solve the reduced weighted least squares optimization
            if variance is None:
                fit_result = self._reduced_model.fit(y, params=guess, x=x, y=y,
                                                     variance=[],
                                                     scale_covar=True)
            else:
                fit_result = self._reduced_model.fit(y, params=guess, x=x, y=y,
                                                     variance=variance,
                                                     weights=1 / np.sqrt(variance),
                                                     scale_covar=False)
            # find the linear amplitude coefficients
            amplitudes = self._reduced_model.eval(fit_result.params, x=x, y=y,
                                                  variance=variance,
                                                  return_amplitudes=True)
            # set the minimum amplitude to be 0 for the full fit
            for amplitude in amplitudes.values():
                amplitude.set(min=0)
            # create a new guess with the least squares solution
            guess = fit_result.params.copy()
            guess.add_many(*[amplitudes[key] for key in amplitudes.keys()])

        except (NotImplementedError, AttributeError):
            # skip if reduced_fit_function is not implemented
            pass

        # solve the full system to get the full covariance matrix
        if variance is None:
            self.fit_result = self._full_model.fit(y, params=guess, x=x,
                                                   scale_covar=True)
        else:
            self.fit_result = self._full_model.fit(y, params=guess, x=x,
                                                   weights=1 / np.sqrt(variance),
                                                   scale_covar=False)

        # save the data in best_fit_result if it is better than previous fits
        self.used_last_fit = (self.best_fit_result is None or
                              self.fit_result.chisqr < self.best_fit_result.chisqr)
        if self.used_last_fit:
            self.best_fit_result = self.fit_result
            self.best_fit_result_guess = self.initial_guess

    @property
    def signal_center(self):
        self._check_fit()
        return self.best_fit_result.params['signal_center'].value

    @property
    def signal_center_standard_error(self):
        self._check_fit()
        if not self.best_fit_result.errorbars:
            message = ("The best fit for this model isn't good enough to have computed"
                       "errors on its parameters")
            raise RuntimeError(message)
        return self.best_fit_result.params['signal_center'].stderr

    def plot(self, axis=None, legend=True, title=True, x_label=True, y_label=True,
             best_fit=True, text=True):
        # set up plot basics
        if axis is None:
            if legend:
                size = (8.4, 4.8)
            else:
                size = (6.4, 4.8)
            fig, axis = plt.subplots(figsize=size)
        if self.has_good_solution():
            color = "green"
            label = "fit accepted"
        else:
            color = "red"
            label = "fit rejected"
        if x_label:
            axis.set_xlabel('phase [degrees]')
        if title:
            axis.set_title(("Model '{}'" + os.linesep + "Pixel {} : ResID {}")
                           .format(type(self).__name__, self.pixel, self.res_id))

        # no data
        if self.x is None or self.y is None:
            if text:
                plot_text(axis, self.flag, color)
            if y_label:
                axis.set_ylabel('counts')
            return axis

        # plot data
        cycle = self._cycler()
        difference = np.diff(self.x)
        widths = np.hstack([difference[0], difference])
        axis.bar(self.x, self.y, widths)
        if y_label:
            axis.set_ylabel('counts per {:.1f} degrees'.format(np.mean(widths)))

        # no fit
        if self.best_fit_result is None:
            if text:
                plot_text(axis, self.flag, color)
            return axis

        # choose fit to plot
        if best_fit:
            fit_result = self.best_fit_result
        else:
            fit_result = self.fit_result

        # plot fit
        xx = np.linspace(self.x.min(), self.x.max(), 1000)
        yy = fit_result.eval(x=xx)
        axis.plot(xx, yy, color=color, label=label, zorder=3)
        all_parameters = self._full_model.make_params()
        reduced_parameters = self._reduced_model.make_params()
        amplitude_names = []
        for parameter_name in all_parameters.keys():
            if parameter_name not in reduced_parameters:
                amplitude_names.append(parameter_name)
        for amplitude_name in amplitude_names:
            fit_parameters = fit_result.params.copy()
            for parameter_name in amplitude_names:
                if parameter_name != amplitude_name:
                    fit_parameters[parameter_name].value = 0
            axis.plot(xx, fit_result.eval(fit_parameters, x=xx),
                      color=next(cycle)['color'], linestyle='--',
                      label=amplitude_name)

        # make legend
        if legend:
            axis.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
            plt.tight_layout()

        # add text
        if text:
            plot_text(axis, self.flag, color)

        return axis

    def has_good_solution(self):
        # no fit
        if self.best_fit_result is None:
            return False

        # solver failed
        success = self.best_fit_result.success
        if not success:
            return success

        # bad fit to data
        p = self.best_fit_result.params.valuesdict()
        # p_value = chi2.sf(*self.chi2())
        chi_squared, df = self.chi2()
        high_chi2 = chi_squared / df > 30
        no_errors = not self.best_fit_result.errorbars
        max_phase = np.min([-10, np.max(self.x) * 1.2])
        min_phase = np.min(self.x)
        out_of_bounds_peak = (p['signal_center'] > max_phase or
                              p['signal_center'] < min_phase)
        large_sigma = 2 * p['signal_sigma'] > np.max(self.x) - np.min(self.x)
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
        p = self.best_fit_result.params.valuesdict()
        center = p['signal_center']
        sigma = p['signal_sigma']
        left = center - 2 * sigma
        right = center + 2 * sigma
        x = self.x[self.y != 0]
        logic = np.logical_and(x > left, x < right)
        chi_squared = np.sum(self.best_fit_result.residual[logic]**2)
        df = np.sum(logic) - self.best_fit_result.nvarys
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
        success = super(__class__, self).has_good_solution()
        if not success:
            return success
        p = self.best_fit_result.params.valuesdict()

        g = p['signal_amplitude'] * gaussian(p['signal_center'], p['signal_center'],
                                             p['signal_sigma'])
        e = p['trigger_amplitude'] * exponential(p['signal_center'], p['trigger_tail'])
        swamped_peak = g < 2 * e
        success = not swamped_peak
        return success

    def guess(self, index=0):
        parameters = lm.Parameters()
        if index == 0:
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
        parameters.add('signal_center', value=signal_center, min=-np.inf,
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
        success = super(__class__, self).has_good_solution()
        if not success:
            return success
        p = self.best_fit_result.params.valuesdict()
        # switch center and background if fit backwards
        if p['signal_center'] > p['background_center']:
            success = switch_centers(self)
            if not success:
                return success

        g = p['signal_amplitude'] * gaussian(p['signal_center'], p['signal_center'],
                                             p['signal_sigma'])
        g2 = p['background_amplitude'] * gaussian(p['signal_center'],
                                                  p['background_center'],
                                                  p['background_sigma'])
        swamped_peak = g < 2 * g2
        large_background_sigma = (p['background_sigma'] >
                                  (np.max(self.x) - np.min(self.x)) / 4)
        success = not (swamped_peak or large_background_sigma)

        return success

    def guess(self, index=0):
        parameters = lm.Parameters()
        if index == 0:
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
        parameters.add('signal_center', value=signal_center, min=-np.inf,
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
        success = super(__class__, self).has_good_solution()
        if not success:
            return success
        p = self.best_fit_result.params.valuesdict()
        # switch center and background if fit backwards
        if p['signal_center'] > p['background_center']:
            success = switch_centers(self)
            if not success:
                return success

        g = p['signal_amplitude'] * gaussian(p['signal_center'], p['signal_center'],
                                             p['signal_sigma'])
        e = p['trigger_amplitude'] * exponential(p['signal_center'], p['trigger_tail'])
        g2 = p['background_amplitude'] * gaussian(p['signal_center'],
                                                  p['background_center'],
                                                  p['background_sigma'])
        swamped_peak = g < 2 * (g2 + e)
        large_background_sigma = (p['background_sigma'] >
                                  (np.max(self.x) - np.min(self.x)) / 4)
        success = not (swamped_peak or large_background_sigma)

        return success

    def guess(self, index=0):
        parameters = lm.Parameters()
        if index == 0:
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
        parameters.add('signal_center', value=signal_center, min=-np.inf,
                       max=np.max(self.x))
        parameters.add('signal_sigma', value=signal_sigma, min=sigma_min, max=np.inf)
        parameters.add('background_center', value=background_center, min=-np.inf, max=0)
        parameters.add('background_sigma', value=background_sigma,
                       min=background_sigma_min, max=np.inf)
        parameters.add('trigger_tail', value=trigger_tail, min=0, max=np.inf)
        return parameters


class SkewedGaussianAndGaussianExponential(PartialLinearModel):
    """Gaussian signal plus gaussian background"""
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
        success = super(__class__, self).has_good_solution()
        if not success:
            return success
        p = self.best_fit_result.params.valuesdict()
        # switch center and background if fit backwards
        if p['signal_center'] > p['background_center']:
            success = switch_centers(self)
            if not success:
                return success

        g = p['signal_amplitude'] * gaussian(p['signal_center'], p['signal_center'],
                                             p['signal_sigma'])
        e = p['trigger_amplitude'] * exponential(p['signal_center'], p['trigger_tail'])
        g2 = p['background_amplitude'] * gaussian(p['signal_center'],
                                                  p['background_center'],
                                                  p['background_sigma'])
        swamped_peak = g < 2 * (g2 + e)
        large_background_sigma = (p['background_sigma'] >
                                  (np.max(self.x) - np.min(self.x)) / 4)
        success = not (swamped_peak or large_background_sigma)

        return success

    def guess(self, index=0):
        parameters = lm.Parameters()
        if index == 0:
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
        parameters.add('signal_center', value=signal_center, min=-np.inf,
                       max=np.max(self.x))
        parameters.add('signal_sigma', value=signal_sigma, min=sigma_min, max=np.inf)
        parameters.add('signal_gamma', value=signal_gamma, min=0)
        parameters.add('background_center', value=background_center, min=-np.inf, max=0)
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
    def __init__(self, pixel=None, res_id=None):
        self.pixel = pixel
        self.res_id = res_id
        self.initial_guess = None
        self.used_last_fit = None
        self.best_fit_result = None
        self.best_fit_result_guess = None
        self.fit_result = None
        self.x = None
        self.y = None
        self.variance = None
        self.flag = None

    def fit(self, guess):
        self._check_data()
        self.initial_guess = guess.copy()
        scale = True
        variance = self.variance
        if self.variance is None:
            variance = np.ones(y.shape)
            scale = False
        arguments = (self.x, self.y, variance, self.fit_function, self.dfdx)
        self.fit_result = lm.minimize(self.chi_squared, guess, args=arguments,
                                      scale_covar=scale)

        # save the data in best_fit_result if it is better than previous fits
        self.used_last_fit = (self.best_fit_result is None or
                              self.fit_result.chisqr < self.best_fit_result.chisqr)
        if self.used_last_fit:
            self.best_fit_result = self.fit_result
            self.best_fit_result_guess = self.initial_guess

    @staticmethod
    def chi_squared(parameters, x, y, variance, f, dfdx):
        return (f(x, parameters) - y) / (dfdx(x, parameters) * np.sqrt(variance))

    def plot(self, axis=None, legend=True, title=True, x_label=True, y_label=True,
             best_fit=True, text=True):
        # set up plot basics
        if axis is None:
            fig, axis = plt.subplots()
        if self.has_good_solution():
            color = "green"
            label = "fit accepted"
        else:
            color = "red"
            label = "fit rejected"
        if x_label:
            axis.set_xlabel('phase [degrees]')
        if y_label:
            axis.set_ylabel('energy [eV]')
        if title:
            axis.set_title(("Model '{}'" + os.linesep + "Pixel {} : ResID {}")
                           .format(type(self).__name__, self.pixel, self.res_id))

        # no data
        if self.x is None or self.y is None:
            if text:
                plot_text(axis, self.flag, color)
            return axis

        # plot data
        axis.errorbar(self.x, self.y, xerr=np.sqrt(self.variance), linestyle='--',
                      marker='o', markersize=5, markeredgecolor='black',
                      markeredgewidth=0.5, ecolor='black', capsize=3, elinewidth=0.5)

        # no fit
        if self.best_fit_result is None:
            if text:
                plot_text(axis, self.flag, color)
            return axis

        # choose fit to plot
        if best_fit:
            fit_result = self.best_fit_result
        else:
            fit_result = self.fit_result

        # plot fit
        y_limit = [0.9 * min(self.y), max(self.y) * 1.1]
        axis.set_ylim(y_limit)
        x_limit = [1.05 * min(self.x - np.sqrt(self.variance)),
                   0.95 * max(self.x + np.sqrt(self.variance))]
        axis.set_xlim(x_limit)
        xx = np.linspace(x_limit[0], x_limit[1], 1000)
        axis.plot(xx, self.fit_function(xx, fit_result.params), color=color, label=label)

        if text:
            plot_text(axis, self.flag, color)
        if legend:
            axis.legend(loc="lower left")



    def has_good_solution(self):
        return self.fit_result.success

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
        poly = np.polyfit(self.x, self.y, 2)
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
        poly = np.polyfit(self.x, self.y, 1)
        parameters = lm.Parameters()
        parameters.add('c0', value=poly[1])
        parameters.add('c1', value=poly[0])
        return parameters
