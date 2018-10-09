#!/bin/env python3
import os
import ast
import sys
import atexit
import argparse
import numpy as np
import multiprocessing as mp
from datetime import datetime
from astropy.constants import c, h
from six.moves.configparser import ConfigParser
from progressbar import Bar, ETA, Percentage, ProgressBar, Timer

from mkidpipeline.hdf import bin2hdf
import mkidcore.corelog as pipelinelog
from mkidpipeline.hdf.darkObsFile import ObsFile
import mkidpipeline.calibration.wavecal_models as models

log = pipelinelog.getLogger('mkidpipeline.calibration.wavecal', setup=False)


class Configuration(object):
    """Configuration class for the wavelength calibration analysis."""
    def __init__(self, configuration_path, solution_name='solution.npz'):
        # parse arguments
        self.solution_name = solution_name
        self.configuration_path = configuration_path
        assert os.path.isfile(self.configuration_path), \
            self.configuration_path + " is not a valid configuration file"

        # load in the configuration file
        self.config = ConfigParser()
        self.config.read(self.configuration_path)

        # check the configuration file format and load the parameters
        self.check_sections()

        # load in the parameters
        self.x_pixels = ast.literal_eval(self.config['Data']['x_pixels'])
        self.y_pixels = ast.literal_eval(self.config['Data']['y_pixels'])
        self.bin_directory = ast.literal_eval(self.config['Data']['bin_directory'])
        self.start_times = ast.literal_eval(self.config['Data']['start_times'])
        self.exposure_times = ast.literal_eval(self.config['Data']['exposure_times'])
        self.beam_map_path = ast.literal_eval(self.config['Data']['beam_map_path'])
        self.h5_directory = ast.literal_eval(self.config['Data']['h5_directory'])
        self.h5_file_names = ast.literal_eval(self.config['Data']['h5_file_names'])
        self.wavelengths = ast.literal_eval(self.config['Data']['wavelengths'])
        self.histogram_model_names = ast.literal_eval(
            self.config['Fit']['histogram_model_names'])
        self.bin_width = ast.literal_eval(self.config['Fit']['bin_width'])
        self.histogram_fit_attempts = ast.literal_eval(
            self.config['Fit']['histogram_fit_attempts'])
        self.calibration_model_names = ast.literal_eval(
            self.config['Fit']['calibration_model_names'])
        self.dt = ast.literal_eval(self.config['Fit']['dt'])
        self.parallel = ast.literal_eval(self.config['Fit']['parallel'])
        self.out_directory = ast.literal_eval(self.config['Output']['out_directory'])
        self.summary_plot = ast.literal_eval(self.config['Output']['summary_plot'])
        self.templar_configuration_path = ast.literal_eval(
            self.config['Output']['templar_configuration_path'])
        self.verbose = ast.literal_eval(self.config['Output']['verbose'])
        self.logging = ast.literal_eval(self.config['Output']['logging'])

        # check the parameter formats
        self.check_parameters()
        
        # enforce consistency between h5 and bin file start times
        self._config_changed = False
        self.enforce_consistency()

        # write new config file if enforce_consistency() updated any parameters
        if self._config_changed:
            while True:
                if os.path.isfile(self.configuration_path):
                    directory = os.path.dirname(self.configuration_path)
                    base_name = "".join(
                        os.path.basename(self.configuration_path).split(".")[:-1])
                    suffix = str(os.path.basename(self.configuration_path).split(".")[-1])
                    self.configuration_path = os.path.join(directory,
                                                           base_name + "_new." + suffix)
                else:
                    break
            self.write(self.configuration_path)

    def check_sections(self):
        """Check if all sections and parameters exist in the configuration file."""
        section = "'{0}' must be a configuration file section"
        param = "'{0}' must be a parameter in the '{1}' section of the configuration file"

        assert 'Data' in self.config.sections(), section.format('Data')
        assert 'x_pixels' in self.config['Data'].keys(), \
            param.format('x_pixels', 'Data')
        assert 'y_pixels' in self.config['Data'].keys(), \
            param.format('y_pixels', 'Data')
        assert 'bin_directory' in self.config['Data'].keys(), \
            param.format('bin_directory', 'Data')
        assert 'start_times' in self.config['Data'].keys(), \
            param.format('start_times', 'Data')
        assert 'exposure_times' in self.config['Data'].keys(), \
            param.format('exposure_times', 'Data')
        assert 'beam_map_path' in self.config['Data'].keys(), \
            param.format('beam_map_path', 'Data')
        assert 'h5_directory' in self.config['Data'].keys(), \
            param.format('h5_directory', 'Data')
        assert 'h5_file_names' in self.config['Data'].keys(), \
            param.format('h5_file_names', 'Data')
        assert 'wavelengths' in self.config['Data'].keys(), \
            param.format('wavelengths', 'Data')

        assert 'Fit' in self.config.sections(), section.format('Fit')
        assert 'histogram_model_names' in self.config['Fit'].keys(), \
            param.format('histogram_model_names', 'Fit')
        assert 'bin_width' in self.config['Fit'].keys(), \
            param.format('bin_width', 'Fit')
        assert 'histogram_fit_attempts' in self.config['Fit'].keys(), \
            param.format('histogram_fit_attempts', 'Fit')
        assert 'calibration_model_names' in self.config['Fit'].keys(), \
            param.format('calibration_model_names', 'Fit')
        assert 'dt' in self.config['Fit'].keys(), \
            param.format('dt', 'Fit')
        assert 'parallel' in self.config['Fit'].keys(), \
            param.format('parallel', 'Fit')

        assert 'Output' in self.config.sections(), section.format('Output')
        assert 'out_directory' in self.config['Output'], \
            param.format('out_directory', 'Output')
        assert 'summary_plot' in self.config['Output'], \
            param.format('summary_plot', 'Output')
        assert 'templar_configuration_path' in self.config['Output'], \
            param.format('templar_configuration_path', 'Output')
        assert 'verbose' in self.config['Output'], \
            param.format('verbose', 'Output')
        assert 'logging' in self.config['Output'], \
            param.format('logging', 'Output')

    def check_parameters(self):
        """Type check configuration file parameters."""
        assert type(self.x_pixels) is int, "x_pixels parameter must be an integer"
        assert type(self.y_pixels) is int, "y_pixels parameter must be an integer"
        assert os.path.isdir(self.bin_directory),\
            "bin_directory parameter must be a string and a valid directory"
        message = "start_times parameter must be a list of integers."
        assert type(self.start_times) is list, message
        for st in self.start_times:
            assert type(st) is int, message
        message = "exposure_times parameter must be a list of integers"
        assert type(self.exposure_times) is list, message
        for et in self.exposure_times:
            assert type(et) is int, message
        assert os.path.isfile(self.beam_map_path),\
            "beam_map_path parameter must be a string and a valid path to a file"
        assert os.path.isdir(self.h5_directory), \
            "h5_directory parameter must be a string and a valid directory"
        message = "h5_file_names parameter must be a list of strings or None."
        assert isinstance(self.h5_file_names, (list, type(None))), message
        if isinstance(self.h5_file_names, list):
            for name in self.h5_file_names:
                assert isinstance(name, str), message
        message = "wavelengths parameter must be a list of numbers"
        assert isinstance(self.wavelengths, (list, np.ndarray)), message
        try:
            self.wavelengths = np.array([float(wavelength)
                                         for wavelength in self.wavelengths])
        except ValueError:
            raise AssertionError(message)
        message = ("histogram_model_names parameter must be a list of subclasses of "
                   "PartialLinearModel and be in wavecal_models.py")
        assert isinstance(self.histogram_model_names, list), message
        for model in self.histogram_model_names:
            assert issubclass(getattr(models, model), models.PartialLinearModel), message
        try:
            self.bin_width = float(self.bin_width)
        except ValueError:
            raise AssertionError("bin_width parameter must be an integer or float")
        assert isinstance(self.histogram_fit_attempts, int), \
            "histogram_fit_attempts parameter must be an integer"
        message = ("calibration_model_names parameter must be a list of subclasses of "
                   "XErrorsModel and be in wavecal_models.py")
        assert isinstance(self.calibration_model_names, list), message
        for model in self.calibration_model_names:
            assert issubclass(getattr(models, model), models.XErrorsModel), message
        try:
            self.dt = float(self.dt)
        except ValueError:
            raise AssertionError("dt parameter must be an integer or float")
        assert type(self.parallel) is bool, "parallel parameter must be a boolean"
        assert os.path.isdir(self.out_directory), \
            "out_directory parameter must be a string and a valid directory"
        assert type(self.summary_plot) is bool, "summary_plot parameter must be a boolean"
        assert os.path.isfile(self.templar_configuration_path),\
            "templar_configuration_path parameter must be a string and a valid file path"
        assert type(self.verbose) is bool, "verbose parameter bust be a boolean"
        assert type(self.logging) is bool, "logging parameter must be a boolean"

    def hdf_exist(self):
        file_paths = [os.path.join(self.h5_directory, file_)
                      for file_ in self.h5_file_names]
        return all(map(os.path.isfile, file_paths))

    def _compute_hdf_names(self):
        return ['%d' % st + '.h5' for st in self.start_times]

    def _sort_wavelengths(self):
        indices = np.argsort(self.wavelengths)
        self.wavelengths = list(np.array(self.wavelengths)[indices])
        self.exposure_times = list(np.array(self.exposure_times)[indices])
        self.start_times = list(np.array(self.start_times)[indices])
        self.h5_file_names = list(np.array(self.h5_file_names)[indices])

    def enforce_consistency(self):
        # check to see if h5 files were specified and compute their names otherwise
        if self.h5_file_names is None:
            self._config_changed = True
            self.h5_file_names = self._compute_hdf_names()
        # check that wavelengths are in ascending order and sort otherwise
        if (sorted(self.wavelengths) != self.wavelengths).all():
            self._config_changed = True
            self._sort_wavelengths()

    def write(self, file_):
        with open(file_, 'w') as f:
            f.write('[Data]' + os.linesep +
                    'x_pixels = {}'.format(self.x_pixels) + os.linesep +
                    'y_pixels = {}'.format(self.y_pixels) + os.linesep +
                    'bin_directory = "{}"'.format(self.bin_directory) + os.linesep +
                    'start_times = {}'.format(self.start_times) + os.linesep +
                    'exposure_times = {}'.format(self.exposure_times) + os.linesep +
                    'beam_map_path = "{}"'.format(self.beam_map_path) + os.linesep +
                    'h5_directory = "{}"'.format(self.h5_directory) + os.linesep +
                    'h5_file_names = {}'.format(self.h5_file_names) + os.linesep +
                    'wavelengths = {}'.format(list(self.wavelengths)) + os.linesep +
                    os.linesep +
                    '[Fit]' + os.linesep +
                    'histogram_model_names = {}'.format(self.histogram_model_names) +
                    os.linesep +
                    'bin_width = {}'.format(self.bin_width) + os.linesep +
                    'histogram_fit_attempts = {}'.format(self.histogram_fit_attempts) +
                    os.linesep +
                    'calibration_model_names = {}'.format(self.calibration_model_names) +
                    os.linesep +
                    'dt = {}'.format(self.dt) + os.linesep +
                    'parallel = {}'.format(self.parallel) + os.linesep +
                    os.linesep +
                    '[Output]' + os.linesep +
                    'out_directory = "{}"'.format(self.out_directory) + os.linesep +
                    'summary_plot = {}'.format(self.summary_plot) + os.linesep +
                    ('templar_configuration_path = "{}"'
                     .format(self.templar_configuration_path)) + os.linesep +
                    'verbose = {}'.format(self.verbose) + os.linesep +
                    'logging = {}'.format(self.logging))


class Calibrator(object):
    def __init__(self, configuration):
        # save configuration
        self.cfg = configuration

        # get beam map
        obs_files = []
        for index, _ in enumerate(self.cfg.wavelengths):
            file_name = os.path.join(self.cfg.h5_directory,
                                     self.cfg.h5_file_names[index])
            obs_files.append(ObsFile(file_name))
            message = "The beam map does not match the array dimensions"
            assert (obs_files[-1].beamImage.shape ==
                    (self.cfg.x_pixels, self.cfg.y_pixels)), message
        beam_map = obs_files[0].beamImage.copy()
        del obs_files

        # initialize fit array
        fit_array = np.empty((self.cfg.x_pixels, self.cfg.y_pixels), dtype=object)
        self.solution = Solution(fit_array=fit_array, configuration=self.cfg,
                                 beam_map=beam_map)
        self.progress = None
        self.progress_iteration = None

    def run(self, pixels=None, wavelengths=None, verbose=True, parallel=True, save=True,
            plot=True):
        pixels = self._parse_pixels(pixels)
        try:
            log.info("Computing phase histograms")
            self._run("make_phase_histogram", pixels=pixels, wavelengths=wavelengths,
                      parallel=parallel, verbose=verbose, h5_safe=True)
            log.info("Fitting phase histograms")
            self._run("fit_phase_histogram", pixels=pixels, wavelengths=wavelengths,
                      parallel=parallel, verbose=verbose)
            log.info("Fitting phase-energy calibration")
            self._run("fit_phase_energy_curve", pixels=pixels, wavelengths=wavelengths,
                      parallel=parallel, verbose=verbose)

            if save:
                log.info("Saving solution")
                self.solution.save()

            if plot:
                log.info("Making summary plot")
                self.solution.plot_summary()
        except KeyboardInterrupt:
            log.info(os.linesep + "Keyboard shutdown requested ... exiting")

    def make_phase_histogram(self, pixels=None, wavelengths=None, verbose=True):
        # check inputs and setup progress bar
        pixels, wavelength_indices, wavelengths = self._setup(pixels, wavelengths,
                                                              verbose)
        # make ObsFiles
        obs_files = []
        for index in wavelength_indices:
            file_name = os.path.join(self.cfg.h5_directory, self.cfg.h5_file_names[index])
            obs_files.append(ObsFile(file_name))
        # make histograms for each pixel in pixels and wavelength in wavelengths
        for pixel in pixels:
            # update progress bar
            self._update_progress(pixels, verbose=verbose)
            # histogram get models
            x_ind, y_ind = pixel[0], pixel[1]
            histogram_models = self.solution[x_ind, y_ind]['histogram']
            for index, wavelength in enumerate(wavelengths):
                model = histogram_models[wavelength_indices[index]]
                # flag the model as modified from the default conditions
                # used for parallel computation
                model.modified = True
                # load the data
                photon_list = obs_files[index].getPixelPhotonList(x_ind, y_ind)
                if photon_list.size == 0:
                    model.flag = 1
                    message = "({}, {}) : {} nm : there are no photons"
                    log.debug(message.format(x_ind, y_ind, wavelength))
                    continue
                # remove hot pixels
                rate = (len(photon_list['Wavelength']) /
                        (max(photon_list['Time']) - min(photon_list['Time']))) * 1e6
                if rate > 2000:
                    model.flag = 2
                    message = ("({}, {}) : {} nm : removed for being too hot "
                               "({:.2f} > 2000 cps)")
                    log.debug(message.format(x_ind, y_ind, wavelength, rate))
                    continue
                # remove photons too close together in time
                photon_list = self._remove_tail_riding_photons(photon_list)
                if photon_list.size == 0:
                    model.flag = 3
                    message = ("({}, {}) : {} nm : all the photons were removed after "
                               "the arrival time cut")
                    log.debug(message.format(x_ind, y_ind, wavelength))
                    continue
                # remove photons with positive peak heights
                phase_list = photon_list['Wavelength']
                phase_list = phase_list[phase_list < 0]
                if phase_list.size == 0:
                    model.flag = 4
                    message = ("({}, {}) : {} nm : all the photons were removed after "
                               "the negative phase only cut")
                    log.debug(message.format(x_ind, y_ind, wavelength))
                    continue
                # make histogram
                centers, counts = self._histogram(phase_list)
                # assign x, y and variance data to the fit model
                model.x = centers
                model.y = counts
                # gaussian mle for the variance of poisson distributed data
                # https://doi.org/10.1016/S0168-9002(00)00756-7
                model.variance = np.sqrt(counts**2 + 0.25) - 0.5
                message = "({}, {}) : {} nm : histogram successfully computed"
                log.debug(message.format(x_ind, y_ind, wavelength))
        # update progress bar
        self._update_progress(pixels, finish=True, verbose=verbose)

        # close all observation files
        for obs_file in obs_files:
            obs_file.file.close()

    def fit_phase_histogram(self, pixels=None, wavelengths=None, verbose=True):
        # check inputs and setup progress bar
        pixels, wavelength_indices, wavelengths = self._setup(pixels, wavelengths,
                                                              verbose)
        # fit histograms for each pixel in pixels and wavelength in wavelengths
        for pixel in pixels:
            # update progress bar
            self._update_progress(pixels, verbose=verbose)
            x_ind, y_ind = pixel[0], pixel[1]
            model_list = self.solution[x_ind, y_ind]['histogram']
            # fit the histograms of the higher energy data sets first and use good
            # fits to inform the guesses to the lower energy data sets
            for index, wavelength in enumerate(wavelengths):
                model = model_list[wavelength_indices[index]]
                # flag the model as modified from the default conditions
                # used for parallel computation
                model.modified = True
                if model.x is None or model.y is None:
                    message = ("({}, {}) : {} nm : histogram fit failed because there is "
                               "no data")
                    log.debug(message.format(x_ind, y_ind, wavelength))
                    continue
                message = "({}, {}) : {} nm : beginning histogram fitting"
                log.debug(message.format(x_ind, y_ind, wavelength))
                # try models in order specified in the config file
                tried_models = []
                for histogram_model in self.solution.histogram_model_list:
                    # update the model if needed
                    if not isinstance(model, histogram_model):
                        model = self._update_histogram_model(wavelength_indices[index],
                                                             histogram_model, pixel)
                    # if there are any good fits intelligently guess the signal_center
                    # parameter and set the other parameters equal to the average of those
                    # in the good fits
                    good_solutions = [model.has_good_solution() for model in model_list]
                    if np.any(good_solutions):
                        guess = self._guess(pixel, wavelength_indices[index],
                                            good_solutions)
                        model.fit(guess)
                        # if the fit worked continue with the next wavelength
                        if model.has_good_solution():
                            tried_models.append(model.copy())
                            message = ("({}, {}) : {} nm : histogram fit successful with "
                                       "computed guess and model '{}'")
                            log.debug(message.format(x_ind, y_ind, wavelength,
                                                     type(model).__name__))
                            continue
                    # try a guess based on the model if the intelligent guess didn't work
                    for fit_index in range(self.cfg.histogram_fit_attempts):
                        guess = model.guess(fit_index)
                        model.fit(guess)
                        if model.has_good_solution():
                            tried_models.append(model.copy())
                            message = ("({}, {}) : {} nm : histogram fit successful with "
                                       "guess number {} and model '{}'")
                            log.debug(message.format(x_ind, y_ind, wavelength, fit_index,
                                                     type(model).__name__))
                            break
                    else:
                        # trying next model since no good fit was found
                        tried_models.append(model.copy())
                        continue
                # find model with the best fit and save that one
                self._assign_best_histogram_model(model_list, tried_models,
                                                  wavelength_indices[index], pixel)

            # recheck fits that didn't work with better guesses if there exist
            # lower energy fits that did work
            good_solutions = [model.has_good_solution() for model in model_list]
            for index, wavelength in enumerate(wavelengths):
                model = model_list[wavelength_indices[index]]
                if model.x is None or model.y is None:
                    continue
                if model.has_good_solution():
                    continue
                if np.any(good_solutions[wavelength_indices[index] + 1:]):
                    tried_models = []
                    for histogram_model in self.solution.histogram_model_list:
                        if not isinstance(model, histogram_model):
                            model = self._update_histogram_model(
                                wavelength_indices[index], histogram_model, pixel)
                        guess = self._guess(pixel, wavelength_indices[index],
                                            good_solutions)
                        model.fit(guess)
                        if model.has_good_solution():
                            message = ("({}, {}) : {} nm : histogram fit recomputed and "
                                       "successful with model '{}'")
                            log.debug(message.format(x_ind, y_ind, wavelength,
                                                     type(model).__name__))
                        tried_models.append(model.copy())
                    else:
                        # find the model with the best bad fit and save that one
                        self._assign_best_histogram_model(
                            model_list, tried_models, wavelength_indices[index], pixel)
        # update progress bar
        self._update_progress(pixels, finish=True, verbose=verbose)

    def fit_phase_energy_curve(self, pixels=None, wavelengths=None, verbose=True):
        # check inputs and setup progress bar
        pixels, wavelength_indices, wavelengths = self._setup(pixels, wavelengths,
                                                              verbose)
        for pixel in pixels:
            # update progress bar
            self._update_progress(pixels, verbose=verbose)
            x_ind, y_ind = pixel[0], pixel[1]
            model = self.solution[x_ind, y_ind]['calibration']
            # get data from histogram fits
            histogram_models = self.solution[x_ind, y_ind]['histogram']
            phases, variance, energies = [], [], []
            for index, wavelength in enumerate(wavelengths):
                histogram_model = histogram_models[wavelength_indices[index]]
                if histogram_model.has_good_solution:
                    phases.append(histogram_model.signal_center)
                    variance.append(histogram_model.signal_center_standard_error**2)
                    energies.append(h.to('eV s').value * c.to('nm/s').value /
                                    wavelength)
            # give data to model
            if variance:
                model.x = np.array(phases)
                model.y = np.array(energies)
                model.variance = np.array(variance)

            # don't fit if there's not enough data
            if len(variance) < 3:
                model.flag = 11
                message = "({}, {}) : {} data points is not enough to make a calibration"
                log.debug(message.format(x_ind, y_ind, len(variance)))
                continue

            diff = np.diff(phases)
            sigma = np.sqrt(variance)
            if (diff < -4 * (sigma[:-1] + sigma[1:])).any():
                model.flag = 12
                message = ("({}, {}) : fitted phase values are not monotonic enough "
                           "to make a calibration")
                log.debug(message.format(x_ind, y_ind))

            # fit the data
            message = "({}, {}) : beginning phase-energy calibration fitting"
            log.debug(message.format(x_ind, y_ind))
            tried_models = []
            for calibration_model in self.solution.calibration_model_list:
                # update the model if needed
                if not isinstance(model, calibration_model):
                    model = self._update_calibration_model(calibration_model, pixel)
                guess = model.guess()
                model.fit(guess)
                tried_models.append(model.copy())
                if model.has_good_solution():
                    message = ("({}, {}) : phase-energy calibration fit successful with "
                               "model '{}'")
                    log.debug(message.format(x_ind, y_ind, type(model).__name__))

            # find model with the best fit and save that one
            self._assign_best_calibration_model(tried_models, pixel)
        # update progress bar
        self._update_progress(pixels, finish=True, verbose=verbose)

    def _run(self, method, pixels=(), wavelengths=None, verbose=True, parallel=True,
             h5_safe=False):
        if parallel:
            self._parallel(method, pixels=pixels, wavelengths=wavelengths,
                           verbose=verbose, h5_safe=h5_safe)
        else:
            getattr(self, method)(pixels=pixels, wavelengths=wavelengths, verbose=verbose)

    def _parallel(self, method, pixels=None, wavelengths=None, verbose=True,
                  h5_safe=False):
        # set up progress bar
        self._update_progress(pixels, initialize=True, verbose=verbose)

        # configure number of processes
        worker_list = []
        if h5_safe:
            cpu_count = np.min([len(self.cfg.wavelengths), np.ceil(mp.cpu_count() / 2)])
            cpu_count = cpu_count.astype(int)
        else:
            cpu_count = np.ceil(mp.cpu_count() / 2).astype(int)

        # make input and output queues
        output_queue = mp.Queue()
        input_queues = []
        if h5_safe:
            for _ in range(cpu_count):
                input_queues.append(mp.Queue())
        else:
            input_queues.append(mp.Queue())
        queue_length = len(input_queues)

        # make workers to process the data
        for index in range(cpu_count):
            worker_list.append(Worker(self.cfg, method,
                                      input_queues[index % queue_length], output_queue))

        # assign data to workers
        if h5_safe:
            fit_array_list = []
            for pixel in pixels:
                fit_array_list.append(self.solution[pixel[0], pixel[1]])
            for wavelength_index, wavelength in enumerate(wavelengths):
                kwargs = {"pixels": pixels, "wavelengths": wavelength,
                          "fit_array_list": fit_array_list}
                input_queues[wavelength_index % queue_length].put(kwargs)

            for queue in input_queues:
                queue.put({"stop": True})


        else:
            for index, pixel in pixels:
                fit_array_list = [self.solution[pixel[0], pixel[1]]]
                kwargs = {"pixels": pixels, "wavelengths": wavelengths,
                          "fit_array_list": fit_array_list}
                input_queues[0].put(kwargs)

        # collect data from workers
        pass

    def _parse_pixels(self, pixels):
        if pixels is not None:
            pixels = np.atleast_2d(np.array(pixels))
            if not np.issubdtype(pixels.dtype, np.integer) or pixels.shape[1] != 2:
                raise ValueError("pixels must be a list of pairs of integers")
        else:
            x_pixels = range(self.cfg.x_pixels)
            y_pixels = range(self.cfg.y_pixels)
            pixels = np.array([[x, y] for x in x_pixels for y in y_pixels])
        return pixels

    def _parse_wavelengths(self, wavelengths):
        if wavelengths is None:
            wavelengths = self.cfg.wavelengths
        elif not isinstance(wavelengths, (list, tuple, np.ndarray)):
            wavelengths = [wavelengths]
        wavelength_indices = []
        for wavelength in wavelengths:
            if wavelength in self.cfg.wavelengths:
                index = np.where(wavelength == self.cfg.wavelengths)[0][0]
                wavelength_indices.append(index)
            else:
                raise ValueError('{} nm is not a valid wavelength'.format(wavelength))
        wavelength_indices = np.array(wavelength_indices)
        wavelengths = np.array(wavelengths)
        return wavelength_indices, wavelengths

    def _remove_tail_riding_photons(self, photon_list):
        indices = np.argsort(photon_list['Time'])
        photon_list = photon_list[indices]

        indices = np.where(np.diff(photon_list['Time']) > self.cfg.dt)[0] + 1
        photon_list = photon_list[indices]
        return photon_list

    def _histogram(self, phase_list):
        # initialize variables
        min_phase = np.min(phase_list)
        max_phase = np.max(phase_list)
        max_count = 0
        update = 0
        centers = None
        counts = None
        # make histogram
        while max_count < 400 and update < 3:
            # update bin_width
            bin_width = self.cfg.bin_width * (2 ** update)

            # define bin edges being careful to start at the threshold cut
            bin_edges = np.arange(max_phase, min_phase - bin_width,
                                  -bin_width)[::-1]

            # make histogram
            counts, x0 = np.histogram(phase_list, bins=bin_edges)
            centers = (x0[:-1] + x0[1:]) / 2.0

            # update counters
            max_count = np.max(counts)
            update += 1

        return centers, counts

    def _update_histogram_model(self, wavelength_index, histogram_model, pixel):
        model_list = self.solution[pixel[0], pixel[1]]['histogram']
        # save old data
        x = model_list[wavelength_index].x
        y = model_list[wavelength_index].y
        variance = model_list[wavelength_index].variance
        pixel = model_list[wavelength_index].pixel
        res_id = model_list[wavelength_index].res_id
        # swap model
        model_list[wavelength_index] = histogram_model(pixel=pixel, res_id=res_id)
        # set new data
        model_list[wavelength_index].x = x
        model_list[wavelength_index].y = y
        model_list[wavelength_index].variance = variance
        return model_list[wavelength_index]

    def _update_calibration_model(self, calibration_model, pixel):
        # save old data
        x = self.solution[pixel[0], pixel[1]]['calibration'].x
        y = self.solution[pixel[0], pixel[1]]['calibration'].y
        variance = self.solution[pixel[0], pixel[1]]['calibration'].variance
        pixel = self.solution[pixel[0], pixel[1]]['calibration'].pixel
        res_id = self.solution[pixel[0], pixel[1]]['calibration'].res_id
        # swap model
        self.solution[pixel[0], pixel[1]]['calibration'] = calibration_model(
            pixel=pixel, res_id=res_id)
        # set new data
        self.solution[pixel[0], pixel[1]]['calibration'].x = x
        self.solution[pixel[0], pixel[1]]['calibration'].y = y
        self.solution[pixel[0], pixel[1]]['calibration'].variance = variance
        return self.solution[pixel[0], pixel[1]]['calibration']

    def _guess(self, pixel, wavelength_index, good_solutions):
        """If there are any good fits for this pixel intelligently guess the
        signal_center parameter and set the other parameters equal to the average of
        those in the good fits"""
        # get initial guess
        histogram_models = self.solution[pixel[0], pixel[1]]['histogram']
        wavelengths = self.cfg.wavelengths
        model = histogram_models[wavelength_index]
        guess = model.guess()
        # get index of closest shorter wavelength good solution
        shorter_index = None
        for index, good in enumerate(good_solutions[:wavelength_index]):
            if good:
                shorter_index = index
        # get index of closest longer wavelength good solution
        longer_index = None
        for index, good in enumerate(good_solutions[wavelength_index + 1:]):
            if good:
                longer_index = wavelength_index + 1 + index
                break
        # get data from shorter fit
        if shorter_index is not None:
            shorter_model = histogram_models[shorter_index]
            shorter_params = shorter_model.best_fit_result.params
            shorter_center = (shorter_params['signal_center'].value *
                              wavelengths[shorter_index] / wavelengths[wavelength_index])
            shorter_guesses = {}
            if isinstance(shorter_model, type(model)):
                for parameter in shorter_params.values():
                    if parameter.name != "signal_center":
                        shorter_guesses.update({parameter.name: parameter.value})
        else:
            shorter_center = None
            shorter_guesses = {}
        # get data from longer fit
        if longer_index is not None:
            longer_model = histogram_models[longer_index]
            longer_params = longer_model.best_fit_result.params
            longer_center = (longer_params['signal_center'].value *
                             wavelengths[longer_index] / wavelengths[wavelength_index])
            longer_guesses = {}
            if isinstance(longer_model, type(model)):
                for parameter in longer_params.values():
                    if parameter.name != "signal_center":
                        longer_guesses.update({parameter.name: parameter.value})
        else:
            longer_center = None
            longer_guesses = {}
        if shorter_index is None and longer_index is None:
            raise RuntimeError("There were no good solutions to base a fit guess on.")

        # set center parameter
        if shorter_center is not None and longer_center is not None:
            guess['signal_center'].set(value=np.mean([shorter_center, longer_center]))
        elif shorter_center is not None:
            guess['signal_center'].set(value=shorter_center)
        elif longer_center is not None:
            guess['signal_center'].set(value=longer_center)
        # set other parameters
        for parameter in guess.values():
            name = parameter.name
            if name in shorter_guesses.keys() and name in longer_guesses.keys():
                guess[name].set(value=np.mean([longer_guesses[name],
                                               shorter_guesses[name]]))
            elif name in shorter_guesses.keys():
                guess[name].set(value=shorter_guesses[name])
            elif name in longer_guesses.keys():
                guess[name].set(value=longer_guesses[name])

        return guess

    def _assign_best_histogram_model(self, model_list, tried_models, wavelength_index,
                                     pixel):
        wavelength = self.cfg.wavelengths[wavelength_index]
        x_ind, y_ind = pixel[0], pixel[1]
        best_model = tried_models[0]
        lowest_aic_model = tried_models[0]
        for model in tried_models[1:]:
            lower_aic = model.best_fit_result.aic < best_model.best_fit_result.aic
            good_fit = model.has_good_solution()
            if lower_aic and good_fit:
                best_model = model
            if lower_aic:
                lowest_aic_model = model

        if best_model.has_good_solution():
            best_model.flag = 0
            model_list[wavelength_index] = best_model
            message = ("({}, {}) : {} nm : histogram model '{}' chosen as "
                       "the best successful fit")
            log.debug(message.format(x_ind, y_ind, wavelength,
                                     type(best_model).__name__))
        else:
            if not lowest_aic_model.best_fit_result.success:
                lowest_aic_model.flag = 5  # did not converge
            else:
                lowest_aic_model.flag = 6  # converged but failed validation
            model_list[wavelength_index] = lowest_aic_model
            message = ("({}, {}) : {} nm : histogram fit failed with all "
                       "models")
            log.debug(message.format(x_ind, y_ind, wavelength))

    def _assign_best_calibration_model(self, tried_models, pixel):
        x_ind, y_ind = pixel[0], pixel[1]
        best_model = tried_models[0]
        lowest_aic_model = tried_models[0]
        for model in tried_models[1:]:
            lower_aic = model.best_fit_result.aic < best_model.best_fit_result.aic
            good_fit = model.has_good_solution()
            if lower_aic and good_fit:
                best_model = model
            if lower_aic:
                lowest_aic_model = model
        if best_model.has_good_solution():
            best_model.flag = 10
            self.solution[x_ind, y_ind]['calibration'] = best_model
            message = ("({}, {}) : energy-phase calibration model '{}' chosen as "
                       "the best successful fit")
            log.debug(message.format(x_ind, y_ind, type(best_model).__name__))
        else:
            if not lowest_aic_model.best_fit_result.success:
                lowest_aic_model.flag = 13  # did not converge
            else:
                lowest_aic_model.flag = 14  # converged but failed validation
            self.solution[x_ind, y_ind]['calibration'] = lowest_aic_model
            message = "({}, {}) : energy-phase calibration fit failed with all models"
            log.debug(message.format(x_ind, y_ind))

    def _update_progress(self, pixels, initialize=False, finish=False, verbose=True):
        if verbose:
            if initialize:
                self.progress = ProgressBar(widgets=[Percentage(), Bar(), '  (', Timer(),
                                                     ') ', ETA(), ' '],
                                            max_value=len(pixels)).start()
                self.progress_iteration = -1
            elif finish:
                self.progress_iteration += 1
                self.progress.update(self.progress_iteration)
                self.progress.finish()
            else:
                self.progress_iteration += 1
                self.progress.update(self.progress_iteration)

    def _setup(self, pixels, wavelengths, verbose):
        # check inputs
        pixels = self._parse_pixels(pixels)
        wavelength_indices, wavelengths = self._parse_wavelengths(wavelengths)
        # set up progress bar
        self._update_progress(pixels, initialize=True, verbose=verbose)
        return pixels, wavelength_indices, wavelengths


class Worker(mp.Process):
    """Worker class for running the wavelength calibration in parallel"""
    def __init__(self, configuration, method, input_queue, output_queue):
        super(Worker, self).__init__()
        self.calibrator = Calibrator(configuration)
        self.method = method
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.start()

    def run(self):
        try:
            calibrator = Calibrator(self.cfg)
            while True:
                # get next bit of data to analyze
                kwargs = self.input_queue.get()
                # check for stopping condition
                stop = kwargs.pop("stop", False)
                if stop:
                    break
                # get fit_array_element to overload
                fit_array_list = kwargs.pop("fit_array_list", None)
                pixels = kwargs.pop("pixels")
                wavelengths = kwargs.pop("wavelengths")
                if fit_array_list is not None:
                    for index, pixel in enumerate(pixels):
                        pixel = tuple(pixel)
                        self.calibrator.solution[pixel] = fit_array_list[index]

                getattr(self.calibrator, self.method)(verbose=False, pixels=pixels,
                                                      wavelengths=wavelengths)
                for index, pixel in enumerate(pixels):
                    fit_array_list[index] = self.calibrator.solution[pixel[0], pixel[1]]
                self.output_queue.put({"pixels": pixels, "wavelengths": wavelengths,
                                       "fit_array_list": fit_array_list})

        except KeyboardInterrupt:
            pass


class Solution(object):
    """Solution class for the wavelength calibration. Initialize with either the file_name
    argument or both the fit_array and configuration arguments."""
    def __init__(self, file_path=None, fit_array=None, configuration=None, beam_map=None):
        # load in solution and configuration objects
        if fit_array is not None and configuration is not None and beam_map is not None:
            self._fit_array = fit_array
            self.cfg = configuration
            self.beam_map = beam_map
        elif file_path is not None:
            self.load(file_path)
        else:
            message = ('provide either a file_path or both the fit_array and '
                       'configuration arguments')
            raise ValueError(message)

        # load in fitting models
        self.histogram_model_list = [getattr(models, name) for _, name in
                                     enumerate(self.cfg.histogram_model_names)]
        self.calibration_model_list = [getattr(models, name) for _, name in
                                       enumerate(self.cfg.calibration_model_names)]

    def __getitem__(self, item):
        results = self._fit_array[item]
        if isinstance(results, np.ndarray):
            empty = (results == np.array([None]))
            if empty.any():
                for item, entry in np.ndenumerate(results):
                    if empty[item]:
                        res_id = self.beam_map[item][item]
                        start0 = item[0].start if item[0].start is not None else 0
                        start1 = item[1].start if item[1].start is not None else 0
                        step0 = item[0].step if item[0].step is not None else 1
                        step1 = item[1].step if item[1].step is not None else 1
                        pixel = (item[0] * step0 + start0, item[1] * step1 + start1)
                        histogram_models = [self.histogram_model_list[0](pixel=pixel,
                                                                         res_id=res_id)
                                            for _ in range(len(self.cfg.wavelengths))]
                        calibration_model = self.calibration_model_list[0](pixel=item,
                                                                           res_id=res_id)
                        results[item] = {'histogram': histogram_models,
                                          'calibration': calibration_model}
        else:
            if results is None:
                res_id = self.beam_map[item]
                histogram_models = [self.histogram_model_list[0](pixel=item,
                                                                 res_id=res_id)
                                    for _ in range(len(self.cfg.wavelengths))]
                calibration_model = self.calibration_model_list[0](pixel=item,
                                                                   res_id=res_id)
                self._fit_array[item] = {'histogram': histogram_models,
                                           'calibration': calibration_model}
        return self._fit_array[item]

    def __setitem__(self, key, value):
        self._fit_array[key] = value


    def save(self):
        np.savez(self.cfg.solution_name, fit_array=self._fit_array,
                 configuration=self.cfg, beam_map=self.beam_map)

    def load(self, file_path):
        try:
            npz_file = np.load(file_path)
            self._fit_array = npz_file['fit_array']
            self.cfg = npz_file['configuration'].item()
            self.beam_map = npz_file['beam_map']
        except (OSError, KeyError):
            message = "Failed to interpret '{}' as a wavecal solution object"
            raise OSError(message.format(file_path))


if __name__ == "__main__":
    timestamp = datetime.utcnow().timestamp()

    # read in command line arguments
    parser = argparse.ArgumentParser(description='MKID Wavelength Calibration Utility')
    parser.add_argument('cfg_file', type=str, help='The configuration file')
    parser.add_argument('--vet', action='store_true', dest='vet_only',
                        help='Only verify the configuration file')
    parser.add_argument('--h5', action='store_true', dest='h5_only',
                        help='Only make the h5 files')
    parser.add_argument('--force', action='store_true', dest='force_h5',
                        help='Force h5 file creation')
    parser.add_argument('-nc', type=int, dest='n_cpu', default=0,
                        help="Number of CPUs to use for bin2hdf, " 
                             "default is number of wavelengths")
    parser.add_argument('--quiet', action='store_true', dest='quiet',
                        help='Disable logging')
    args = parser.parse_args()

    # load the configuration file
    config = Configuration(args.cfg_file,
                           solution_name='wavecal_solution_{}.npz'.format(timestamp))
    # set up logging
    if not args.quiet and config.logging:
        log_directory = os.path.join(config.out_directory, 'logs')
        log_file = os.path.join(log_directory, '{:.0f}.log'.format(timestamp))
        log_format = '%(asctime)s : %(funcName)s : %(levelname)s : %(message)s'
        log = pipelinelog.create_log('mkidpipeline.calibration.wavecal',
                                     logfile=log_file, console=False, fmt=log_format,
                                     level="DEBUG")
    if not args.quiet and config.verbose:
        log_format = "%(funcName)s : %(message)s"
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter(fmt))
        handler.setLevel('INFO')
        log.addHandler(handler)

    # print execution time on exit
    atexit.register(lambda x: print('Execution took {:.2f} minutes'
                                    .format((datetime.utcnow().timestamp() - x) / 60)),
                    timestamp)

    # set up bin2hdf
    if args.n_cpu == 0:
        args.n_cpu = len(config.wavelengths)
    if args.vet_only:
        exit()
    if not config.hdf_exist() or args.force_h5:
        b2h_configs = []
        for wave, start_t, int_t in zip(config.wavelengths, config.start_times,
                                        config.exposure_times):
            b2h_configs.append(bin2hdf.Bin2HdfConfig(datadir=config.bin_directory,
                                                     beamfile=config.beam_map_path,
                                                     outdir=config.h5_directory,
                                                     starttime=start_t, inttime=int_t,
                                                     x=config.x_pixels,
                                                     y=config.y_pixels))
        bin2hdf.makehdf(b2h_configs, maxprocs=min(args.n_cpu, mp.cpu_count()))
    if args.h5_only:
        exit()

    # run the wavelength calibration
    Calibrator(config).run(parallel=config.parallel, plot=config.summary_plot,
                           verbose=config.verbose)
