#!/bin/env python3
import os
import ast
import atexit
import argparse
import numpy as np
import multiprocessing as mp
from datetime import datetime
from six.moves.configparser import ConfigParser

from mkidpipeline.hdf import bin2hdf
import mkidcore.corelog as pipelinelog
from mkidpipeline.hdf.darkObsFile import ObsFile
import mkidpipeline.calibration.wavecal_models as models

log = pipelinelog.getLogger('mkidpipeline.calibration.wavecal', setup=False)
file_log = pipelinelog.getLogger('mkidpipeline.calibration.wavecal.file_log', setup=False)


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
        message = ("histogram_model_names parameter must be a list of subclasses in "
                   "wavecal_models.py of PartialLinearModel from wavecal_models.py")
        assert isinstance(self.histogram_model_names, list), message
        for model in self.histogram_model_names:
            assert issubclass(getattr(models, model), models.PartialLinearModel), message
        try:
            self.bin_width = float(self.bin_width)
        except ValueError:
            raise AssertionError("bin_width parameter must be an integer or float")
        message = ("calibration_model_names parameter must be a list of subclasses in "
                   "wavecal_models.py of XErrorsModel from wavecal_models.py")
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

        # initialize fit array
        solution = np.empty((self.cfg.y_pixels, self.cfg.x_pixels), dtype=object)
        self.solution = Solution(solution=solution, configuration=self.cfg)
        self.cpu_count = None

    def run(self, pixels=(), parallel=True):
        try:
            if parallel:
                pass
            else:
                self.make_phase_histogram(pixels=pixels)
                self.fit_phase_histogram(pixels=pixels)
                self.fit_phase_energy_curve(pixels=pixels)
            self.save()
            if self.cfg.summary_plot:
                sol = Solution(solution=self.solution, configuration=self.cfg)
                sol.plot_summary()
        except KeyboardInterrupt:
            log.info(os.linesep + "Keyboard shutdown requested ... exiting")

    def make_phase_histogram(self, pixels=(), wavelengths=None):
        pixels = self._check_pixel_inputs(pixels)
        # grab the wavelength indices referenced to the config file and
        # check wavelengths parameter
        wavelength_indices, wavelengths = self._parse_wavelengths(wavelengths)
        # make ObsFiles
        obs_files = []
        for index in wavelength_indices:
            file_name = os.path.join(self.cfg.h5_directory, self.cfg.h5_file_names[index])
            obs_files.append(ObsFile(file_name))
        # make histograms for each pixel in pixels
        for pixel in pixels:
            x_ind, y_ind = pixel[0], pixel[1]
            histogram_models = self.solution[y_ind, x_ind]['histogram']
            for index, wavelength in enumerate(wavelengths):
                # load the data
                photon_list = obs_files[index].getPixelPhotonList(x_ind, y_ind)
                if photon_list.size == 0:
                    message = "({}, {}) : {} nm : there are no photons"
                    file_log.debug(message.format(x_ind, y_ind, wavelength))
                    break
                # remove hot pixels
                rate = (len(photon_list['Wavelength']) /
                        (max(photon_list['Time']) - min(photon_list['Time']))) * 1e6
                if rate > 1800:
                    message = ("({}, {}) : {} nm : removed for being too hot "
                               "(>1800 cps)")
                    file_log.debug(message.format(x_ind, y_ind, wavelength))
                    break
                # remove photons too close together in time
                photon_list = self._remove_tail_riding_photons(photon_list)
                if photon_list.size == 0:
                    message = ("({}, {}) : {} nm : all the photons were removed after "
                               "the arrival time cut")
                    file_log.debug(message.format(x_ind, y_ind, wavelength))
                    break
                # remove photons with positive peak heights
                phase_list = photon_list['Wavelength']
                phase_list = phase_list[phase_list < 0]
                if phase_list.size == 0:
                    message = ("({}, {}) : {} nm : all the photons were removed after "
                               "the negative phase only cut")
                    file_log.debug(message.format(x_ind, y_ind, wavelength))
                    break
                # make histogram
                centers, counts = self._histogram(phase_list)
                # assign x, y and variance data to the fit model
                model = histogram_models[wavelength_indices[index]]
                model.x = centers
                model.y = counts
                # gaussian mle for the variance of poisson distributed data
                # https://doi.org/10.1016/S0168-9002(00)00756-7
                model.variance = np.sqrt(counts**2 + 0.25) - 0.5

    def fit_phase_histogram(self, pixels=()):
        for pixel in pixels:
            for wavelength_index, wavelength in enumerate(self.cfg.wavelengths):
                model = self.solution[pixel]['histogram'][wavelength_index]
                for fit_index in range(self.histogram_fit_attempts):
                    # get a rough guess from the model
                    guess = model.guess(fit_index)
                    # if there are any good fits intelligently guess the signal_center
                    # and set the other parameters equal to the average of those in the
                    # good fits
                    guess = self._update_guess(guess, fit_index, wavelength_index,
                                               wavelength)
                    model.fit(guess)
                    if model.has_good_solution():
                        break

    def fit_phase_energy_curve(self, pixels=()):
        for pixel in pixels:
            guess = self._phase_energy_guess()
            self.solution[pixel]['calibration'].fit(guess)

    def save(self):
        np.savez(self.cfg.solution_name, solution=self.solution, configuration=self.cfg)

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
        while max_count < 400 and update < 2:
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

    def _check_pixel_inputs(self, pixels):
        if pixels:
            pixels = np.atleast_2d(np.array(pixels))
            if not np.issubdtype(pixels.dtype, np.integer):
                raise ValueError("pixels must be a list of pairs of integers")
        else:
            x_pixels = range(self.cfg.x_pixels)
            y_pixels = range(self.cfg.y_pixels)
            pixels = np.array([[x, y] for x in x_pixels for y in y_pixels])
        return pixels

    def _update_guess(self, guess, fit_index, wavelength_index, wavelength):
        """If there are any good fits for this pixel intelligently guess the
        signal_center parameter and set the other parameters equal to the average of
        those in the good fits"""

        return guess

class Solution(object):
    """Solution class for the wavelength calibration. Initialize with either the file_name
    argument or both the solution and configuration arguments."""
    def __init__(self, file_name=None, solution=None, configuration=None):
        # load in solution and configuration objects
        if solution is not None and configuration is not None:
            self._solution = solution
            self.cfg = configuration
        elif file_name is not None:
            npz_file = np.load(file_name)
            self._solution = npz_file['solution']
            self.cfg = npz_file['configuration']
        else:
            message = ('provide either a file_name or both the solution and '
                       'configuration arguments')
            raise ValueError(message)

        # load in fitting models
        self.histogram_model_list = [getattr(models, name) for _, name in
                                     enumerate(self.cfg.histogram_model_names)]
        self.calibration_model_list = [getattr(models, name) for _, name in
                                       enumerate(self.cfg.calibration_model_names)]

    def __getitem__(self, values):
        results = self._solution[values]
        empty = (results == np.array([None]))
        if empty.any():
            for index, entry in np.ndenumerate(results):
                if empty[index]:
                    histogram_models = [self.histogram_model_list[0]()
                                        for _ in range(len(self.cfg.wavelengths))]
                    calibration_models = self.calibration_model_list[0]()
                    self._solution[index] = {'histogram': histogram_models,
                                             'calibration': calibration_models}
            results = self._solution[values]
        return results


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
        file_log = pipelinelog.create_log('mkidpipeline.calibration.wavecal.file_log',
                                          logfile=log_file, console=False,
                                          fmt=log_format)
    if not args.quiet and config.verbose:
        log_format = "%(funcName)s : %(message)s"
        log = pipelinelog.create_log('mkidpipeline.calibration.wavecal', console=True,
                                     fmt=log_format)
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
    Calibrator(config).run(parallel=config.parallel)
