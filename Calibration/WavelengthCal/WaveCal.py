import os
import ast
import warnings
import numpy as np
import scipy.optimize as opt
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from configparser import ConfigParser
from RawDataProcessing.darkObsFile import ObsFile


class WaveCal:
    '''
    Class for creating wavelength calibrations for ObsFile formated data.
    '''
    def __init__(self, config_file='default.cfg'):
        # define the configuration file path
        directory = os.path.dirname(__file__)
        self.config_directory = os.path.join(directory, 'Params', config_file)

        # check the configuration file path and read it in
        self.__configCheck(0)
        self.config = ConfigParser()
        self.config.read(self.config_directory)

        # check the configuration file format and load the parameters
        self.__configCheck(1)
        self.wavelengths = ast.literal_eval(self.config['Data']['wavelengths'])
        self.file_names = ast.literal_eval(self.config['Data']['file_names'])
        self.directory = ast.literal_eval(self.config['Data']['directory'])
        self.model_name = ast.literal_eval(self.config['Fit']['model_name'])
        self.bin_width = ast.literal_eval(self.config['Fit']['bin_width'])
        self.dt = ast.literal_eval(self.config['Fit']['dt'])
        self.out_directory = ast.literal_eval(self.config['Output']['out_directory'])
        self.save_plots = ast.literal_eval(self.config['Output']['save_plots'])
        self.plot_file_name = ast.literal_eval(self.config['Output']['plot_file_name'])

        # check the parameter formats
        self.__configCheck(2)

        # arrange the files in increasing wavelength order and open all of the h5 files
        indices = np.argsort(self.wavelengths)
        self.wavelengths = np.array(self.wavelengths)[indices]
        self.file_names = np.array(self.file_names)[indices]
        self.obs = []
        for file_ in self.file_names:
            self.obs.append(ObsFile(os.path.join(self.directory, file_)))

        # get the array size from the beam map and check that all files are the same
        self.rows, self.columns = np.shape(self.obs[0].beamImage)
        self.__configCheck(3)

        # initialize plotting variables
        if self.save_plots:
            self.pdf = PdfPages(os.path.join(self.out_directory, self.plot_file_name))
            self.plot_counter = 0
            self.plots_x = 3
            self.plots_y = 4
            self.plots_per_page = self.plots_x * self.plots_y

    def loadPhotonData(self, row, column, wavelength_index):
        '''
        Get a photon list for a single pixel and wavelength.
        '''
        photon_list = self.obs[wavelength_index].getPixelPhotonList(row, column)
        return photon_list

    def removeTailRidingPhotons(self, photon_list, dt):
        '''
        Remove photons that arrive too close together.
        '''
        # enforce time ordering (will remove once this is enforced in h5 file creation)
        indices = np.argsort(photon_list['Time'])
        photon_list = photon_list[indices]

        indices = np.where(np.diff(photon_list['Time']) > dt)[0] + 1
        photon_list = photon_list[indices]

        return photon_list

    def histogramPhotons(self, phase_list, fit_list, wavelength_index):
        '''
        Create a histogram of the phase data for a specified bin width.
        '''
        phase_list = phase_list[phase_list < 0]
        min_phase = np.min(phase_list)
        max_phase = np.max(phase_list)

        recent_fit, recent_index, success = self.findLastGoodFit(fit_list)
        if success:
            # rescale binwidth -> old width * (expected center / old center)
            self.bin_width = (self.bin_width * self.wavelengths[recent_index] /
                              self.wavelengths[wavelength_index])

        # define bin edges being careful to start at the threshold cut
        bin_edges = np.arange(max_phase, min_phase - self.bin_width,
                              -self.bin_width)[::-1]

        counts, x0 = np.histogram(phase_list, bins=bin_edges)
        centers = (x0[:-1] + x0[1:]) / 2.0

        # removing last bin because it messes up the noise tail fit
        phase_hist = {'centers': centers, 'counts': counts}

        return phase_hist

    def fitSetup(self, phase_hist):
        '''
        Get a good initial guess (and bounds) for the fitting model.
        '''
        if self.model_name == 'gaussian_and_exp':
            box = np.ones(10) / 10.0
            phase_smoothed = np.convolve(phase_hist['counts'], box, mode='same')

            limits = ([0, -np.inf, 0, 1.1 * np.min(phase_hist['centers']), 0.1],
                      [np.inf, np.inf, 1.1 * np.max(phase_hist['counts']), 0, np.inf])

            guess = [np.amax(phase_hist['counts']), -0.05,
                     1.1 * np.max(phase_hist['counts']),
                     np.min([phase_hist['centers'][np.argmax(phase_smoothed)], -10]), 10]
            setup = (guess, limits)
        else:
            raise ValueError("{0} is not a valid fit model name".format(self.model_name))

        return setup

    def fitUpdate(self, setup, fit_list, wavelength_index):
        '''
        Update the guess (and bounds) after successfully fitting the shortest wavelength.
        '''
        if self.model_name == 'gaussian_and_exp':
            recent_fit, recent_index, success = self.findLastGoodFit(fit_list)
            if success:
                # update gaussian center guess
                setup[0][3] = (recent_fit[0][3] * self.wavelengths[recent_index] /
                               self.wavelengths[wavelength_index])
                # update standard deviation guess
                setup[0][4] = recent_fit[0][4]

        return setup

    def evaluateFit(self, phase_hist, fit_result, fit_list, wavelength_index):
        '''
        Evaluate the result of the fit and return a flag for different conditions.
        '''
        if self.model_name == 'gaussian_and_exp':
            peak_upper_lim = np.min([-10, np.max(phase_hist['centers']) * 1.2])
            peak_lower_lim = np.min(phase_hist['centers'])

            # change peak_upper_lim if good fits exist for higher wavelengths
            recent_fit, recent_index, success = self.findLastGoodFit(fit_list)
            if success:
                peak_upper_lim = (0.2 * recent_fit[0][3] * self.wavelengths[recent_index]
                                  / self.wavelengths[wavelength_index])

            if not fit_result:
                flag = 1  # fit did not converge
            else:
                center = fit_result[0][3]
                gaussian_height = self.fitModels('gaussian')(center, *fit_result[0][2:])
                exp_height = self.fitModels('exp')(center, *fit_result[0][:2])

                bad_fit_conditions = (center > peak_upper_lim) & \
                                     (center < peak_lower_lim) & \
                                     (gaussian_height < 2 * exp_height) & \
                                     (fit_result[0][2] < 10) & \
                                     np.abs(fit_result[0][4] < 2)
                if bad_fit_conditions:
                    flag = 2  # fit converged to a bad solution
                else:
                    flag = 0  # fit converged

        else:
            raise ValueError("{0} is not a valid fit model name".format(self.model_name))

        return flag

    def fitPhaseHistogram(self, phase_hist, fit_function, setup):
        '''
        Fit the phase histogram to the specified fit fit_function
        '''
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                if len(setup) == 2:
                    popt, pcov = opt.curve_fit(fit_function, phase_hist['centers'],
                                               phase_hist['counts'], setup[0],
                                               bounds=setup[1])
                else:
                    popt, pcov = opt.curve_fit(fit_function, phase_hist['centers'],
                                               phase_hist['counts'], setup)
                fit_result = (popt, pcov)
            except (RuntimeError, opt.OptimizeWarning, RuntimeWarning):
                # RuntimeError catches failed minimization
                # OptimizeWarning catches if covariance can't be calculated
                # RuntimeWarning catches overflow errors
                fit_result = False
            except ValueError:
                print("curve_fit not set up correctly. Either ydata or xdata " +
                      "contain NaNs or incompatible options were used.")
                fit_result = False

        return fit_result

    def getPhaseHeights(self, pixels=()):
        '''
        Returns the phase heights at all of the specified wavelengths for a specified
        pixel.
        '''
        # check inputs
        if len(pixels) == 0:
            rows = range(self.rows)
            columns = range(self.columns)
            pixels = ((r, c) for r in rows for c in columns)
        else:
            for pixel in pixels:
                assert type(pixel) is tuple or type(pixel) is list, \
                    "pixels must be a list of pairs of integers"
                assert len(pixel) == 2, "pixels must be a list of pairs of integers"
                assert type(pixel[0]) is int, "pixels must be a list of pairs of integers"
                assert type(pixel[1]) is int, "pixels must be a list of pairs of integers"
                assert pixel[0] > 0 & pixel[0] < self.rows, \
                    "rows in pixels must be between 0 and {0}".format(self.rows)
                assert pixel[1] > 0 & pixel[1] < self.columns, \
                    "columns in pixels must be between 0 and {0}".format(self.columns)

        # initialize fit_data structure
        fit_data = np.empty((self.rows, self.columns), dtype=object)
        for ind, _ in np.ndenumerate(fit_data):
            fit_data[ind] = []

        # loop over pixels and fit the phase histograms
        for row, column in pixels:
            for wavelength_index, _ in enumerate(self.wavelengths):
                fit_list = fit_data[row, column]

                photon_list = self.loadPhotonData(row, column, wavelength_index)
                photon_list = self.removeTailRidingPhotons(photon_list, self.dt)

                phase_hist = self.histogramPhotons(photon_list['Wavelength'],
                                                   fit_list, wavelength_index)

                setup = self.fitSetup(phase_hist)
                setup = self.fitUpdate(setup, fit_list, wavelength_index)

                fit_function = self.fitModels(self.model_name)
                fit_result = self.fitPhaseHistogram(phase_hist, fit_function, setup)

                flag = self.evaluateFit(phase_hist, fit_result, fit_list,
                                        wavelength_index)
                fit_data[row, column].append((fit_result[0], fit_result[1], flag))

                self.plotFit(phase_hist, fit_result, fit_function)

        # save last page and close
        if self.save_plots:
            if not self.saved:
                self.pdf.savefig(self.fig)
            self.pdf.close()

        return fit_data

    def calculateCoefficients(self, use_zero_point=True):
        '''
        Loop through 'getPhaseHeights()' for all pixels and fit to a parabola
        '''
        raise NotImplementedError

    def exportData(self, filename):
        '''
        Saves data in the WaveCal format to the filename
        '''
        raise NotImplementedError

    def plotFit(self, phase_hist, fit_result, fit_function):
        '''
        Plots the histogram data against the model fit for comparison and saves to pdf
        '''
        if not self.save_plots:
            return
        # reset figure if needed
        if self.plot_counter % self.plots_per_page == 0:
            self.fig, self.axes = plt.subplots(self.plots_x, self.plots_y,
                                               figsize=(8.25, 10), dpi=100)
            self.axes = self.axes.flatten()
            self.saved = False

        # get index of plot on current page
        index = self.plot_counter % self.plots_per_page

        # create plot
        self.axes[index].bar(phase_hist['centers'], phase_hist['counts'],
                             align='center', width=2)
        self.axes[index].plot(phase_hist['centers'],
                              fit_function(phase_hist['centers'], *fit_result[0]),
                              color='orange')

        # save page if all the plots have been made
        if self.plot_counter % self.plots_per_page == self.plots_per_page - 1:
            self.pdf.savefig(self.fig)
            self.saved = True

        # update plot counter
        self.plot_counter += 1

    @staticmethod
    def fitModels(model_name):
        '''
        Returns the specified fit model from a library of possible functions
        '''
        if model_name == 'gaussian_and_exp':
            fit_function = lambda x, a, b, c, d, f: \
                a * np.exp(-b * x) + c * np.exp(-1 / 2.0 * ((x - d) / f)**2)
        elif model_name == 'gaussian':
            fit_function = lambda x, c, d, f: c * np.exp(-1 / 2.0 * ((x - d) / f)**2)
        elif model_name == 'exp':
            fit_function = lambda x, a, b: a * np.exp(-b * x)

        return fit_function

    @staticmethod
    def findLastGoodFit(fit_list):
        '''
        Find the most recent fit and index from a list of fits.
        '''
        if fit_list:  # fit_list not empty
            for index, fit in enumerate(fit_list):
                if fit[2] == 0:
                    recent_fit = fit
                    recent_index = index
            if 'recent_fit' in locals():
                return recent_fit, recent_index, True
        return None, None, False

    def __configCheck(self, index):
        '''
        Checks the variables loaded in from the configuration file for type and
        consistencey. Run in the '__init__()' method.
        '''
        if index == 0:
            # check for configuration file
            assert os.path.isfile(self.config_directory), \
                self.config_directory + " is not a valid configuration file"

        elif index == 1:
            # check if all sections and parameters exist in the configuration file
            section = "{0} must be a configuration section"
            param = "{0} must be a parameter in the configuration file '{1}' section"

            assert 'Data' in self.config.sections(), section.format('Array')
            assert 'directory' in self.config['Data'].keys(), \
                param.format('directory', 'Data')
            assert 'wavelengths' in self.config['Data'].keys(), \
                param.format('wavelengths', 'Data')
            assert 'file_names' in self.config['Data'].keys(), \
                param.format('file_names', 'Data')

            assert 'Fit' in self.config.sections(), section.format('Fit')
            assert 'model_name' in self.config['Fit'].keys(), \
                param.format('model_name', 'Fit')
            assert 'bin_width' in self.config['Fit'].keys(), \
                param.format('bin_width', 'Fit')
            assert 'dt' in self.config['Fit'].keys(), \
                param.format('dt', 'Fit')

            assert 'Output' in self.config.sections(), section.format('Output')
            assert 'out_directory' in self.config['Output'], \
                param.format('out_directory', 'Output')
            assert 'save_plots' in self.config['Output'], \
                param.format('save_plots', 'Output')
            assert 'plot_file_name' in self.config['Output'], \
                param.format('plot_file_name', 'Output')

        elif index == 2:
            # type check parameters
            assert type(self.wavelengths) is list, "wavelengths parameter must be a list."
            assert type(self.file_names) is list, "file_names parameter must be a list."
            assert type(self.model_name) is str, "model_name parameter must be a string."
            assert type(self.save_plots) is bool, "save_plots parameter must be a boolean"
            assert type(self.directory) is str, "directory parameter must be a string"
            assert type(self.plot_file_name) is str, \
                "plot_file_name parameter must be a string"
            assert type(self.out_directory) is str, \
                "out_directory parameter must be a string"
            assert os.path.isdir(self.out_directory), \
                "{0} is not a valid output directory".format(self.out_directory)

            assert len(self.wavelengths) == len(self.file_names), \
                "wavelengths and file_names parameters must be the same length."
            if type(self.bin_width) is int:
                self.bin_width = float(self.bin_width)
            assert type(self.bin_width) is float, \
                "bin_width parameter must be an integer or float"

            if type(self.dt) is int:
                self.dt = float(self.dt)
            assert type(self.dt) is float, "dt parameter must be an integer or float"

            for index, lambda_ in enumerate(self.wavelengths):
                if type(lambda_) is int:
                    self.wavelengths[index] = float(self.wavelengths[index])
                assert type(self.wavelengths[index]) is float, \
                    "elements in wavelengths parameter must be floats or integers."

            for file_ in self.file_names:
                assert type(file_) is str, "elements in filenames " + \
                                           "parameter must be strings."
                assert os.path.isfile(os.path.join(self.directory, file_)), \
                    os.path.join(self.directory, file_) + " is not a valid file name."

        elif index == 3:
            # check that all beammaps are the same
            for obs in self.obs:
                assert np.shape(obs.beamImage) == (self.rows, self.columns), \
                    "All files must have the same beam map shape."
        else:
            raise ValueError("index must be 0, 1, 2 or 3")
