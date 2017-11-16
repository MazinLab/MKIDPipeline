import os
import ast
import warnings
import pickle
import numpy as np
import scipy.optimize as opt
from datetime import datetime
from matplotlib import lines
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from PyPDF2 import PdfFileMerger, PdfFileReader
from configparser import ConfigParser
from astropy.constants import h, c
from progressbar import ProgressBar, Bar, ETA, Timer, Percentage
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
        self.verbose = ast.literal_eval(self.config['Output']['verbose'])
        self.logging = ast.literal_eval(self.config['Output']['logging'])
        self.cal_file_name = ast.literal_eval(self.config['Output']['cal_file_name'])

        # check the parameter formats
        self.__configCheck(2)

        # arrange the files in increasing wavelength order and open all of the h5 files
        indices = np.argsort(self.wavelengths)
        self.wavelengths = np.array(self.wavelengths)[indices]
        self.file_names = np.array(self.file_names)[indices]
        self.obs = []
        for file_ in self.file_names:
            self.obs.append(ObsFile(os.path.join(self.directory, file_), mode='write'))

        # get the array size from the beam map and check that all files are the same
        self.rows, self.columns = np.shape(self.obs[0].beamImage)
        self.__configCheck(3)

        # initialize output flag definitions
        self.flag_dict = {0: "fit converged", 1: "fit did not converge",
                          2: "fit converged to a bad solution", 3: "no data to fit",
                          4: "energy fit to quadratic function",
                          5: "energy fit to linear function",
                          6: "energy not fit - not enough data points",
                          7: "energy not fit - data not monotonic enough"}

        # initialize logging directory if it doesn't exist
        if self.logging:
            if not os.path.isdir(os.path.join(self.out_directory, 'logs')):
                os.mkdir(os.path.join(self.out_directory, 'logs'))
            self.logger("WaveCal object created", new=True)

    def loadPhotonData(self, row, column, wavelength_index):
        '''
        Get a photon list for a single pixel and wavelength.
        '''
        try:
            photon_list = self.obs[wavelength_index].getPixelPhotonList(row, column)
        except:
            return np.array([], dtype=[('Time', '<u4'), ('Wavelength', '<f4'),
                                       ('SpecWeight', '<f4'), ('NoiseWeight', '<f4')])
            # later check for if the pixel exists in the beam map and if not raise error
            raise ValueError("could not load pixel in ({0}, {1}) for {2} nm wavelength"
                             .format(row, column, self.wavelengths[wavelength_index]))
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
        if len(phase_list) == 0:
            phase_hist = {'centers': np.array([]), 'counts': np.array([])}
            return phase_hist
        min_phase = np.min(phase_list)
        max_phase = np.max(phase_list)

        # if no fits have been done, reload the bin width
        if not fit_list:
            self.bin_width = ast.literal_eval(self.config['Fit']['bin_width'])
        # check for most recent fit to rescale the bin width
        else:
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

        phase_hist = {'centers': np.array(centers), 'counts': np.array(counts)}

        return phase_hist

    def fitSetup(self, phase_hist):
        '''
        Get a good initial guess (and bounds) for the fitting model.
        '''
        if len(phase_hist['centers']) == 0:
            return (None, None)
        if self.model_name == 'gaussian_and_exp':
            threshold = max(phase_hist['centers'])
            exp_amplitude = (phase_hist['counts'][phase_hist['centers'] == threshold][0] /
                             np.exp(threshold * 0.2))

            box = np.ones(10) / 10.0
            phase_smoothed = np.convolve(phase_hist['counts'], box, mode='same')
            gaussian_center = phase_hist['centers'][np.argmax(phase_smoothed)]

            if (gaussian_center > 1.4 * threshold):  # remember both numbers are negative
                gaussian_center = np.max([1.5 * threshold, np.min(phase_hist['centers'])])

            gaussian_amplitude = 1.1 * np.max(phase_hist['counts']) / 2

            limits = ([0, -1, 0, np.min(phase_hist['centers']), 0.1],
                      [np.inf, np.inf, 1.1 * np.max(phase_hist['counts']), 0, np.inf])
            guess = [exp_amplitude, 0.2, gaussian_amplitude, gaussian_center, 10]

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
                setup[0][3] = (recent_fit[1][3] * self.wavelengths[recent_index] /
                               self.wavelengths[wavelength_index])
                # update standard deviation guess
                setup[0][4] = recent_fit[1][4]

        return setup

    def evaluateFit(self, phase_hist, fit_result, fit_list, wavelength_index):
        '''
        Evaluate the result of the fit and return a flag for different conditions.
        '''
        if len(phase_hist['centers']) == 0:
            flag = 3  # no data to fit
            return flag
        if self.model_name == 'gaussian_and_exp':
            peak_upper_lim = np.min([-10, np.max(phase_hist['centers']) * 1.2])
            peak_lower_lim = np.min(phase_hist['centers'])

            # change peak_upper_lim if good fits exist for higher wavelengths
            recent_fit, recent_index, success = self.findLastGoodFit(fit_list)
            if success:
                peak_upper_lim = (0.2 * recent_fit[1][3] * self.wavelengths[recent_index]
                                  / self.wavelengths[wavelength_index])

            if fit_result[0] is False:
                flag = 1  # fit did not converge
            else:
                center = fit_result[0][3]
                sigma = fit_result[0][4]
                gauss = self.fitModels('gaussian')(center, *fit_result[0][2:])
                exp = self.fitModels('exp')(center, *fit_result[0][:2])
                center_ind = np.argmin(np.abs(phase_hist['centers'] - center))
                counts = phase_hist['counts'][center_ind]
                height = gauss + exp
                bad_fit_conditions = (center > peak_upper_lim) or \
                                     (center < peak_lower_lim) or \
                                     (gauss < 2 * exp) or \
                                     (gauss < 10) or \
                                     np.abs(sigma < 2) or \
                                     np.abs(counts - height) > 5 * np.sqrt(counts)
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
            error = np.sqrt(phase_hist['counts'] + 0.25) + 0.5
            try:
                if len(setup) == 2:
                    popt, pcov = opt.curve_fit(fit_function, phase_hist['centers'],
                                               phase_hist['counts'], setup[0],
                                               sigma=np.sqrt(phase_hist['counts']) + 1,
                                               bounds=setup[1])
                else:
                    popt, pcov = opt.curve_fit(fit_function, phase_hist['centers'],
                                               phase_hist['counts'], setup,
                                               sigma=np.sqrt(phase_hist['counts']) + 1)
                fit_result = (popt, pcov)
            except (RuntimeError, opt.OptimizeWarning,
                    RuntimeWarning, TypeError) as error:
                # RuntimeError catches failed minimization
                # OptimizeWarning catches if covariance can't be calculated
                # RuntimeWarning catches overflow errors
                # TypeError catches when no data is passed to curve_fit
                if self.logging:
                    self.logger(str(error))
                fit_result = (False, False)
            except ValueError as error:
                if self.logging:
                    self.logger(str(error))
                fit_result = (False, False)

        return fit_result

    def getPhaseHeights(self, pixels=()):
        '''
        Returns the phase heights at all of the specified wavelengths for a specified
        list of pixels.
        '''
        # check inputs
        pixels = self.checkPixelInputs(pixels)

        # initialize plotting, logging, and verbose
        if self.save_plots:
            self.setupPlots()
        if self.logging:
            self.logger("fitting phase histograms")
        if self.verbose:
            print('fitting phase histograms')
            N = len(pixels) * len(self.wavelengths)
            self.pbar = ProgressBar(widgets=[Percentage(), Bar(), '  (',
                                             Timer(), ') ', ETA(), ' '],
                                    maxval=N).start()
            self.pbar_iter = 0

        # initialize fit_data structure
        fit_data = np.empty((self.rows, self.columns), dtype=object)
        for ind, _ in np.ndenumerate(fit_data):
            fit_data[ind] = []

        # loop over pixels and fit the phase histograms
        for row, column in pixels:
            for wavelength_index, _ in enumerate(self.wavelengths):
                if self.logging:
                    start_time = datetime.now()

                # pull out fits already done for this wavelength
                fit_list = fit_data[row, column]

                # load data and cut photons too close together in time
                photon_list = self.loadPhotonData(row, column, wavelength_index)
                photon_list = self.removeTailRidingPhotons(photon_list, self.dt)

                # make the phase histogram
                phase_hist = self.histogramPhotons(photon_list['Wavelength'],
                                                   fit_list, wavelength_index)

                # set up the fit parameters and update them with information from fit_list
                setup = self.fitSetup(phase_hist)
                setup = self.fitUpdate(setup, fit_list, wavelength_index)

                # fit the histogram
                fit_function = self.fitModels(self.model_name)
                fit_result = self.fitPhaseHistogram(phase_hist, fit_function, setup)

                # evaluate how the fit did and save to fit_data structure
                flag = self.evaluateFit(phase_hist, fit_result, fit_list,
                                        wavelength_index)
                fit_data[row, column].append((flag, fit_result[0], fit_result[1]))

                # plot data (will skip if save_plots is set to be true)
                self.plotFit(phase_hist, fit_result, fit_function, flag, row, column)

                # update progress bar and log
                if self.verbose:
                    self.pbar_iter += 1
                    self.pbar.update(self.pbar_iter)
                if self.logging:
                    dt = round((datetime.now() - start_time).total_seconds(), 2)
                    dt = str(dt) + ' s'
                    self.logger("({0}, {1}): {2} : {3}"
                                .format(row, column, self.flag_dict[flag], dt))

        # close progress bar
        if self.verbose:
            self.pbar.finish()
        # close and save last plots
        if self.save_plots:
            self.closePlots()

        return fit_data

    def calculateCoefficients(self, fit_data, pixels=(), use_zero_point=False):
        '''
        Loop through the results of 'getPhaseHeights()' and fit energy vs phase height
        to a parabola
        '''
        # check inputs
        pixels = self.checkPixelInputs(pixels)
        assert np.shape(fit_data) == (self.rows, self.columns), \
            "fit_data must be a ({0}, {1}) numpy array".format(self.rows, self.columns)

        # initialize verbose and logging
        if self.verbose:
            print('calculating phase to energy solution')
            self.pbar = ProgressBar(widgets=[Percentage(), Bar(), '  (',
                                             Timer(), ') ', ETA(), ' '],
                                    maxval=len(pixels)).start()
            self.pbar_iter = 0
        if self.logging:
            self.logger('calculating phase to energy solution')

        # initialize wavelength_cal structure
        wavelength_cal = np.empty((self.rows, self.columns), dtype=object)

        for row, column in pixels:
            if self.logging:
                start_time = datetime.now()
            fit_results = fit_data[row, column]

            # count the number of good fits and save their data
            count = 0
            wavelengths = []
            phases = []
            std = []
            errors = []
            for index, fit_result in enumerate(fit_results):
                if fit_result[0] == 0:
                    count += 1
                    wavelengths.append(self.wavelengths[index])
                    if self.model_name == 'gaussian_and_exp':
                        phases.append(fit_result[1][3])
                        std.append(fit_result[1][4])
                        errors.append(np.sqrt(fit_result[2][3, 3]))
                    else:
                        raise ValueError("{0} is not a valid fit model name"
                                         .format(self.model_name))

            # if there are enough points fit the wavelengths
            if count > 2:
                energies = h.value * c.value / (np.array(wavelengths) * 1e-9)

                if use_zero_point:
                    fit_function = self.fitModels('parabola_zero')
                    guess = [0, energies[0] / phases[0]]  # guess straight line
                else:
                    fit_function = self.fitModels('parabola')
                    guess = [0, energies[0] / phases[0], 0]  # guess straight line
                popt, pcov = opt.curve_fit(fit_function, phases, energies, guess, errors)

                # refit if vertex is between wavelengths or slope is positive
                vertex = -popt[1] / (2 * popt[0])
                min_slope = 2 * popt[0] * min(phases) + popt[1]
                max_slope = 2 * popt[0] * max(phases) + popt[1]
                conditions = vertex < max(phases) and vertex > min(phases) and \
                            (min_slope > 0 or max_slope > 0)
                if conditions:
                    fit_function = self.fitModels('linear')
                    guess = [energies[0] / phases[0], 0]
                    popt, pcov = opt.curve_fit(fit_function, phases, energies,
                                               guess, errors)
                    if popt[0] > 0:
                        flag = 7  # linear fit unsuccessful
                        wavelength_cal[row, column] = (flag, False, False)
                    else:
                        flag = 5  # linear fit successful
                        wavelength_cal[row, column] = (flag, popt, pcov)
                else:
                    flag = 4  # quadratic fit successful
                    wavelength_cal[row, column] = (flag, popt, pcov)
            else:
                flag = 6  # no fit done because of lack of data
                wavelength_cal[row, column] = (flag, False, False)

            # update progress bar and log
            if self.verbose:
                self.pbar_iter += 1
                self.pbar.update(self.pbar_iter)
            if self.logging:
                dt = round((datetime.now() - start_time).total_seconds(), 2)
                dt = str(dt) + ' s'
                self.logger("({0}, {1}): {2} : {3}"
                            .format(row, column, self.flag_dict[flag], dt))
        # close progress bar
        if self.verbose:
            self.pbar.finish()

        return wavelength_cal

    def exportData(self, wavelength_cal, fit_data, phase_hist, fit_function):
        '''
        Saves data in the WaveCal format to the filename
        '''
        cal_file_name = os.path.join(self.out_directory, self.cal_file_name)
        if os.path.isfile(cal_file_name):
            answer = self.query("{0} already exists. Overwrite?".format(cal_file_name),
                                yes_or_no=True)
            if answer is False:
                answer = self.query("Provide a new file name (type exit to quit):")
                if answer == 'exit':
                    print("No data was saved.")
                    return
                cal_file_name = os.path.join(self.out_directory, answer)
                while os.path.isfile(cal_file_name):
                    question = "{0} already exists. Choose a new file name " + \
                               "(type exit to quit):"
                    answer = self.query(question.format(cal_file_name))
                    if answer == 'exit':
                        print("No data was saved.")
                        return
                    cal_file_name = os.path.join(self.out_directory, answer)

        with open(cal_file_name, "wb") as file_:
            data = {'wavelegth_cal': wavelength_cal, 'fit_data': fit_data,
                    'phase_hist': phase_hist, 'fit_function': fit_function}
            pickle.dump(data, file_)

    def plotFit(self, phase_hist, fit_result, fit_function, flag, row, column):
        '''
        Plots the histogram data against the model fit for comparison and saves to pdf
        '''
        if not self.save_plots:
            return
        # reset figure if needed
        if self.plot_counter % self.plots_per_page == 0:
            self.fig, self.axes = plt.subplots(self.plots_x, self.plots_y,
                                               figsize=(8.25, 10), dpi=100)
            self.fig.text(0.01, 0.5, 'Counts', va='center', rotation='vertical')
            self.fig.text(0.5, 0.01, 'Phase [degrees]', ha='center')

            self.axes = self.axes.flatten()
            plt.tight_layout(rect=[0.02, 0.02, 1, 0.95])

            fit_accepted = lines.Line2D([], [], color='green', label='fit accepted')
            fit_rejected = lines.Line2D([], [], color='red', label='fit rejected')
            gaussian = lines.Line2D([], [], color='orange',
                                    linestyle='--', label='gaussian')
            noise = lines.Line2D([], [], color='purple',
                                 linestyle='--', label='noise')
            self.axes[0].legend(handles=[fit_accepted, fit_rejected, gaussian, noise],
                                loc=3, bbox_to_anchor=(0, 1.02, 1, .102),
                                ncol=2)

            self.saved = False

        # get index of plot on current page
        index = self.plot_counter % self.plots_per_page

        # create plot
        if flag == 0:
            color = 'green'
        else:
            color = 'red'

        if self.model_name == 'gaussian_and_exp':
            self.axes[index].bar(phase_hist['centers'], phase_hist['counts'],
                                 align='center', width=self.bin_width)

            if fit_result[0] is not False:
                g_func = self.fitModels('gaussian')
                e_func = self.fitModels('exp')
                phase = np.arange(np.min(phase_hist['centers']),
                                  np.max(phase_hist['centers']), 0.1)
                self.axes[index].plot(phase, e_func(phase, *fit_result[0][:2]),
                                      color='purple', linestyle='--')
                self.axes[index].plot(phase, g_func(phase, *fit_result[0][2:]),
                                      color='orange', linestyle='--')
                self.axes[index].plot(phase, fit_function(phase, *fit_result[0]),
                                      color=color)
                ymax = self.axes[index].get_ylim()[1]
                xmin = self.axes[index].get_xlim()[0]
            else:
                ymax = self.axes[index].get_ylim()[1]
                xmin = self.axes[index].get_xlim()[0]
                self.axes[index].text(xmin * 0.98, ymax * 0.5, 'Fit Error', color='red')

            self.axes[index].text(xmin * 0.98, ymax * 0.98, '({0}, {1})'
                                  .format(row, column), ha='left', va='top')

        else:
            raise ValueError("{0} is not a valid fit model name for plotting"
                             .format(self.model_name))

        # save page if all the plots have been made
        if self.plot_counter % self.plots_per_page == self.plots_per_page - 1:
            pdf = PdfPages(os.path.join(self.out_directory, 'temp.pdf'))
            pdf.savefig(self.fig)
            pdf.close()
            self.mergePlots()
            self.saved = True
            plt.close('all')

        # update plot counter
        self.plot_counter += 1

    def setupPlots(self):
        '''
        Initialize plotting variables
        '''
        self.plot_counter = 0
        self.plots_x = 3
        self.plots_y = 4
        self.plots_per_page = self.plots_x * self.plots_y
        plot_file = os.path.join(self.out_directory, self.plot_file_name)
        if os.path.isfile(plot_file):
            answer = self.query("{0} already exists. Overwrite?".format(plot_file),
                                yes_or_no=True)
            if answer is False:
                answer = self.query("Provide a new file name (type exit to quit):")
                if answer == 'exit':
                    raise IOError("User doesn't want to overwrite the plot file")
                plot_file = os.path.join(self.out_directory, answer)
                while os.path.isfile(plot_file):
                    question = "{0} already exists. Choose a new file name " + \
                               "(type exit to quit):"
                    answer = self.query(question.format(plot_file))
                    if answer == 'exit':
                        raise IOError("User doesn't want to overwrite the plot file")
                    plot_file = os.path.join(self.out_directory, answer)
                self.plot_file_name = plot_file
            else:
                os.remove(plot_file)

    def mergePlots(self):
        '''
        Merge recently created temp.pdf with the main file
        '''
        plot_file = os.path.join(self.out_directory, self.plot_file_name)
        temp_file = os.path.join(self.out_directory, 'temp.pdf')
        if os.path.isfile(plot_file):
            merger = PdfFileMerger()
            merger.append(PdfFileReader(open(plot_file, 'rb')))
            merger.append(PdfFileReader(open(temp_file, 'rb')))
            merger.write(plot_file)
            merger.close()
            os.remove(temp_file)
        else:
            os.rename(temp_file, plot_file)

    def closePlots(self):
        '''
        Safely close plotting variables after plotting since the last page is only saved
        if it is full.
        '''
        if not self.saved:
            pdf = PdfPages(os.path.join(self.out_directory, 'temp.pdf'))
            pdf.savefig(self.fig)
            pdf.close()
            self.mergePlots()
        plt.close('all')

    def logger(self, message, new=False):
        '''
        Method for writing information to a log file
        '''
        if new:
            self.log_time = datetime.now().strftime("%Y%m%d-%H%M%S")
        file_name = os.path.join(self.out_directory, 'logs', str(self.log_time) + '.log')
        with open(file_name, 'a') as log_file:
            log_file.write(message + os.linesep)

    @staticmethod
    def fitModels(model_name):
        '''
        Returns the specified fit model from a library of possible functions
        '''
        if model_name == 'gaussian_and_exp':
            fit_function = lambda x, a, b, c, d, f: \
                a * np.exp(b * x) + c * np.exp(-1 / 2.0 * ((x - d) / f)**2)
        elif model_name == 'gaussian':
            fit_function = lambda x, c, d, f: c * np.exp(-1 / 2.0 * ((x - d) / f)**2)
        elif model_name == 'exp':
            fit_function = lambda x, a, b: a * np.exp(b * x)
        elif model_name == 'parabola':
            fit_function = lambda x, a, b, c: a * x**2 + b * x + c
        elif model_name == 'parabola_zero':
            fit_function = lambda x, a, b: a * x**2 + b * x
        elif model_name == 'linear':
            fit_function = lambda x, a, b: a * x + b

        return fit_function

    @staticmethod
    def findLastGoodFit(fit_list):
        '''
        Find the most recent fit and index from a list of fits.
        '''
        if fit_list:  # fit_list not empty
            for index, fit in enumerate(fit_list):
                if fit[0] == 0:
                    recent_fit = fit
                    recent_index = index
            if 'recent_fit' in locals():
                return recent_fit, recent_index, True
        return None, None, False

    @staticmethod
    def query(question, yes_or_no=False, default="no"):
        '''Ask a question via raw_input() and return their answer.

        "question" is a string that is presented to the user.
        "yes_or_no" specifies if it is a yes or no question
        "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning an answer is required of
        the user). Only used if yes_or_no=True.

        The "answer" return value is the user input for a general question. For a yes or
        no question it is True for "yes" and False for "no".
        '''
        if yes_or_no:
            valid = {"yes": True, "y": True, "ye": True,
                     "no": False, "n": False}
        if not yes_or_no:
            prompt = ""
            default = None
        elif default is None:
            prompt = " [y/n] "
        elif default == "yes":
            prompt = " [Y/n] "
        elif default == "no":
            prompt = " [y/N] "
        else:
            raise ValueError("invalid default answer: '%s'" % default)

        while True:
            print(question + prompt)
            choice = input().lower()
            if not yes_or_no:
                return choice
            elif default is not None and choice == '':
                return valid[default]
            elif choice in valid:
                return valid[choice]
            else:
                print("Please respond with 'yes' or 'no' (or 'y' or 'n').")

    def checkPixelInputs(self, pixels):
        '''
        Check inputs for getPhaseHeights and calculateCoefficients
        '''
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
                assert pixel[0] >= 0 & pixel[0] < self.rows, \
                    "rows in pixels must be between 0 and {0}".format(self.rows)
                assert pixel[1] >= 0 & pixel[1] < self.columns, \
                    "columns in pixels must be between 0 and {0}".format(self.columns)
        return pixels

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
            assert 'verbose' in self.config['Output'], \
                param.format('verbose', 'Output')
            assert 'logging' in self.config['Output'], \
                param.format('logging', 'Output')
            assert 'cal_file_name' in self.config['Output'], \
                param.format('cal_file_name', 'Output')

        elif index == 2:
            # type check parameters
            assert type(self.wavelengths) is list, "wavelengths parameter must be a list."
            assert type(self.file_names) is list, "file_names parameter must be a list."
            assert type(self.model_name) is str, "model_name parameter must be a string."
            assert type(self.save_plots) is bool, "save_plots parameter must be a boolean"
            assert type(self.verbose) is bool, "verbose parameter bust be a boolean"
            assert type(self.directory) is str, "directory parameter must be a string"
            assert type(self.logging) is bool, "logging parameter must be a boolean"
            assert type(self.plot_file_name) is str, \
                "plot_file_name parameter must be a string"
            assert type(self.out_directory) is str, \
                "out_directory parameter must be a string"
            assert type(self.cal_file_name) is str, \
                "cal_file_name parameter must be a string"
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
