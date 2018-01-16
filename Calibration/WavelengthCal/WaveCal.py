import os
import ast
import sys
import glob
import time
import pickle
import warnings
import lmfit as lm
import numpy as np
import tables as tb
import multiprocessing as mp
import scipy.optimize as opt
from matplotlib import lines
from datetime import datetime
from astropy.constants import h, c
from matplotlib import pyplot as plt
from configparser import ConfigParser
from PyPDF2 import PdfFileMerger, PdfFileReader
from matplotlib.backends.backend_pdf import PdfPages
from progressbar import ProgressBar, Bar, ETA, Timer, Percentage
from Headers import pipelineFlags
from RawDataProcessing.darkObsFile import ObsFile
from Headers.CalHeaders import WaveCalDescription, WaveCalHeader, WaveCalDebugDescription


class WaveCal:
    '''
    Class for creating wavelength calibrations for ObsFile formated data.
    '''
    def __init__(self, config_file='default.cfg', log_file=False, lock=None):
        # define start time for use in __logger and saving calsol file
        if log_file is False:
            self.log_file = str(round(datetime.utcnow().timestamp()))
        else:
            self.log_file = log_file

        # set lock for multiprocessing (internal use only)
        self.lock = lock

        # define the configuration file path
        self.config_file = config_file
        directory = os.path.dirname(__file__)
        self.config_directory = os.path.join(directory, 'Params', self.config_file)

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
        self.parallel = ast.literal_eval(self.config['Fit']['parallel'])
        self.out_directory = ast.literal_eval(self.config['Output']['out_directory'])
        self.save_plots = ast.literal_eval(self.config['Output']['save_plots'])
        self.plot_file_name = ast.literal_eval(self.config['Output']['plot_file_name'])
        self.verbose = ast.literal_eval(self.config['Output']['verbose'])
        self.logging = ast.literal_eval(self.config['Output']['logging'])

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

        # initialize output flag definitions
        self.flag_dict = pipelineFlags.waveCal

        # initialize logging directory if it doesn't exist
        if self.logging:
            if not os.path.isdir(os.path.join(self.out_directory, 'logs')):
                os.mkdir(os.path.join(self.out_directory, 'logs'))
            message = "WaveCal object created: UTC " + str(datetime.utcnow()) + \
                " : Local " + str(datetime.now())
            self.__logger(message)
            self.__logger("## configuration file used")
            with open(self.config_directory, "r") as file_:
                config = file_.read()
            self.__logger(config)

    def makeCalibration(self, pixels=[]):
        '''
        Compute the wavelength calibration for the pixels in 'pixels' and save the data
        in the standard format.
        '''
        with warnings.catch_warnings():
            # ignore unclosed file warnings from PyPDF2
            warnings.simplefilter("ignore", category=ResourceWarning)
            try:
                if self.parallel:
                    self.__checkParallelOptions()
                    self.cpu_count = int(np.ceil(mp.cpu_count() / 2))
                    self.getPhaseHeightsParallel(self.cpu_count, pixels=pixels)
                    self.__cleanLogs()
                else:
                    self.getPhaseHeights(pixels=pixels)
                self.calculateCoefficients(pixels=pixels)
                self.exportData(pixels=pixels)
            except (KeyboardInterrupt, BrokenPipeError):
                print(os.linesep + "Shutdown requested ... exiting")
            except UserError as err:
                print(err)

    def getPhaseHeightsParallel(self, n_processes, pixels=[]):
        '''
        Computes the fitted phase height at all of the specified wavelengths for a
        specified list of pixels. Uses more than one process to speed up the computation.
        '''
        # check inputs
        pixels = self.__checkPixelInputs(pixels)

        # create progress bar
        progress_queue = mp.Queue()
        N = len(pixels)
        progress = ProgressWorker(progress_queue, N)

        # create workers
        in_queue = mp.Queue()
        out_queue = mp.Queue()
        workers = []
        lock = mp.Lock()
        for i in range(n_processes):
            workers.append(Worker(in_queue, out_queue, progress_queue,
                                  self.config_file, i, lock))

        try:
            # give workers data ending in n_processes close commands
            for pixel in pixels:
                in_queue.put(pixel)
            for i in range(n_processes):
                in_queue.put(None)

            # collect all results into a single result_dict
            result_dict = {}
            for i in range(len(pixels)):
                result = out_queue.get()
                result_dict.update(result)

            # wait for all worker processes to finish
            progress.join()
            for w in workers:
                w.join()
        except (KeyboardInterrupt, BrokenPipeError):
            print(os.linesep + "PID {0} ... exiting".format(progress.pid))
            progress.terminate()
            progress.join()
            for w in workers:
                print("PID {0} ... exiting".format(w.pid))
                w.terminate()
                w.join()

            log_file = os.path.join(self.out_directory,
                                    'logs/worker*.log')
            for file_ in glob.glob(log_file):
                os.remove(file_)
            raise KeyboardInterrupt

        # populate fit_data with results from workers
        self.fit_data = np.empty((80, 125), dtype=object)
        for ind, _ in np.ndenumerate(self.fit_data):
            self.fit_data[ind] = []
        for (row, column) in result_dict.keys():
            self.fit_data[row, column] = result_dict[(row, column)]

    def getPhaseHeights(self, pixels=[]):
        '''
        Computes the fitted phase height at all of the specified wavelengths for a
        specified list of pixels.
        '''
        # check inputs
        pixels = self.__checkPixelInputs(pixels)

        # initialize plotting, logging, and verbose
        if self.save_plots:
            self.__setupPlots()
        if self.logging:
            self.__logger("## fitting phase histograms")
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

                # load data
                photon_list = self.loadPhotonData(row, column, wavelength_index)

                # if there is no data go to next loop
                if len(photon_list['Wavelength']) == 0:
                    fit_data[row, column].append((3, False, False,
                                                  {'centers': np.array([]),
                                                   'counts': np.array([])}))
                    # update progress bar and log
                    if self.logging:
                        dt = round((datetime.now() - start_time).total_seconds(), 2)
                        dt = str(dt) + ' s'
                        self.__logger("({0}, {1}): {2} : {3}"
                                      .format(row, column, self.flag_dict[3], dt))
                    if self.verbose:
                        self.pbar_iter += 1
                        self.pbar.update(self.pbar_iter)
                    continue

                # cut photons too close together in time
                photon_list = self.__removeTailRidingPhotons(photon_list, self.dt)

                # make the phase histogram
                phase_hist = self.__histogramPhotons(photon_list['Wavelength'])

                # get fit model
                fit_function = fitModels(self.model_name)

                # determine iteration range based on if there are other wavelength fits
                _, _, success = self.__findLastGoodFit(fit_list)
                if success and self.model_name == 'gaussian_and_exp':
                    fit_numbers = range(6)
                elif self.model_name == 'gaussian_and_exp':
                    fit_numbers = range(5)
                else:
                    raise ValueError('invalid model_name')

                fit_results = []
                flags = []
                for fit_number in fit_numbers:
                    # get guess for fit
                    setup = self.__fitSetup(phase_hist, fit_list, wavelength_index,
                                            fit_number)
                    # fit data
                    fit_results.append(self.__fitPhaseHistogram(phase_hist,
                                                                  fit_function,
                                                                  setup, row, column))
                    # evaluate how the fit did
                    flags.append(self.__evaluateFit(phase_hist, fit_results[-1], fit_list,
                                                    wavelength_index))
                    if flags[-1] == 0:
                        break
                # find best fit
                fit_result, flag = self.__findBestFit(fit_results, flags, phase_hist)

                # save data in fit_data object
                fit_data[row, column].append((flag, fit_result[0], fit_result[1],
                                              phase_hist))

                # plot data (will skip if save_plots is set to be true)
                self.__plotFit(phase_hist, fit_result, fit_function, flag, row, column)

                # update progress bar and log
                if self.verbose:
                    self.pbar_iter += 1
                    self.pbar.update(self.pbar_iter)
                if self.logging:
                    dt = round((datetime.now() - start_time).total_seconds(), 2)
                    dt = str(dt) + ' s'
                    self.__logger("({0}, {1}): {2} : {3}"
                                  .format(row, column, self.flag_dict[flag], dt))

        # close progress bar
        if self.verbose:
            self.pbar.finish()
        # close and save last plots
        if self.save_plots:
            self.__closePlots()

        self.fit_data = fit_data

    def calculateCoefficients(self, pixels=[]):
        '''
        Loop through the results of 'getPhaseHeights()' and fit energy vs phase height
        to a parabola
        '''
        # check inputs
        pixels = self.__checkPixelInputs(pixels)
        assert hasattr(self, 'fit_data'), "run getPhaseHeights() first"
        assert np.shape(self.fit_data) == (self.rows, self.columns), \
            "fit_data must be a ({0}, {1}) numpy array".format(self.rows, self.columns)

        # initialize verbose and logging
        if self.verbose:
            print('calculating phase to energy solution')
            self.pbar = ProgressBar(widgets=[Percentage(), Bar(), '  (',
                                             Timer(), ') ', ETA(), ' '],
                                    maxval=len(pixels)).start()
            self.pbar_iter = 0
        if self.logging:
            self.__logger('## calculating phase to energy solution')

        # initialize wavelength_cal structure
        wavelength_cal = np.empty((self.rows, self.columns), dtype=object)

        for row, column in pixels:
            fit_results = self.fit_data[row, column]

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
            phases = np.array(phases)
            std = np.array(std)
            errors = np.array(errors)
            if count > 0 and (np.diff(phases) < 0.0 * min(phases)).any():
                flag = 7  # data not monotonic enough
                wavelength_cal[row, column] = (flag, False, False)

            # if there are enough points fit the wavelengths
            elif count > 2:
                energies = h.to('eV s').value * c.to('nm/s').value / np.array(wavelengths)

                guess = [0, phases[0] / energies[0], 0]  # guess straight line
                try:
                    phase_list = [np.max(fit_results[ind][3]['centers'])
                                  for ind, _ in enumerate(fit_results)]
                except Exception as error:
                    print(" (row, column):", row, ",", column)
                    raise error
                self.current_threshold = np.max(phase_list)
                phase_list = [np.min(fit_results[ind][3]['centers'])
                              for ind, _ in enumerate(fit_results)]
                self.current_min = np.min(phase_list)
                popt, pcov = self.__fitEnergy('quadratic', phases, energies, guess,
                                              errors, row, column)

                # refit if vertex is between wavelengths or slope is positive
                if popt is False:
                    conditions = True
                else:
                    ind_max = np.argmax(phases)
                    ind_min = np.argmin(phases)
                    max_phase = phases[ind_max] + std[ind_max]
                    min_phase = phases[ind_min] - std[ind_min]
                    vertex = -popt[1] / (2 * popt[0])
                    min_slope = 2 * popt[0] * min_phase + popt[1]
                    max_slope = 2 * popt[0] * max_phase + popt[1]
                    conditions = vertex < max_phase and vertex > min_phase and \
                        (min_slope > 0 or max_slope > 0)
                    conditions = False
                if conditions:
                    guess = [phases[0] / energies[0], 0]
                    popt, pcov = self.__fitEnergy('linear', phases, energies, guess,
                                                  errors, row, column)

                    if popt is False or popt[0] > 0:
                        flag = 8  # linear fit unsuccessful
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
                self.__logger("({0}, {1}): {2}".format(row, column, self.flag_dict[flag]))
        # close progress bar
        if self.verbose:
            self.pbar.finish()

        self.wavelength_cal = wavelength_cal

    def exportData(self, pixels=[]):
        '''
        Saves data in the WaveCal format to the filename
        '''
        # check inputs
        pixels = self.__checkPixelInputs(pixels)

        # load wavecal header
        wavecal_description = WaveCalDescription(len(self.wavelengths))

        # create output file name
        cal_file_name = 'calsol_' + self.log_file + '.h5'
        cal_file = os.path.join(self.out_directory, cal_file_name)

        # initialize verbose and logging
        if self.verbose:
            print('exporting data')
            self.pbar = ProgressBar(widgets=[Percentage(), Bar(), '  (',
                                             Timer(), ') ', ETA(), ' '],
                                    maxval=2 * len(pixels)).start()
            self.pbar_iter = 0
        if self.logging:
            self.__logger("## exporting data to {0}".format(cal_file))

        # create folders in file
        file_ = tb.open_file(cal_file, mode='w')
        header = file_.create_group(file_.root, 'header', 'Calibration information')
        wavecal = file_.create_group(file_.root, 'wavecal',
                                     'Table of calibration parameters for each pixel')
        debug = file_.create_group(file_.root, 'debug',
                                   'Detailed fitting information for debugging')

        # populate header
        info = file_.create_table(header, 'info', WaveCalHeader)
        file_.create_vlarray(header, 'obsFiles', obj=self.file_names)
        file_.create_vlarray(header, 'wavelengths', obj=self.wavelengths)
        file_.create_array(header, 'beamMap', obj=self.obs[0].beamImage)
        info.row['model_name'] = self.model_name
        info.row.append()
        info.flush()

        # populate wavecal
        calsoln = file_.create_table(wavecal, 'calsoln', wavecal_description,
                                     title='Wavelength Calibration Table')
        for row, column in pixels:
            calsoln.row['resid'] = self.obs[0].beamImage[row][column]
            calsoln.row['pixel_row'] = row
            calsoln.row['pixel_col'] = column
            if (self.wavelength_cal[row, column][0] == 4 or
               self.wavelength_cal[row, column][0] == 5):
                calsoln.row['polyfit'] = self.wavelength_cal[row, column][1]
            else:
                calsoln.row['polyfit'] = [-1, -1, -1]
            wavelengths = []
            sigma = []
            R = []
            for index, wavelength in enumerate(self.wavelengths):
                if ((self.wavelength_cal[row, column][0] == 4 or
                     self.wavelength_cal[row, column][0] == 5) and
                     self.fit_data[row, column][index][0] == 0):
                    if self.model_name == 'gaussian_and_exp':
                        mu = self.fit_data[row, column][index][1][3]
                        std = self.fit_data[row, column][index][1][4]
                    else:
                        raise ValueError("{0} is not a valid fit model name"
                                         .format(self.model_name))
                    poly = self.wavelength_cal[row, column][1]
                    dE = (np.polyval(poly, mu - std) - np.polyval(poly, mu + std)) / 2
                    E = h.to('eV s').value * c.to('nm/s').value / wavelength
                    sigma.append(dE)
                    R.append(E / dE)
                    wavelengths.append(wavelength)
                else:
                    sigma.append(-1)
                    R.append(-1)
            calsoln.row['sigma'] = sigma
            calsoln.row['R'] = R
            if len(wavelengths) == 0:
                calsoln.row['soln_range'] = [-1, -1]
            else:
                calsoln.row['soln_range'] = [min(wavelengths), max(wavelengths)]
            calsoln.row['wave_flag'] = self.wavelength_cal[row, column][0]
            calsoln.row.append()
            # update progress bar
            if self.verbose:
                self.pbar_iter += 1
                self.pbar.update(self.pbar_iter)
        calsoln.flush()

        if self.logging:
            self.__logger("wavecal table saved")

        # find max number of bins in histograms
        lengths = []
        for row, column in pixels:
            for index, wavelength in enumerate(self.wavelengths):
                fit_list = self.fit_data[row, column][index]
                phase_centers = fit_list[3]['centers']
                lengths.append(len(phase_centers))
        max_l = np.max(lengths)

        # make debug table
        if self.model_name == 'gaussian_and_exp':
            n_param = 5
        else:
            raise ValueError("{0} is not a valid fit model name"
                             .format(self.model_name))
        debug_description = WaveCalDebugDescription(len(self.wavelengths), n_param, max_l)
        debug_info = file_.create_table(debug, 'debug_info', debug_description,
                                        title='Debug Table')
        for row, column in pixels:
            res_id = self.obs[0].beamImage[row][column]
            debug_info.row['resid'] = res_id
            debug_info.row['pixel_row'] = row
            debug_info.row['pixel_col'] = column
            hist_flags = []
            has_data = []
            bin_widths = []
            for index, wavelength in enumerate(self.wavelengths):
                fit_list = self.fit_data[row, column][index]
                hist_flags.append(fit_list[0])
                if len(fit_list[3]['counts']) > 0 and np.max(fit_list[3]['counts']) > 0:
                    has_data.append(True)
                else:
                    has_data.append(False)
                phase_centers = fit_list[3]['centers']
                if len(phase_centers) == 0 or len(phase_centers) == 1:
                    bin_widths.append(0)
                else:
                    bin_widths.append(np.min(np.diff(phase_centers)))
                hist_fit = fit_list[1]
                hist_cov = fit_list[2]
                if hist_fit is False:
                    hist_fit = np.ones(n_param) * -1
                    hist_cov = np.ones((n_param, n_param)) * -1
                if self.model_name == 'gaussian_and_exp' and hist_cov.size == 16:
                    hist_cov = np.insert(np.insert(hist_cov, 1, [0, 0, 0, 0], axis=1),
                                         1, [0, 0, 0, 0, 0], axis=0)
                debug_info.row["hist_fit" + str(index)] = hist_fit
                debug_info.row["hist_cov" + str(index)] = hist_cov.flatten()
                phase_centers = np.ones(max_l)
                phase_counts = np.ones(max_l) * -1
                phase_centers[:len(fit_list[3]['centers'])] = fit_list[3]['centers']
                phase_counts[:len(fit_list[3]['counts'])] = fit_list[3]['counts']
                debug_info.row["phase_centers" + str(index)] = phase_centers
                debug_info.row["phase_counts" + str(index)] = phase_counts
            debug_info.row['hist_flag'] = hist_flags
            debug_info.row['has_data'] = has_data
            debug_info.row['bin_width'] = bin_widths
            poly_cov = self.wavelength_cal[row][column][2]
            if poly_cov is False or poly_cov is None:
                poly_cov = np.ones((3, 3)) * -1
            debug_info.row['poly_cov'] = poly_cov.flatten()
            debug_info.row.append()
            # update progress bar
            if self.verbose:
                self.pbar_iter += 1
                self.pbar.update(self.pbar_iter)
        debug_info.flush()
        # for row, column in pixels:
        #     res_id = self.obs[0].beamImage[row][column]
        #     # skip if already done (res_id repeats mean pixel wasn't beam maped)
        #     if 'res' + str(res_id) in file_.root.debug:
        #         continue
        #     data = file_.create_group(file_.root.debug, 'res' + str(res_id))
        #     poly_cov = self.wavelength_cal[row][column][2]
        #     if poly_cov is False or poly_cov is None:
        #         poly_cov = []
        #     file_.create_array(data, 'poly_cov', obj=poly_cov)
        #     for index, wavelength in enumerate(self.wavelengths):
        #         fit_list = self.fit_data[row, column][index]
        #         folder = file_.create_group(data, 'wvl' + str(index))
        #         phase_centers = fit_list[3]['centers']
        #         phase_counts = fit_list[3]['counts']
        #         hist_fit = fit_list[1]
        #         hist_cov = fit_list[2]
        #         if hist_fit is False:
        #             hist_fit = []
        #             hist_cov = []
        #         file_.create_vlarray(folder, 'phase_centers', obj=phase_centers)
        #         file_.create_vlarray(folder, 'phase_counts', obj=phase_counts)
        #         file_.create_array(folder, 'hist_cov', obj=hist_cov)
        #     # update progress bar
        #     if self.verbose:
        #         self.pbar_iter += 1
        #         self.pbar.update(self.pbar_iter)

        if self.logging:
            self.__logger("debug information saved")

        # close file and progress bar
        file_.close()
        if self.verbose:
            self.pbar.finish()

    def loadPhotonData(self, row, column, wavelength_index):
        '''
        Get a photon list for a single pixel and wavelength.
        '''
        max_tries = 10
        for i in range(max_tries):
            try:
                if self.lock is None:
                    photon_list = self.obs[wavelength_index].getPixelPhotonList(
                        row, column)
                else:
                    self.lock.acquire()
                    photon_list = self.obs[wavelength_index].getPixelPhotonList(
                        row, column)
                    self.lock.release()
                return photon_list
            except KeyboardInterrupt:
                try:
                    self.lock.release()
                except:
                    pass
                raise KeyboardInterrupt
            except Exception as error:
                try:
                    self.lock.release()
                except:
                    pass
        return np.array([], dtype=[('Time', '<u4'), ('Wavelength', '<f4'),
                                    ('SpecWeight', '<f4'), ('NoiseWeight', '<f4')])

    def __checkParallelOptions(self):
        '''
        Check to make sure options that are incompatible with parallel computing are not
        enabled
        '''
        assert self.save_plots is False, "Cannot save histogram plots while " + \
            "running in parallel. save_plots must be False in the configuration file"

    def __cleanLogs(self):
        '''
        Combine logs when running in parallel after getPhaseHeightsParallel()
        '''
        worker_logs = ['worker' + str(num) + '.log' for num in range(self.cpu_count)]
        self.__logger("## fitting phase histograms")
        # loop through worker logs
        for log in worker_logs:
            with open(os.path.join(self.out_directory, 'logs', log)) as file_:
                contents = file_.readlines()
                contents = [line.strip() for line in contents]
                # log lines in the main log that start with pixel (row, column)
                for line in contents:
                    if len(line) != 0 and line[0] == '(':
                        self.__logger(line)

        # remove all worker logs
        log_file = os.path.join(self.out_directory, 'logs/worker*.log')
        for file_ in glob.glob(log_file):
                os.remove(file_)

    def __removeTailRidingPhotons(self, photon_list, dt):
        '''
        Remove photons that arrive too close together.
        '''
        # enforce time ordering (will remove once this is enforced in h5 file creation)
        indices = np.argsort(photon_list['Time'])
        photon_list = photon_list[indices]

        indices = np.where(np.diff(photon_list['Time']) > dt)[0] + 1
        photon_list = photon_list[indices]

        return photon_list

    def __histogramPhotons(self, phase_list):
        '''
        Create a histogram of the phase data for a specified bin width.
        '''
        phase_list = phase_list[phase_list < 0]
        if len(phase_list) == 0:
            phase_hist = {'centers': np.array([]), 'counts': np.array([])}
            return phase_hist
        min_phase = np.min(phase_list)
        max_phase = np.max(phase_list)

        # reload default bin_width
        self.bin_width = ast.literal_eval(self.config['Fit']['bin_width'])

        # make histogram and try twice to make the bin width larger if needed
        max_count = 0
        update = 0
        while max_count < 400 and update < 2:
            # update bin_width
            bin_width = self.bin_width * (2**update)

            # define bin edges being careful to start at the threshold cut
            bin_edges = np.arange(max_phase, min_phase - bin_width,
                                  -bin_width)[::-1]

            # make histogram
            counts, x0 = np.histogram(phase_list, bins=bin_edges)
            centers = (x0[:-1] + x0[1:]) / 2.0

            # update counters
            if len(counts) == 0:
                phase_hist = {'centers': np.array([]), 'counts': np.array([])}
                return phase_hist
            max_count = np.max(counts)
            update += 1
        # record final bin_width (needed for in situ plotting)
        self.bin_width = bin_width

        phase_hist = {'centers': np.array(centers), 'counts': np.array(counts)}

        return phase_hist

    def __fitSetup(self, phase_hist, fit_list, wavelength_index, fit_number):
        '''
        Get a good initial guess (and bounds) for the fitting model.
        '''
        if len(phase_hist['centers']) == 0:
            return None
        # check for successful fits for this pixel with a different wavelength
        recent_fit, recent_index, success = self.__findLastGoodFit(fit_list)
        if self.model_name == 'gaussian_and_exp':
            if fit_number == 0 and not success:
                # box smoothed guess fit with varying b
                params = self.__boxGuess(phase_hist)
            elif fit_number == 1 and not success:
                # median center guess fit with varying b
                params = self.__medianGuess(phase_hist)
            elif fit_number == 2 and not success:
                # fixed number guess fit with varying b
                params = self.__numberGuess(phase_hist, 0)
            elif fit_number == 3 and not success:
                # fixed number guess fit with varying b
                params = self.__numberGuess(phase_hist, 1)
            elif fit_number == 4 and not success:
                # fixed number guess fit with varying b
                params = self.__numberGuess(phase_hist, 2)

            elif fit_number == 0 and success:
                # wavelength scaled fit with fixed b
                params = self.__wavelengthGuess(phase_hist, recent_fit, recent_index,
                                                wavelength_index, b=recent_fit[1][1])
            elif fit_number == 1 and success:
                # box smoothed fit with fixed b
                params = self.__boxGuess(phase_hist, b=recent_fit[1][1])
            elif fit_number == 2 and success:
                # median center guess fit with fixed b
                params = self.__medianGuess(phase_hist, b=recent_fit[1][1])
            elif fit_number == 3 and success:
                # wavelength scaled fit with varying b
                params = self.__wavelengthGuess(phase_hist, recent_fit, recent_index,
                                                wavelength_index)
            elif fit_number == 4 and success:
                # box smoothed guess fit with varying b
                params = self.__boxGuess(phase_hist)
            elif fit_number == 5 and success:
                # median center guess fit with varying b
                params = self.__medianGuess(phase_hist)
            else:
                raise ValueError('fit_number not valid for this pixel')
            setup = params
        else:
            raise ValueError("{0} is not a valid fit model name".format(self.model_name))

        return setup

    def __boxGuess(self, phase_hist, b=None):
        '''
        Returns parameter guess based on a box smoothed histogram
        '''
        if b is None:
            vary = True
            b = 0.2
        else:
            vary = False
        threshold = max(phase_hist['centers'])
        exp_amplitude = (phase_hist['counts'][phase_hist['centers'] == threshold][0] /
                         np.exp(threshold * 0.2))

        box = np.ones(10) / 10.0
        phase_smoothed = np.convolve(phase_hist['counts'], box, mode='same')
        gaussian_center = phase_hist['centers'][np.argmax(phase_smoothed)]

        if (gaussian_center > 1.4 * threshold):  # remember both numbers are negative
            gaussian_center = np.max([1.5 * threshold, np.min(phase_hist['centers'])])

        gaussian_amplitude = 1.1 * np.max(phase_hist['counts']) / 2
        standard_deviation = 10

        params = lm.Parameters()
        params.add('a', value=exp_amplitude, min=0, max=np.inf)
        params.add('b', value=b, min=-1, max=np.inf, vary=vary)
        params.add('c', value=gaussian_amplitude, min=0,
                   max=1.1 * np.max(phase_hist['counts']))
        params.add('d', value=gaussian_center, min=np.min(phase_hist['centers']), max=0)
        params.add('f', value=standard_deviation, min=0.1, max=np.inf)

        return params

    def __wavelengthGuess(self, phase_hist, recent_fit, recent_index, wavelength_index,
                          b=None):
        '''
        Returns parameter guess based on previous wavelength solutions
        '''
        if b is None:
            vary = True
            b = recent_fit[1][1]
        else:
            vary = False
        # values derived from last wavelength
        exp_amplitude = recent_fit[1][0]
        gaussian_center = (recent_fit[1][3] * self.wavelengths[recent_index] /
                           self.wavelengths[wavelength_index])
        standard_deviation = recent_fit[1][4]

        # values derived from data (same as __boxGuess)
        gaussian_amplitude = 1.1 * np.max(phase_hist['counts']) / 2

        params = lm.Parameters()
        params.add('a', value=exp_amplitude, min=0, max=np.inf)
        params.add('b', value=b, min=-1, max=np.inf, vary=vary)
        params.add('c', value=gaussian_amplitude, min=0,
                   max=1.1 * np.max(phase_hist['counts']))
        params.add('d', value=gaussian_center, min=np.min(phase_hist['centers']), max=0)
        params.add('f', value=standard_deviation, min=0.1, max=np.inf)

        return params

    def __medianGuess(self, phase_hist, b=None):
        '''
        Returns parameter guess based on median histogram center
        '''
        if b is None:
            vary = True
            b = 0.2
        else:
            vary = False
        # new values
        centers = phase_hist['centers']
        counts = phase_hist['counts']
        gaussian_center = (np.min(centers) + np.max(centers)) / 2
        gaussian_amplitude = (np.min(counts) + np.max(counts)) / 2
        exp_amplitude = 0

        # old values (same as __boxGuess)
        standard_deviation = 10

        params = lm.Parameters()
        params.add('a', value=exp_amplitude, min=0, max=np.inf)
        params.add('b', value=b, min=-1, max=np.inf, vary=vary)
        params.add('c', value=gaussian_amplitude, min=0,
                   max=1.1 * np.max(phase_hist['counts']))
        params.add('d', value=gaussian_center, min=np.min(phase_hist['centers']), max=0)
        params.add('f', value=standard_deviation, min=0.1, max=np.inf)

        return params

    def __numberGuess(self, phase_hist, attempt):
        '''
        Hard coded numbers used as guess parameters
        '''
        if attempt == 0:
            a = 0
            b = 0.2
            c = min([1000, 1.1 * np.max(phase_hist['counts'])])
            d = max([-80, np.min(phase_hist['centers'])])
            f = 6
        elif attempt == 1:
            a = 1e7
            b = 0.2
            c = min([3e3, 1.1 * np.max(phase_hist['counts'])])
            d = max([-90, np.min(phase_hist['centers'])])
            f = 15
        elif attempt == 2:
            a = 4e6
            b = 0.15
            c = min([1000, 1.1 * np.max(phase_hist['counts'])])
            d = max([-80, np.min(phase_hist['centers'])])
            f = 10

        params = lm.Parameters()
        params.add('a', value=a, min=0, max=np.inf)
        params.add('b', value=b, min=-1, max=np.inf)
        params.add('c', value=c, min=0,
                   max=1.1 * np.max(phase_hist['counts']))
        params.add('d', value=d, min=np.min(phase_hist['centers']), max=0)
        params.add('f', value=f, min=0.1, max=np.inf)

        return params

    def __fitPhaseHistogram(self, phase_hist, fit_function, setup, row, column):
        '''
        Fit the phase histogram to the specified fit fit_function
        '''
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            error = np.sqrt(phase_hist['counts'] + 0.25) + 0.5
            model = lm.Model(fit_function)
            try:
                # fit data
                result = model.fit(phase_hist['counts'], setup, x=phase_hist['centers'],
                                   weights=np.sqrt(phase_hist['counts']) + 1)
                # lm fit doesn't error if covariance wasn't calculated so check here
                # replace with gaussian width if covariance couldn't be calculated
                if result.covar is None:
                    if self.model_name == 'gaussian_and_exp':
                        result.covar = np.ones((5, 5)) * result.best_values['f'] / 2
                # unpack results
                fit_result = (list(result.best_values.values()), result.covar)

            except (RuntimeError, RuntimeWarning, ValueError) as error:
                # RuntimeError catches failed minimization
                # RuntimeWarning catches overflow errors
                # ValueError catches if ydata or xdata contain Nans
                if self.logging:
                    self.__logger('({0}, {1}): '.format(row, column) + str(error))
                fit_result = (False, False)
            except TypeError as error:
                # TypeError catches when no data is passed to params.add()
                if self.logging:
                    self.__logger('({0}, {1}): '.format(row, column) + "No data passed "
                                  + "to the fit function")
                fit_result = (False, False)

        return fit_result

    def __evaluateFit(self, phase_hist, fit_result, fit_list, wavelength_index):
        '''
        Evaluate the result of the fit and return a flag for different conditions.
        '''
        if len(phase_hist['centers']) == 0:
            flag = 3  # no data to fit
            return flag
        if self.model_name == 'gaussian_and_exp':
            max_phase = max(phase_hist['centers'])
            min_phase = min(phase_hist['centers'])
            peak_upper_lim = np.min([-10, max_phase * 1.2])
            peak_lower_lim = min_phase

            # change peak_upper_lim if good fits exist for higher wavelengths
            recent_fit, recent_index, success = self.__findLastGoodFit(fit_list)
            if success:
                guess = (recent_fit[1][3] * self.wavelengths[recent_index] /
                         self.wavelengths[wavelength_index])
                peak_upper_lim = min([0.5 * guess, 1.1 * max_phase])

            if fit_result[0] is False:
                flag = 1  # fit did not converge
            else:
                centers = phase_hist['centers']
                counts = phase_hist['counts']
                center = fit_result[0][3]
                sigma = fit_result[0][4]
                gauss = lambda x: fitModels('gaussian')(x, *fit_result[0][2:])
                exp = lambda x: fitModels('exp')(x, *fit_result[0][:2])
                c_ind = np.argmin(np.abs(centers - center))
                c_p_ind = np.argmin(np.abs(centers - (center + sigma)))
                c_n_ind = np.argmin(np.abs(centers - (center - sigma)))
                c = counts[c_ind]
                c_p = counts[c_p_ind]
                c_n = counts[c_n_ind]
                h = gauss(centers[c_ind]) + exp(centers[c_ind])
                if max_phase < center + sigma:
                    h_p = gauss(max_phase) + exp(max_phase)
                else:
                    h_p = gauss(centers[c_p_ind]) + exp(centers[c_p_ind])
                if min_phase > center - sigma:
                    h_n = gauss(min_phase) + exp(min_phase)
                else:
                    h_n = gauss(centers[c_n_ind]) + exp(centers[c_n_ind])

                bad_fit_conditions = ((center > peak_upper_lim) or
                                      (center < peak_lower_lim) or
                                      (gauss(center) < 2 * exp(center)) or
                                      (gauss(center) < 10) or
                                      np.abs(sigma < 2) or
                                      np.abs(c - h) > 4 * np.sqrt(c) or
                                      np.abs(c_p - h_p) > 4 * np.sqrt(c_p) or
                                      np.abs(c_n - h_n) > 4 * np.sqrt(c_n))

                if bad_fit_conditions:
                    flag = 2  # fit converged to a bad solution
                else:
                    flag = 0  # fit converged

        else:
            raise ValueError("{0} is not a valid fit model name".format(self.model_name))

        return flag

    def __findBestFit(self, fit_results, flags, phase_hist):
        '''
        Finds the best fit out of a list based on chi squared
        '''
        centers = phase_hist['centers']
        counts = phase_hist['counts']
        chi2 = []
        for fit_result in fit_results:
            if fit_result[0] is False:
                chi2.append(np.inf)
            else:
                fit = fitModels(self.model_name)(centers, *fit_result[0])
                errors = np.sqrt(counts + 0.25) + 0.5
                chi2.append(np.sum(((counts - fit) / errors)**2))
        index = np.argmin(chi2)

        return fit_results[index], flags[index]

    def __fitEnergy(self, fit_type, phases, energies, guess, errors, row, column):
        '''
        Fit the phase histogram to the specified fit fit_function
        '''
        fit_function = fitModels(fit_type)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                params = lm.Parameters()
                if fit_type == 'linear':
                    params.add('b', value=guess[0])
                    params.add('c', value=guess[1])
                    output = lm.minimize(self.__energyChi2, params, method='leastsq',
                                         args=(phases, energies, errors, fit_function))
                elif fit_type == 'quadratic':
                    # params.add('p', value=guess[0] - guess[1] / 360, min=0)
                    # params.add('b', value=guess[1], max=0)
                    # params.add('a', expr='p + b/360')
                    # params.add('c', value=guess[2], min=-20)
                    # output = lm.minimize(self.__energyChi2, params, method='leastsq',
                    #                      args=(phases, energies, errors, fit_function))
                    params.add('vertex', value=-180, max=-180)
                    params.add('b', value=guess[1])
                    params.add('c', value=guess[2])
                    params.add('a', expr='-b/(2*vertex)')
                    output1 = lm.minimize(self.__energyChi2, params, method='leastsq',
                                          args=(phases, energies, errors, fit_function))

                    params['vertex'].set(max=np.inf)
                    params['vertex'].set(value=180, min=1e-4)
                    output2 = lm.minimize(self.__energyChi2, params, method='leastsq',
                                          args=(phases, energies, errors, fit_function))
                    if output1.success and output1.chisqr < output2.chisqr:
                        output = output1
                    else:
                        output = output2
                else:
                    raise ValueError('{0} is not a valid fit type'.format(fit_type))
                if output.success:
                    p = output.params.valuesdict()
                    if fit_type == 'linear':
                        popt = (0, p['b'], p['c'])
                    else:
                        popt = (p['a'], p['b'], p['c'])
                    fit_result = (popt, output.covar)
                else:
                    if self.logging:
                        self.__logger('({0}, {1}): '.format(row, column) + output.message)
                        fit_result = (False, False)

            except (Exception, Warning) as error:
                # shouldn't have any exceptions or warnings
                if self.logging:
                    self.__logger('({0}, {1}): '.format(row, column) + str(error))
                fit_result = (False, False)
                raise error

        return fit_result

    @staticmethod
    def __energyChi2(params, phases, energies, errors, fit_function):
        p = params.valuesdict()
        if 'a' not in p.keys():
            dfdx = p['b']
        else:
            dfdx = 2 * p['a'] * phases + p['b']

        chi2 = ((fit_function(p, phases) - energies) / (dfdx * errors))**2
        return chi2

    def __plotFit(self, phase_hist, fit_result, fit_function, flag, row, column):
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
            plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])

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
                g_func = fitModels('gaussian')
                e_func = fitModels('exp')
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
            self.__mergePlots()
            self.saved = True
            plt.close('all')

        # update plot counter
        self.plot_counter += 1

    def __setupPlots(self):
        '''
        Initialize plotting variables
        '''
        self.plot_counter = 0
        self.plots_x = 3
        self.plots_y = 4
        self.plots_per_page = self.plots_x * self.plots_y
        plot_file = os.path.join(self.out_directory, self.plot_file_name)
        if os.path.isfile(plot_file):
            answer = self.__query("{0} already exists. Overwrite?".format(plot_file),
                                  yes_or_no=True)
            if answer is False:
                answer = self.__query("Provide a new file name (type exit to quit):")
                if answer == 'exit':
                    raise UserError("User doesn't want to overwrite the plot file " +
                                    "... exiting")
                plot_file = os.path.join(self.out_directory, answer)
                while os.path.isfile(plot_file):
                    question = "{0} already exists. Choose a new file name " + \
                               "(type exit to quit):"
                    answer = self.__query(question.format(plot_file))
                    if answer == 'exit':
                        raise UserError("User doesn't want to overwrite the plot file " +
                                        "... exiting")
                    plot_file = os.path.join(self.out_directory, answer)
                self.plot_file_name = plot_file
            else:
                os.remove(plot_file)

    def __mergePlots(self):
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

    def __closePlots(self):
        '''
        Safely close plotting variables after plotting since the last page is only saved
        if it is full.
        '''
        if not self.saved:
            pdf = PdfPages(os.path.join(self.out_directory, 'temp.pdf'))
            pdf.savefig(self.fig)
            pdf.close()
            self.__mergePlots()
        plt.close('all')

    def __logger(self, message):
        '''
        Method for writing information to a log file
        '''
        file_name = os.path.join(self.out_directory, 'logs', self.log_file + '.log')
        with open(file_name, 'a') as log_file:
            log_file.write(message + os.linesep)

    @staticmethod
    def __findLastGoodFit(fit_list):
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
    def __query(question, yes_or_no=False, default="no"):
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

    def __checkPixelInputs(self, pixels):
        '''
        Check inputs for getPhaseHeights, calculateCoefficients and exportData
        '''
        if len(pixels) == 0:
            rows = range(self.rows)
            columns = range(self.columns)
            pixels = [(r, c) for r in rows for c in columns]
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
            # check for configuration file, log file, and lock type
            assert os.path.isfile(self.config_directory), \
                self.config_directory + " is not a valid configuration file"
            assert type(self.log_file) is str, "log_file must be a string"
            assert self.lock is None or type(self.lock) is mp.synchronize.Lock, \
                "lock must be a multiprocessing Lock() or None"
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
            assert 'parallel' in self.config['Fit'].keys(), \
                param.format('parallel', 'Fit')

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

        elif index == 2:
            # type check parameters
            assert type(self.wavelengths) is list, "wavelengths parameter must be a list."
            assert type(self.file_names) is list, "file_names parameter must be a list."
            assert type(self.model_name) is str, "model_name parameter must be a string."
            assert type(self.save_plots) is bool, "save_plots parameter must be a boolean"
            assert type(self.verbose) is bool, "verbose parameter bust be a boolean"
            assert type(self.directory) is str, "directory parameter must be a string"
            assert type(self.logging) is bool, "logging parameter must be a boolean"
            assert type(self.parallel) is bool, "parallel parameter must be a boolean"
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
    elif model_name == 'quadratic':
        fit_function = lambda p, x: p['a'] * x**2 + p['b'] * x + p['c']
    elif model_name == 'linear':
        fit_function = lambda p, x: p['b'] * x + p['c']

    return fit_function


class UserError(Exception):
    '''
    Custom error used to exit the waveCal program without traceback
    '''
    pass


class Worker(mp.Process):
    '''
    Worker class to send pixels to and do the histogram fits. Run by
    getPhaseHeightsParallel.
    '''
    def __init__(self, in_queue, out_queue, progress_queue, config_file, num, lock):
        super(Worker, self).__init__()
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.progress_queue = progress_queue
        self.config_file = config_file
        self.num = num
        self.lock = lock
        self.daemon = True
        self.start()

    def run(self):
        try:
            w = WaveCal(config_file=self.config_file, log_file='worker' + str(self.num),
                        lock=self.lock)
            w.verbose = False
            while True:
                time = round(datetime.utcnow().timestamp())
                pixel = self.in_queue.get()
                if pixel is None:
                    break
                w.getPhaseHeights(pixels=[pixel])
                pixel_dict = {pixel: w.fit_data[pixel[0], pixel[1]]}
                self.progress_queue.put(True)

                self.out_queue.put(pixel_dict)
        except (KeyboardInterrupt, BrokenPipeError):
            pass


class ProgressWorker(mp.Process):
    '''
    Worker class to make progress bar when using multiprocessing. Run by
    getPhaseHeightsParallel.
    '''
    def __init__(self, progress_queue, N):
        super(ProgressWorker, self).__init__()
        self.progress_queue = progress_queue
        self.N = N
        self.daemon = True
        self.start()

    def run(self):
        try:
            pbar = ProgressBar(widgets=[Percentage(), Bar(), '  (', Timer(), ') ',
                                        ETA(), ' '], maxval=self.N).start()
            pbar_iter = 0
            while pbar_iter < self.N:
                if self.progress_queue.get():
                    pbar_iter += 1
                    pbar.update(pbar_iter)
            pbar.finish()
        except (KeyboardInterrupt, BrokenPipeError):
            pass


if __name__ == '__main__':
    if len(sys.argv) == 1:
        w = WaveCal()
    else:
        w = WaveCal(config_file=sys.argv[1])

    w.makeCalibration()
