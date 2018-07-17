import os
import ast
import sys
import pickle
import warnings
import numpy as np
import tables as tb
from datetime import datetime
from matplotlib import pyplot as plt
from configparser import ConfigParser
from scipy.interpolate import interp1d

from Utils.utils import medianStack
from Utils.arrayPopup import plotArray
from ImageReg.loadStack import loadIMGStack
from Cleaning.HotPix import darkHotPixMask as dhpm


class Measurement:
    '''
    Class for holding the information of a QE measurement and running the analysis. After
    the Measurement object is innitialized with the configuration file, runAnalysis()
    should be run to compute the QE solution.

    Args:
        config_file: full path and file name of the configuration file for the QE analysis
                     (string)
    '''
    def __init__(self, config_file):
        # record start time in log file name
        self.log_file = str(round(datetime.utcnow().timestamp()))

        # define the configuration file path
        self.config_directory = config_file

        # check the configuration file path and read it in
        self.__configCheck(0)
        self.config = ConfigParser()
        self.config.read(self.config_directory)

        # check the configuration file format and load the parameters
        self.__configCheck(1)
        self.QE_file = ast.literal_eval(self.config['Data']['QE_file'])
        self.img_directory = ast.literal_eval(self.config['Data']['img_directory'])
        self.wavelengths = ast.literal_eval(self.config['Data']['wavelengths'])
        self.light_int = ast.literal_eval(self.config['Data']['light'])
        self.integration_time = ast.literal_eval(self.config['Data']['integration_time'])
        self.mkid_area = ast.literal_eval(self.config['Array']['mkid_area'])
        self.rows = ast.literal_eval(self.config['Array']['rows'])
        self.columns = ast.literal_eval(self.config['Array']['columns'])
        self.masks = ast.literal_eval(self.config['Masks']['masks'])
        if 'wavelength_cal' in self.masks:
            param = "{0} must be a parameter in the configuration file '{1}' section"
            assert 'waveCal_file' in self.config['Masks'].keys(), \
                param.format('masks', 'Masks')
            self.waveCal_file = ast.literal_eval(self.config['Masks']['waveCal_file'])
        self.opt_directory = ast.literal_eval(self.config['Optics']['opt_directory'])
        self.QE_factors = ast.literal_eval(self.config['Optics']['QE_factors'])
        self.out_directory = ast.literal_eval(self.config['Output']['out_directory'])
        self.logging = ast.literal_eval(self.config['Output']['logging'])
        self.verbose = ast.literal_eval(self.config['Output']['verbose'])

        # check the parameter formats
        self.__configCheck(2)

        # initialize logging directory if it doesn't exist and start log file
        if self.logging:
            if not os.path.isdir(os.path.join(self.out_directory, 'logs')):
                os.mkdir(os.path.join(self.out_directory, 'logs'))
            message = "QE Measurement object created: UTC " + str(datetime.utcnow()) + \
                " : Local " + str(datetime.now())
            self.__logger(message)
            self.__logger("## configuration file used")
            with open(self.config_directory, "r") as file_:
                config = file_.read()
            self.__logger(config)

        # make light and dark lists
        self.light = []
        self.dark = []
        for index, integer in enumerate(self.light_int):
            self.light.append([integer, integer + self.integration_time])
            if index != len(self.light_int) - 1:
                nextinteger = self.light_int[index + 1]
                offset = int((nextinteger - (integer + self.integration_time)) / 7)
                self.dark.append([integer + self.integration_time + offset,
                                  nextinteger - offset])
                previousinteger = integer
            else:
                self.dark.append([integer + self.integration_time + offset,
                                  integer + self.integration_time + offset + integer -
                                  previousinteger])
        # make wavelengths into numpy array
        self.wavelengths = np.array(self.wavelengths)

    def runAnalysis(self, plot=True, save=True):
        '''
        Run through all the steps in the QE analysis.

        Args:
            plot: determines if a QE plot is generated (boolean)
            save: determines if solution file (and plot) is (are) saved
        '''
        try:
            self.loadData()
            self.computeQE()
            self.computeTheory()

            if save:
                self.saveData()
            if plot:
                self.plotQE(save=save)
        except KeyboardInterrupt:
            print(os.linesep + "Shutdown requested ... exiting")

    def loadData(self):
        '''
        Load measurement text file and save photodiode irradiance.
        '''
        # load txt
        QE_data = np.loadtxt(self.QE_file)

        # make mask to ignore wavelengths not specified in the configuration file
        QE_wavelengths = QE_data[:, 0]
        mask = np.in1d(QE_wavelengths, self.wavelengths)

        # photo diode properties
        pd_flux = QE_data[mask, 4] * 1e7  # photons / s
        pd_area = 0.00155179165  # m^2
        self.pd_irradiance = pd_flux / pd_area  # photons / m^2 / s

        if self.logging:
            self.__logger("loadData(): loaded {0}".format(self.QE_file))

    def computeQE(self):
        '''
        Compute the quantum efficiency at each wavelength
        '''
        self.QE = []
        self.QE_masks = []
        for index, wavelength in enumerate(self.wavelengths):
            if self.verbose:
                print('computing QE for {0} nm'.format(wavelength))
            # load light and dark stacks
            light_stack = loadIMGStack(self.img_directory, self.light[index][0],
                                       self.light[index][1], nCols=self.columns,
                                       nRows=self.rows, verbose=False)
            dark_stack = loadIMGStack(self.img_directory, self.dark[index][0],
                                      self.dark[index][1], nCols=self.columns,
                                      nRows=self.rows, verbose=False)

            # median data
            median_light = medianStack(light_stack)
            median_dark = medianStack(dark_stack)

            # mask data
            self.QE_masks.append(self.makeMask(dark_stack, light_stack, wavelength))
            median_light[self.QE_masks[-1]] = np.nan
            median_dark[self.QE_masks[-1]] = np.nan

            mkid_irradiance = (median_light - median_dark) / self.mkid_area
            QE = mkid_irradiance / self.pd_irradiance[index]

            with warnings.catch_warnings():
                # nan values will give an unnecessary RuntimeWarning
                warnings.simplefilter("ignore", category=RuntimeWarning)
                logic = np.logical_or(QE < 0, QE > 1)
            self.QE_masks[-1][logic] = 1
            QE[logic] = np.nan

            if self.verbose:
                print('{0} pixels passed the cuts'.format(np.sum(self.QE_masks[-1] == 0)))
                # print(np.where(QE < 0.10))
                # plt.figure()
                # plt.hist(QE.flatten()[np.isfinite(QE.flatten())], 20)
                # plt.show()

            self.QE.append(QE)

    def makeMask(self, dark_stack, light_stack, wavelength):
        '''
        Make a mask for the stack to remove unwanted pixels
        '''
        masks = []
        if 'hot_and_cold' in self.masks:
            masks.append(dhpm.makeDHPMask(stack=dark_stack, maxCut=2400, coldCut=False))
            masks.append(dhpm.makeDHPMask(stack=light_stack, maxCut=2400, coldCut=True))

        if 'hot' in self.masks:
            masks.append(dhpm.makeDHPMask(stack=dark_stack, maxCut=2400, coldCut=False))
            masks.append(dhpm.makeDHPMask(stack=light_stack, maxCut=2400, coldCut=False))

        if 'dark_threshold' in self.masks:
            dark_mask = np.zeros(dark_stack[0].shape, dtype=int)
            for (row, column), _ in np.ndenumerate(dark_stack[0]):
                if max(dark_stack[:, row, column]) > 100:
                    dark_mask[row, column] = 1
            masks.append(dark_mask)

        if 'wavelength_cal' in self.masks:
            # load data
            wave_cal = tb.open_file(self.waveCal_file, mode='r')
            wavelengths = wave_cal.root.header.wavelengths.read()[0]
            calsoln = wave_cal.root.wavecal.calsoln.read()
            beamImage = wave_cal.root.header.beamMap.read()
            wave_cal.close()

            wavelength_index = np.argmin(np.abs(wavelengths - wavelength))
            waveCal_mask = np.zeros(dark_stack[0].shape, dtype=int)
            for (row, column), _ in np.ndenumerate(dark_stack[0]):
                res_id = beamImage[column][row]  # beam map is transposed from stack
                index = np.where(res_id == np.array(calsoln['resid']))
                wave_flag = calsoln['wave_flag'][index][0]
                # mask data if energy - phase fit failed
                if wave_flag != 4 and wave_flag != 5:
                    waveCal_mask[row, column] = 1
                # mask data if histogram fit failed
                elif calsoln['R'][index][0][wavelength_index] == -1:
                    waveCal_mask[row, column] = 1

            # add wavecal mask to the masks list
            masks.append(waveCal_mask)

        if len(masks) == 0:
            mask = np.ones(dark_stack[0].shape, dtype=int)
            return mask
        else:
            mask = masks[0]
            for current_mask in masks:
                mask = np.logical_or(current_mask, mask)
            return mask

    def plotMask(self, wavelength_index):
        '''
        Plot and show mask grid. Ones are bad pixels. Zeros are good pixels.
        '''
        fig, ax = plt.subplots()
        self.plotArray = self.QE_masks[wavelength_index]
        im = ax.imshow(np.array(self.plotArray), aspect='equal')
        ax.format_coord = Formatter(im)
        fig.colorbar(im)
        plt.show(block=False)

    def plotQEGrid(self, wavelength_index):
        '''
        Plot QE on the array grid.
        '''
        fig, ax = plt.subplots()
        self.plotArray = self.QE[wavelength_index]
        im = ax.imshow(self.plotArray, aspect='equal')
        ax.format_coord = Formatter(im)
        plt.colorbar(im)
        plt.show(block=False)

    def computeTheory(self):
        '''
        Compute the theoretical QE.
        '''
        self.wvl_theory = np.linspace(min(self.wavelengths), max(self.wavelengths),
                                      len(self.wavelengths) * 100)
        QE = np.ones(len(self.wvl_theory))
        for factor in self.QE_factors:
            if type(factor) is int or type(factor) is float:
                QE *= factor
            elif type(factor) is str:
                # load file
                file_ = os.path.join(self.opt_directory, factor)
                try:
                    data = np.loadtxt(file_)
                except:
                    message = 'could not load {0} with numpy.loadtxt()'
                    raise ValueError(message.format(file_))
                wavelengths = data[:, 0]
                multipliers = data[:, 1]

                # check that loaded data makes sense
                if np.logical_or(multipliers > 1, multipliers < 0).any():
                    message = "second column of {0} not normalized between 0 and 1"
                    raise ValueError(message.format(file_))
                max_wvl = max(wavelengths)
                min_wvl = min(wavelengths)
                if np.logical_or(min_wvl > min(self.wavelengths),
                                 max_wvl < max(self.wavelengths)):
                    message = "{0} only covers the range ({1}, {2}) nm. Data was " + \
                              "requested outside of the bounds and extrapolated"
                    warnings.warn(message.format(file_, min_wvl, max_wvl))

                # interpolate for each requested wavelength
                ind = np.argsort(wavelengths)
                wavelengths = wavelengths[ind]
                multipliers = multipliers[ind]
                interp = interp1d(wavelengths, multipliers, fill_value='extrapolate')

                QE *= interp(self.wvl_theory)

        self.QE_theory = QE

    def plotQE(self, save=False):
        '''
        Plot the measured and theoretical QE
        '''
        fig, ax = plt.subplots()

        QE_median = np.array([np.nanmedian(QE.flatten()) for QE in self.QE])
        QE_upper = np.array([QE_median[ind] + np.nanstd(QE.flatten())
                             for ind, QE in enumerate(self.QE)])
        QE_lower = np.array([QE_median[ind] - np.nanstd(QE.flatten())
                             for ind, QE in enumerate(self.QE)])

        ax.plot(self.wavelengths, QE_median * 100, linewidth=3, color='black',
                label=r'Measured')
        ax.fill_between(self.wavelengths, QE_lower * 100, QE_upper * 100,
                        where=QE_upper >= QE_lower, color='green', facecolor='green',
                        interpolate='True', alpha=0.1)
        ax.plot(self.wvl_theory, 100 * self.QE_theory, linestyle='-.', linewidth=2,
                color='black', label=r'Theoretical')

        ax.set_xlabel(r'Wavelength (nm)')
        ax.set_ylabel(r'QE (%)')
        ax.legend()
        ax.set_xlim([min(self.wavelengths), max(self.wavelengths)])
        ax.set_ylim([0, max(QE_upper) * 100 * 1.2])

        if save:
            file_name = os.path.join(self.out_directory, self.log_file + '.pdf')
            plt.savefig(file_name, format='pdf')
            plt.close(fig)
        else:
            plt.show(block=False)

    def saveData(self):
        '''
        Save Measurement instance in a pickle file
        '''
        with open(os.path.join(self.out_directory, self.log_file + '.p'), 'wb') as file_:
            pickle.dump(self, file_)

    def __formatCoords(self, x, y):
        shape = self.QE[0].shape
        col = int(x + 0.5)
        row = int(y + 0.5)
        if col >= 0 and col < shape[1] and row >= 0 and row < shape[0]:
            z = self.plotArray[row, col]
            return '(%1.0f, %1.0f) z=%1.0f' % (col, row, z)
        else:
            return '(%1.0f, %1.0f)' % (col, row)

    def __logger(self, message):
        '''
        Method for writing information to a log file
        '''
        file_name = os.path.join(self.out_directory, 'logs', self.log_file + '.log')
        with open(file_name, 'a') as log_file:
            log_file.write(message + os.linesep)

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

            assert 'Data' in self.config.sections(), section.format('Data')
            assert 'QE_file' in self.config['Data'].keys(), \
                param.format('QE_file', 'Data')
            assert 'img_directory' in self.config['Data'].keys(), \
                param.format('img_directory', Data)
            assert 'wavelengths' in self.config['Data'].keys(), \
                param.format('wavelengths', 'Data')
            assert 'light' in self.config['Data'].keys(), \
                param.format('light', 'Data')
            assert 'integration_time' in self.config['Data'].keys(), \
                param.format('dark', 'Data')

            assert 'Array' in self.config.sections(), section.format('Array')
            assert 'mkid_area' in self.config['Array'].keys(), \
                param.format('mkid_area', 'Array')
            assert 'rows' in self.config['Array'].keys(), \
                param.format('rows', 'Array')
            assert 'columns' in self.config['Array'].keys(), \
                param.format('columns', 'Array')

            assert 'Masks' in self.config.sections(), section.format('Masks')
            assert 'masks' in self.config['Masks'].keys(), \
                param.format('masks', 'Masks')

            assert 'Optics' in self.config.sections(), section.format('Optics')
            assert 'opt_directory' in self.config['Optics'].keys(), \
                param.format('opt_directory', 'Optics')
            assert 'QE_factors' in self.config['Optics'].keys(), \
                param.format('QE_factors', 'Optics')

            assert 'Output' in self.config.sections(), section.format('Output')
            assert 'out_directory' in self.config['Output'], \
                param.format('out_directory', 'Output')
            assert 'logging' in self.config['Output'], \
                param.format('logging', 'Output')
            assert 'verbose' in self.config['Output'], \
                param.format('verbose', 'Output')

        elif index == 2:
            # type check parameters
            assert type(self.QE_file) is str, "QE_file parameter must be a string"
            assert os.path.isfile(self.QE_file), \
                "{0} parameter is not a valid file".format(self.QE_file)

            assert type(self.img_directory) is str, \
                "img_directory parameter must be a string"
            assert os.path.isdir(self.img_directory), \
                "{0} is not a valid directory".format(self.img_directory)

            assert type(self.wavelengths) is list, "wavelengths parameter must be a list."
            for index, lambda_ in enumerate(self.wavelengths):
                if type(lambda_) is int:
                    self.wavelengths[index] = float(self.wavelengths[index])
                assert type(self.wavelengths[index]) is float, \
                    "elements in wavelengths parameter must be floats or integers."

            assert type(self.light_int) is list, "light parameter must be a list"
            for _, timestamp in enumerate(self.light_int):
                assert type(timestamp) is int, \
                    "elements in light parameter must be integers"
            assert type(self.integration_time) is int, \
                "integration_time parameter must be a int"

            if type(self.mkid_area) is int:
                self.mkid_area = float(self.mkid_area)
            assert type(self.mkid_area) is float, \
                "mkid_area parameter must be a float or integer"

            assert type(self.rows) is int, "rows parameter must be an integer"
            assert type(self.columns) is int, "columns parameter must be an integer"

            assert type(self.masks) is list, "masks parameter must be a list of strings"
            for mask in self.masks:
                assert type(mask) is str, "masks parameter must be a list of strings"
                mask_list = ['hot_and_cold', 'hot', 'wavelength_cal', 'dark_threshold']
                assert mask in mask_list, "{0} is not a valid mask name".format(mask)
                if mask == 'wavelength_cal':
                    assert os.path.isfile(self.waveCal_file), \
                        "{0} is not a valid file".format(self.waveCal_file)

            assert type(self.opt_directory) is str, \
                "opt_directory parameter must be a string"
            assert os.path.isdir(self.opt_directory), \
                "{0} is not a valid directory".format(self.opt_directory)

            message = "QE_factors parameter must be a list of file names and numbers" + \
                      " between 0 and 1"
            assert type(self.QE_factors) is list, message
            for element in self.QE_factors:
                if type(element) is str:
                    directory = os.path.join(self.opt_directory, element)
                    assert os.path.isfile(directory), \
                        "{0} is not a valid directory".format(directory)
                elif type(element) is int or type(element) is float:
                    assert (element >= 0 and element <= 1), message
                else:
                    raise ValueError(message)

            assert type(self.logging) is bool, "logging parameter must be a boolean"

            assert type(self.out_directory) is str, \
                "out_directory parameter must be a string"
            assert os.path.isdir(self.out_directory), \
                "{0} is not a valid directory".format(self.out_directory)

            assert type(self.verbose) is bool, "verbose parameter must be a boolean"

        else:
            raise ValueError("index must be 0, 1, or 2")


class Formatter(object):
    '''
    Custom formatting class for mouse over text
    '''
    def __init__(self, im):
        self.im = im

    def __call__(self, x, y):
        return '({:1.0f}, {:1.0f})'.format(x, y)


if __name__ == '__main__':
    qe = Measurement(config_file=sys.argv[1])
    qe.runAnalysis(plot=True, save=True)
