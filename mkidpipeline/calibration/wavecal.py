#!/bin/env python3
import argparse
import ast
import atexit
import multiprocessing as mp
import os
import subprocess as sp
import time
import warnings
from configparser import ConfigParser
from datetime import datetime

import lmfit as lm
import numpy as np
import tables as tb
from PyPDF2 import PdfFileMerger, PdfFileReader
from astropy.constants import c, h
from matplotlib import lines, pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from progressbar import Bar, ETA, Percentage, ProgressBar, Timer

import mkidpipeline.utils.pipelinelog as pipelinelog
from mkidpipeline.wavecal import fitModels, plotSummary
from mkidpipeline.core import pixelflags
from mkidpipeline.core.headers import (WaveCalDebugDescription, WaveCalDescription, WaveCalHeader)
from mkidpipeline.hdf.darkObsFile import ObsFile
from mkidpipeline.utils.pipelinelog import getLogger

DEFAULT_CONFIG_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                   'Params', 'default.cfg')
BIN2HDF_DEFAULT_PATH = '/mnt/data0/DarknessPipeline/RawDataProcessing'


def makeHDFscripts(wavecfg):
    scriptpath = wavecfg.h5directory + 'makeh5files_{}.sh'

    scripts = []
    for wave, startt, intt  in zip(wavecfg.wavelengths, wavecfg.startTimes, wavecfg.expTimes):
        wavepath = '{}{}nm.txt'.format(wavecfg.h5directory, wave)

        BIN2HDFConfig(wavepath, datadir=wavecfg.dataDir, beamdir=wavecfg.beamDir,
                      outdir=wavecfg.h5directory, starttime=startt, inttime=intt, x=wavecfg.xpix, y=wavecfg.ypix).write()

        # TODO bin2hdf should with a path and require running in the directory with its python scripts
        scripts.append(scriptpath.format(wave))
        with open(scripts[-1], 'w') as script:
            getLogger('WaveCal').info('Creating {}'.format(scripts[-1]))
            script.write('#!/bin/bash\n'
                         'cd {}\n'.format(wavecfg.bin2hdf_path)+
                         '{} {}\n'.format('./Bin2HDF ', wavepath)+
                         'cd -')
    return scripts


def findDifferences(solution1, solution2):
    """
    Determines the pixels that were fit differently between the two solution files. This
    function is useful for comparing solution files made with different WaveCal versions
    and solution files made at different times during an observation.

    Args:
        solution1: the file name of the first wavelength calibration .h5 file (string)
        solution2: the file name of the second wavelength calibration .h5 file (string)

    Returns:
        good_to_bad: list of tuples containing pixels (row, column) that were good in
                     solution 1 but bad in solution 2
        bad_to_good: list of tuples containing pixels (row, column) that were bad in
                     solution 1 but good in solution 2
    """
    wave_cal1 = tb.open_file(solution1, mode='r')
    wave_cal2 = tb.open_file(solution2, mode='r')
    calsoln1 = wave_cal1.root.wavecal.calsoln.read()
    calsoln2 = wave_cal2.root.wavecal.calsoln.read()
    wave_cal1.close()
    wave_cal2.close()
    flag1 = calsoln1['wave_flag']
    flag2 = calsoln2['wave_flag']
    res_id1 = calsoln1['resid']
    res_id2 = calsoln2['resid']

    good_to_bad = []
    for index, res_id in enumerate(res_id1):
        if flag1[index] == 4 or flag1[index] == 5:
            index2 = np.where(res_id == res_id2)
            if len(index2[0]) == 1 and (flag2[index2][0] != 4 and flag2[index2][0] != 5):
                row = calsoln1['pixel_row'][index]
                column = calsoln1['pixel_col'][index]
                good_to_bad.append((row, column))

    bad_to_good = []
    for index, res_id in enumerate(res_id1):
        if flag1[index] != 4 and flag1[index] != 5:
            index2 = np.where(res_id == res_id2)
            if len(index2[0]) == 1 and (flag2[index2][0] == 4 or flag2[index2][0] == 5):
                row = calsoln1['pixel_row'][index]
                column = calsoln1['pixel_col'][index]
                bad_to_good.append((row, column))

    return good_to_bad, bad_to_good


class BIN2HDFConfig(object):
    #TODO this should not be part of wavecal
    template = ('{x} {y}\n'
                '{datadir}\n'
                '{starttime}\n'
                '{inttime}\n'
                '{beamdir}\n'
                '1\n'
                '{outdir}')

    def __init__(self, file, datadir='./', beamdir = './', starttime = None, inttime = None,
                 outdir = './', x=140, y=145):
        self.file = file
        self.datadir = datadir
        self.starttime = starttime
        self.inttime = inttime
        self.beamdir = beamdir
        self.outdir = outdir
        self.x = x
        self.y = y

    def write(self, file=None):
        with open(file if isinstance(file, str) else self.file, 'w') as wavefile:
            wavefile.write(BIN2HDFConfig.template.format(datadir=self.datadir, starttime=self.starttime,
                                                  inttime=self.inttime, beamdir=self.beamdir,
                                                  outdir=self.outdir,x=self.x,y=self.y))

    def load(self):
        raise NotImplementedError


class WaveCalConfig:
    def __init__(self, file='default.cfg', cal_file_name = 'calsol_default.h5'):

        # define the configuration file path
        self.file = DEFAULT_CONFIG_FILE if file == 'default.cfg' else file

        assert os.path.isfile(self.file), \
            self.file + " is not a valid configuration file"

        self.config = ConfigParser()
        self.config.read(self.file)

        # Prevent accidental default overwrite
        if self.file == DEFAULT_CONFIG_FILE:
            self.file = os.path.join(os.getcwd(),'default.cfg')

        # check the configuration file format and load the parameters
        self.checksections()

        #From runwavecal
        self.startTimes = ast.literal_eval(self.config['Data']['startTimes'])
        self.xpix= ast.literal_eval(self.config['Data']['xpix'])
        self.ypix= ast.literal_eval(self.config['Data']['ypix'])
        self.expTimes = ast.literal_eval(self.config['Data']['expTimes'])
        self.dataDir = ast.literal_eval(self.config['Data']['dataDir'])
        self.beamDir = ast.literal_eval(self.config['Data']['beamDir'])

        self.wavelengths = [l for l in ast.literal_eval(self.config['Data']['wavelengths'])]
        self.file_names = ast.literal_eval(self.config['Data']['file_names'])
        self.h5directory = ast.literal_eval(self.config['Data']['h5directory'])
        self.model_name = ast.literal_eval(self.config['Fit']['model_name'])
        self.bin_width = ast.literal_eval(self.config['Fit']['bin_width'])
        self.dt = ast.literal_eval(self.config['Fit']['dt'])
        self.parallel = ast.literal_eval(self.config['Fit']['parallel'])
        self.out_directory = ast.literal_eval(self.config['Output']['out_directory'])
        self.save_plots = ast.literal_eval(self.config['Output']['save_plots'])
        self.plot_file_name = ast.literal_eval(self.config['Output']['plot_file_name'])
        self.verbose = ast.literal_eval(self.config['Output']['verbose'])
        self.logging = ast.literal_eval(self.config['Output']['logging'])
        self.summary_plot = ast.literal_eval(self.config['Output']['summary_plot'])
        self.templar_config = ast.literal_eval(self.config['Output']['templar_config'])

        if self.config.has_option('Data', 'bin2hdf_path'):
            self.bin2hdf_path = ast.literal_eval(self.config['Data']['bin2hdf_path'])

        else:
            self.bin2hdf_path = BIN2HDF_DEFAULT_PATH

        self.cal_file_name = cal_file_name

        # check the parameter formats
        self.checktypes()

    def checksections(self):
        # check if all sections and parameters exist in the configuration file
        section = "{0} must be a configuration section"
        param = "{0} must be a parameter in the configuration file '{1}' section"

        assert 'Data' in self.config.sections(), section.format('Data')
        assert 'h5directory' in self.config['Data'].keys(), \
            param.format('h5directory', 'Data')
        assert 'wavelengths' in self.config['Data'].keys(), \
            param.format('wavelengths', 'Data')
        assert 'file_names' in self.config['Data'].keys(), \
            param.format('file_names', 'Data')
        assert 'startTimes' in self.config['Data'].keys(), \
            param.format('startTimes', 'Data')
        assert 'expTimes' in self.config['Data'].keys(), \
            param.format('expTimes', 'Data')
        assert 'dataDir' in self.config['Data'].keys(), \
            param.format('dataDir', 'Data')
        assert 'beamDir' in self.config['Data'].keys(), \
            param.format('beamDir', 'Data')
        assert 'xpix' in self.config['Data'].keys(), \
            param.format('xpix', 'Data')
        assert 'ypix' in self.config['Data'].keys(), \
            param.format('ypix', 'Data')

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
        assert 'summary_plot' in self.config['Output'], \
            param.format('summary_plot', 'Output')
        assert 'templar_config' in self.config['Output'], \
            param.format('templar_config', 'Output')

    def checktypes(self):
        # type check parameters
        assert type(self.startTimes) is list, "startTimes parameter must be a list."
        assert type(self.expTimes) is list, "expTimes parameter must be a list."

        assert type(self.dataDir) is str, "Data directory parameter must be a string"
        assert type(self.beamDir) is str, "Beam directory parameter must be a string"
        assert type(self.h5directory) is str, "H5 directory parameter must be a string"
        assert os.path.isdir(self.h5directory), \
            "{0} is not a valid output directory".format(self.h5directory)
        assert type(self.xpix) is int, "Number of X Pix parameter must be an integer"
        assert type(self.ypix) is int, "Number of Y Pix parameter must be an integer"

        assert type(self.wavelengths) is list, "wavelengths parameter must be a list."
        assert type(self.file_names) is list, "file_names parameter must be a list."
        assert type(self.model_name) is str, "model_name parameter must be a string."
        assert type(self.save_plots) is bool, "save_plots parameter must be a boolean"
        assert type(self.verbose) is bool, "verbose parameter bust be a boolean"
        assert type(self.logging) is bool, "logging parameter must be a boolean"
        assert type(self.parallel) is bool, "parallel parameter must be a boolean"
        assert type(self.summary_plot) is bool, "summary_plot parameter must be a boolean"
        assert type(self.plot_file_name) is str, \
            "plot_file_name parameter must be a string"
        assert type(self.templar_config) is str, \
            "templar_config parameter must be a string"
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

        assert len(self.wavelengths) == len(self.startTimes), \
            "wavelengths and startTimes parameters must be the same length."

        assert len(self.wavelengths) == len(self.expTimes), \
            "wavelengths and expTimes parameters must be the same length."

        try:
            self.wavelengths = [l for l in map(float,self.wavelengths)]
        except:
            raise AssertionError("elements in wavelengths parameter must be floats or integers.")

        for file_ in self.file_names:
            assert type(file_) is str, "elements in filenames " + \
                                       "parameter must be strings."

    def hdfexist(self):
        fqps = [os.path.join(self.h5directory, file_) for file_ in self.file_names]
        return all(map(os.path.isfile,fqps))

    def _computeHDFnames(self):
        self.file_names = ['%d' % st + '.h5' for st in self.startTimes]

    def enforceselfconsistency(self):
        self._computeHDFnames()

    def write(self, file, forceconsistency=True):
        if forceconsistency:
            self.enforceselfconsistency() #Force self consistency
        with open(file,'w') as f:
            f.write('[Data]\n'
                    '\n'
                    'h5directory = "{}"\n'.format(self.h5directory) +
                    'wavelengths = {}\n'.format(self.wavelengths)+
                    'file_names = {}\n'.format(str(self.file_names))+
                    'startTimes = {}\n'.format(self.startTimes) +
                    'expTimes = {}\n'.format(self.expTimes) +
                    'dataDir = "{}"\n'.format(self.dataDir) +
                    'beamDir = "{}"\n'.format(self.beamDir) +
                    'xpix = {}\n'.format(self.xpix) +
                    'ypix = {}\n'.format(self.ypix) +
                    '\n'
                    '[Fit]\n'
                    '\n'
                    'model_name = "{}"\n'.format(self.model_name)+
                    'bin_width = {}\n'.format(self.bin_width)+
                    'dt = {}\n'.format(self.dt)+
                    'parallel = {}\n'.format(self.parallel)+
                    '\n'
                    '[Output]\n'
                    '\n'
                    'out_directory = "{}"\n'.format(self.out_directory)+
                    'save_plots = {}\n'.format(self.save_plots) +
                    'plot_file_name = "{}"\n'.format(self.plot_file_name) +
                    'summary_plot = {}\n'.format(self.summary_plot) +
                    'templar_config = "{}"\n'.format(self.templar_config) +
                    'verbose = {}\n'.format(self.verbose) +
                    'logging = {}'.format(self.logging))


class WaveCal:
    """
    Class for creating wavelength calibrations for ObsFile formated data. After the
    WaveCal object is innitialized with the configuration file, makeCalibration() should
    be run to compute the calibration solution .h5 file.

    Args:
        Public Options
        config_file: full path and file name of the configuration file for the wavelength
                     calibration (string)

        Private Options
        These are used internally for parallel processing. Use the configuration file to
        specify parallel computations.
        Only one of the following three arguments can be true
        master: determines if the object is the master of a parallel computation (boolean)
        data_slave: determines if the object in charge of accessing the .h5 files
                    (boolean)
        worker_slave: determines if the object is in charge of computing the histogram
                      fits for the pixels assigned to it

        If the object is a worker_slave, five more arguments are required.
        pid: Unique number to identify the process internally (integer)
        request_data: multiprocessing queue object used to request pixels from the
                      data_slave.
        load_data: multiprocessing queue used to retrieve .h5 file contents from the data
                   slave
        rows: number of rows in the array (needed because the .h5 files can't be opened
              by the worker_slave)
        columns: number of columns in the array (needed because the .h5 files can't be
                 opened by the worker_slave)

    Created by: Nicholas Zobrist, January 2018
    """
    def __init__(self, config='default.cfg', master=True, data_slave=False,
                 worker_slave=False, pid=None, request_data=None, load_data=None,
                 rows=None, columns=None, filelog=None):
        # determine info about master/slave settings for parallel computing
        # (internal use only)

        self.master = master
        self.data_slave = data_slave
        self.worker_slave = worker_slave
        self.pid = pid
        self.request_data = request_data
        self.load_data = load_data
        self.rows = rows
        self.columns = columns
        self._checkbasics()

        #load configuration
        self.cfg = config if isinstance(config, WaveCalConfig) else WaveCalConfig(config)
        self.bin_width = self.cfg.bin_width

        if filelog in (None, False):
            self._log = getLogger('devnull')
            self._log.disabled = True
        elif filelog is True:
            self._log = pipelinelog.createFileLog('WaveCal.logfile', os.path.join(os.getcwd(),
                                                  'WaveCal{:.0f}.log'.format(datetime.utcnow().timestamp())))
        else:
            self._log = filelog

        self._clog = getLogger('WaveCal')

        if not self.master and not (self.worker_slave or self.data_slave):
            raise ValueError('WaveCal must be either a master or a slave')

        # create output file name
        self.cal_file = os.path.join(self.cfg.out_directory,
                                     self.cfg.cal_file_name)

        # arrange the files in increasing wavelength order and open all of the h5 files
        indices = np.argsort(self.cfg.wavelengths)
        self.wavelengths = np.array(self.cfg.wavelengths)[indices]
        self.file_names = np.array(self.cfg.file_names)[indices]
        if self.master or self.data_slave:
            self.obs = [ObsFile(os.path.join(self.cfg.h5directory, f)) for f in self.file_names]

            # get the array size from the beam map and check that all files are the same
            self.rows, self.columns = np.shape(self.obs[0].beamImage)

        self._checkbeammaps()

        # close obs files if running in parallel
        if self.master and self.cfg.parallel:
            for obs in self.obs:
                obs.file.close()  #TODO Should the obsfile object be changed so it is inherently safe

        # initialize output flag definitions
        self.flag_dict = pixelflags.waveCal

        message = "WaveCal object created: UTC " + str(datetime.utcnow()) + \
            " : Local " + str(datetime.now())
        self._log.info(message)

        if self.master:
            with open(self.cfg.file, "r") as file_:
                config = file_.read()
            self._log.info("## configuration file used\n"+config)

    def _checkbasics(self):
        """
        Checks some basics for consistency. Run in the '__init__()' method.
        """
        # check for configuration file, and any other keyword args

        assert type(self.master) is bool, "master keyword must be a boolean"
        assert type(self.data_slave) is bool, "data_slave keyword must be a boolean"
        assert type(self.worker_slave) is bool, \
            "worker_slave keyword must be a boolean"
        assert np.sum([self.master, self.data_slave, self.worker_slave]) == 1, \
            "WaveCal can only be one of the master/slaves at a time"
        assert self.request_data is None or \
            type(self.request_data) is mp.queues.Queue, \
            "request_data keyword must be None or a multiprocessing queue"
        assert self.load_data is None or type(self.load_data) is mp.queues.Queue, \
            "load_data keyword must be None or a multiprocessing queue"
        assert self.rows is None or type(self.rows) is int, \
            "rows keyword must be None or and int"
        assert self.columns is None or type(self.columns) is int, \
            "columns keyword must be None or and int"

    def _checkbeammaps(self):
        # check that all beammaps are the same
        if (self.master and not self.cfg.parallel) or self.data_slave:
            for obs in self.obs:
                assert np.shape(obs.beamImage) == (self.rows, self.columns), \
                    "All files must have the same beam map shape."

    def makeCalibration(self, pixels=[]):
        """
        Compute the wavelength calibration for the pixels in 'pixels' and save the data
        in the standard .h5 format.

        Args:
            pixels: a list of length 2 lists containing the (row, column) of the pixels
                    on which to compute a phase-energy relation. If it isn't specified,
                    all of the pixels in the array are used.

        Returns:
            Nothing is returned but a .h5 solution file is saved in the output directory
            specified in the configuration file.
        """
        with warnings.catch_warnings():
            # ignore unclosed file warnings from PyPDF2
            warnings.simplefilter("ignore", category=ResourceWarning)
            try:
                if self.cfg.parallel:
                    self._checkParallelOptions()
                    self.cpu_count = int(np.ceil(mp.cpu_count() / 2))
                    self.getPhaseHeightsParallel(self.cpu_count, pixels=pixels)
                else:
                    self.getPhaseHeights(pixels=pixels)
                self.calculateCoefficients(pixels=pixels)
                self.exportData(pixels=pixels)
                if self.cfg.summary_plot:
                    self.dataSummary()
            except (KeyboardInterrupt, BrokenPipeError):
                log.info(os.linesep + "Shutdown requested ... exiting")
            except UserError as err:
                log.error(err)

    def getPhaseHeightsParallel(self, n_processes, pixels=[]):
        """
        Fits the phase height histogram to a model for a specified list of pixels. Uses
        more than one process to speed up the computation.

        Args:
            n_processes: number of processes to generate to compute the histogram fits.
                         Three additional processes are needed for the main file,
                         accessing the .h5 files and printing information to the terminal
                         (if verbose is True in the config file).
            pixels: a list of length 2 lists containing the (row, column) of the pixels
                    on which to compute a phase-energy relation. If it isn't specified,
                    all of the pixels in the array are used.

        Returns:
            Nothing is returned, but a self.fit_data attribute is created. It is a numpy
            array of shape (self.row, self.columns). Each index contains the information
            about the fits for the pixel in (self.row, self.column). For each pixel a
            list of the fit information for each wavelength is stored. In that list is
            saved the fit flag, the fit result, the fit covariance, and a dictionary
            containing the phase histogram in that order.
        """
        # check inputs
        pixels = self._checkPixelInputs(pixels)

        if self.cfg.verbose:
            log.info('fitting phase histograms')

        # create progress bar
        if self.cfg.verbose:
            progress_queue = mp.Queue()
            N = len(pixels)
            progress = ProgressWorker(progress_queue, N)
        else:
            progress_queue = None

        # make request photon data queue
        request_data = mp.Queue()
        # make process specific load data queue and save in list
        load_data = []
        for i in range(n_processes):
            load_data.append(mp.Queue())
        # start process to handle accessing the .h5 files
        N = len(pixels) * len(self.wavelengths)
        gate_keeper = GateWorker(self.cfg.file, N, request_data, load_data)

        # make pixel in and result out queues
        in_queue = mp.Queue()
        out_queue = mp.Queue()
        # make workers to process the data
        workers = []
        for i in range(n_processes):
            workers.append(Worker(in_queue, out_queue, progress_queue,
                                  self.cfg.file, i, request_data, load_data[i],
                                  self.rows, self.columns, self._log))

        try:
            # give workers pixels to compute ending in n_processes close commands
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
            if self.cfg.verbose:
                progress.join()
            for w in workers:
                w.join()
            gate_keeper.join()
        except (KeyboardInterrupt, BrokenPipeError):
            # close queues
            while not in_queue.empty():
                in_queue.get()
            in_queue.close()
            while not out_queue.empty():
                out_queue.get()
            out_queue.close()
            if self.cfg.verbose:
                while not progress_queue.empty():
                    progress_queue.get()
                progress_queue.close()
            while not request_data.empty():
                request_data.get()
            request_data.close()
            for q in load_data:
                while not q.empty():
                    q.get()
                q.close()
            # close processes
            if self.cfg.verbose:
                log.info(os.linesep + "PID {0} ... exiting".format(progress.pid))
                progress.terminate()
                progress.join()
            for w in workers:
                log.info("PID {0} ... exiting".format(w.pid))
                w.terminate()
                w.join()
            log.info("PID {0} ... exiting".format(gate_keeper.pid))
            gate_keeper.terminate()
            gate_keeper.join()

            raise KeyboardInterrupt

        # populate fit_data with results from workers
        self.fit_data = np.empty((self.rows, self.columns), dtype=object)
        for ind, _ in np.ndenumerate(self.fit_data):
            self.fit_data[ind] = []
        for (row, column) in result_dict.keys():
            self.fit_data[row, column] = result_dict[(row, column)]

    def getPhaseHeights(self, pixels=[]):
        """
        Fits the phase height histogram to a model for a specified list of pixels.

        Args:
            pixels: a list of length 2 lists containing the (row, column) of the pixels
                    on which to compute a phase-energy relation. If it isn't specified,
                    all of the pixels in the array are used.

        Returns:
            Nothing is returned, but a self.fit_data attribute is created. It is a numpy
            array of shape (self.row, self.columns). Each index contains the information
            about the fits for the pixel in (row, column). For each pixel a list of the
            fit information for each wavelength is stored. In that list is saved the fit
            flag, the fit result, the fit covariance, and a dictionary containing the
            phase histogram in that order.
        """
        # check inputs
        pixels = self._checkPixelInputs(pixels)

        # initialize plotting, logging, and verbose
        if self.cfg.save_plots:
            self._setupPlots()
        self._log.info("## fitting phase histograms")
        if self.cfg.verbose:
            log.info('fitting phase histograms')
            self.pbar = ProgressBar(widgets=[Percentage(), Bar(), '  (',
                                             Timer(), ') ', ETA(), ' '],
                                    max_value=len(pixels)).start()
            self.pbar_iter = 0

        # initialize fit_data structure
        fit_data = np.empty((self.rows, self.columns), dtype=object)
        for ind, _ in np.ndenumerate(fit_data):
            fit_data[ind] = []

        # loop over pixels and fit the phase histograms
        for row, column in pixels:
            # initialize rate parameter
            rate = 2000
            for wavelength_index, wavelength in enumerate(self.wavelengths):

                start_time = datetime.now()
                # pull out fits already done for this wavelength
                fit_list = fit_data[row, column]

                # load data
                photon_list = self.loadPhotonData(row, column, wavelength_index)

                # recalculate event rate [#/s] if it's been flaged as hot before
                if rate > 1800 and len(photon_list['Wavelength']) > 1:
                    rate = (len(photon_list['Wavelength']) /
                            (max(photon_list['Time']) - min(photon_list['Time']))) * 1e6

                # if there is no data go to next loop
                if len(photon_list['Wavelength']) <= 1:
                    fit_data[row, column].append((3, False, False,
                                                  {'centers': np.array([]),
                                                   'counts': np.array([])}))
                    # update progress bar and log
                    dt = str(round((datetime.now() - start_time).total_seconds(), 2)) + ' s'
                    self._log.info("({0}, {1}) {2}nm: {3} : {4}".format(row, column, wavelength,
                                    self.flag_dict[3], dt))
                    if self.cfg.verbose and wavelength_index == len(self.wavelengths) - 1:
                        self.pbar_iter += 1
                        self.pbar.update(self.pbar_iter)
                    continue

                # cut photons too close together in time
                photon_list = self._removeTailRidingPhotons(photon_list, self.cfg.dt)

                # make the phase histogram
                phase_hist = self._histogramPhotons(photon_list['Wavelength'])

                # if there is not enough data or too much go to next loop
                if len(phase_hist['centers']) == 0 or np.max(phase_hist['counts']) < 20:
                    flag = 3
                elif rate > 1800:
                    flag = 10
                else:
                    flag = 0  # for now
                if flag == 3 or flag == 10:
                    fit_data[row, column].append((flag, False, False, phase_hist))
                    # update progress bar and log
                    dt = str(round((datetime.now() - start_time).total_seconds(), 2)) + ' s'
                    self._log.info("({0}, {1}) {2}nm: {3} : {4}".format(row, column, wavelength,
                                    self.flag_dict[3], dt))
                    if self.cfg.verbose and wavelength_index == len(self.wavelengths) - 1:
                        self.pbar_iter += 1
                        self.pbar.update(self.pbar_iter)
                    continue

                # get fit model
                fit_function = fitModels(self.cfg.model_name)

                # determine iteration range based on if there are other wavelength fits
                _, _, success = self._findLastGoodFit(fit_list)
                if success and self.cfg.model_name == 'gaussian_and_exp':
                    fit_numbers = range(6)
                elif self.cfg.model_name == 'gaussian_and_exp':
                    fit_numbers = range(5)
                else:
                    raise ValueError('invalid model_name')

                fit_results = []
                flags = []
                for fit_number in fit_numbers:
                    # get guess for fit
                    setup = self._setupFit(phase_hist, fit_list, wavelength_index,
                                           fit_number)
                    # fit data
                    fit_results.append(self._fitPhaseHistogram(phase_hist,
                                                                fit_function,
                                                                setup, row, column))
                    # evaluate how the fit did
                    flags.append(self._evaluateFit(phase_hist, fit_results[-1], fit_list,
                                                    wavelength_index))
                    if flags[-1] == 0:
                        break
                # find best fit
                fit_result, flag = self._findBestFit(fit_results, flags, phase_hist)

                # save data in fit_data object
                fit_data[row, column].append((flag, fit_result[0], fit_result[1],
                                              phase_hist))

                # plot data (will skip if save_plots is set to be true)
                self._plotFit(phase_hist, fit_result, fit_function, flag, row, column)

                # update log
                dt = str(round((datetime.now() - start_time).total_seconds(), 2)) + ' s'
                self._log.info("({0}, {1}) {2}nm: {3} : {4}".format(row, column, wavelength,
                                self.flag_dict[flag], dt))
            # check to see if fits at longer wavelengths can be used to fix fits at
            # shorter wavelengths
            fit_list = fit_data[row, column]
            fit_list = self._reexamineFits(fit_list, row, column)

            # try to fit all of the histograms at once enforcing monotonicity
            # full_fit = self._simultaneousFit(fit_list, row, column, vary=True)
            # if full_fit is not None:
            #     fit_list = full_fit

            fit_data[row, column] = fit_list

            # update progress bar
            if self.cfg.verbose:
                self.pbar_iter += 1
                self.pbar.update(self.pbar_iter)

        # close progress bar
        if self.cfg.verbose:
            self.pbar.finish()
        # close and save last plots
        if self.cfg.save_plots:
            self._closePlots()

        self.fit_data = fit_data

    def calculateCoefficients(self, pixels=[]):
        """
        Loop through the results of 'getPhaseHeights()' and fit energy vs phase height
        to a parabola.

        Args:
            pixels: a list of length 2 lists containing the (row, column) of the pixels
                    on which to compute a phase-energy relation. If it isn't specified,
                    all of the pixels in the array are used.

        Returns:
            Nothing is returned, but a self.wavelength_cal attribute is created. It is a
            numpy array of the shape (self.rows, self.columns). Each entry contains a list
            with the fit information for the pixel in (row, column). In that list is saved
            the fit flag, fit result, and fit covariance in that order.
        """
        # check inputs
        pixels = self._checkPixelInputs(pixels)
        assert hasattr(self, 'fit_data'), "run getPhaseHeights() first"
        assert np.shape(self.fit_data) == (self.rows, self.columns), \
            "fit_data must be a ({0}, {1}) numpy array".format(self.rows, self.columns)

        # initialize verbose and logging
        self._log.info('## calculating phase to energy solution')
        if self.cfg.verbose:
            log.info('calculating phase to energy solution')
            self.pbar = ProgressBar(widgets=[Percentage(), Bar(), '  (',
                                             Timer(), ') ', ETA(), ' '],
                                    max_value=len(pixels)).start()
            self.pbar_iter = 0

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
                    if self.cfg.model_name == 'gaussian_and_exp':
                        phases.append(fit_result[1][3])
                        std.append(fit_result[1][4])
                        if fit_result[2][3, 3] <= 0:
                            errors.append(np.sqrt(fit_result[1][4]))
                        else:
                            errors.append(np.sqrt(fit_result[2][3, 3]))
                    else:
                        raise ValueError("{0} is not a valid fit model name"
                                         .format(self.cfg.model_name))
            phases = np.array(phases)
            std = np.array(std)
            errors = np.array(errors)

            # mask out data points that are within error for monotonic consideration
            if count > 1:
                dE = np.diff(wavelengths) / np.mean(wavelengths)**2  # proportional to
                diff = np.diff(phases)
                mask = np.ones(diff.shape, dtype=bool)
                for ind, _ in enumerate(mask):
                    if diff[ind] < 0 and (-diff[ind] < errors[ind] or
                                          -diff[ind] < errors[ind + 1]):
                        mask[ind] = False

            if count > 1 and ((diff < -2e8 * dE / np.mean(wavelengths))[mask].any()
                              or sum(mask) == 0):
                flag = 7  # data not monotonic enough
                wavelength_cal[row, column] = (flag, False, False)

            # if there are enough points fit the wavelengths
            elif count > 2:
                energies = h.to('eV s').value * c.to('nm/s').value / np.array(wavelengths)

                phase_list1 = []
                phase_list2 = []
                bin_widths = []
                for ind, _ in enumerate(fit_results):
                    if len(fit_results[ind][3]['centers']) > 1:
                        phase_list1.append(np.max(fit_results[ind][3]['centers']))
                        phase_list2.append(np.min(fit_results[ind][3]['centers']))
                        bin_widths.append(np.diff(fit_results[ind][3]['centers'])[0])
                max_width = np.max(bin_widths)
                self.current_threshold = np.max(phase_list1) + max_width / 2
                self.current_min = np.min(phase_list2) - max_width / 2
                popt, pcov = self._fitEnergy('quadratic', phases, energies,
                                              errors, row, column)

                # refit if vertex is between wavelengths or slope is positive
                ind_max = np.argmax(phases)
                ind_min = np.argmin(phases)
                max_phase = phases[ind_max] + std[ind_max]
                min_phase = phases[ind_min] - std[ind_min]
                if popt is False:
                    conditions = True
                else:
                    vertex = -popt[1] / (2 * popt[0])
                    min_slope = 2 * popt[0] * min_phase + popt[1]
                    max_slope = 2 * popt[0] * max_phase + popt[1]
                    vertex_val = np.polyval(popt, vertex)
                    max_val = np.polyval(popt, max_phase)
                    min_val = np.polyval(popt, min_phase)
                    conditions = (vertex < max_phase and vertex > min_phase) or \
                        (min_slope > 0 or max_slope > 0)
                    conditions = conditions or (vertex_val < 0 or max_val < 0 or
                                                min_val < 0)
                if conditions:
                    popt, pcov = self._fitEnergy('linear', phases, energies,
                                                  errors, row, column)

                    if popt is False or popt[1] > 0 or (max_phase > -popt[2] / popt[1]
                                                        and popt[1] < 0):
                        popt, pcov = self._fitEnergy('linear_zero', phases, energies,
                                                      errors, row, column)
                        if popt is False or popt[1] > 0:
                            flag = 8  # linear fit unsuccessful
                            wavelength_cal[row, column] = (flag, False, False)
                        else:
                            flag = 9  # linear fit through zero successful
                            wavelength_cal[row, column] = (flag, popt, pcov)
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
            if self.cfg.verbose:
                self.pbar_iter += 1
                self.pbar.update(self.pbar_iter)
            self._log.info("({0}, {1}): {2}".format(row, column, self.flag_dict[flag]))
        # close progress bar
        if self.cfg.verbose:
            self.pbar.finish()

        self.wavelength_cal = wavelength_cal

    def exportData(self, pixels=[]):
        """
        Saves data in the WaveCal format to the filename.

        Args:
            pixels: a list of length 2 lists containing the (row, column) of the pixels
                    on which to compute a phase-energy relation. If it isn't specified,
                    all of the pixels in the array are used.

        Returns:
            Nothing is returned, but a .h5 file is created with the fit information
            computed with calculateCoefficients() and getPhaseHeights() (or
            getPhaseHeightsParallel()). The .h5 file is saved as calsol_timestamp.h5,
            where the timestamp is the utc timestamp for when the WaveCal object was
            created.
        """
        # check inputs
        pixels = self._checkPixelInputs(pixels)

        # load wavecal header
        wavecal_description = WaveCalDescription(len(self.wavelengths))

        # initialize verbose and logging
        if self.cfg.verbose:
            log.info('exporting data')
            self.pbar = ProgressBar(widgets=[Percentage(), Bar(), '  (',
                                             Timer(), ') ', ETA(), ' '],
                                    max_value=2 * len(pixels)).start()
            self.pbar_iter = 0

        self._log.info("## exporting data to {0}".format(self.cal_file))

        # create folders in file
        file_ = tb.open_file(self.cal_file, mode='w')
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
        info.row['model_name'] = self.cfg.model_name
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
               self.wavelength_cal[row, column][0] == 5 or
               self.wavelength_cal[row, column][0] == 9):
                calsoln.row['polyfit'] = self.wavelength_cal[row, column][1]
            else:
                calsoln.row['polyfit'] = [-1, -1, -1]
            wavelengths = []
            sigma = []
            R = []
            for index, wavelength in enumerate(self.wavelengths):
                if ((self.wavelength_cal[row, column][0] == 4 or
                     self.wavelength_cal[row, column][0] == 5 or
                     self.wavelength_cal[row, column][0] == 9) and
                     self.fit_data[row, column][index][0] == 0):
                    if self.cfg.model_name == 'gaussian_and_exp':
                        mu = self.fit_data[row, column][index][1][3]
                        std = self.fit_data[row, column][index][1][4]
                    else:
                        raise ValueError("{0} is not a valid fit model name"
                                         .format(self.cfg.model_name))
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
            if self.cfg.verbose:
                self.pbar_iter += 1
                self.pbar.update(self.pbar_iter)
        calsoln.flush()

        self._log.info("wavecal table saved")

        # find max number of bins in histograms
        lengths = []
        for row, column in pixels:
            for index, wavelength in enumerate(self.wavelengths):
                fit_list = self.fit_data[row, column][index]
                phase_centers = fit_list[3]['centers']
                lengths.append(len(phase_centers))
        max_l = np.max(lengths)

        # make debug table
        if self.cfg.model_name == 'gaussian_and_exp':
            n_param = 5
        else:
            raise ValueError("{0} is not a valid fit model name"
                             .format(self.cfg.model_name))
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
                if len(fit_list[3]['counts']) > 0 and np.max(fit_list[3]['counts']) > 20:
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
                if self.cfg.model_name == 'gaussian_and_exp' and hist_cov.size == 16:
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
            if self.cfg.verbose:
                self.pbar_iter += 1
                self.pbar.update(self.pbar_iter)
        debug_info.flush()

        self._log.info("debug information saved")

        # close file and progress bar
        file_.close()
        if self.cfg.verbose:
            self.pbar.finish()

    def dataSummary(self):
        """
        Generates a summary plot of the data to the output directory. During calibration
        WaveCal will use this function to generate a summary if the summary_plot
        configuration option is set.
        """
        self._log.debug("## saving summary plot")
        self._clog.debug('saving summary plot')
        try:
            save_name = self.cfg.cal_file_name + '.summary.pdf'
            save_dir = os.path.join(self.cfg.out_directory, save_name)
            plotSummary(self.cal_file, self.cfg.templar_config,
                        save_name=save_name, verbose=self.cfg.verbose)
            self._log.info("summary plot saved as {0}".format(save_dir))
        except KeyboardInterrupt:
            self._clog.info(os.linesep + "Shutdown requested ... exiting")
        except Exception as error:
            self._clog.error('Summary plot generation failed. It can be remade by ' +
                           'using plotSummary() in plotWaveCal.py', exc_info=True)
            self._log.error("summary plot failed", exc_info=True)

    def loadPhotonData(self, row, column, wavelength_index):
        """
        Get a photon list for a single pixel and wavelength.
        """
        try:
            if self.worker_slave:
                self.request_data.put([row, column, wavelength_index, self.pid])
                photon_list = self.load_data.get()
            else:
                photon_list = self.obs[wavelength_index].getPixelPhotonList(row, column)
            return photon_list
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as error:
            return np.array([], dtype=[('Time', '<u4'), ('Wavelength', '<f4'),
                                       ('SpecWeight', '<f4'), ('NoiseWeight', '<f4')])

    def _checkParallelOptions(self):
        """
        Check to make sure options that are incompatible with parallel computing are not
        enabled
        """
        #TODO instead of this (or in addition) I would put the guard on the actions that are incompatible
        #e.g. the internal plotting function should just return
        assert self.cfg.save_plots is False, "Cannot save histogram plots while " + \
            "running in parallel. save_plots must be False in the configuration file"

    def _removeTailRidingPhotons(self, photon_list, dt):
        """
        Remove photons that arrive too close together.
        """
        # enforce time ordering (will remove once this is enforced in h5 file creation)
        indices = np.argsort(photon_list['Time'])
        photon_list = photon_list[indices]

        indices = np.where(np.diff(photon_list['Time']) > dt)[0] + 1
        photon_list = photon_list[indices]

        return photon_list

    def _histogramPhotons(self, phase_list):
        """
        Create a histogram of the phase data for a specified bin width.
        """
        phase_list = phase_list[phase_list < 0]
        if len(phase_list) == 0:
            phase_hist = {'centers': np.array([]), 'counts': np.array([])}
            return phase_hist
        min_phase = np.min(phase_list)
        max_phase = np.max(phase_list)

        # reload default bin_width
        self.bin_width = self.cfg.bin_width

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

    def _setupFit(self, phase_hist, fit_list, wavelength_index, fit_number):
        """
        Get a good initial guess (and bounds) for the fitting model.
        """
        if len(phase_hist['centers']) == 0:
            return None
        # check for successful fits for this pixel with a different wavelength
        recent_fit, recent_index, success = self._findLastGoodFit(fit_list)
        if self.cfg.model_name == 'gaussian_and_exp':
            if fit_number == 0 and not success:
                # box smoothed guess fit with varying b
                params = self._boxGuess(phase_hist)
            elif fit_number == 1 and not success:
                # median center guess fit with varying b
                params = self._medianGuess(phase_hist)
            elif fit_number == 2 and not success:
                # fixed number guess fit with varying b
                params = self._numberGuess(phase_hist, 0)
            elif fit_number == 3 and not success:
                # fixed number guess fit with varying b
                params = self._numberGuess(phase_hist, 1)
            elif fit_number == 4 and not success:
                # fixed number guess fit with varying b
                params = self._numberGuess(phase_hist, 2)

            elif fit_number == 0 and success:
                # wavelength scaled fit with fixed b
                params = self._wavelengthGuess(phase_hist, recent_fit, recent_index,
                                                wavelength_index, b=recent_fit[1][1])
            elif fit_number == 1 and success:
                # box smoothed fit with fixed b
                params = self._boxGuess(phase_hist, b=recent_fit[1][1])
            elif fit_number == 2 and success:
                # median center guess fit with fixed b
                params = self._medianGuess(phase_hist, b=recent_fit[1][1])
            elif fit_number == 3 and success:
                # wavelength scaled fit with varying b
                params = self._wavelengthGuess(phase_hist, recent_fit, recent_index,
                                                wavelength_index)
            elif fit_number == 4 and success:
                # box smoothed guess fit with varying b
                params = self._boxGuess(phase_hist)
            elif fit_number == 5 and success:
                # median center guess fit with varying b
                params = self._medianGuess(phase_hist)

            elif fit_number == 10:
                # after all histogram fits are done use good fits to refit the others
                params = self._allDataGuess(phase_hist, fit_list, wavelength_index)
            else:
                raise ValueError('fit_number not valid for this pixel')
            setup = params
        else:
            raise ValueError("{0} is not a valid fit model name".format(self.cfg.model_name))

        return setup

    def _boxGuess(self, phase_hist, b=None):
        """
        Returns parameter guess based on a box smoothed histogram
        """
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

    def _wavelengthGuess(self, phase_hist, recent_fit, recent_index, wavelength_index,
                          b=None):
        """
        Returns parameter guess based on previous wavelength solutions
        """
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

        # values derived from data (same as _boxGuess)
        gaussian_amplitude = 1.1 * np.max(phase_hist['counts']) / 2

        params = lm.Parameters()
        params.add('a', value=exp_amplitude, min=0, max=np.inf)
        params.add('b', value=b, min=-1, max=np.inf, vary=vary)
        params.add('c', value=gaussian_amplitude, min=0,
                   max=1.1 * np.max(phase_hist['counts']))
        params.add('d', value=gaussian_center, min=np.min(phase_hist['centers']), max=0)
        params.add('f', value=standard_deviation, min=0.1, max=np.inf)

        return params

    def _medianGuess(self, phase_hist, b=None):
        """
        Returns parameter guess based on median histogram center
        """
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

        # old values (same as _boxGuess)
        standard_deviation = 10

        params = lm.Parameters()
        params.add('a', value=exp_amplitude, min=0, max=np.inf)
        params.add('b', value=b, min=-1, max=np.inf, vary=vary)
        params.add('c', value=gaussian_amplitude, min=0,
                   max=1.1 * np.max(phase_hist['counts']))
        params.add('d', value=gaussian_center, min=np.min(phase_hist['centers']), max=0)
        params.add('f', value=standard_deviation, min=0.1, max=np.inf)

        return params

    def _numberGuess(self, phase_hist, attempt):
        """
        Hard coded numbers used as guess parameters
        """
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

    def _allDataGuess(self, phase_hist, fit_list, wavelength_index):
        """
        Returns parameter guess based on all wavelength solutions
        """
        # determine which fits worked
        flags = np.array([fit_list[ind][0] for ind in range(len(self.wavelengths))])
        sucessful = (flags == 0)

        # get index of closest good fit with longer wavelength (must exist)
        for ind, s in enumerate(sucessful[wavelength_index + 1:]):
            if s:
                longer_ind = wavelength_index + ind + 1
                break
        # get index of closest good fit with shorter wavelength (may not exist)
        if wavelength_index > 0 and any(sucessful[:wavelength_index]):
            for ind, s in enumerate(sucessful[:wavelength_index]):
                if s:
                    shorter_ind = ind
                    break
        else:
            shorter_ind = None
        if self.cfg.model_name == 'gaussian_and_exp':
            a_long = fit_list[longer_ind][1][0]
            b_long = fit_list[longer_ind][1][1]
            c_long = fit_list[longer_ind][1][2]
            d_long = fit_list[longer_ind][1][3]
            f_long = fit_list[longer_ind][1][4]
            if shorter_ind is not None:
                a_short = fit_list[shorter_ind][1][0]
                b_short = fit_list[shorter_ind][1][1]
                c_short = fit_list[shorter_ind][1][2]
                d_short = fit_list[shorter_ind][1][3]
                f_short = fit_list[shorter_ind][1][4]
                a = np.mean([a_short, a_long])
                b = np.mean([b_short, b_long])
                c = np.mean([c_short, c_long])
                d = np.mean([d_short * self.wavelengths[wavelength_index] /
                             self.wavelengths[shorter_ind],
                             d_long * self.wavelengths[wavelength_index] /
                             self.wavelengths[longer_ind]])
                f = np.mean([f_short, f_long])
            else:
                a = fit_list[longer_ind][1][0]
                b = fit_list[longer_ind][1][1]
                c = fit_list[longer_ind][1][2]
                d = (d_long * self.wavelengths[wavelength_index] /
                     self.wavelengths[longer_ind])
                f = fit_list[longer_ind][1][4]

        else:
            raise ValueError("{0} is not a valid fit model name".format(self.cfg.model_name))
        params = lm.Parameters()
        params.add('a', value=a, min=0, max=np.inf)
        params.add('b', value=b, min=-1, max=np.inf)
        params.add('c', value=c, min=0,
                   max=1.1 * np.max(phase_hist['counts']))
        params.add('d', value=d, min=np.min(phase_hist['centers']), max=0)
        params.add('f', value=f, min=0.1, max=np.inf)

        return params

    def _fitPhaseHistogram(self, phase_hist, fit_function, setup, row, column):
        """
        Fit the phase histogram to the specified fit fit_function
        """
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            error = np.sqrt(phase_hist['counts'] + 0.25) + 0.5
            model = lm.Model(fit_function)
            try:
                # fit data
                result = model.fit(phase_hist['counts'], setup, x=phase_hist['centers'],
                                   weights=1 / error)
                # lm fit doesn't error if covariance wasn't calculated so check here
                # replace with gaussian width if covariance couldn't be calculated
                if result.covar is None:
                    if self.cfg.model_name == 'gaussian_and_exp':
                        result.covar = np.ones((5, 5)) * result.best_values['f'] / 2
                if self.cfg.model_name == 'gaussian_and_exp':
                    if result.covar[3, 3] == 0:
                        result.covar = np.ones((5, 5)) * result.best_values['f'] / 2
                # unpack results
                if self.cfg.model_name == 'gaussian_and_exp':
                    parameters = ['a', 'b', 'c', 'd', 'f']
                    popt = [result.best_values[p] for p in parameters]
                    current_order = result.var_names
                    indices = []
                    for p in parameters:
                        for index, o in enumerate(current_order):
                            if p == o:
                                indices.append(index)
                    pcov = result.covar[indices, :][:, indices]
                fit_result = (popt, pcov)

            except (RuntimeError, RuntimeWarning, ValueError) as error:
                # RuntimeError catches failed minimization
                # RuntimeWarning catches overflow errors
                # ValueError catches if ydata or xdata contain Nans
                self._log.error('({0}, {1}): '.format(row, column), exc_info=True)
                fit_result = (False, False)
            except TypeError:
                # TypeError catches when not enough data is passed to params.add()
                self._log.error('({0}, {1}): '.format(row, column) + "Not enough data "
                                + "passed to the fit function")
                fit_result = (False, False)

        return fit_result

    def _evaluateFit(self, phase_hist, fit_result, fit_list, wavelength_index):
        """
        Evaluate the result of the fit and return a flag for different conditions.
        """
        if len(phase_hist['centers']) == 0:
            flag = 3  # no data to fit
            return flag
        if self.cfg.model_name == 'gaussian_and_exp':
            max_phase = max(phase_hist['centers'])
            min_phase = min(phase_hist['centers'])
            peak_upper_lim = np.min([-10, max_phase * 1.2])

            # change peak_upper_lim if good fits exist for higher wavelengths
            recent_fit, recent_index, success = self._findLastGoodFit(fit_list)
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

                if wavelength_index == 0:
                    snr = 4
                else:
                    snr = 2
                peak_lower_lim = min_phase + sigma
                fit_quality = np.sum([np.abs(c - h) > 5 * np.sqrt(c),
                                      np.abs(c_p - h_p) > 5 * np.sqrt(c_p),
                                      np.abs(c_n - h_n) > 5 * np.sqrt(c_n)])
                bad_fit_conditions = ((center > peak_upper_lim) or
                                      (center < peak_lower_lim) or
                                      (gauss(center) < snr * exp(center)) or
                                      (gauss(center) < 10) or
                                      np.abs(sigma) < 2 or
                                      fit_quality >= 2 or
                                      2 * sigma > peak_upper_lim - peak_lower_lim)
                # if wavelength_index == 0:
                #     log.info(center > peak_upper_lim, center < peak_lower_lim,
                #           gauss(center) < snr * exp(center), gauss(center) < 10,
                #           np.abs(sigma) < 2, fit_quality >= 2,
                #           2 * sigma > peak_upper_lim - peak_lower_lim)
                if bad_fit_conditions:
                    flag = 2  # fit converged to a bad solution
                else:
                    flag = 0  # fit converged

        else:
            raise ValueError("{0} is not a valid fit model name".format(self.cfg.model_name))

        return flag

    def _findBestFit(self, fit_results, flags, phase_hist):
        """
        Finds the best fit out of a list based on chi squared
        """
        centers = phase_hist['centers']
        counts = phase_hist['counts']
        chi2 = []
        for fit_result in fit_results:
            if fit_result[0] is False:
                chi2.append(np.inf)
            else:
                fit = fitModels(self.cfg.model_name)(centers, *fit_result[0])
                errors = np.sqrt(counts + 0.25) + 0.5
                chi2.append(np.sum(((counts - fit) / errors)**2))
        index = np.argmin(chi2)

        return fit_results[index], flags[index]

    def _reexamineFits(self, fit_list, row, column):
        """
        Recalculate unsuccessful fits using fit information from all successful
        wavelength fits. The main loop is only able to use shorter wavelengths to inform
        longer wavelength fits. This step mainly catches when the first wavelength fit
        fails because there isn't enough information about the pixel sensitivity.
        """
        start_time = datetime.now()

        # determine which fits worked
        flags = np.array([fit_list[ind][0] for ind in range(len(self.wavelengths))])
        successful = (flags == 0)

        # determine which fits to retry
        indices = []
        for index, success in enumerate(successful):
            # only recalculate if there is a longer wavelength fit availible
            if not success and any(successful[index + 1:]):
                indices.append(index)

        # return the fit_list if nothing can be done
        if len(indices) == 0:
            return fit_list
        # loop through bad fits and refit them
        for wavelength_index in reversed(indices):
            # setup fit
            phase_hist = fit_list[wavelength_index][3]
            setup = self._setupFit(phase_hist, fit_list, wavelength_index, 10)

            # get fit model
            fit_function = fitModels(self.cfg.model_name)

            # fit histogram
            fit_result = self._fitPhaseHistogram(phase_hist, fit_function, setup,
                                                  row, column)

            # evaluate fit
            flag = self._evaluateFit(phase_hist, fit_result, fit_list, wavelength_index)

            # find best fit even if the fit failed
            fit_results = [fit_result, fit_list[wavelength_index][1:3]]
            fit_flags = [flag, fit_list[wavelength_index][0]]
            fit_result, flag = self._findBestFit(fit_results, fit_flags, phase_hist)

            # save data
            fit_list[wavelength_index] = (flag, fit_result[0], fit_result[1],
                                          phase_hist)
            if flag == 0:
                dt = round((datetime.now() - start_time).total_seconds(), 2)
                dt = str(dt) + ' s'
                message = "({0}, {1}) {2}nm: histogram fit recalculated " + \
                          "- converged and validated : {3}"
                self._log.info(message.format(row, column,
                                             self.wavelengths[wavelength_index], dt))
        return fit_list

    def _simultaneousFit(self, fit_list, row, column, vary=False):
        """
        Try to fit all of the histograms simultaneously with the condition that the energy
        phase relation be monotonic. The previous sucessful fits will be used as guesses.
        If the fit fails or a single histogram fit fails evaluation, None will be
        returned.
        """
        start_time = datetime.now()

        new_fit_list = fit_list

        # determine which fits worked
        flags = np.array([fit_list[ind][0] for ind in range(len(self.wavelengths))])
        successful = (flags == 0)

        # if there are less than three good fits, nothing can be done
        if np.sum(successful) < 3:
            return None

        # determine which fits to include in composite model
        indices = []
        good_fits = []
        for index, success in enumerate(successful):
            if success:
                indices.append(index)
                good_fits.append(fit_list[index])

        # get fit function
        fit_function = fitModels(self.cfg.model_name)

        # initialize the parameter object
        params = lm.Parameters()

        # make the noise fall off a constant over all sets
        if vary is False:
            # find the average noise fall off
            b = []
            for ind, wavelength_index in enumerate(indices):
                if self.cfg.model_name == 'gaussian_and_exp':
                    b.append(fit_list[wavelength_index][1][1])
            b = np.mean(b)
            params.add('b', value=b, min=-1, max=np.inf, vary=False)

        # add the parameters
        for ind, wavelength_index in enumerate(indices):
            fit_result = new_fit_list[wavelength_index][1]
            phase_hist = new_fit_list[wavelength_index][3]
            prefix = 'm' + str(ind) + '_'
            if self.cfg.model_name == 'gaussian_and_exp':
                params.add(prefix + 'a', value=fit_result[0], min=0, max=np.inf)
                if vary:
                    params.add(prefix + 'b', value=fit_result[1], min=-1, max=np.inf)
                params.add(prefix + 'c', value=fit_result[2], min=0,
                           max=1.1 * np.max(phase_hist['counts']))
                params.add(prefix + 'f', value=fit_result[4], min=0.1, max=np.inf)
                if ind == 0:
                    params.add(prefix + 'd', value=fit_result[3],
                               min=np.min(phase_hist['centers']), max=0)
                else:
                    previous_fit = new_fit_list[indices[ind - 1]][1]
                    delta = previous_fit[3] - fit_result[3]
                    # move delta to 0 if not monotonic
                    if delta > 0:
                        if ind < len(indices) - 1:
                            next_fit = new_fit_list[indices[ind + 1]][1]
                            e1 = 1 / self.wavelengths[indices[ind - 1]]
                            p1 = previous_fit[3]
                            e2 = 1 / self.wavelengths[indices[ind + 1]]
                            p2 = next_fit[3]
                            e0 = 1 / self.wavelengths[wavelength_index]
                            new_fit_list[wavelength_index][1][3] = ((p2 - p1) / (e2 - e1)
                                                                    * (e0 - e1)) + p1
                            delta = previous_fit[3] - new_fit_list[wavelength_index][1][3]
                        else:
                            new_fit_list[wavelength_index][1][3] = previous_fit[3] * 0.95
                            delta = 0.05 * previous_fit[3]
                    previous_prefix = 'm' + str(ind - 1) + '_'
                    params.add(previous_prefix + 'delta', value=delta, min=fit_result[3],
                               max=0)
                    expression = previous_prefix + 'd -' + previous_prefix + 'delta'
                    params.add(prefix + 'd', expr=expression)
        # try to fit
        try:
            result = lm.minimize(self._histogramChi2, params, method='leastsq',
                                 args=(good_fits, fit_function, vary))

            # exit if fit failed
            if result.success is False:
                return None

            # loop through output fits
            for ind, wavelength_index in enumerate(indices):
                # repackage solution into a fit_result
                prefix = 'm' + str(ind) + '_'
                if self.cfg.model_name == 'gaussian_and_exp':
                    if vary:
                        parameters = [prefix + 'a', prefix + 'b', prefix + 'c',
                                      prefix + 'd', prefix + 'f']
                    else:
                        parameters = [prefix + 'a', 'b', prefix + 'c',
                                      prefix + 'd', prefix + 'f']
                popt = [result.params[p].value for p in parameters]
                pcov = np.zeros((len(parameters), len(parameters)))
                # only fill the diagonal elements (don't care about the rest for now)
                for ind, param in enumerate(parameters):
                    # exit if covariance couldn't be calculated
                    # this happens when peak centers converge on top of each other
                    if result.params[param].stderr == 0:
                        return None
                    pcov[ind, ind] = result.params[param].stderr**2
                fit_result = (popt, pcov)

                # evaluate fits
                phase_hist = new_fit_list[wavelength_index][3]
                flag = self._evaluateFit(phase_hist, fit_result, new_fit_list,
                                          wavelength_index)
                # exit if one of the fits fails any evaluation step
                if flag != 0:
                    return None
                # replace old fit with simultaneous fit
                new_fit_list[wavelength_index] = (flag, fit_result[0], fit_result[1],
                                                  phase_hist)

            dt = str(round((datetime.now() - start_time).total_seconds(), 2))+' s'
            if vary:
                message = "histograms refit to a single model enforcing monotonicity"
            else:
                message = "histograms refit to a single model enforcing " + \
                          "monotonicity with a constant exponential fall time"
            self._log.info("({0}, {1}): {2} : {3}".format(row, column, message, dt))
            return new_fit_list

        except Exception as error:
            # do some error handeling
            raise error
            return None

    def _histogramChi2(self, params, fit_list, fit_function, vary):
        """
        Calculates the normalized chi squared residual for the simultaneous histogram fit
        """
        p = params.valuesdict()

        chi2 = np.array([])
        for index, fit_result in enumerate(fit_list):
            centers = fit_result[3]['centers']
            counts = fit_result[3]['counts']
            error = np.sqrt(counts + 0.25) + 0.5
            if self.cfg.model_name == 'gaussian_and_exp':
                prefix = 'm' + str(index) + '_'
                if vary:
                    b = p[prefix + 'b']
                else:
                    b = p['b']
                fit = fit_function(centers, p[prefix + 'a'], b, p[prefix + 'c'],
                                   p[prefix + 'd'], p[prefix + 'f'])

                nu_free = np.max([len(counts) - 5, 1])
            chi2 = np.append(chi2, ((counts - fit) / error) / np.sqrt(nu_free))
        return chi2

    def _fitEnergy(self, fit_type, phases, energies, errors, row, column):
        """
        Fit the phase histogram to the specified fit fit_function
        """
        fit_function = fitModels(fit_type)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            try:
                params = lm.Parameters()
                if fit_type == 'linear':
                    guess = np.polyfit(phases, energies, 1)
                    params.add('b', value=guess[0])
                    params.add('c', value=guess[1])
                    output = lm.minimize(self._energyChi2, params, method='leastsq',
                                         args=(phases, energies, errors, fit_function))
                elif fit_type == 'linear_zero':
                    guess = np.polyfit(phases, energies, 1)
                    params.add('b', value=guess[0])
                    output = lm.minimize(self._energyChi2, params, method='leastsq',
                                         args=(phases, energies, errors, fit_function))
                elif fit_type == 'quadratic':
                    guess = np.polyfit(phases, energies, 2)
                    params.add('a', value=guess[0])
                    params.add('b', value=guess[1])
                    params.add('c', value=guess[2])
                    output = lm.minimize(self._energyChi2, params, method='leastsq',
                                          args=(phases, energies, errors, fit_function))
                else:
                    raise ValueError('{0} is not a valid fit type'.format(fit_type))
                if output.success:
                    p = output.params.valuesdict()
                    if fit_type == 'linear':
                        popt = (0, p['b'], p['c'])
                        if output.covar is not False and output.covar is not None:
                            pcov = np.insert(np.insert(output.covar, 0, [0, 0],
                                                       axis=1), 0, [0, 0, 0], axis=0)
                    elif fit_type == 'linear_zero':
                        popt = (0, p['b'], 0)
                        if output.covar is not False and output.covar is not None:
                            cov = np.ndarray.flatten(np.array(output.covar))[0]
                            pcov = np.array([[0, 0, 0], [0, cov, 0], [0, 0, 0]])
                    else:
                        popt = (p['a'], p['b'], p['c'])
                        pcov = output.covar
                    fit_result = (popt, pcov)
                else:
                    self._log.info('({0}, {1}): '.format(row, column) + output.message)
                    if self.cfg.logging:
                        fit_result = (False, False) #TODO is this an indendation error

            except (Exception, Warning) as error:
                # shouldn't have any exceptions or warnings
                self._log.error('({0}, {1}): '.format(row, column), exc_info=True)
                fit_result = (False, False)
                raise error

        return fit_result

    @staticmethod
    def _energyChi2(params, phases, energies, errors, fit_function):
        """
        Calculates the chi squared residual for the energy - phase fit using x-errors
        """
        p = params.valuesdict()
        if 'a' not in p.keys():
            dfdx = p['b']
        else:
            dfdx = 2 * p['a'] * phases + p['b']

        chi2 = ((fit_function(p, phases) - energies) / (dfdx * errors))**2
        return chi2

    def _plotFit(self, phase_hist, fit_result, fit_function, flag, row, column):
        """
        Plots the histogram data against the model fit for comparison and saves to pdf
        """
        if not self.cfg.save_plots:
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

        if self.cfg.model_name == 'gaussian_and_exp':
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
                             .format(self.cfg.model_name))

        # save page if all the plots have been made
        if self.plot_counter % self.plots_per_page == self.plots_per_page - 1:
            pdf = PdfPages(os.path.join(self.cfg.out_directory, 'temp.pdf'))
            pdf.savefig(self.fig)
            pdf.close()
            self._mergePlots()
            self.saved = True
            plt.close('all')

        # update plot counter
        self.plot_counter += 1

    def _setupPlots(self):
        """
        Initialize plotting variables
        """
        self.plot_counter = 0
        self.plots_x = 3
        self.plots_y = 4
        self.plots_per_page = self.plots_x * self.plots_y
        plot_file = os.path.join(self.cfg.out_directory, self.cfg.plot_file_name)
        if os.path.isfile(plot_file):
            answer = self._query("{0} already exists. Overwrite?".format(plot_file),
                                  yes_or_no=True)
            if answer is False:
                answer = self._query("Provide a new file name (type exit to quit):")
                if answer == 'exit':
                    raise UserError("User doesn't want to overwrite the plot file " +
                                    "... exiting")
                plot_file = os.path.join(self.cfg.out_directory, answer)
                while os.path.isfile(plot_file):
                    question = "{0} already exists. Choose a new file name " + \
                               "(type exit to quit):"
                    answer = self._query(question.format(plot_file))
                    if answer == 'exit':
                        raise UserError("User doesn't want to overwrite the plot file " +
                                        "... exiting")
                    plot_file = os.path.join(self.cfg.out_directory, answer)
                self.cfg.plot_file_name = plot_file
            else:
                os.remove(plot_file)

    def _mergePlots(self):
        """
        Merge recently created temp.pdf with the main file
        """
        plot_file = os.path.join(self.cfg.out_directory, self.cfg.plot_file_name)
        temp_file = os.path.join(self.cfg.out_directory, 'temp.pdf')
        if os.path.isfile(plot_file):
            merger = PdfFileMerger()
            merger.append(PdfFileReader(open(plot_file, 'rb')))
            merger.append(PdfFileReader(open(temp_file, 'rb')))
            merger.write(plot_file)
            merger.close()
            os.remove(temp_file)
        else:
            os.rename(temp_file, plot_file)

    def _closePlots(self):
        """
        Safely close plotting variables after plotting since the last page is only saved
        if it is full.
        """
        if not self.saved:
            pdf = PdfPages(os.path.join(self.cfg.out_directory, 'temp.pdf'))
            pdf.savefig(self.fig)
            pdf.close()
            self._mergePlots()
        plt.close('all')

    @staticmethod
    def _findLastGoodFit(fit_list):
        """
        Find the most recent fit and index from a list of fits.
        """
        if fit_list:  # fit_list not empty
            for index, fit in enumerate(fit_list):
                if fit[0] == 0:
                    recent_fit = fit
                    recent_index = index
            if 'recent_fit' in locals():
                return recent_fit, recent_index, True
        return None, None, False

    @staticmethod
    def _query(question, yes_or_no=False, default="no"):
        """Ask a question via raw_input() and return their answer.

        "question" is a string that is presented to the user.
        "yes_or_no" specifies if it is a yes or no question
        "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning an answer is required of
        the user). Only used if yes_or_no=True.

        The "answer" return value is the user input for a general question. For a yes or
        no question it is True for "yes" and False for "no".
        """
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

    def _checkPixelInputs(self, pixels):
        """
        Check inputs for getPhaseHeights, calculateCoefficients and exportData
        """
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


class UserError(Exception):
    """
    Custom error used to exit the waveCal program without traceback
    """
    pass


class Worker(mp.Process):
    """
    Worker class to send pixels to and do the histogram fits. Run by
    getPhaseHeightsParallel.
    """
    def __init__(self, in_queue, out_queue, progress_queue, config_file, num,
                 request_data, load_data, rows, columns, log):
        super(Worker, self).__init__()
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.progress_queue = progress_queue
        self.config_file = config_file
        self.num = num
        self.request_data = request_data
        self.load_data = load_data
        self.rows = rows
        self.columns = columns
        self.daemon = True
        self._log = log
        self.start()

    def run(self):
        try:
            w = WaveCal(config=self.config_file, master=False, worker_slave=True,
                        pid=self.num, request_data=self.request_data,
                        load_data=self.load_data, rows=self.rows, columns=self.columns,
                        filelog=self._log)
            w.cfg.verbose = False
            w.cfg.summary_plot = False

            while True:
                pixel = self.in_queue.get()
                if pixel is None:
                    break
                w.getPhaseHeights(pixels=[pixel])
                pixel_dict = {tuple(pixel): w.fit_data[pixel[0], pixel[1]]}
                if self.progress_queue is not None:
                    self.progress_queue.put(True)

                self.out_queue.put(pixel_dict)
        except (KeyboardInterrupt, BrokenPipeError):
            pass


class GateWorker(mp.Process):
    """
    Worker class in charge of opening and closing and reading .h5 files.
    """
    def __init__(self, config_file, num, request_data, load_data):
        super(GateWorker, self).__init__()
        self.config_file = config_file
        self.num = num
        self.request_data = request_data
        self.load_data = load_data
        self.daemon = True
        self.start()

    def run(self):
        try:
            w = WaveCal(config=self.config_file, master=False, data_slave=True)
            w.cfg.verbose = False
            w.cfg.summary_plot = False  #TODO i don't think this is needed

            # we know how many data sets we will be loading
            for i in range(self.num):
                # get a request
                request = self.request_data.get()
                # do request
                photon_list = w.loadPhotonData(request[0], request[1], request[2])
                # return data from request to the correct process
                self.load_data[request[3]].put(photon_list)
        except (KeyboardInterrupt, BrokenPipeError):
            pass


class ProgressWorker(mp.Process):
    """
    Worker class to make progress bar when using multiprocessing. Run by
    getPhaseHeightsParallel.
    """
    def __init__(self, progress_queue, N):
        super(ProgressWorker, self).__init__()
        self.progress_queue = progress_queue
        self.N = N
        self.daemon = True
        self.start()

    def run(self):
        try:
            pbar = ProgressBar(widgets=[Percentage(), Bar(), '  (', Timer(), ') ',
                                        ETA(), ' '], max_value=self.N).start()
            pbar_iter = 0
            while pbar_iter < self.N:
                if self.progress_queue.get():
                    pbar_iter += 1
                    pbar.update(pbar_iter)
            pbar.finish()
        except (KeyboardInterrupt, BrokenPipeError):
            pass



if __name__ == '__main__':

    pipelinelog.setup_logging()

    log = getLogger('WaveCal')

    timestamp = datetime.utcnow().timestamp()


    parser = argparse.ArgumentParser(description='MKID Wavelength Calibration Utility')
    parser.add_argument('cfgfile', type=str, help='The config file')
    parser.add_argument('--vet', action='store_true', dest='vetonly', default=False,
                        help='Only verify config file')
    parser.add_argument('--h5script', action='store_true', dest='scriptsonly', default=False,
                        help='Only make HDF scripts')
    parser.add_argument('--h5', action='store_true', dest='h5only', default=False,
                        help='Only make h5 files')
    parser.add_argument('--forceh5', action='store_true', dest='forcehdf', default=False,
                        help='Force HDF creation')
    parser.add_argument('-nc', type=int, dest='ncpu', default=0,
                        help='Number of CPUs to use, default is number of wavelengths')
    parser.add_argument('-s', type=str, dest='summary',
                        help='Generate a summary of the specified solution')
    parser.add_argument('--nolog', action='store_true', dest='nolog', default=False,
                        help='Disable logging')
    args = parser.parse_args()

    if args.nolog:
        flog = None
    else:
        flog = pipelinelog.createFileLog('WaveCal.logfile',
                                         os.path.join(os.getcwd(),
                                                      '{:.0f}.log'.format(timestamp)))

    atexit.register(lambda x:print('Execution took {:.0f}s'.format(time.time()-x)), time.time())

    config = WaveCalConfig(args.cfgfile, cal_file_name='calsol_{}.h5'.format(timestamp))

    if args.ncpu == 0:
        args.ncpu = len(config.wavelengths)

    if args.vetonly:
        exit()

    if args.summary:
        WaveCal(config).dataSummary()
        exit()

    if not config.hdfexist() or args.forcehdf or args.scriptsonly:

        config.write(config.file+'.bak', forceconsistency=False)
        config.write(config.file)  # Make sure the file is consistent and save

        scripts = makeHDFscripts(config)

        if args.scriptsonly:
            exit()

        if args.ncpu > 1:
            pool = mp.Pool(processes=min(args.ncpu, mp.cpu_count()))
            pool.map(sp.call, zip(['bash']*len(scripts), scripts))
            pool.close()
        else:
            for s in scripts:
                sp.call(('bash', s))

    if args.h5only:
        exit()

    WaveCal(config, filelog=flog).makeCalibration()

