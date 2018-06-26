import numpy as np
import sys, os
from DarknessPipeline.Calibration.WavelengthCal import WaveCal as W
import ast
from configparser import ConfigParser
import cmd
class RunWaveCal:
    '''
    Created by: Isabel Lipartito, May 2018
    A simple wrapper to make a wavecal solution file from a single wavecal
    Inputs:  Wavelengths of lasers used in this wavecal, each laser's start time and exposure time.
    See defaultRunWaveCal.cfg for a complete list of input parameters (you also need to input the fit/output parameters 
    received by WaveCal.py
    Replaces the indivdual processes of:
              1.  making individual H5 cfg files for each laser
              2.  Running Bin2HDF for each laser
              3.  Making a wavecal cfg file
              4.  Running WaveCal.py on the wavecal cfg file
    '''
    def __init__(self, config_file='default.cfg'):

        # define the configuration file path
        self.config = config_file
        if self.config == 'defaultRunWaveCal.cfg':
            directory = os.path.dirname(os.path.realpath(__file__))
            self.config_directory = os.path.join(directory, 'Params', self.config)
        else:
            self.config_directory = self.config

        # check the configuration file path and read it in
        self.__configCheck(0)
        self.config = ConfigParser()
        self.config.read(self.config_directory)
        self.config = ConfigParser()
        self.config.read(self.config_directory)

        # check the configuration file format and load the parameters
        self.__configCheck(1)
        self.wavelengths = ast.literal_eval(self.config['Data']['wavelengths'])
        self.startTimes = ast.literal_eval(self.config['Data']['startTimes'])
        self.expTimes = ast.literal_eval(self.config['Data']['expTimes'])
        self.dataDir= ast.literal_eval(self.config['Data']['dataDir'])
        self.beamDir= ast.literal_eval(self.config['Data']['beamDir'])
        self.model_name = ast.literal_eval(self.config['Fit']['model_name'])
        self.bin_width = ast.literal_eval(self.config['Fit']['bin_width'])
        self.dt = ast.literal_eval(self.config['Fit']['dt'])
        self.parallel = ast.literal_eval(self.config['Fit']['parallel'])
        self.save_plots = ast.literal_eval(self.config['Output']['save_plots'])
        self.plot_file_name = ast.literal_eval(self.config['Output']['plot_file_name'])
        self.verbose = ast.literal_eval(self.config['Output']['verbose'])
        self.logging = ast.literal_eval(self.config['Output']['logging'])
        self.summary_plot = ast.literal_eval(self.config['Output']['summary_plot'])
        self.outH5Dir= ast.literal_eval(self.config['Output']['outH5Dir'])
        self.outCalSolnDir= ast.literal_eval(self.config['Output']['outCalSolnDir'])
        self.templar_config= ast.literal_eval(self.config['Output']['templar_config'])

        # check the parameter formats
        self.__configCheck(2)

    def makeH5CfgFiles(self):
        path='/mnt/data0/isabel/DarknessPipeline/RawDataProcessing'
        file1path=self.outH5Dir+'makeh5files.sh'
        file1=open(file1path, 'w')       
        file1.write('#!/bin/bash') 
        for i in np.arange(len(self.wavelengths)):
            wavepath=self.outH5Dir+str(self.wavelengths[i])+'nm.txt'
            cwd = os.getcwd()
            file=open(wavepath,'w')
            file.write('140 145')
            file.write('\n'+self.dataDir)
            file.write('\n'+'%d' %self.startTimes[i])
            file.write('\n'+'%d' %self.expTimes[i])
            file.write('\n'+self.beamDir)
            file.write('\n'+str(1))
            file.write('\n'+self.outH5Dir)
            file.close()
            os.chdir(path)
            os.system('./Bin2HDF '+wavepath)
            os.chdir(cwd)
            file1.write('\n'+'./Bin2HDF '+wavepath)   
        file1.close() 
            
 
    def makeWaveCalCfgFile(self):
        self.file_names=[]
        for i in np.arange(len(self.startTimes)):
            filename='%d' %self.startTimes[i]+'.h5'
            self.file_names.append(filename)
        self.cfgpath=self.outH5Dir+'WvCalCfg.cfg'
        file=open(self.cfgpath, 'w')
        file.write('[Data]')
        file.write('\n')
        file.write('\n'+'directory = '+'"'+self.outH5Dir+'"')
        file.write('\n'+'wavelengths= '+str(self.wavelengths))
        file.write('\n'+'file_names= '+str(self.file_names))
        file.write('\n')
        file.write('\n'+'[Fit]')
        file.write('\n')
        file.write('\n'+'model_name = '+'"'+str(self.model_name)+'"')
        file.write('\n'+'bin_width = '+str(self.bin_width))
        file.write('\n'+'dt = '+str(self.dt))
        file.write('\n'+'parallel = '+str(self.parallel))
        file.write('\n')
        file.write('\n'+'[Output]')
        file.write('\n')
        file.write('\n'+'out_directory = '+'"'+self.outCalSolnDir+'"')
        file.write('\n'+'save_plots = '+str(self.save_plots))
        file.write('\n'+'plot_file_name = ''"'+str(self.plot_file_name)+'"')
        file.write('\n'+'summary_plot = '+str(self.summary_plot))
        file.write('\n'+'templar_config = '+'"'+self.templar_config+'"')
        file.write('\n'+'verbose = '+str(self.verbose))
        file.write('\n'+'logging = '+str(self.logging))
        file.close()
        
    def runWaveCal(self):
        w = W.WaveCal(config_file=self.cfgpath)
        w.makeCalibration()

    def __configCheck(self, index):
        '''
        Checks the variables loaded in from the configuration file for type and
        consistencey. Run in the '__init__()' method.
        '''
        if index == 0:
            # check for configuration file, and any other keyword args
            assert os.path.isfile(self.config_directory), \
                self.config_directory + " is not a valid configuration file"

        elif index == 1:
            # check if all sections and parameters exist in the configuration file
            section = "{0} must be a configuration section"
            param = "{0} must be a parameter in the configuration file '{1}' section"

            assert 'Data' in self.config.sections(), section.format('Data')
            assert 'wavelengths' in self.config['Data'].keys(), \
                param.format('wavelengths', 'Data')
            assert 'startTimes' in self.config['Data'].keys(), \
                param.format('startTimes', 'Data')
            assert 'expTimes' in self.config['Data'].keys(), \
                param.format('expTimes', 'Data')
            assert 'dataDir' in self.config['Data'].keys(), \
                param.format('dataDir', 'Data')
            assert 'beamDir' in self.config['Data'].keys(), \
                param.format('beamDir', 'Data')

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
            assert 'outH5Dir' in self.config['Output'], \
                param.format('outH5Dir', 'Output')
            assert 'outCalSolnDir' in self.config['Output'], \
                param.format('outCalSolnDir', 'Output')
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

        elif index == 2:
            # type check parameters
            assert type(self.wavelengths) is list, "wavelengths parameter must be a list."
            assert type(self.startTimes) is list, "startTimes parameter must be a list."
            assert type(self.expTimes) is list, "expTimes parameter must be a list."
            assert type(self.model_name) is str, "model_name parameter must be a string."
            assert type(self.save_plots) is bool, "save_plots parameter must be a boolean"
            assert type(self.verbose) is bool, "verbose parameter bust be a boolean"
            assert type(self.dataDir) is str, "Data directory parameter must be a string"
            assert type(self.beamDir) is str, "Beam directory parameter must be a string"
            assert type(self.outH5Dir) is str, "Output H5 directory parameter must be a string"
            assert type(self.outCalSolnDir) is str, "Output Cal Solution file directory parameter must be a string"
            assert type(self.logging) is bool, "logging parameter must be a boolean"
            assert type(self.parallel) is bool, "parallel parameter must be a boolean"
            assert type(self.summary_plot) is bool, "summary_plot parameter must be a boolean"
            assert type(self.plot_file_name) is str, \
                "plot_file_name parameter must be a string"
            assert type(self.templar_config) is str, \
                "templar_config parameter must be a string"
            assert type(self.outCalSolnDir) is str, \
                "out_directory parameter must be a string"
            assert os.path.isdir(self.outCalSolnDir), \
                "{0} is not a valid output directory".format(self.outCalSolnDir)
            assert type(self.outH5Dir) is str, \
                "out_directory parameter must be a string"
            assert os.path.isdir(self.outH5Dir), \
                "{0} is not a valid output directory".format(self.outH5Dir)

            assert len(self.wavelengths) == len(self.startTimes), \
                "wavelengths and file_names parameters must be the same length."
            if type(self.bin_width) is int:
                self.bin_width = float(self.bin_width)
            assert type(self.bin_width) is float, \
                "bin_width parameter must be an integer or float"

            if type(self.dt) is int:
                self.dt = float(self.dt)
            assert type(self.dt) is float, "dt parameter must be an integer or float"

if __name__ == '__main__':
   if len(sys.argv) == 1:
       runwavecal = RunWaveCal()
   else:
       runwavecal = RunWaveCal(config_file=sys.argv[1])
   runwavecal.makeH5CfgFiles()
   runwavecal.makeWaveCalCfgFile()
   runwavecal.runWaveCal()
        


