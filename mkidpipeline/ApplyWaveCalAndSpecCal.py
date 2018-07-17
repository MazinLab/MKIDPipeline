import ast
import glob
import os
import sys
from configparser import ConfigParser

import numpy as np

from mkidpipeline.calibration import wavecal as W


class ApplyWaveCalAndSpecCal:
    '''
    Created by: Isabel Lipartito, June 2018
    A wrapper to complete basic image reduction for a single observation OR dither (set of observations)
    Currently:  Makes and applies wavecal and flatcal (optional).  Full spectral calibration in the future
    Inputs:  Wavelengths of lasers used in this wavecal, each laser's start time and exposure time.
             EITHER the path to the appropriate dither config file OR the start and exposure time of the observation
             Flat Observation start and exposure times (optional)
    See defaultApplyWaveCalAndSpecCal.cfg for a complete list of input parameters (you also need to input the fit/output parameters 
    received by WaveCal.py
    Replaces the individual processes of:
              1.  Making individual H5 cfg files for the science data, each laser for wavecal, and flat data (optional)
              2.  Running Bin2HDF for the science+cal data
              3.  Making a wavecal cfg file
              4.  Running WaveCal.py on the wavecal cfg file
              5.  Applying the wavecal solution to the science data
                  (Optional)
              6.  Making a flatcal solution file 
              7.  Running FlatCal.py on the flatcal cfg file 
              8.  Applying the flatcal solution file to the science data
    '''
    def __init__(self, config_file='default.cfg'):

        # define the configuration file path
        self.config = config_file
        print(self.config)
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

        self.dataDir= ast.literal_eval(self.config['DataObs']['dataDir'])
        self.beamDir= ast.literal_eval(self.config['DataObs']['beamDir'])
        self.XPIX= ast.literal_eval(self.config['DataObs']['XPIX'])
        self.YPIX= ast.literal_eval(self.config['DataObs']['YPIX'])
        self.ditherBool= ast.literal_eval(self.config['DataObs']['ditherBool'])
        self.flatBool= ast.literal_eval(self.config['DataObs']['flatBool'])
        self.mapFlag= ast.literal_eval(self.config['DataObs']['mapFlag'])
        self.filePrefix= ast.literal_eval(self.config['DataObs']['filePrefix'])
        self.b2hPath=ast.literal_eval(self.config['DataObs']['b2hPath'])
        self.ditherStackFile=ast.literal_eval(self.config['DataObs']['ditherStackFile'])
        self.startTimeObs=ast.literal_eval(self.config['DataObs']['startTimeObs'])
        self.expTimeObs=ast.literal_eval(self.config['DataObs']['expTimeObs'])

        self.startTimeFlat=ast.literal_eval(self.config['DataFlatcal']['startTimeFlat'])
        self.expTimeFlat=ast.literal_eval(self.config['DataFlatcal']['expTimeFlat'])
        self.intTimeFlat=ast.literal_eval(self.config['DataFlatcal']['intTimeFlat'])
        self.wvlStart=ast.literal_eval(self.config['DataFlatcal']['wvlStart'])
        self.wvlStop=ast.literal_eval(self.config['DataFlatcal']['wvlStop'])

        self.wavelengths = ast.literal_eval(self.config['DataWavecal']['wavelengths'])
        self.startTimesWave = ast.literal_eval(self.config['DataWavecal']['startTimesWave'])
        self.expTimesWave = ast.literal_eval(self.config['DataWavecal']['expTimesWave'])

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
        self.templar_config= ast.literal_eval(self.config['Output']['templar_config'])

        # check the parameter formats
        self.__configCheck(2)

        self.path=self.b2hPath

        self.wavepath=self.outH5Dir+'wavecal/'
        if not os.path.isdir(self.wavepath):
            self.wavepath=os.mkdir(self.wavepath)

        self.flatpath=self.outH5Dir+'flatcal/'
        if not os.path.isdir(self.flatpath):
            self.flatpath=os.mkdir(self.flatpath)

        self.obspath=self.outH5Dir+'ScienceObs/'
        if not os.path.isdir(self.obspath):
            self.obspath=os.mkdir(self.obspath)

    def makeWaveCalH5CfgFiles(self):
        scriptpath=str(self.wavepath)+'makeh5files.sh'
        scriptfile=open(scriptpath, 'w')       
        scriptfile.write('#!/bin/bash') 
        for i in np.arange(len(self.wavelengths)):
            wavelengthpath=str(self.wavepath)+str(self.wavelengths[i])+'nm.txt'
            cwd = os.getcwd()
            file=open(wavelengthpath,'w')
            file.write(str(self.XPIX)+str(' ')+ str(self.YPIX))
            file.write('\n'+self.dataDir)
            file.write('\n'+'%d' %self.startTimesWave[i])
            file.write('\n'+'%d' %self.expTimesWave[i])
            file.write('\n'+self.beamDir)
            file.write('\n'+str(1))
            file.write('\n'+str(self.wavepath))
            file.close()
            os.chdir(self.path)
            os.system('./Bin2HDF '+wavelengthpath)
            os.chdir(cwd)
            scriptfile.write('\n'+'./Bin2HDF '+wavelengthpath)   
        scriptfile.close() 
             
    def makeWaveCalCfgFile(self):
        self.wvcalfilenames=[]
        for i in np.arange(len(self.startTimesWave)):
            filename='%d' %self.startTimesWave[i]+'.h5'
            self.wvcalfilenames.append(filename)
        self.wvcfgpath=str(self.wavepath)+'WvCalCfg.cfg'
        file=open(self.wvcfgpath, 'w')
        file.write('[Data]')
        file.write('\n')
        file.write('\n'+'directory = '+'"'+self.wavepath+'"')
        file.write('\n'+'wavelengths= '+str(self.wavelengths))
        file.write('\n'+'file_names= '+str(self.wvcalfilenames))
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
        file.write('\n'+'out_directory = '+'"'+self.wavepath+'"')
        file.write('\n'+'save_plots = '+str(self.save_plots))
        file.write('\n'+'plot_file_name = ''"'+str(self.plot_file_name)+'"')
        file.write('\n'+'summary_plot = '+str(self.summary_plot))
        file.write('\n'+'templar_config = '+'"'+self.templar_config+'"')
        file.write('\n'+'verbose = '+str(self.verbose))
        file.write('\n'+'logging = '+str(self.logging))
        file.close()
        
    def runWaveCal(self):
        w = W.WaveCal(config_file=self.wvcfgpath)
        w.makeCalibration()
        self.waveCalSolnFile = max(glob.iglob(str(self.wavepath)+'calsol'+'*.h5'), key=os.path.getctime)

    def makeFlatCalH5CfgFile(self):
        flatfieldpath=self.flatpath+'Flat.txt'
        file=open(flatfieldpath,'w')
        file.write(str(self.XPIX)+str(' ')+ str(self.YPIX))
        file.write('\n'+self.dataDir)
        file.write('\n'+'%d' %self.startTimeFlat)
        file.write('\n'+'%d' %self.expTimeFlat)
        file.write('\n'+self.beamDir)
        file.write('\n'+str(1))
        file.write('\n'+self.flatpath)
        file.close()
        os.chdir(self.path)
        os.system('./Bin2HDF '+flatfieldpath)
        os.chdir(cwd)
        
    def makeSingleObsH5CfgFile(self):
        observationpath=self.obspath+'Obs.txt'
        file=open(observationpath,'w')
        file.write(str(self.XPIX)+str(' ')+ str(self.YPIX))
        file.write('\n'+self.dataDir)
        file.write('\n'+'%d' %self.startTimeObs)
        file.write('\n'+'%d' %self.expTimeObs)
        file.write('\n'+self.beamDir)
        file.write('\n'+str(1))
        file.write('\n'+self.obspath)
        file.close()
        os.chdir(self.path)
        os.system('./Bin2HDF '+observationpath)
        os.chdir(cwd)
        self.obsFiles=[]
        self.obsFiles.append(self.obspath+str(self.startTimeObs)+'.h5')

    def makeDitherObsHFCfgFile(self):
        observationpath=self.obspath+'ObsDither.cfg'
        file=open(observationpath,'w')
        file.write('[Data]')
        file.write('\n')
        file.write('\n'+'XPIX = '+str(self.XPIX))
        file.write('\n'+'YPIX = '+str(self.YPIX))
        file.write('\n'+'binPath = '+'"'+self.dataDir+'"')
        file.write('\n'+'outPath = '+'"'+self.outH5Dir+'"')
        file.write('\n'+'beamFile = '+'"'+self.beamDir+'"')
        file.write('\n'+'mapFlag = '+str(1))
        file.write('\n'+'filePrefix = '+'"'+'a'+'"')
        file.write('\n'+'b2hPath = '+'"'+self.path+'"')
        file.write('\n'+'ditherStackFile = '+'"'+self.ditherStackFile+'"')
        file.close()    
        os.chdir(self.path)
        os.system('python Dither2H5.py '+observationpath)
        os.chdir(cwd)
        self.obsFiles=glob.glob(self.obspath+'*.h5')  

    def makeFlatCalH5CfgFile(self):
        self.flatcfgpath=self.outFlatCalSolnDir+'FlatCalCfg.cfg'
        self.flatCalSolnpath=self.flatpath+"flatcalsoln.h5"
        file=open(self.flatcfgpath, 'w')  
        file.write('[Data]')
        file.write('\n')
        file.write('\n'+'wvlCalFile = '+'"'+self.waveCalSolnFile+'"')
        file.write('\n'+'flatPath= +''"'+self.flatpath+'"')
        file.write('\n'+'calSolnPath= '+'"'+self.flatCalSolnpath+'"')
        file.write('\n'+'intTime = '+str(self.intTimeFlat))
        file.write('\n'+'expTime = '+str(self.expTimeFlat))
        file.write('\n')
        file.write('\n'+'[Instrument]')
        file.write('\n')
        file.write('\n'+'deadtime = '+str(0.000001))
        file.write('\n'+'energyBinWidth = '+str(0.0100000))
        file.write('\n'+'wvlStart = '+str(self.wvlStart))
        file.write('\n'+'wvlStop = '+str(self.wvlStop))
        file.write('\n')
        file.write('\n'+'[Calibration]')
        file.write('\n')
        file.write('\n'+'countRateCutoff = '+str(20000))
        file.write('\n'+'fractionOfChunksToTrim = '+str(0))
        file.close()

    def runFlatCal(self):
        f = f.FlatCal(config_file=self.flatcfgpath)

    def applyWaveCal(self):
        self.obsandFlatFiles=self.obsFiles.append(self.flatpath+str(self.startTimeFlat)+'.h5')
        for ObsFN in self.obsandFlatFiles:
            obsfile=obs(ObsFN, mode='write')
            obsfilecal=obs.applyWaveCal(obsfile,self.waveCalSolnFile)        
        
    def applyFlatCal(self):
        for ObsFN in self.obsFiles:
            obsfile=obs(ObsFN, mode='write')
            obsfilecal=obs.applyFlatCal(obsfile,self.flatCalSolnFile)


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

            assert 'DataObs' in self.config.sections(), section.format('DataObs')
            assert 'dataDir' in self.config['DataObs'].keys(), \
                param.format('dataDir', 'DataObs')
            assert 'beamDir' in self.config['DataObs'].keys(), \
                param.format('beamDir', 'DataObs')
            assert 'XPIX' in self.config['DataObs'].keys(), \
                param.format('XPIX', 'DataObs')
            assert 'YPIX' in self.config['DataObs'].keys(), \
                param.format('YPIX', 'DataObs')
            assert 'ditherBool' in self.config['DataObs'].keys(), \
                param.format('ditherBool', 'DataObs')
            assert 'flatBool' in self.config['DataObs'].keys(), \
                param.format('flatBool', 'DataObs')
            assert 'mapFlag' in self.config['DataObs'].keys(), \
                param.format('mapFlag', 'DataObs')
            assert 'filePrefix' in self.config['DataObs'].keys(), \
                param.format('filePrefix', 'DataObs')
            assert 'b2hPath' in self.config['DataObs'].keys(), \
                param.format('b2hPath', 'DataObs')
            assert 'ditherStackFile' in self.config['DataObs'].keys(), \
                param.format('ditherStackFile', 'DataObs')

            assert 'startTimeFlat' in self.config['DataFlatcal'].keys(), \
                param.format('startTimeFlat', 'DataFlatcal')
            assert 'expTimeFlat' in self.config['DataFlatcal'].keys(), \
                param.format('expTimeFlat', 'DataFlatcal')
            assert 'intTimeFlat' in self.config['DataFlatcal'].keys(), \
                param.format('intTimeFlat', 'DataFlatcal')
            assert 'wvlStart' in self.config['DataFlatcal'].keys(), \
                param.format('wvlStart', 'DataFlatcal')
            assert 'wvlStop' in self.config['DataFlatcal'].keys(), \
                param.format('wvlStop', 'DataFlatcal')

            assert 'wavelengths' in self.config['DataWavecal'].keys(), \
                param.format('wavelengths', 'DataWavecal')
            assert 'startTimesWave' in self.config['DataWavecal'].keys(), \
                param.format('startTimesWave', 'DataWavecal')
            assert 'expTimesWave' in self.config['DataWavecal'].keys(), \
                param.format('expTimesWave', 'DataWavecal')

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
            assert 'outH5Dir' in self.config['Output'], \
                param.format('outH5Dir', 'Output')
            assert 'templar_config' in self.config['Output'], \
                param.format('templar_config', 'Output')

        elif index == 2:
            # type check parameters
            assert type(self.dataDir) is str, "Data directory parameter must be a string"
            assert type(self.beamDir) is str, "Beam directory parameter must be a string"
            assert type(self.XPIX) is int, "Number of X Pix parameter must be an integer"
            assert type(self.YPIX) is int, "Number of Y Pix parameter must be an integer"
            assert type(self.ditherBool) is bool, "ditherBool parameter must be a boolean"
            assert type(self.flatBool) is bool, "flatBool parameter must be a boolean"
            assert type(self.b2hPath) is str, "Bin 2 HDF directory parameter must be a string"
            assert type(self.mapFlag) is int, "mapFlat parameter must be an integer"
            assert type(self.filePrefix) is str, "filePrefix parameter must be a string"
            assert type(self.ditherStackFile) is str, "Directory where the dither stack file is stored must be a string"
            assert type(self.startTimeObs) is int, "Observation start time parameter must be an integer"
            assert type(self.expTimeObs) is int, "Observation exposure time parameter must be an integer"

            assert type(self.startTimeFlat) is int, "Flat start time parameter must be an integer"
            assert type(self.expTimeFlat) is int, "Flat exposure time parameter must be an integer"
            assert type(self.intTimeFlat) is int, "Flat integration time parameter must be an integer"
            assert type(self.wvlStart) is int, "Starting wavelength for flat calibration must be an integer"
            assert type(self.wvlStop) is int, "Stopping wavelength for flat calibration must be an integer"

            assert type(self.wavelengths) is list, "wavelengths parameter must be a list."
            assert type(self.startTimesWave) is list, "startTimes parameter must be a list."
            assert type(self.expTimesWave) is list, "expTimes parameter must be a list."

            assert type(self.model_name) is str, "model_name parameter must be a string."
            assert type(self.bin_width) is int, "bin_width parameter must be an integer."
            assert type(self.dt) is int, "dt parameter must be an integer."
            assert type(self.parallel) is bool, "parallel parameter must be a boolean"
            assert type(self.save_plots) is bool, "save_plots parameter must be a boolean"
            assert type(self.plot_file_name) is str, "plot_file_name parameter must be a string"
            assert type(self.verbose) is bool, "verbose parameter bust be a boolean"
            assert type(self.logging) is bool, "logging parameter must be a boolean"
            assert type(self.summary_plot) is bool, "summary plot parameter must be a boolean"
            assert type(self.outH5Dir) is str, "Output H5 directory parameter must be a string"
            assert type(self.templar_config) is str, "templar_config parameter must be a string"

            assert os.path.isdir(self.outH5Dir), \
                "{0} is not a valid output directory".format(self.outH5Dir)

            assert len(self.wavelengths) == len(self.startTimesWave), \
                "wavelengths and file_names parameters must be the same length."

if __name__ == '__main__':
   if len(sys.argv) == 1:
       ApplyWaveCalAndSpecCal = ApplyWaveCalAndSpecCal()
   else:
       ApplyWaveCalAndSpecCal = ApplyWaveCalAndSpecCal(config_file=sys.argv[1])
       #ApplyWaveCalAndSpecCal.makeWaveCalH5CfgFiles()
       ApplyWaveCalAndSpecCal.makeWaveCalCfgFile()
       ApplyWaveCalAndSpecCal.runWaveCal()
        


