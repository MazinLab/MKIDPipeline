"""
Author: Isabel Lipartito        Date: July 19 2018

Quick routine to calibrate an observation or set of dithered observations.  Assuming full performance, this code will:
-Load in data (dithered or single obs), dark, and flat img files.
Note:  Code can handle a separate dark for the data files and the flat files
-Make master darks and a HPM
-Dark subtract the flat and make a master flat calibration weight file
-Dark subtract, flat field, and HPM correct the observation/dither set
-For dithers:  align the frames to first-order and make a median final image
-For single observations:  make a median final image
-Save the final image and final calibration files to a multi-dimensional fits image
"""

import glob
import os
import sys

import numpy as np

import mkidpipeline.hotpix.darkHotPixMask as dhpm
import mkidpipeline.utils.irUtils as irUtils
from mkidpipeline.utils.arrayPopup import plotArray
from mkidpipeline.utils.loadStack import loadIMGStack

class ImgReducerConfig:

    """
    Opens the image processing pipeline config file and loads in flat, dark, and science obs information
    """

    def __init__(cfg, file='default.cfg'):
        # define the configuration file path
        self.file = DEFAULT_CONFIG_FILE if file == 'default.cfg' else file

        assert os.path.isfile(self.file), \
            self.file + " is not a valid configuration file"
        self.config = ConfigParser()
        self.config.read(self.config_file)

        # check the configuration file format and load the parameters
        self.checksections()
        self.darkSpanFlat = ast.literal_eval(self.config['Data']['darkSpanFlat'])
        self.darkSpanImg = ast.literal_eval(self.config['Data']['darkSpanImg'])
        self.flatSpan = ast.literal_eval(self.config['Data']['flatSpan'])
        self.startTimes = ast.literal_eval(self.config['Data']['startTimes'])
        self.stopTimes = ast.literal_eval(self.config['Data']['stopTimes'])
        self.nPos = ast.literal_eval(self.config['Data']['nPos'])
        self.xPos = ast.literal_eval(self.config['Data']['xPos'])
        self.yPos = ast.literal_eval(self.config['Data']['yPos'])
        self.numRows = ast.literal_eval(self.config['Obs']['numRows'])
        self.numCols = ast.literal_eval(self.config['Obs']['numCols'])
        self.date = ast.literal_eval(self.config['Obs']['date'])
        self.run = ast.literal_eval(self.config['Obs']['run'])
        self.beammapFile = ast.literal_eval(self.config['Obs']['beammapFile'])
        self.divideFlat= ast.literal_eval(self.config['Proc']['divideFlat'])
        self.subtractDark= ast.literal_eval(self.config['Proc']['subtractDark'])
        self.doHPM= ast.literal_eval(self.config['Proc']['doHPM'])
        self.coldCut= ast.literal_eval(self.config['Proc']['coldCut'])
        self.outputDir= ast.literal_eval(self.config['Output']['outputDir'])
        self.savePlots= ast.literal_eval(self.config['Output']['savePlots'])

        if self.config.has_option('Data', 'imgPath'):
            self.img_path = ast.literal_eval(self.config['Data']['imgPath'])

        else:
            runDir = os.path.join('/mnt/data0/ScienceDataImgs/', self.run)
            self.img_path = os.path.join(runDir, self.date)

        # check the parameter formats
        self.checktypes()

    def checksections(self):
        # check if all sections and parameters exist in the configuration file
        section = "{0} must be a configuration section"
        param = "{0} must be a parameter in the configuration file '{1}' section"

        assert 'Data' in self.config.sections(), section.format('Data')
        assert 'darkSpanFlat' in self.config['Data'].keys(), \
            param.format('darkSpanFlat', 'Data')
        assert 'darkSpanImg' in self.config['Data'].keys(), \
            param.format('darkSpanImg', 'Data')
        assert 'flatSpan' in self.config['Data'].keys(), \
            param.format('flatSpan', 'Data')
        assert 'startTimes' in self.config['Data'].keys(), \
            param.format('startTimes', 'Data')
        assert 'stopTimes' in self.config['Data'].keys(), \
            param.format('stopTimes', 'Data')
        assert 'nPos' in self.config['Data'].keys(), \
            param.format('nPos', 'Data')
        assert 'xPos' in self.config['Data'].keys(), \
            param.format('xPos', 'Data')
        assert 'yPos' in self.config['Data'].keys(), \
            param.format('yPos', 'Data')

        assert 'Obs' in self.config.sections(), section.format('Obs')


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

if __name__ == '__main__':

    pipelinelog.setup_logging()

    log = getLogger('ImgReducer')

    timestamp = datetime.utcnow().timestamp()

    config=loadimgcfg()

    flatdata=loadIMGStack(cfg.flatinfo)

    fdarkdata=loadIMGStack(cfg.fdarkinfo)
    fdark=make_dark(fdarkdata)

    masterdarkdata=loadIMGStack(cfg.masterdarkinfo)
    masterdark=make_dark(masterdarkdata)

    masterflat=make_flat(flatdata-fdark)

    for img in cfg.imginfo:
        reducedimg= (loadIMGstack(cfg.imginfo)-masterdark)/masterflat
        image=ReducedIM(reducedimg, masterdark, masterflat)
        if save_individual:
            image.save_to_fits





