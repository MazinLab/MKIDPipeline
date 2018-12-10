#!/bin/env python3
"""
Author: Isabel Lipartito        Date:Dec 4, 2017
Opens a twilight flat h5 and breaks it into INTTIME (5 second suggested) blocks.
For each block, this program makes the spectrum of each pixel.
Then takes the median of each energy over all pixels
A factor is then calculated for each energy in each pixel of its
twilight count rate / median count rate
The factors are written out in an h5 file for each block (You'll get EXPTIME/INTTIME number of files)
Plotting options:
Entire array: both wavelength slices and masked wavelength slices
Per pixel:  plots of weights vs wavelength next to twilight spectrum OR
            plots of weights vs wavelength, twilight spectrum, next to wavecal solution
            (has _WavelengthCompare_ in the name)
"""
import argparse
import ast
import atexit
import os
import time
from configparser import ConfigParser
from typing import Any, Union
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tables
from PyPDF2 import PdfFileMerger, PdfFileReader

from matplotlib.backends.backend_pdf import PdfPages
from mkidpipeline.calibration import wavecal
from mkidcore.headers  import FlatCalSoln_Description
from mkidpipeline.hdf.photontable import ObsFile
from mkidcore.corelog import getLogger
import mkidcore.corelog
from mkidpipeline.hdf import bin2hdf

DEFAULT_CONFIG_FILE = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                   'Params', 'default.cfg')

np.seterr(divide='ignore', invalid='ignore')

class FlatCal(object):
    """
    Opens flat file using parameters from the param file, sets wavelength binnning parameters, and calculates flat
    weights for flat file.  Writes these weights to a h5 file and plots weights both by pixel
    and in wavelength-sliced images.
    """
    def __init__(self, config_file='default.cfg', cal_file_name = 'calsol_default.h5'):
        """
        Reads in the param file and opens appropriate flat file.  Sets wavelength binning parameters.
        """
        self.config_file = DEFAULT_CONFIG_FILE if config_file == 'default.cfg' else config_file
        assert os.path.isfile(self.config_file), \
            self.config_file + " is not a valid configuration file"
        # check the configuration file path and read it in
        self.config = ConfigParser()
        self.config.read(self.config_file)
        # check the configuration file format and load the parameters
        self.checksections()
        self.wvlCalFile = ast.literal_eval(self.config['Data']['wvlCalFile'])
        self.h5directory = ast.literal_eval(self.config['Data']['h5directory'])
        self.file_name = ast.literal_eval(self.config['Data']['file_name'])
        self.intTime = ast.literal_eval(self.config['Data']['intTime'])
        self.expTime = ast.literal_eval(self.config['Data']['expTime'])
        self.dataDir = ast.literal_eval(self.config['Data']['dataDir'])
        self.beamDir = ast.literal_eval(self.config['Data']['beamDir'])
        self.startTime = ast.literal_eval(self.config['Data']['startTime'])
        self.xpix = ast.literal_eval(self.config['Data']['xpix'])
        self.ypix = ast.literal_eval(self.config['Data']['ypix'])
        self.deadtime = ast.literal_eval(self.config['Instrument']['deadtime'])
        self.energyBinWidth = ast.literal_eval(self.config['Instrument']['energyBinWidth'])
        self.wvlStart = ast.literal_eval(self.config['Instrument']['wvlStart'])
        self.wvlStop = ast.literal_eval(self.config['Instrument']['wvlStop'])
        self.countRateCutoff = ast.literal_eval(self.config['Calibration']['countRateCutoff'])
        self.fractionOfChunksToTrim = ast.literal_eval(self.config['Calibration']['fractionOfChunksToTrim'])
        self.logging = ast.literal_eval(self.config['Output']['logging'])
        self.out_directory = ast.literal_eval(self.config['Output']['out_directory'])
        self.save_plots = ast.literal_eval(self.config['Output']['save_plots'])
        self.summary_plot = ast.literal_eval(self.config['Output']['summary_plot'])
        self.cal_file_name=cal_file_name
        if self.save_plots:
            answer = self._query("Save Plots flag set to 'yes', this will add ~30 min to the code.  "
                                 "Are you sure you want to save plots?", yes_or_no=True)
            if answer is False:
                self.save_plots = False
                getLogger(__name__).info("Setting save_plots parameter to FALSE")
        self.timeSpacingCut = None

        self.checktypes()
        self.checksections()
        self.checktypes()
        getLogger(__name__).info("Computing Factors for FlatCal")

    def checksections(self):
        """
        Checks the variables loaded in from the configuration file for type and consistency.
        """
        section = "{0} must be a configuration section"
        param = "{0} must be a parameter in the configuration file '{1}' section"

        assert 'Data' in self.config.sections(), section.format('Data')
        assert 'h5directory' in self.config['Data'].keys(), \
            param.format('h5directory', 'Data')
        assert 'file_name' in self.config['Data'].keys(), \
            param.format('file_name', 'Data')
        assert 'startTime' in self.config['Data'].keys(), \
            param.format('startTimes', 'Data')
        assert 'dataDir' in self.config['Data'].keys(), \
            param.format('dataDir', 'Data')
        assert 'beamDir' in self.config['Data'].keys(), \
            param.format('beamDir', 'Data')
        assert 'wvlCalFile' in self.config['Data'].keys(), \
            param.format('wvlCalFile', 'Data')
        assert 'intTime' in self.config['Data'].keys(), \
            param.format('intTime', 'Data')
        assert 'expTime' in self.config['Data'].keys(), \
            param.format('expTime', 'Data')
        assert 'Instrument' in self.config.sections(), section.format('Instrument')
        assert 'deadtime' in self.config['Instrument'].keys(), \
            param.format('deadtime', 'Instrument')
        assert 'energyBinWidth' in self.config['Instrument'].keys(), \
            param.format('energyBinWidth', 'Instrument')
        assert 'wvlStart' in self.config['Instrument'].keys(), \
            param.format('wvlStart', 'Instrument')
        assert 'wvlStop' in self.config['Instrument'].keys(), \
            param.format('wvlStop', 'Instrument')
        assert 'Calibration' in self.config.sections(), section.format('Calibration')
        assert 'countRateCutoff' in self.config['Calibration'], \
            param.format('countRateCutoff', 'Calibration')
        assert 'fractionOfChunksToTrim' in self.config['Calibration'], \
            param.format('fractionOfChunksToTrim', 'Calibration')
        assert 'out_directory' in self.config['Output'].keys(), \
            param.format('out_directory', 'Output')
        assert 'save_plots' in self.config['Output'].keys(), \
            param.format('save_plots', 'Output')
        assert 'summary_plot' in self.config['Output'].keys(), \
            param.format('summary_plot', 'Output')
        assert 'logging' in self.config['Output'], \
            param.format('logging', 'Output')

    def checktypes(self):
        """
        type check parameters
        """
        if self.h5directory != '':
            assert type(self.h5directory) is str, "Flat Path parameter must be a string."
            assert os.path.exists(self.h5directory), "Please confirm the Flat File path provided is correct"
        if self.out_directory != '':
            assert type(self.out_directory) is str, "Cal Solution Path parameter must be a string."
        if self.wvlCalFile != '':
            assert type(self.wvlCalFile) is str, "WaveCal Solution Path parameter must be a string."
        assert type(self.intTime) is int, "integration time parameter must be an integer"
        assert type(self.expTime) is int, "Exposure time parameter must be an integer"
        assert type(self.startTime) is int, "Start time parameter must be an integer."

        assert type(self.dataDir) is str, "Data directory parameter must be a string"
        assert type(self.file_name) is str, "File Name parameter must be a string"
        assert type(self.beamDir) is str, "Beam directory parameter must be a string"
        if type(self.deadtime) is int:
            self.deadtime = float(self.deadtime)
        assert type(self.deadtime) is float, "Dead time parameter must be an integer or float"
        if type(self.energyBinWidth) is int:
            self.energyBinWidth = float(self.energyBinWidth)
        assert type(self.energyBinWidth) is float, "Energy Bin Width parameter must be an integer or float"
        if type(self.wvlStart) is int:
            self.wvlStart = float(self.wvlStart)
        assert type(self.wvlStart) is float, "Starting Wavelength must be an integer or float"
        if type(self.wvlStop) is int:
            self.wvlStop = float(self.wvlStop)
        assert type(self.wvlStop) is float, "Stopping Wavelength must be an integer or float"
        if type(self.countRateCutoff) is int:
            self.countRateCutoff = float(self.countRateCutoff)
        assert type(self.countRateCutoff) is float, "Count Rate Cutoff must be an integer or float"
        assert type(self.fractionOfChunksToTrim) is int, "Fraction of Chunks to Trim must be an integer"
        assert type(self.save_plots) is bool, "Save Plots indicator must be a bool"
        assert type(self.summary_plot) is bool, "Save Summary Plot indicator must be a bool"
        assert type(self.logging) is bool, "logging parameter must be a boolean"

    def hdfexist(self):
        return(os.path.isfile(self.h5directory+self.file_name))

    def makeCalibration(self):
        """
        wvlBinEdges includes both lower and upper limits, so number of bins is 1 less than number of edges
        """
        self.obs = ObsFile(self.h5directory)
        self.flatCalFileName = self.out_directory + self.cal_file_name
        self.beamImage = self.obs.beamImage
        self.wvlFlags = self.obs.beamFlagImage
        self.xpix = self.obs.nXPix
        self.ypix = self.obs.nYPix
        self.wvlBinEdges = ObsFile.makeWvlBins(self.energyBinWidth, self.wvlStart, self.wvlStop)
        self.nWvlBins = len(self.wvlBinEdges) - 1

        self.loadFlatSpectra()
        self.checkCountRates()
        self.calculateWeights()

    def loadFlatSpectra(self):
        """
        Reads the flat data into a spectral cube whose dimensions are determined
        by the number of x and y pixels and the number of wavelength bins.
        Each element will be the spectral cube for a time chunk
        Find factors to correct nonlinearity due to deadtime in firmware
        """
        self.frames = []
        self.spectralCubes = []
        self.cubeEffIntTimes = []
        for firstSec in range(0, self.expTime, self.intTime):
            #for each time chunk
            cubeDict = self.obs.getSpectralCube(firstSec=firstSec, integrationTime=self.intTime, applySpecWeight=False,
                                               applyTPFWeight=False, wvlBinEdges=self.wvlBinEdges, energyBinWidth=None,
                                               timeSpacingCut=self.timeSpacingCut)
            cube = np.array(cubeDict['cube'], dtype=np.double)
            effIntTime = cubeDict['effIntTime']
            # add third dimension for broadcasting
            effIntTime3d = np.reshape(effIntTime, np.shape(effIntTime) + (1,))
            cube /= effIntTime3d
            cube[np.isnan(cube)] = 0
            rawFrameDict = self.obs.getPixelCountImage(firstSec=firstSec, integrationTime=self.intTime,
                                                      scaleByEffInt=True)
            rawFrame = np.array(rawFrameDict['image'], dtype=np.double)
            rawFrame /= rawFrameDict['effIntTimes']
            nonlinearFactors = 1. / (1. - rawFrame * self.deadtime)
            nonlinearFactors[np.isnan(nonlinearFactors)] = 0.
            frame = np.sum(cube, axis=2)  # in counts per sec
            frame = frame * nonlinearFactors
            nonlinearFactors = np.reshape(nonlinearFactors, np.shape(nonlinearFactors) + (1,))
            cube = cube * nonlinearFactors
            self.frames.append(frame)
            self.spectralCubes.append(cube)
            self.cubeEffIntTimes.append(effIntTime3d)
            getLogger(__name__).info('Loaded Flat Spectra for seconds {} to {}'.format(int(firstSec), int(firstSec) + int(self.intTime)))
        self.obs.file.close()
        self.spectralCubes = np.array(self.spectralCubes)
        self.cubeEffIntTimes = np.array(self.cubeEffIntTimes)
        self.countCubes = self.cubeEffIntTimes * self.spectralCubes

    def checkCountRates(self):
        """
        mask out frames, or cubes from integration time chunks with count rates too high
        """
        medianCountRates = np.array([np.median(frame[frame != 0]) for frame in self.frames])
        boolIncludeFrames: Union[bool, Any] = medianCountRates <= self.countRateCutoff
        self.spectralCubes = np.array([cube for cube, boolIncludeFrame in zip(self.spectralCubes, boolIncludeFrames)
                                       if boolIncludeFrame == True])
        self.frames = [frame for frame, boolIncludeFrame in zip(self.frames, boolIncludeFrames)
                       if boolIncludeFrame == True]

    def calculateWeights(self):
        """
        finds flat cal factors as medians/pixelSpectra for each pixel.  Normalizes these weights at each wavelength bin.
        Trim the beginning and end off the sorted weights for each wvl for each pixel, to exclude extremes from averages
        """
        self.flatWeightsList = []
        cubeWeightsList = []
        self.averageSpectra = []
        deltaWeightsList = []
        for iCube, cube in enumerate(self.spectralCubes):
            effIntTime = self.cubeEffIntTimes[iCube]
            # for each time chunk
            wvlAverages = np.zeros(self.nWvlBins)
            spectra2d = np.reshape(cube, [self.xpix * self.ypix, self.nWvlBins])
            for iWvl in range(self.nWvlBins):
                wvlSlice = spectra2d[:, iWvl]
                goodPixelWvlSlice = np.array(wvlSlice[wvlSlice != 0])
                # dead pixels need to be taken out before calculating averages
                wvlAverages[iWvl] = np.median(goodPixelWvlSlice)
            weights = np.divide(wvlAverages, cube)
            weights[weights == 0] = np.nan
            weights[weights == np.inf] = np.nan
            cubeWeightsList.append(weights)

            """
            To get uncertainty in weight:
            Assuming negligible uncertainty in medians compared to single pixel spectra,
            then deltaWeight=weight*deltaSpectrum/Spectrum
            deltaWeight=weight*deltaRawCounts/RawCounts
            with deltaRawCounts=sqrt(RawCounts)#Assuming Poisson noise
            deltaWeight=weight/sqrt(RawCounts)
            but 'cube' is in units cps, not raw counts so multiply by effIntTime before sqrt
            """

            deltaWeights = weights / np.sqrt(effIntTime * cube)
            deltaWeightsList.append(deltaWeights)
            self.averageSpectra.append(wvlAverages)
        cubeWeights = np.array(cubeWeightsList)
        deltaCubeWeights = np.array(deltaWeightsList)
        cubeWeightsMask = np.isnan(cubeWeights)
        self.maskedCubeWeights = np.ma.array(cubeWeights, mask=cubeWeightsMask, fill_value=1.)
        nCubes = np.shape(self.maskedCubeWeights)[0]
        self.maskedCubeDeltaWeights = np.ma.array(deltaCubeWeights, mask=cubeWeightsMask)

        # sort maskedCubeWeights and rearrange spectral cubes the same way

        if self.fractionOfChunksToTrim and nCubes > 1:
            sortedIndices = np.ma.argsort(self.maskedCubeWeights, axis=0)
            identityIndices = np.ma.indices(np.shape(self.maskedCubeWeights))
            sortedWeights = self.maskedCubeWeights[sortedIndices, identityIndices[1], identityIndices[2], identityIndices[3]]
            countCubesReordered = self.countCubes[sortedIndices, identityIndices[1], identityIndices[2], identityIndices[3]]
            cubeDeltaWeightsReordered = self.maskedCubeDeltaWeights[sortedIndices, identityIndices[1], identityIndices[2], identityIndices[3]]
            trimmedWeights = sortedWeights[self.fractionOfChunksToTrim * nCubes:(1 - self.fractionOfChunksToTrim) * nCubes, :, :, :]
            trimmedCountCubesReordered = countCubesReordered[self.fractionOfChunksToTrim * nCubes:(1 - self.fractionOfChunksToTrim) * nCubes,:, :, :]
            trimmedCubeDeltaWeightsReordered = cubeDeltaWeightsReordered[self.fractionOfChunksToTrim * nCubes:(1 - self.fractionOfChunksToTrim) * nCubes,:, :, :]
            self.totalCube = np.ma.sum(trimmedCountCubesReordered, axis=0)
            self.totalFrame = np.ma.sum(self.totalCube, axis=-1)
            self.flatWeights, summedAveragingWeights = np.ma.average(trimmedWeights, axis=0,weights=trimmedCubeDeltaWeightsReordered ** -2.,returned=True)
            self.countCubesToSave = np.ma.sum(trimmedCountCubesReordered, axis=0)
        else:
            self.totalCube = np.ma.sum(self.countCubes, axis=0)
            self.totalFrame = np.ma.sum(self.totalCube, axis=-1)
            self.flatWeights, summedAveragingWeights = np.ma.average(self.maskedCubeWeights, axis=0,weights=self.maskedCubeDeltaWeights ** -2., returned=True)
            self.countCubesToSave = np.ma.sum(self.countCubes, axis=0)

        """
        Uncertainty in weighted average is sqrt(1/sum(averagingWeights))
        Normalize weights at each wavelength bin
        """

        self.deltaFlatWeights = np.sqrt(summedAveragingWeights ** -1.)
        self.flatFlags = self.flatWeights.mask
        wvlWeightMedians = np.ma.median(np.reshape(self.flatWeights, (-1, self.nWvlBins)), axis=0)
        self.flatWeights = np.divide(self.flatWeights, wvlWeightMedians)
        self.flatWeightsforplot = np.ma.sum(self.flatWeights, axis=-1)
        self.writeWeights()
        if self.save_plots:
            self.plotWeightsWvlSlices()
            getLogger(__name__).info('Plotted Weights by Wvl Slices at WvlSlices_{}'.format(timestamp))
            self.plotWeightsByPixelWvlCompare()
            getLogger(__name__).info('Plotted Weights by Pixel against the Wavelength Solution at WavelengthCompare_{}'.format(timestamp))
        if self.summary_plot:
            self.makeSummary()
            getLogger(__name__).info('Made Summary Plot')

    def makeh5(self):

        flatpath = '{}{}.txt'.format(self.h5directory, 'flat')
        b2h_config=bin2hdf.Bin2HdfConfig(datadir=self.dataDir,
                                                     beamfile=self.beamDir, outdir=self.h5directory,
                                                     starttime=self.startTime, inttime=self.expTime, x=self.xpix,
                                                     y=self.ypix, writeto=flatpath)
        getLogger(__name__).info('Made h5 file at {}.h5'.format(self.startTime))

        bin2hdf.makehdf(b2h_config, maxprocs=1)
        self.h5directory=self.h5directory+str(self.startTime)+'.h5'
        getLogger(__name__).info('Applied Wavecal {} to {}.h5'.format(self.wvlCalFile, self.startTime))
        obsfile = ObsFile(self.h5directory, mode='write')
        ObsFile.applyWaveCal(obsfile, self.wvlCalFile)

    def plotWeightsByPixelWvlCompare(self):
        """
        Plot weights of each wavelength bin for every single pixel
        Makes a plot of wavelength vs weights, twilight spectrum, and wavecal solution for each pixel
        """
        if not self.save_plots:
            return
        if self.save_plots:
            self.plotName = 'WavelengthCompare_{}'.format(timestamp)
            self._setupPlots()
        # path to your wavecal solution file
        wavesol=wavecal.Solution(self.wvlCalFile)
        matplotlib.rcParams['font.size'] = 4
        wvls = self.wvlBinEdges[0:-1]
        nCubes = len(self.maskedCubeWeights)
        for iRow in range(self.xpix):
            for iCol in range(self.ypix):
                weights = self.flatWeights[iRow, iCol, :]
                deltaWeights = self.deltaFlatWeights[iRow, iCol, :]
                if not weights.mask.all():
                    if self.iPlot % self.nPlotsPerPage == 0:
                        self.fig = plt.figure(figsize=(10, 10), dpi=100)
                    ax = self.fig.add_subplot(self.nPlotsPerCol, self.nPlotsPerRow, self.iPlot % self.nPlotsPerPage + 1)
                    ax.set_ylim(.5, 2.)
                    for iCube in range(nCubes):
                        cubeWeights = self.maskedCubeWeights[iCube, iRow, iCol]
                        ax.plot(wvls, cubeWeights.data, label='weights %d' % iCube, alpha=.7,
                                color=matplotlib.cm.Paired((iCube + 1.) / nCubes))
                        ax.errorbar(wvls, weights.data, yerr=deltaWeights.data, label='weights', color='k')
                    ax.set_title('p %d,%d' % (iRow, iCol))
                    ax.set_ylabel('weight')
                    ax.set_xlabel(r'$\lambda$ ($\AA$)')
                    if self.iPlot % self.nPlotsPerPage == self.nPlotsPerPage - 1 or (
                            iRow == self.xpix - 1 and iCol == self.ypix - 1):
                        pdf = PdfPages(os.path.join(self.out_directory, 'temp.pdf'))
                        pdf.savefig(self.fig)
                    self.iPlot += 1
                    # Put a plot of twilight spectrums for this pixel
                    if self.iPlot % self.nPlotsPerPage == 0:
                        self.fig = plt.figure(figsize=(10, 10), dpi=100)
                    ax = self.fig.add_subplot(self.nPlotsPerCol, self.nPlotsPerRow, self.iPlot % self.nPlotsPerPage + 1)
                    for iCube in range(nCubes):
                        spectrum = self.spectralCubes[iCube, iRow, iCol]
                        ax.plot(wvls, spectrum, label='spectrum %d' % iCube, alpha=.7,
                                color=matplotlib.cm.Paired((iCube + 1.) / nCubes))
                    ax.set_title('p %d,%d' % (iRow, iCol))
                    ax.set_xlabel(r'$\lambda$ ($\AA$)')
                    if self.iPlot % self.nPlotsPerPage == self.nPlotsPerPage - 1 or (
                            iRow == self.xpix - 1 and iCol == self.ypix - 1):
                        pdf = PdfPages(os.path.join(self.out_directory, 'temp.pdf'))
                        pdf.savefig(self.fig)
                    self.iPlot += 1
                    # Plot wavecal solution
                    if self.iPlot % self.nPlotsPerPage == 0:
                        self.fig = plt.figure(figsize=(10, 10), dpi=100)
                    ax = self.fig.add_subplot(self.nPlotsPerCol, self.nPlotsPerRow, self.iPlot % self.nPlotsPerPage + 1)
                    ax.set_ylim(.5, 2.)
                    for iCube in range(nCubes):
                        my_pixel = [iRow, iCol]
                        wavesol.plot_energy_solution(pixel=my_pixel, axis=ax)
                    ax.set_title('p %d,%d' % (iRow, iCol))
                    if self.iPlot % self.nPlotsPerPage == self.nPlotsPerPage - 1 or (
                            iRow == self.xpix - 1 and iCol == self.ypix - 1):
                        pdf = PdfPages(os.path.join(self.out_directory, 'temp.pdf'))
                        pdf.savefig(self.fig)
                        pdf.close()
                        self._mergePlots()
                        self.saved = True
                        plt.close('all')
                    self.iPlot += 1
        self._closePlots()

    def plotWeightsWvlSlices(self):
        """
        Plot weights in images of a single wavelength bin (wavelength-sliced images)
        """
        self.plotName = 'WvlSlices_{}'.format(timestamp)
        self._setupPlots()
        matplotlib.rcParams['font.size'] = 4
        wvls = self.wvlBinEdges[0:-1]
        for iWvl, wvl in enumerate(wvls):
            if self.iPlot % self.nPlotsPerPage == 0:
                self.fig = plt.figure(figsize=(10, 10), dpi=100)
            ax = self.fig.add_subplot(self.nPlotsPerCol, self.nPlotsPerRow, self.iPlot % self.nPlotsPerPage + 1)
            ax.set_title(r'Weights %.0f $\AA$' % wvl)
            image = self.flatWeights[:, :, iWvl]
            cmap = matplotlib.cm.hot
            cmap.set_bad('#222222')
            handleMatshow = ax.matshow(image, cmap=cmap, origin='lower', vmax=2., vmin=.5)
            self.fig.colorbar(handleMatshow)
            if self.iPlot % self.nPlotsPerPage == self.nPlotsPerPage - 1:
                pdf = PdfPages(os.path.join(self.out_directory, 'temp.pdf'))
                pdf.savefig(self.fig)
            self.iPlot += 1
            ax = self.fig.add_subplot(self.nPlotsPerCol, self.nPlotsPerRow, self.iPlot % self.nPlotsPerPage + 1)
            ax.set_title(r'Twilight Image %.0f $\AA$' % wvl)
            image = self.totalCube[:, :, iWvl]
            nSdev = 3.
            goodImage = image[np.isfinite(image)]
            vmax = np.mean(goodImage) + nSdev * np.std(goodImage)
            handleMatshow = ax.matshow(image, cmap=cmap, origin='lower', vmax=vmax)
            self.fig.colorbar(handleMatshow)
            if self.iPlot % self.nPlotsPerPage == self.nPlotsPerPage - 1:
                pdf = PdfPages(os.path.join(self.out_directory, 'temp.pdf'))
                pdf.savefig(self.fig)
                pdf.close()
                self._mergePlots()
                self.saved = True
                plt.close('all')
            self.iPlot += 1
        self._closePlots()

    def plotMaskWvlSlices(self):
        """
        Plot mask in images of a single wavelength bin (wavelength-sliced images)
        """
        if not self.save_plots:
            return
        if self.save_plots:
            self.plotName = 'MaskWvlSlices_{}'.format(timestamp)
            self._setupPlots()
        matplotlib.rcParams['font.size'] = 4
        wvls = self.wvlBinEdges[0:-1]
        for iWvl, wvl in enumerate(wvls):
            if self.iPlot % self.nPlotsPerPage == 0:
                self.fig = plt.figure(figsize=(10, 10), dpi=100)
            ax = self.fig.add_subplot(self.nPlotsPerCol, self.nPlotsPerRow, self.iPlot % self.nPlotsPerPage + 1)
            ax.set_title(r'%.0f $\AA$' % wvl)
            image = self.flatFlags[:, :, iWvl]
            image = image * 1
            self.wvlFlags = np.array(self.wvlFlags)
            # image += 2*self.wvlFlags
            image = 3 - image
            cmap = matplotlib.cm.gnuplot2
            handleMatshow = ax.matshow(image, cmap=cmap, origin='lower', vmax=2., vmin=.5)
            self.fig.colorbar(handleMatshow)
            if self.iPlot % self.nPlotsPerPage == self.nPlotsPerPage - 1:
                pdf = PdfPages(os.path.join(self.out_directory, 'temp.pdf'))
                pdf.savefig(self.fig)
                pdf.close()
                self._mergePlots()
                self.saved = True
                plt.close('all')
            self.iPlot += 1
        self._closePlots()

    def writeWeights(self):
        """
        Writes an h5 file to put calculated flat cal factors in
        """
        if not os.path.exists(self.out_directory):
            os.makedirs(self.out_directory)
        try:
            flatCalFile = tables.open_file(self.flatCalFileName, mode='w')
        except:
            getLogger(__name__).info('Error: Couldn\'t create flat cal file,{} ', self.flatCalFileName)
            return
        header = flatCalFile.create_group(flatCalFile.root, 'header', 'Calibration information')
        tables.Array(header, 'beamMap', obj=self.beamImage)
        tables.Array(header, 'xpix', obj=self.xpix)
        tables.Array(header, 'ypix', obj=self.ypix)
        calgroup = flatCalFile.create_group(flatCalFile.root, 'flatcal',
                                            'Table of flat calibration weights by pixel and wavelength')
        tables.Array(calgroup, 'weights', obj=self.flatWeights.data,
                     title='Flat calibration Weights indexed by pixelRow,pixelCol,wavelengthBin')
        tables.Array(calgroup, 'spectrum', obj=self.countCubesToSave.data,
                     title='Twilight spectrum indexed by pixelRow,pixelCol,wavelengthBin')
        tables.Array(calgroup, 'flags', obj=self.flatFlags,
                     title='Flat cal flags indexed by pixelRow,pixelCol,wavelengthBin. 0 is Good')
        tables.Array(calgroup, 'wavelengthBins', obj=self.wvlBinEdges,
                     title='Wavelength bin edges corresponding to third dimension of weights array')
        descriptionDict = FlatCalSoln_Description(self.nWvlBins)
        caltable = flatCalFile.create_table(calgroup, 'calsoln', descriptionDict, title='Flat Cal Table')
        for iRow in range(self.xpix):
            for iCol in range(self.ypix):
                weights = self.flatWeights[iRow, iCol, :]
                spectrum = self.countCubesToSave[iRow, iCol, :]
                deltaWeights = self.deltaFlatWeights[iRow, iCol, :]
                flags = self.flatFlags[iRow, iCol, :]
                flag = np.any(self.flatFlags[iRow, iCol, :])
                pixelName = self.beamImage[iRow, iCol]
                entry = caltable.row
                entry['resid'] = pixelName
                entry['pixelrow'] = iRow
                entry['pixelcol'] = iCol
                entry['weights'] = weights
                entry['weightUncertainties'] = deltaWeights
                entry['spectrum'] = spectrum
                entry['weightFlags'] = flags
                entry['flag'] = flag
                entry.append()
        flatCalFile.flush()
        flatCalFile.close()
        getLogger(__name__).info("Wrote to {}".format(self.flatCalFileName))

    def makeSummary(self):
        summaryPlot(calsolnName=self.flatCalFileName, save_plot=True)

    def _setupPlots(self):
        """
        Initialize plotting variables
        """
        self.nPlotsPerRow = 3
        self.nPlotsPerCol = 4
        self.nPlotsPerPage = self.nPlotsPerRow * self.nPlotsPerCol
        self.iPlot = 0
        self.pdfFullPath = self.out_directory + self.plotName + '.pdf'
        if os.path.isfile(self.pdfFullPath):
            answer = self._query("{0} already exists. Overwrite?".format(self.pdfFullPath), yes_or_no=True)
            if answer is False:
                answer = self._query("Provide a new file name (type exit to quit):")
                if answer == 'exit':
                    raise UserError("User doesn't want to overwrite the plot file " + "... exiting")
                self.pdfFullPath = self.out_directory + str(answer) + '.pdf'
            else:
                os.remove(self.pdfFullPath)

    def _mergePlots(self):
        """
        Merge recently created temp.pdf with the main file
        """
        temp_file = os.path.join(self.out_directory, 'temp.pdf')
        if os.path.isfile(self.pdfFullPath):
            merger = PdfFileMerger()
            merger.append(PdfFileReader(open(self.pdfFullPath, 'rb')))
            merger.append(PdfFileReader(open(temp_file, 'rb')))
            merger.write(self.pdfFullPath)
            merger.close()
            os.remove(temp_file)
        else:
            os.rename(temp_file, self.pdfFullPath)

    def _closePlots(self):
        """
        Safely close plotting variables after plotting since the last page is only saved if it is full.
        """
        if not self.saved:
            pdf = PdfPages(os.path.join(self.out_directory, 'temp.pdf'))
            pdf.savefig(self.fig)
            pdf.close()
            self._mergePlots()
        plt.close('all')

    @staticmethod
    def _query(question, yes_or_no=False, default="no"):
        """
        Ask a question via raw_input() and return their answer.
        "question" is a string that is presented to the user.
        "yes_or_no" specifies if it is a yes or no question
        "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning an answer is required of
        the user). Only used if yes_or_no=True.
        The "answer" return value is the user input for a general question. For a yes or
        no question it is True for "yes" and False for "no".
        """
        if yes_or_no:
            valid = {"yes": True, "y": True, "ye": True, "no": False, "n": False}
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

class UserError(Exception):
    """
    Custom error used to exit the flatCal program without traceback
    """
    pass

def summaryPlot(calsolnName, save_plot=False):
    """
        Writes a summary plot of the Flat Fielding
        """
    assert os.path.exists(calsolnName), "{0} does not exist".format(calsolnName)
    flat_cal = tables.open_file(calsolnName, mode='r')
    calsoln = flat_cal.root.flatcal.calsoln.read()
    weightArrPerPixel = flat_cal.root.flatcal.weights.read()
    beamImage = flat_cal.root.header.beamMap.read()
    xpix = flat_cal.root.header.xpix.read()
    ypix = flat_cal.root.header.ypix.read()
    wavelengths = flat_cal.root.flatcal.wavelengthBins.read()
    meanWeightList = np.zeros((xpix, ypix))
    meanSpecList = np.zeros((xpix, ypix))
    fig = plt.figure(figsize=(10, 10), dpi=100)
    for iRow in range(xpix):
        for iCol in range(ypix):
            res_id = beamImage[iRow][iCol]
            index = np.where(res_id == np.array(calsoln['resid']))
            weights = calsoln['weights'][index]
            weightFlags = calsoln['weightFlags'][index]
            weightUncertainties = calsoln['weightUncertainties'][index]
            spectrum = calsoln['spectrum'][index]
            weights = np.array(weights)
            weights = weights.flatten()
            meanWeight = np.nanmean(weights)
            meanWeightList[iRow, iCol] = meanWeight
            spectrum = np.array(spectrum)
            spectrum = spectrum.flatten()
            meanSpec = np.nanmean(spectrum)
            meanSpecList[iRow, iCol] = meanSpec
            weightUncertainties = np.array(weightUncertainties)
            weightUncertainties = weightUncertainties.flatten()
    weightArrPerPixel[weightArrPerPixel == 0] = np.nan
    weightArrAveraged = np.nanmean(weightArrPerPixel, axis=(0, 1))
    weightArrStd = np.nanstd(weightArrPerPixel, axis=(0, 1))
    meanSpecList[meanSpecList == 0] = np.nan
    meanWeightList[meanWeightList == 0] = np.nan
    ax = fig.add_subplot(2, 2, 1)
    ax.set_title('Mean Flat weight across the array')
    maxValue = np.nanmean(meanWeightList) + 1 * np.nanstd(meanWeightList)
    minValue = np.nanmean(meanWeightList) - 1 * np.nanstd(meanWeightList)
    plt.imshow(meanWeightList, cmap=plt.get_cmap('rainbow'), vmin=0, vmax=maxValue)
    plt.colorbar()
    ax = fig.add_subplot(2, 2, 2)
    ax.set_title('Mean Flat value across the array')
    maxValue = np.nanmean(meanSpecList) + 1 * np.nanstd(meanSpecList)
    minValue = np.nanmean(meanSpecList) - 1 * np.nanstd(meanSpecList)
    plt.imshow(meanSpecList, cmap=plt.get_cmap('rainbow'), vmin=0, vmax=maxValue)
    plt.colorbar()
    ax = fig.add_subplot(2, 2, 3)
    ax.plot(wavelengths[0:len(wavelengths) - 1], weightArrAveraged)
    ax.set_title('Mean Weight Versus Wavelength')
    ax.set_ylabel('Mean Weight')
    ax.set_xlabel(r'$\lambda$ ($\AA$)')
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(wavelengths[0:len(wavelengths) - 1], weightArrStd)
    ax.set_title('Standard Deviation of Weight Versus Wavelength')
    ax.set_ylabel('Standard Deviation')
    ax.set_xlabel(r'$\lambda$ ($\AA$)')
    if not save_plot:
        plt.show()
    else:
        pdf = PdfPages(os.path.join(os.getcwd(), 'SummaryPlot_{}.pdf'.format(timestamp)))
        pdf.savefig(fig)
        pdf.close()

if __name__ == '__main__':

    timestamp = datetime.utcnow().timestamp()

    parser = argparse.ArgumentParser(description='MKID Flat Calibration Utility')
    parser.add_argument('cfgfile', type=str, help='The config file')
    parser.add_argument('--vet', action='store_true', dest='vetonly', default=False,
                        help='Only verify config file')
    parser.add_argument('--h5', action='store_true', dest='h5only', default=False,
                        help='Only make h5 files')
    parser.add_argument('--forceh5', action='store_true', dest='forcehdf', default=False,
                        help='Force HDF creation')
    parser.add_argument('--quiet', action='store_true', dest='quiet',
                        help='Disable logging')
    args = parser.parse_args()

    if not args.quiet:
        mkidcore.corelog.create_log('flatcalib', logfile='flatcalib_{}.log'.format(timestamp), console=False, propagate=False,
                                    fmt='%(levelname)s %(message)s', level=mkidcore.corelog.INFO)

    atexit.register(lambda x:print('Execution took {:.0f}s'.format(time.time()-x)), time.time())

    args = parser.parse_args()


    flatobject=FlatCal(args.cfgfile, cal_file_name='calsol_{}.h5'.format(timestamp))

    if not flatobject.hdfexist():
        flatobject.makeh5()

    else:
        flatobject.h5directory=flatobject.h5directory+flatobject.file_name

    if args.h5only:
        exit()

    else:

        flatobject.makeCalibration()