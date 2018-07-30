#!/bin/python
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

import ast
import os
import sys
from configparser import ConfigParser
from typing import Any, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tables
from PyPDF2 import PdfFileMerger, PdfFileReader
from matplotlib.backends.backend_pdf import PdfPages
from progressbar import Bar, ETA, Percentage, ProgressBar, Timer

from mkidpipeline.calibration import wavecalplots
from mkidpipeline.core.headers import FlatCalSoln_Description
from mkidpipeline.hdf.darkObsFile import ObsFile
from mkidpipeline.utils.pipelinelog import getLogger

np.seterr(divide='ignore', invalid='ignore')


class FlatCalConfig:
    """
    Opens flat file using parameters from the param file, sets wavelength binnning parameters, and calculates flat
    weights for flat file.  Writes these weights to a h5 file and plots weights both by pixel
    and in wavelength-sliced images.
    """

    def __init__(self, config_file='default.cfg', cal_file_name = 'calsol_default.h5'):
        """
        Reads in the param file and opens appropriate flat file.  Sets wavelength binning parameters.
        """
        # define the configuration file path
        self.config_file = DEFAULT_CONFIG_FILE if config_file == 'default.cfg' else config_file
        assert os.path.isfile(self.config_file), \
            self.config_file + " is not a valid configuration file"

        # check the configuration file path and read it in
        self.config = ConfigParser()
        self.config.read(self.config_file)

        # check the configuration file format and load the parameters
        self.checksections()
        self.wvlCalFile = ast.literal_eval(self.config['Data']['wvlCalFile'])
        self.flatPath = ast.literal_eval(self.config['Data']['flatPath'])
        self.intTime = ast.literal_eval(self.config['Data']['intTime'])
        self.expTime = ast.literal_eval(self.config['Data']['expTime'])
        self.deadtime = ast.literal_eval(self.config['Instrument']['deadtime'])
        self.energyBinWidth = ast.literal_eval(self.config['Instrument']['energyBinWidth'])
        self.wvlStart = ast.literal_eval(self.config['Instrument']['wvlStart'])
        self.wvlStop = ast.literal_eval(self.config['Instrument']['wvlStop'])
        self.countRateCutoff = ast.literal_eval(self.config['Calibration']['countRateCutoff'])
        self.fractionOfChunksToTrim = ast.literal_eval(self.config['Calibration']['fractionOfChunksToTrim'])
        self.verbose = ast.literal_eval(self.config['Output']['verbose'])
        self.logging = ast.literal_eval(self.config['Output']['logging'])
        self.calSolnPath = ast.literal_eval(self.config['Output']['calSolnPath'])
        self.save_plots = ast.literal_eval(self.config['Output']['save_plots'])
        if self.save_plots:
            answer = self._query("Save Plots flag set to 'yes', this will add ~30 min to the code.  "
                                 "Are you sure you want to save plots?", yes_or_no=True)
            if answer is False:
                self.save_plots = False
                print('Setting save_plots parameter to FALSE')
        self.timeSpacingCut = None

        # check the parameter formats
        self.checktypes()

        self.obsList = [ObsFile(self.flatPath)]
        self.flatCalFileName = self.calSolnPath + 'flatcalsoln.h5'
        self.out_directory = self.calSolnPath

        # get beammap from first obs
        self.beamImage = self.obsList[0].beamImage
        self.wvlFlags = self.obsList[0].beamFlagImage
        self.nxpix = self.obsList[0].nXPix
        self.nypix = self.obsList[0].nYPix
        self.wvlBinEdges = ObsFile.makeWvlBins(self.energyBinWidth, self.wvlStart, self.wvlStop)

        # wvlBinEdges includes both lower and upper limits, so number of bins is 1 less than number of edges
        self.nWvlBins = len(self.wvlBinEdges) - 1
        if self.verbose:
            print('Computing Factors for FlatCal')
            self.pbar = ProgressBar(widgets=[Percentage(), Bar(), '  (', Timer(), ') ', ETA(), ' '],
                                    max_value=4 * len(range(0, self.expTime, self.intTime))).start()
            self.pbar_iter = 0

    def checksections(self):
        """
        Checks the variables loaded in from the configuration file for type and
        consistencey.
        """
        assert 'Data' in self.config.sections(), section.format('Data')
        assert 'flatPath' in self.config['Data'].keys(), \
            param.format('flatPath', 'Data')
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

        assert 'calSolnPath' in self.config['Output'].keys(), \
            param.format('calSolnPath', 'Output')
        assert 'verbose' in self.config['Output'].keys(), \
            param.format('verbose', 'Output')
        assert 'save_plots' in self.config['Output'].keys(), \
            param.format('save_plots', 'Output')
        assert 'logging' in self.config['Output'], \
            param.format('logging', 'Output')

    def checktypes(self):
            # type check parameters
        if self.flatPath != '':
            assert type(self.flatPath) is str, "Flat Path parameter must be a string."
            assert os.path.exists(self.flatPath), "Please confirm the Flat File path provided is correct"
        if self.calSolnPath != '':
            assert type(self.calSolnPath) is str, "Cal Solution Path parameter must be a string."
        if self.wvlCalFile != '':
            assert type(self.wvlCalFile) is str, "WaveCal Solution Path parameter must be a string."
        assert type(self.intTime) is int, "integration time parameter must be an integer"
        assert type(self.expTime) is int, "Exposure time parameter must be an integer"

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
        assert type(self.verbose) is bool, "Verbose indicator must be a bool"
        assert type(self.save_plots) is bool, "Save Plots indicator must be a bool"
        assert type(self.logging) is bool, "logging parameter must be a boolean"

class FlatCal:
    def makeCalibration(self):
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
        for iObs, obs in enumerate(self.obsList):
            for firstSec in range(0, self.expTime, self.intTime):
                cubeDict = obs.getSpectralCube(firstSec=firstSec, integrationTime=self.intTime, applySpecWeight=False,
                                               applyTPFWeight=False, wvlBinEdges=self.wvlBinEdges, energyBinWidth=None,
                                               timeSpacingCut=self.timeSpacingCut)
                cube = np.array(cubeDict['cube'], dtype=np.double)
                if self.verbose:
                    self.pbar_iter += 1
                    self.pbar.update(self.pbar_iter)

                effIntTime = cubeDict['effIntTime']
                # add third dimension for broadcasting
                effIntTime3d = np.reshape(effIntTime, np.shape(effIntTime) + (1,))
                cube /= effIntTime3d
                cube[np.isnan(cube)] = 0

                rawFrameDict = obs.getPixelCountImage(firstSec=firstSec, integrationTime=self.intTime,
                                                      scaleByEffInt=True)
                if self.verbose:
                    self.pbar_iter += 1
                    self.pbar.update(self.pbar_iter)
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
            obs.file.close()
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
        if self.verbose:
            self.pbar_iter += 1
            self.pbar.update(self.pbar_iter)

    def calculateWeights(self):
        """
        finds flat cal factors as medians/pixelSpectra for each pixel.  Normalizes these weights at each wavelength bin.
        Trim the beginning and end off the sorted weights for each wvl for each pixel, to exclude extremes from averages
        """
        self.flatWeightsList = []
        for iCube, cube in enumerate(self.spectralCubes):
            cubeWeightsList = []
            self.averageSpectra = []
            deltaWeightsList = []
            effIntTime = self.cubeEffIntTimes[iCube]
            # for each time chunk
            wvlAverages = np.zeros(self.nWvlBins)
            spectra2d = np.reshape(cube, [self.nxpix * self.nypix, self.nWvlBins])
            print('Shape Spectra2d', np.shape(spectra2d))
            for iWvl in range(self.nWvlBins):
                wvlSlice = spectra2d[:, iWvl]
                goodPixelWvlSlice = np.array(wvlSlice[wvlSlice != 0])
                # dead pixels need to be taken out before calculating averages
                wvlAverages[iWvl] = np.median(goodPixelWvlSlice)
            weights = np.divide(wvlAverages, cube)
            print('Shape Weights', np.shape(weights))
            weights[weights == 0] = np.nan
            weights[weights == np.inf] = np.nan
            cubeWeightsList.append(weights)
            deltaWeights = weights / np.sqrt(effIntTime * cube)
            deltaWeightsList.append(deltaWeights)
            self.averageSpectra.append(wvlAverages)

            cubeWeights = np.array(cubeWeightsList)
            deltaCubeWeights = np.array(deltaWeightsList)
            cubeWeightsMask = np.isnan(cubeWeights)
            self.maskedCubeWeights = np.ma.array(cubeWeights, mask=cubeWeightsMask, fill_value=1.)
            self.maskedCubeDeltaWeights = np.ma.array(deltaCubeWeights, mask=cubeWeightsMask)

            # sort maskedCubeWeights and rearange spectral cubes the same way
            sortedIndices = np.ma.argsort(self.maskedCubeWeights, axis=0)
            identityIndices = np.ma.indices(np.shape(self.maskedCubeWeights))

            sortedWeights = self.maskedCubeWeights[
                sortedIndices, identityIndices[1], identityIndices[2], identityIndices[3]]
            countCubesReordered = self.countCubes[
                sortedIndices, identityIndices[1], identityIndices[2], identityIndices[3]]
            cubeDeltaWeightsReordered = self.maskedCubeDeltaWeights[
                sortedIndices, identityIndices[1], identityIndices[2], identityIndices[3]]

            nCubes = np.shape(self.maskedCubeWeights)[0]
            trimmedWeights = sortedWeights[
                             self.fractionOfChunksToTrim * nCubes:(1 - self.fractionOfChunksToTrim) * nCubes, :, :, :]
            trimmedCountCubesReordered = countCubesReordered[self.fractionOfChunksToTrim * nCubes:(
                                                                                                          1 - self.fractionOfChunksToTrim) * nCubes,
                                         :, :, :]

            self.totalCube = np.ma.sum(trimmedCountCubesReordered, axis=0)
            self.totalFrame = np.ma.sum(self.totalCube, axis=-1)

            trimmedCubeDeltaWeightsReordered = cubeDeltaWeightsReordered[self.fractionOfChunksToTrim * nCubes:(
                                                                                                                      1 - self.fractionOfChunksToTrim) * nCubes,
                                               :, :, :]
            """
            Uncertainty in weighted average is sqrt(1/sum(averagingWeights))
            Normalize weights at each wavelength bin
            """
            self.flatWeights, summedAveragingWeights = np.ma.average(trimmedWeights, axis=0,
                                                                     weights=trimmedCubeDeltaWeightsReordered ** -2.,
                                                                     returned=True)
            self.countCubesToSave = np.ma.average(trimmedCountCubesReordered, axis=0)
            self.deltaFlatWeights = np.sqrt(summedAveragingWeights ** -1.)
            self.flatFlags = self.flatWeights.mask

            wvlWeightMedians = np.ma.median(np.reshape(self.flatWeights, (-1, self.nWvlBins)), axis=0)
            self.flatWeights = np.divide(self.flatWeights, wvlWeightMedians)
            self.flatWeightsforplot = np.ma.sum(self.flatWeights, axis=-1)
            self.indexweights = iCube
            flatcal.writeWeights()
            if self.verbose:
                self.pbar_iter += 1
                self.pbar.update(self.pbar_iter)
            if self.save_plots:
                self.indexplot = iCube
                if iCube == 0 or iCube == int((self.expTime / self.intTime) / 2) or iCube == (
                        int(self.expTime / self.intTime) - 1):
                    flatcal.plotWeightsWvlSlices()
                    flatcal.plotWeightsByPixelWvlCompare()

    def plotWeightsByPixelWvlCompare(self):
        """
        Plot weights of each wavelength bin for every single pixel
                Makes a plot of wavelength vs weights, twilight spectrum, and wavecal solution for each pixel
        """
        if not self.save_plots:
            return
        if self.save_plots:
            self.plotName = 'WavelengthCompare_'
            self._setupPlots()
        # path to your wavecal solution file
        file_nameWvlCal = self.wvlCalFile
        if self.verbose:
            print('plotting weights by pixel at ', self.pdfFullPath)
            self.pbar = ProgressBar(widgets=[Percentage(), Bar(), '  (', Timer(), ') ', ETA(), ' '],
                                    max_value=self.nxpix).start()
            self.pbar_iter = 0

        matplotlib.rcParams['font.size'] = 4
        wvls = self.wvlBinEdges[0:-1]
        nCubes = len(self.maskedCubeWeights)

        for iRow in range(self.nxpix):
            for iCol in range(self.nypix):
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
                            iRow == self.nxpix - 1 and iCol == self.nypix - 1):
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
                            iRow == self.nxpix - 1 and iCol == self.nypix - 1):
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
                        ax = wavecalplots.plotEnergySolution(file_nameWvlCal, pixel=my_pixel, axis=ax)

                    ax.set_title('p %d,%d' % (iRow, iCol))
                    if self.iPlot % self.nPlotsPerPage == self.nPlotsPerPage - 1 or (
                            iRow == self.nxpix - 1 and iCol == self.nypix - 1):
                        pdf = PdfPages(os.path.join(self.out_directory, 'temp.pdf'))
                        pdf.savefig(self.fig)
                        pdf.close()
                        self._mergePlots()
                        self.saved = True
                        plt.close('all')
                    self.iPlot += 1
            if self.verbose:
                self.pbar_iter += 1
                self.pbar.update(self.pbar_iter)

        self._closePlots()
        if self.verbose:
            self.pbar.finish()

    def plotWeightsWvlSlices(self):
        """
        Plot weights in images of a single wavelength bin (wavelength-sliced images)
        """
        self.plotName = 'WvlSlices_'
        self._setupPlots()
        matplotlib.rcParams['font.size'] = 4
        wvls = self.wvlBinEdges[0:-1]
        if self.verbose:
            print('plotting weights in wavelength sliced images')
            self.pbar = ProgressBar(widgets=[Percentage(), Bar(), '  (', Timer(), ') ', ETA(), ' '],
                                    max_value=len(wvls)).start()
            self.pbar_iter = 0

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
            if self.verbose:
                self.pbar_iter += 1

                self.pbar.update(self.pbar_iter)

        self._closePlots()
        if self.verbose:
            self.pbar.finish()

    def plotMaskWvlSlices(self):
        """
        Plot mask in images of a single wavelength bin (wavelength-sliced images)
        """
        if not self.save_plots:
            return
        if self.save_plots:
            self.plotName = 'MaskWvlSlices_'
            self._setupPlots()
        matplotlib.rcParams['font.size'] = 4
        wvls = self.wvlBinEdges[0:-1]
        if self.verbose:
            print(self.pdfFullPath)
            print('plotting mask in wavelength sliced images')
            self.pbar = ProgressBar(widgets=[Percentage(), Bar(), '  (', Timer(), ') ', ETA(), ' '],
                                    max_value=len(wvls)).start()
            self.pbar_iter = 0

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
        if self.verbose:
            self.pbar.finish()

    def writeWeights(self):
        """
        Writes an h5 file to put calculated flat cal factors in
        """
        if os.path.isabs(self.flatCalFileName) == True:
            fullFlatCalFileName = self.flatCalFileName
            baseh5path = fullFlatCalFileName.split('.h5')
            fullFlatCalFileName = baseh5path[0] + str(self.indexweights + 1) + '.h5'
        else:
            scratchDir = os.getenv('MKID_PROC_PATH')
            flatDir = os.path.join(scratchDir, 'flatCalSolnFiles')
            fullFlatCalFileName = os.path.join(flatDir, self.flatCalFileName)
            baseh5path = fullFlatCalFileName.split('.h5')
            fullFlatCalFileName = baseh5path[0] + str(self.indexweights + 1) + '.h5'

        if not os.path.exists(fullFlatCalFileName) and self.calSolnPath == '':
            os.makedirs(fullFlatCalFileName)

        try:
            flatCalFile = tables.open_file(fullFlatCalFileName, mode='w')
        except:
            print('Error: Couldn\'t create flat cal file, ', fullFlatCalFileName)
            return

        header = flatCalFile.create_group(flatCalFile.root, 'header', 'Calibration information')
        tables.Array(header, 'beamMap', obj=self.beamImage)
        tables.Array(header, 'nxpix', obj=self.nxpix)
        tables.Array(header, 'nypix', obj=self.nypix)
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

        for iRow in range(self.nxpix):
            for iCol in range(self.nypix):
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

        # close progress bar
        if self.verbose:
            self.pbar.finish()
        if self.verbose:
            print('wrote to', fullFlatCalFileName)

    def _setupPlots(self):
        """
        Initialize plotting variables
        """
        self.nPlotsPerRow = 3
        self.nPlotsPerCol = 4
        self.nPlotsPerPage = self.nPlotsPerRow * self.nPlotsPerCol
        self.iPlot = 0
        self.pdfFullPath = self.calSolnPath + self.plotName + str(self.indexplot + 1) + '.pdf'

        if os.path.isfile(self.pdfFullPath):
            answer = self._query("{0} already exists. Overwrite?".format(self.pdfFullPath), yes_or_no=True)
            if answer is False:
                answer = self._query("Provide a new file name (type exit to quit):")
                if answer == 'exit':
                    raise UserError("User doesn't want to overwrite the plot file " + "... exiting")
                self.pdfFullPath = self.calSolnPath + str(answer) + str(self.indexplot + 1) + '.pdf'
                print(self.pdfFullPath)
            else:
                os.remove(self.pdfFullPath)
                print(self.pdfFullPath)

    def _mergePlots(self):
        """
        Merge recently created temp.pdf with the main file
        """
        temp_file = os.path.join(self.calSolnPath, 'temp.pdf')
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


def plotSinglePixelSolution(calsolnName, file_nameWvlCal, res_id=None, pixel=[], save_plot=False):
    """
        Plots the weights and twilight spectrum of a single pixel (can be specified through the RES ID or pixel coordinates)
        Plots may be saved to a pdf if save_plot=True.
        Also plots the energy solution for the pixel from Wavecal

        calsolnName= File path and name of wavecal solution
        res_id= RES ID of pixel (if known)
        pixel= Coordinates of pixel (if known)
        Note:  Either RES ID or pixel coordinates must be specified
        save_plot:  Should a plot be saved?  If FALSE, the plot will be displayed.
        If TRUE, the plot will be saved to a pdf in the current working directory
        """

    assert os.path.exists(calsolnName), "{0} does not exist".format(calsolnName)
    flat_cal = tables.open_file(calsolnName, mode='r')
    calsoln = flat_cal.root.flatcal.calsoln.read()
    beamImage = flat_cal.root.header.beamMap.read()
    wavelengths = flat_cal.root.flatcal.wavelengthBins.read()

    if len(pixel) != 2 and res_id is None:
        flat_cal.close()
        raise ValueError('please supply resonator location or res_id')
    if len(pixel) == 2 and res_id is None:
        row = pixel[0]
        column = pixel[1]
        res_id = beamImage[row][column]
        index = np.where(res_id == np.array(calsoln['resid']))
    elif res_id is not None:
        index = np.where(res_id == np.array(calsoln['resid']))
        if len(index[0]) != 1:
            flat_cal.close()
            raise ValueError("res_id must exist and be unique")
        row = calsoln['pixel_row'][index][0]
        column = calsoln['pixel_col'][index][0]

    weights = calsoln['weights'][index]
    weightFlags = calsoln['weightFlags'][index]
    weightUncertainties = calsoln['weightUncertainties'][index]
    spectrum = calsoln['spectrum'][index]

    weights = np.array(weights)
    weights = weights.flatten()

    spectrum = np.array(spectrum)
    spectrum = spectrum.flatten()

    weightUncertainties = np.array(weightUncertainties)
    weightUncertainties = weightUncertainties.flatten()

    fig = plt.figure(figsize=(10, 15), dpi=100)
    ax = fig.add_subplot(3, 1, 1)
    ax.set_ylim(.5, max(weights))
    ax.plot(wavelengths[0:len(wavelengths) - 1], weights, label='weights %d' % index, alpha=.7,
            color=matplotlib.cm.Paired((1 + 1.) / 1))
    ax.errorbar(wavelengths[0:len(wavelengths) - 1], weights, yerr=weightUncertainties, label='weights', color='k')

    ax.set_title('Pixel %d,%d' % (row, column))
    ax.set_ylabel('Weight')
    ax.set_xlabel(r'$\lambda$ ($\AA$)')

    ax = fig.add_subplot(3, 1, 2)
    ax.set_ylim(.5, max(spectrum))
    ax.plot(wavelengths[0:len(wavelengths) - 1], spectrum, label='Twilight Spectrum %d' % index, alpha=.7,
            color=matplotlib.cm.Paired((1 + 1.) / 1))

    ax.set_ylabel('Twilight Spectrum')
    ax.set_xlabel(r'$\lambda$ ($\AA$)')

    ax = fig.add_subplot(3, 1, 3)
    ax.set_ylim(.5, 2.)
    my_pixel = [row, column]
    ax = wavecalplots.plotEnergySolution(file_nameWvlCal, pixel=my_pixel, axis=ax)

    if not save_plot:
        plt.show()
    else:
        pdf = PdfPages(os.path.join(os.getcwd(), str(res_id) + '.pdf'))
        pdf.savefig(fig)
        pdf.close()


def summaryPlot(calsolnName, save_plot=False):
    """
        Writes a summary plot of the Flat Fielding
        """
    assert os.path.exists(calsolnName), "{0} does not exist".format(calsolnName)
    flat_cal = tables.open_file(calsolnName, mode='r')
    calsoln = flat_cal.root.flatcal.calsoln.read()
    weightArrPerPixel = flat_cal.root.flatcal.weights.read()
    beamImage = flat_cal.root.header.beamMap.read()
    nXPix = flat_cal.root.header.nxpix.read()
    nYPix = flat_cal.root.header.nypix.read()
    wavelengths = flat_cal.root.flatcal.wavelengthBins.read()

    meanWeightList = np.zeros((nXPix, nYPix))
    meanSpecList = np.zeros((nXPix, nYPix))
    fig = plt.figure(figsize=(10, 10), dpi=100)

    for iRow in range(nXPix):
        for iCol in range(nYPix):
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
        pdf = PdfPages(os.path.join(os.getcwd(), 'SummaryPlot.pdf'))
        pdf.savefig(fig)
        pdf.close()

class UserError(Exception):
    """
    Custom error used to exit the waveCal program without traceback
    """
    pass

if __name__ == '__main__':
    pipelinelog.setup_logging()
    log = getLogger('FlatCal')
    timestamp = datetime.utcnow().timestamp()
    parser = argparse.ArgumentParser(description='MKID Flat Calibration Utility')
    parser.add_argument('cfgfile', type=str, help='The config file')

    config = FlatCalConfig(args.cfgfile, cal_file_name='calsol_{}.h5'.format(timestamp))
    FlatCal(config, filelog=flog).makeCalibration()