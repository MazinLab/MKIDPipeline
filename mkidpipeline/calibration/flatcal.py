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
import atexit
import os
import time
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tables
from PyPDF2 import PdfFileMerger, PdfFileReader

from matplotlib.backends.backend_pdf import PdfPages
from mkidpipeline.calibration import wavecal
from mkidcore.headers import FlatCalSoln_Description
from mkidpipeline.hdf.photontable import ObsFile
from mkidcore.corelog import getLogger
import mkidcore.corelog
from mkidpipeline.hdf import bin2hdf
import mkidpipeline.config
import pkg_resources as pkg
from mkidcore.utils import query

DEFAULT_CONFIG_FILE = pkg.resource_filename('mkidpipeline.calibration.flatcal', 'flatcal.yml')


# TODO need to create a calibrator factory that works with three options: wavecal, white light, and filtered or laser
#  light. In essence each needs a loadData functionand maybe a loadflatspectra in the parlance of the current
#  structure. The individual cases can be determined by seeing if the input data has a starttime or a wavesol
#  Subclasses for special functions and a factory function for deciding which to instantiate.

class FlatCalibrator(object):

    def makeCalibration(self):
        getLogger(__name__).info("Loading Data")
        self.loadData()
        getLogger(__name__).info("Loading flat spectra")
        self.loadFlatSpectra()
        getLogger(__name__).info("Checking count rates")
        self.checkCountRates()
        getLogger(__name__).info("Calculating weights")
        self.calculateWeights()
        getLogger(__name__).info("Writing weights")
        self.writeWeights()

        if self.summary_plot:
            getLogger(__name__).info('Making a summary plot')
            self.makeSummary()

        if self.save_plots:
            getLogger(__name__).info("Writing detailed plots, go get some tea.")
            getLogger(__name__).info('Plotting Weights by Wvl Slices at WvlSlices_{}'.format(timestamp))
            self.plotWeightsWvlSlices()
            getLogger(__name__).info(
                'Plotting Weights by Pixel against the Wavelength Solution at WavelengthCompare_{}'.format(timestamp))
            self.plotWeightsByPixelWvlCompare()

        getLogger(__name__).info('Done')

    def makeSummary(self):
        summaryPlot(calsolnName=self.flatCalFileName, save_plot=True)

    def writeWeights(self):
        """
        Writes an h5 file to put calculated flat cal factors in
        """
        if not os.path.exists(self.out_directory):
            os.makedirs(self.out_directory)
        try:
            flatCalFile = tables.open_file(self.flatCalFileName, mode='w')
        except:
            getLogger(__name__).error("Couldn't create flat cal file: {} ", self.flatCalFileName)
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
        descriptionDict = FlatCalSoln_Description(self.wvlBinSize)
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

    @property
    def frames(self):
        return self.spectralCubes.sum(axis=2)

    def checkCountRates(self):
        """ mask out frames, or cubes from integration time chunks with count rates too high """
        medianCountRates = np.array([np.median(frame[frame != 0]) for frame in self.frames])
        mask = medianCountRates <= self.countRateCutoff
        self.spectralCubes = np.array([cube for cube, use in zip(self.spectralCubes, mask) if use])

    def calculateWeights(self):
        """
        finds flat cal factors as medians/pixelSpectra for each pixel.  Normalizes these weights at each wavelength bin.
        Trim the beginning and end off the sorted weights for each wvl for each pixel, to exclude extremes from averages
        """
        cubeWeightsList = []
        deltaWeightsList = []
        for iCube, cube in enumerate(self.spectralCubes):
            effIntTime = self.cubeEffIntTimes[iCube]
            # for each time chunk
            wvlAverages = np.zeros(self.wvlBinSize)
            spectra2d = np.reshape(cube, [self.xpix * self.ypix, self.wvlBinSize])
            for iWvl in range(self.wvlBinSize):
                wvlSlice = spectra2d[:, iWvl]
                goodPixelWvlSlice = np.array(wvlSlice[wvlSlice != 0])
                # dead pixels need to be taken out before calculating averages
                wvlAverages[iWvl] = np.median(goodPixelWvlSlice)
            weights = wvlAverages / cube
            weights[(weights == 0) | (weights == np.inf)] = np.nan
            cubeWeightsList.append(weights)

            # To get uncertainty in weight:
            # Assuming negligible uncertainty in medians compared to single pixel spectra,
            # then deltaWeight=weight*deltaSpectrum/Spectrum
            # deltaWeight=weight*deltaRawCounts/RawCounts
            # with deltaRawCounts=sqrt(RawCounts)#Assuming Poisson noise
            # deltaWeight=weight/sqrt(RawCounts)
            # but 'cube' is in units cps, not raw counts so multiply by effIntTime before sqrt

            deltaWeights = weights / np.sqrt(effIntTime * cube)
            deltaWeightsList.append(deltaWeights)

        cubeWeights = np.array(cubeWeightsList)
        deltaCubeWeights = np.array(deltaWeightsList)
        cubeWeightsMask = np.isnan(cubeWeights)
        self.maskedCubeWeights = np.ma.array(cubeWeights, mask=cubeWeightsMask, fill_value=1.)
        nCubes = self.maskedCubeWeights.shape[0]
        self.maskedCubeDeltaWeights = np.ma.array(deltaCubeWeights, mask=cubeWeightsMask)

        # sort maskedCubeWeights and rearrange spectral cubes the same way
        if self.fractionOfChunksToTrim and nCubes > 1:
            sortedIndices = np.ma.argsort(self.maskedCubeWeights, axis=0)
            identityIndices = np.ma.indices(np.shape(self.maskedCubeWeights))
            sortedWeights = self.maskedCubeWeights[
                sortedIndices, identityIndices[1], identityIndices[2], identityIndices[3]]
            countCubesReordered = self.countCubes[
                sortedIndices, identityIndices[1], identityIndices[2], identityIndices[3]]
            cubeDeltaWeightsReordered = self.maskedCubeDeltaWeights[
                sortedIndices, identityIndices[1], identityIndices[2], identityIndices[3]]
            sl = slice(self.fractionOfChunksToTrim * nCubes, (1 - self.fractionOfChunksToTrim) * nCubes)
            trimmedWeights = sortedWeights[sl, :, :, :]
            trimmedCountCubesReordered = countCubesReordered[sl, :, :, :]
            trimmedCubeDeltaWeightsReordered = cubeDeltaWeightsReordered[sl, :, :, :]
            self.totalCube = np.ma.sum(trimmedCountCubesReordered, axis=0)
            self.totalFrame = np.ma.sum(self.totalCube, axis=-1)
            self.flatWeights, summedAveragingWeights = np.ma.average(trimmedWeights, axis=0,
                                                                     weights=trimmedCubeDeltaWeightsReordered ** -2.,
                                                                     returned=True)
            self.countCubesToSave = np.ma.sum(trimmedCountCubesReordered, axis=0)
        else:
            self.totalCube = np.ma.sum(self.countCubes, axis=0)
            self.totalFrame = np.ma.sum(self.totalCube, axis=-1)
            self.flatWeights, summedAveragingWeights = np.ma.average(self.maskedCubeWeights, axis=0,
                                                                     weights=self.maskedCubeDeltaWeights ** -2.,
                                                                     returned=True)
            self.countCubesToSave = np.ma.sum(self.countCubes, axis=0)

        """
        Uncertainty in weighted average is sqrt(1/sum(averagingWeights))
        Normalize weights at each wavelength bin
        """

        self.deltaFlatWeights = np.sqrt(summedAveragingWeights ** -1.)
        self.flatFlags = self.flatWeights.mask
        wvlWeightMedians = np.ma.median(np.reshape(self.flatWeights, (-1, self.wvlBinSize)), axis=0)
        self.flatWeights = np.divide(self.flatWeights, wvlWeightMedians)
        self.flatWeightsforplot = np.ma.sum(self.flatWeights, axis=-1)

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
        wavesol = wavecal.Solution(self.wvlCalFile)
        matplotlib.rcParams['font.size'] = 4
        wvls = self.wvlBinEdges[0:self.wvlBinSize]
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
        self.plotName = 'WvlSlices_{}'.format(timestamp)  # TODO @Isabel this is a bug
        self._setupPlots()
        matplotlib.rcParams['font.size'] = 4
        wvls = self.wvlBinEdges[0:self.wvlBinSize]
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
        wvls = self.wvlBinEdges[0:self.wvlBinSize]
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
            answer = query("{0} already exists. Overwrite?".format(self.pdfFullPath), yes_or_no=True)
            if answer is False:
                answer = query("Provide a new file name (type exit to quit):")
                if answer == 'exit':
                    raise RuntimeError("User doesn't want to overwrite the plot file " + "... exiting")
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


class WhiteCalibrator(FlatCalibrator):
    """
    Opens flat file using parameters from the param file, sets wavelength binnning parameters, and calculates flat
    weights for flat file.  Writes these weights to a h5 file and plots weights both by pixel
    and in wavelength-sliced images.
    """

    def __init__(self, config=None, cal_file_name='flatsol_{start}.h5', log=True):
        """
        Reads in the param file and opens appropriate flat file.  Sets wavelength binning parameters.
        """

        super().__init__()

        self.config_file = DEFAULT_CONFIG_FILE if config is None else config

        self.cfg = mkidpipeline.config.load_task_config(config)

        self.wvlCalFile = ''
        try:
            self.wvlCalFile = self.cfg.wavesol
            if not os.path.exists(self.wvlCalFile):
                self.wvlCalFile = os.path.join(self.cfg.paths.database, self.wvlCalFile)
        except KeyError:
            pass

        self.startTime = self.cfg.start_time
        self.expTime = self.cfg.exposure_time

        self.h5file = self.cfg.get('h5file', os.path.join(self.cfg.paths.out, str(self.startTime) + '.h5'))

        self.dataDir = self.cfg.paths.data
        self.out_directory = self.cfg.paths.out

        self.flatCalFileName = self.cfg.get('flatname', os.path.join(self.cfg.paths.database,
                                                                     cal_file_name.format(start=self.startTime)))

        self.intTime = self.cfg.flatcal.chunk_time

        self.xpix = self.cfg.beammap.ncols
        self.ypix = self.cfg.beammap.nrows
        self.deadtime = self.cfg.instrument.deadtime

        self.energyBinWidth = self.cfg.instrument.energy_bin_width
        self.wvlStart = self.cfg.instrument.wvl_start
        self.wvlStop = self.cfg.instrument.wvl_stop

        self.countRateCutoff = self.cfg.flatcal.rate_cutoff
        self.fractionOfChunksToTrim = self.cfg.flatcal.trim_fraction
        self.timeSpacingCut = None

        self.obs = None
        self.beamImage = None
        self.wvlFlags = None
        self.wvlBinEdges = None
        self.wvlBinSize = None

        self.logging = log
        self.save_plots = self.cfg.flatcal.plots.lower() == 'all'
        self.summary_plot = self.cfg.flatcal.plots.lower() in ('all', 'summary')
        if self.save_plots:
            getLogger(__name__).warning("Comanded to save debug plots, this will add ~30 min to runtime.")

        self.spectralCubes = None
        self.cubeEffIntTimes = None
        self.countCubes = None
        self.maskedCubeWeights = None
        self.maskedCubeDeltaWeights = None
        self.totalCube = None
        self.totalFrame = None
        self.flatWeights = None
        self.countCubesToSave = None
        self.deltaFlatWeights = None
        self.flatFlags = None
        self.flatWeights = None
        self.flatWeightsforplot = None
        self.plotName = None
        self.fig = None  # TODO @Isabel lets talk about if this is really needed as a class attribute

    def loadData(self):
        getLogger(__name__).info('Loading calibration data from {}'.format(self.h5file))
        self.obs = ObsFile(self.h5file)
        if not self.obs.wavelength_calibrated:
            raise RuntimeError('Photon data is not wavelength calibrated.')
        self.beamImage = self.obs.beamImage
        self.wvlFlags = self.obs.beamFlagImage
        self.xpix = self.obs.nXPix
        self.ypix = self.obs.nYPix
        self.wvlBinEdges = ObsFile.makeWvlBins(self.energyBinWidth, self.wvlStart, self.wvlStop)
        self.wvlBinSize = self.wvlBinEdges.size - 1

    def loadFlatSpectra(self):
        """
        Reads the flat data into a spectral cube whose dimensions are determined
        by the number of x and y pixels and the number of wavelength bins.
        Each element will be the spectral cube for a time chunk
        Find factors to correct nonlinearity due to deadtime in firmware

        To be used for whitelight flat data
        """

        self.spectralCubes = []
        self.cubeEffIntTimes = []
        for firstSec in range(0, self.expTime, self.intTime):  # for each time chunk
            cubeDict = self.obs.getSpectralCube(firstSec=firstSec, integrationTime=self.intTime, applySpecWeight=False,
                                                applyTPFWeight=False, wvlBinEdges=self.wvlBinEdges)
            cube = cubeDict['cube'] / cubeDict['effIntTime'][:, :, None]
            cube /= (1 - cube.sum(axis=2) * self.deadtime)[:, :, None]
            bad = np.isnan(cube)  # TODO need to update maskes to note why these 0s appeared
            cube[bad] = 0

            self.spectralCubes.append(cube)
            self.cubeEffIntTimes.append(cubeDict['effIntTime'])
            msg = 'Loaded Flat Spectra for seconds {} to {}'.format(int(firstSec), int(firstSec) + int(self.intTime))
            getLogger(__name__).info(msg)

        self.spectralCubes = np.array(self.spectralCubes)
        self.cubeEffIntTimes = np.array(self.cubeEffIntTimes)
        self.countCubes = self.cubeEffIntTimes[:, :, :, None] * self.spectralCubes


class LaserCalibrator(WhiteCalibrator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def loadData(self):
        getLogger(__name__).info('Loading calibration data from {}'.format(self.wvlCalFile))
        self.sol = wavecal.load_solution(self.wvlCalFile)
        self.beamImage = self.sol.beam_map
        self.wvlFlags = self.sol.beam_map_flags
        self.xpix = self.sol.cfg.x_pixels
        self.ypix = self.sol.cfg.y_pixels
        self.wave_list = self.sol.cfg.wavelengths
        self.wave_inttime_list = self.sol.cfg.exposure_times
        self.laserBinEdges = self.wvl_medians
        self.wvlBinEdges = ObsFile.makeWvlBins(self.energyBinWidth, self.wvlStart, self.wvlStop)
        self.wvlBinSize = self.wvlBinEdges.size - 1

    @property
    def wvl_medians(self):
        wvl_medians = []
        wvl_medians.append(self.wvlStart)
        for index, wavelength in enumerate(self.wave_list[0:-1]):
            wvl_medians.append(self.wave_list[index] + ((self.wave_list[index + 1] - self.wave_list[index]) / 2))
        wvl_medians.append(self.wvlStop)
        return np.array(wvl_medians)

    def loadFlatSpectra(self):
        self.spectralCubes = []
        self.cubeEffIntTimes = []
        cube, eff_time = self.make_spectralcube_from_wavecal()
        cube /= eff_time
        cube[np.isnan(cube)] = 0
        self.spectralCubes.append(cube)
        self.cubeEffIntTimes.append(eff_time)
        self.spectralCubes = np.array(self.spectralCubes)
        self.cubeEffIntTimes = np.array(self.cubeEffIntTimes)
        self.countCubes = self.cubeEffIntTimes * self.spectralCubes

    def make_spectralcube_from_wavecal(self):
        wave_list = self.wave_list
        nWavs = len(self.wave_list)
        spectralcube_wave = np.zeros([self.xpix, self.ypix, nWavs])
        spectralcube_wave[:, :, :] = np.nan
        eff_int_time_3d_wave = np.zeros([self.xpix, self.ypix, nWavs])
        gaussian_names = ['GaussianAndExponential', 'GaussianAndGaussian',
                          'GaussianAndGaussianExponential',
                          'SkewedGaussianAndGaussianExponential']
        for nRow in range(self.xpix):
            for nCol in range(self.ypix):
                good_cal = self.sol.has_good_calibration_solution(pixel=[nRow, nCol])
                if good_cal:
                    params = self.sol.histogram_parameters(pixel=[nRow, nCol])
                    model_names = self.sol.histogram_model_names(pixel=[nRow, nCol])
                    good_sol = self.sol.has_good_histogram_solutions(pixel=[nRow, nCol])
                    for iwvl, wvl in enumerate(wave_list):
                        if good_sol[iwvl] and model_names[iwvl] in gaussian_names:
                            sigma = params[iwvl]['signal_sigma'].value
                            amp_scaled = params[iwvl]['signal_amplitude'].value
                            wvl_intensity = np.sqrt(sigma) * amp_scaled
                            spectralcube_wave[nRow, nCol, iwvl] = wvl_intensity
                            eff_int_time_3d_wave[nRow, nCol, iwvl] = self.wave_inttime_list[iwvl]

        spectralcube = np.zeros([self.xpix, self.ypix, self.wvlBinSize])
        eff_int_time_3d = np.zeros([self.xpix, self.ypix, self.wvlBinSize])
        for j in range(self.laserBinEdges.size):
            for i in range(self.wvlBinSize):
                if self.wvlBinEdges[i] >= self.laserBinEdges[j] and self.wvlBinEdges[i] < self.laserBinEdges[j + 1]:
                    spectralcube[:, :, i] = spectralcube_wave[:, :, j]
                    eff_int_time_3d[:, :, i] = eff_int_time_3d_wave[:, :, j]
        return spectralcube, eff_int_time_3d


def summaryPlot(calsolnName, save_plot=False):
    """ Writes a summary plot of the Flat Fielding """
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
            spectrum = calsoln['spectrum'][index]
            weights = np.array(weights)
            weights = weights.flatten()
            meanWeight = np.nanmean(weights)
            meanWeightList[iRow, iCol] = meanWeight
            spectrum = np.array(spectrum)
            spectrum = spectrum.flatten()
            meanSpec = np.nanmean(spectrum)
            meanSpecList[iRow, iCol] = meanSpec
    weightArrPerPixel[weightArrPerPixel == 0] = np.nan
    weightArrAveraged = np.nanmean(weightArrPerPixel, axis=(0, 1))
    weightArrStd = np.nanstd(weightArrPerPixel, axis=(0, 1))
    meanSpecList[meanSpecList == 0] = np.nan
    meanWeightList[meanWeightList == 0] = np.nan
    ax = fig.add_subplot(2, 2, 1)
    ax.set_title('Mean Flat weight across the array')
    maxValue = np.nanmean(meanWeightList) + 1 * np.nanstd(meanWeightList)
    plt.imshow(meanWeightList, cmap=plt.get_cmap('rainbow'), vmin=0, vmax=maxValue)
    plt.colorbar()
    ax = fig.add_subplot(2, 2, 2)
    ax.set_title('Mean Flat value across the array')
    maxValue = np.nanmean(meanSpecList) + 1 * np.nanstd(meanSpecList)
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
        pdf = PdfPages(calsolnName+'.pdf')
        pdf.savefig(fig)
        pdf.close()


def fetch(solution_descriptors, config=None, ncpu=1, async=False, remake=False):
    cfg = mkidpipeline.config.config if config is None else config

    solutions = []
    for sd in solution_descriptors:
        sf = os.path.join(cfg.paths.database, sd.id)
        if not os.path.exists(sf) or remake:
            if 'flatcal' not in cfg:
                fcfg = mkidpipeline.config.load_task_config(pkg.resource_filename(__name__, 'flatcal.yml'))
            else:
                fcfg = cfg.copy()

            if hasattr(sd, 'wavecal'):
                fcfg.register('wavesol', sd.wavecal, update=True)
                fcfg.register('start_time', -1, update=True) #TODO get rid of this when lasercalibrator's init is fixed
                fcfg.register('exposure_time',-1, update=True)
                fcfg.unregister('flatname')
                fcfg.unregister('h5file')
                flattner = LaserCalibrator(fcfg, cal_file_name=sd.id)
            else:
                fcfg.register('start_time', sd.ob.start, update=True)
                fcfg.register('exposure_time', sd.ob.duration, update=True)
                fcfg.unregister('flatname')
                fcfg.unregister('wavesol')
                fcfg.unregister('h5file')
                flattner = WhiteCalibrator(fcfg, cal_file_name=sd.id)
            flattner.makeCalibration()
        solutions.append(sf)

    return solutions

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
        mkidcore.corelog.create_log('__main__', logfile='flatcalib_{}.log'.format(timestamp), console=True,
                                    propagate=False, fmt='Flatcal: %(levelname)s %(message)s',
                                    level=mkidcore.corelog.INFO)

    atexit.register(lambda x: print('Execution took {:.0f}s'.format(time.time() - x)), time.time())

    args = parser.parse_args()

    flattner = WhiteCalibrator(args.cfgfile, cal_file_name='calsol_{}.h5'.format(timestamp))

    if not os.path.isfile(flattner.cfg.h5file):
        raise RuntimeError('Not up to date')
        b2h_config = bin2hdf.Bin2HdfConfig(datadir=flattner.cfg.paths.data, beamfile=flattner.cfg.beammap.file,
                                           outdir=flattner.paths.out, starttime=flattner.cfg.start_time,
                                           inttime=flattner.cfg.expTime,
                                           x=flattner.cfg.beammap.ncols, y=flattner.cfg.beammap.ncols)
        bin2hdf.makehdf(b2h_config, maxprocs=1)
        getLogger(__name__).info('Made h5 file at {}.h5'.format(flattner.cfg.start_time))

    obsfile = ObsFile(flattner.h5file, mode='write')
    if not obsfile.wavelength_calibrated:
        obsfile.applyWaveCal(wavecal.load_solution(flattner.cfg.wavesol))
        getLogger(__name__).info('Applied Wavecal {} to {}.h5'.format(flattner.cfg.wavesol, flattner.h5file))

    if args.h5only:
        exit()

    flattner.makeCalibration()
