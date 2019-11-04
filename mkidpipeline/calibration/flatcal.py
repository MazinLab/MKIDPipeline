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



Edited by: Sarah Steiger    Date:October 31, 2019

changed the implementation of the laser flat for MEC. Instead of only applying the laser flat to wavelength calibrated
pixels it uses all of the not hot or dead but still photosensitive pixels. It is assumed that the laser light is
monochromatic and a dark frame is subtracted out of the flat calculations to exclude non monochromatic light that
might leak into MEC.
"""
import argparse
import atexit
import os
import time
from datetime import datetime
import multiprocessing as mp

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
import mkidpipeline.config
import pkg_resources as pkg
from mkidcore.utils import query


DEFAULT_CONFIG_FILE = pkg.resource_filename('mkidpipeline.calibration.flatcal', 'flatcal.yml')


# TODO need to create a calibrator factory that works with three options: wavecal, white light, and filtered or laser
#  light. In essence each needs a loadData functionand maybe a loadflatspectra in the parlance of the current
#  structure. The individual cases can be determined by seeing if the input data has a starttime or a wavesol
#  Subclasses for special functions and a factory function for deciding which to instantiate.

class FlatCalibrator(object):

    def __init__(self, config=None):
        self.config_file = DEFAULT_CONFIG_FILE if config is None else config

        self.cfg = mkidpipeline.config.load_task_config(config)

        self.dataDir = self.cfg.paths.data
        self.out_directory = self.cfg.paths.out

        self.intTime = self.cfg.flatcal.chunk_time

        self.dark_start = [self.cfg.flatcal.dark_data['start_times']]
        self.dark_int = [self.cfg.flatcal.dark_data['int_times']]

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
        self.wavelengths = None

        self.save_plots = self.cfg.flatcal.plots.lower() == 'all'
        self.summary_plot = self.cfg.flatcal.plots.lower() in ('all', 'summary')
        if self.save_plots:
            getLogger(__name__).warning("Comanded to save debug plots, this will add ~30 min to runtime.")

        self.spectralCubes = None
        self.cubeEffIntTimes = None
        self.countCubes = None
        self.cubeWeights = None
        self.weightErr = None
        self.totalCube = None
        self.totalFrame = None
        self.flatWeights = None
        self.countCubesToSave = None
        self.flatWeightErr = None
        self.flatFlags = None
        self.flatWeightsforplot = None
        self.plotName = None
        self.fig = None

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
        self.writeWeights(poly_power=self.cfg.flatcal.power)

        if self.summary_plot:
            getLogger(__name__).info('Making a summary plot')
            self.makeSummary()

        if self.save_plots:
            getLogger(__name__).info("Writing detailed plots, go get some tea.")
            getLogger(__name__).info('Plotting Weights by Wvl Slices at WeightsWvlSlices_{0}'.format(self.startTime))
            self.plotWeightsWvlSlices()
            getLogger(__name__).info(
                'Plotting Weights by Pixel against the Wavelength Solution at WeightsByPixel_{0}'.format(
                    self.startTime))
            self.plotWeightsByPixelWvlCompare()

        getLogger(__name__).info('Done')

    def makeSummary(self):
        summaryPlot(flatsol=self.flatCalFileName, save_plot=True)

    def writeWeights(self, poly_power=2):
        """
        Writes an h5 file to put calculated flat cal factors in
        """
        if not os.path.exists(self.out_directory):
            os.makedirs(self.out_directory)
        try:
            flatCalFile = tables.open_file(self.flatCalFileName, mode='w')
        except IOError:
            getLogger(__name__).error("Couldn't create flat cal file: {} ", self.flatCalFileName)
            return
        header = flatCalFile.create_group(flatCalFile.root, 'header', 'Calibration information')
        tables.Array(header, 'beamMap', obj=self.beamImage.residmap)
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
        # TODO this is a misnomer not bins but wavelengths of the calibration
        tables.Array(calgroup, 'wavelengthBins', obj=self.wavelengths,
                     title='Wavelength bin edges corresponding to third dimension of weights array')
        descriptionDict = FlatCalSoln_Description(nWvlBins=len(self.wavelengths), max_power=poly_power)
        caltable = flatCalFile.create_table(calgroup, 'calsoln', descriptionDict, title='Flat Cal Table',
                                            expectedrows=self.xpix*self.ypix)
        for iRow in range(self.xpix):
            for iCol in range(self.ypix):
                entry = caltable.row
                entry['resid'] = self.beamImage.residmap[iRow, iCol]
                entry['y'] = iRow
                entry['x'] = iCol
                entry['weight'] = self.flatWeights.data[iRow, iCol, :]
                entry['err'] = self.flatWeightErr.data[iRow, iCol, :]
                fittable = (entry['weight'] != 0) & np.isfinite(entry['weight']+entry['err'])
                if fittable.sum() < poly_power+1:
                    entry['bad'] = True
                else:
                    entry['coeff'] = np.polyfit(self.wavelengths[fittable], entry['weight'][fittable], poly_power,
                                                w=1 / entry['err'][fittable] ** 2)
                    entry['bad'] = self.flatFlags[iRow, iCol, :].any()
                entry['spectrum'] = self.countCubesToSave.data[iRow, iCol, :]
                entry.append()
        flatCalFile.close()
        getLogger(__name__).info("Wrote to {}".format(self.flatCalFileName))

    def checkCountRates(self):
        """
        masks out frames (time chunks) from the flatcal that are too bright so as to skew the results.
        If a lasercal is used for the flatcal then there is only one frame so this function is irrelevant
        and hits the pass condition.
        """
        frames = np.copy(self.spectralCubes).sum(axis=3) #want to sum over all wavelengths
        if len(frames) == 1:
            pass
        else:
            medianCountRates = np.array([np.nanmedian(frame) for frame in frames])
            mask = medianCountRates <= self.countRateCutoff
            self.spectralCubes = np.array([cube for cube, use in zip(self.spectralCubes, mask) if use])

    def calculateWeights(self):
        """
        finds flat cal factors as medians/pixelSpectra for each pixel.  Normalizes these weights at each wavelength bin.
        Trim the beginning and end off the sorted weights for each wvl for each pixel, to exclude extremes from averages
        """
        cubeWeights = np.zeros_like(self.spectralCubes)
        deltaWeights = np.zeros_like(self.spectralCubes)
        for iCube, cube in enumerate(self.spectralCubes):
            effIntTime = self.cubeEffIntTimes[iCube]
            wvlAverages = np.zeros_like(self.wavelengths)
            for iWvl in range(self.wavelengths.size):
                wvlAverages[iWvl] = np.nanmean(cube[:, :, iWvl])
            weights = cube / wvlAverages
            weights[(weights == 0) | (weights == np.inf)] = np.nan
            cubeWeights[iCube, :, :, :] = weights

            # To get uncertainty in weight:
            # Assuming negligible uncertainty in medians compared to single pixel spectra,
            # then deltaWeight=weight*deltaSpectrum/Spectrum
            # deltaWeight=weight*deltaRawCounts/RawCounts
            # with deltaRawCounts=sqrt(RawCounts)#Assuming Poisson noise
            # deltaWeight=weight/sqrt(RawCounts)
            # but 'cube' is in units cps, not raw counts so multiply by effIntTime before sqrt

            deltaWeights[iCube, :, :, :] = weights / np.sqrt(effIntTime * cube)


        weightsMask = np.isnan(cubeWeights)
        self.cubeWeights = np.ma.array(cubeWeights, mask=weightsMask, fill_value=1.).data
        nCubes = self.cubeWeights.shape[0]
        self.weightErr = np.ma.array(deltaWeights, mask=weightsMask).data

        # sort cubeWeights and rearrange spectral cubes the same way
        if self.fractionOfChunksToTrim and nCubes > 1:
            sortedIndices = np.ma.argsort(self.cubeWeights, axis=0)
            identityIndices = np.ma.indices(np.shape(self.cubeWeights))
            sortedWeights = self.cubeWeights[
                sortedIndices, identityIndices[1], identityIndices[2], identityIndices[3]]
            countCubesReordered = self.countCubes[
                sortedIndices, identityIndices[1], identityIndices[2], identityIndices[3]]
            cubeDeltaWeightsReordered = self.weightErr[
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
            self.flatWeights, summedAveragingWeights = np.ma.average(self.cubeWeights, axis=0,
                                                                     weights=self.weightErr ** -2.,
                                                                     returned=True)
            self.countCubesToSave = np.ma.sum(self.countCubes, axis=0)

        # Uncertainty in weighted average is sqrt(1/sum(averagingWeights)), normalize weights at each wavelength bin -- Is this what we should do??
        self.flatWeightErr = np.sqrt(summedAveragingWeights ** -1.)
        self.flatFlags = self.flatWeights.mask
        self.checkForColdPix() #TODO figure out what this does and fix it flag - wise
        wvlWeightMedians = np.ma.median(np.reshape(self.flatWeights, (-1, self.wavelengths.size)), axis=0)
        self.flatWeights = np.divide(self.flatWeights, wvlWeightMedians)
        self.flatWeightsforplot = np.ma.sum(self.flatWeights, axis=-1)

    def checkForColdPix(self):
        for iWvl in range(self.wavelengths.size):
            weight_slice = self.flatWeights.data[:, :, iWvl]
            # err_slice=self.flatWeightErr.data[:,:,iWvl]
            self.flatFlags[:, :, iWvl][weight_slice >= 2] = True

    def plotFitbyPixel(self, pixbox=50, poly_power=2):
        """
        Plot weights of each wavelength bin for every single pixel
        Makes a plot of wavelength vs weights, twilight spectrum, and wavecal solution for each pixel
        """
        self.plotName = 'FitByPixel_{0}'.format(self.startTime)
        self._setupPlots()
        # path to your wavecal solution file
        matplotlib.rcParams['font.size'] = 3
        wavelengths = self.wavelengths
        nCubes = len(self.cubeWeights)
        if pixbox == None:
            xrange = self.xpix
            yrange = self.ypix
        else:
            xrange = pixbox
            yrange = pixbox
        for iRow in range(xrange):
            for iCol in range(yrange):
                weights = self.flatWeights[iRow, iCol, :].data
                errors = self.flatWeightErr[iRow, iCol, :].data
                if not np.any(self.flatFlags[iRow, iCol, :]):
                    if self.iPlot % self.nPlotsPerPage == 0:
                        self.fig = plt.figure(figsize=(10, 10), dpi=100)
                    ax = self.fig.add_subplot(self.nPlotsPerCol, self.nPlotsPerRow, self.iPlot % self.nPlotsPerPage + 1)
                    ax.scatter(wavelengths, weights, label='weights', alpha=.7, color='green')
                    ax.errorbar(wavelengths, weights, yerr=errors, label='weights', color='k', fmt='o')
                    ax.set_ylim(min(weights) - 2 * np.nanstd(weights), max(weights) + 2 * np.nanstd(weights))
                    wavelengths_expanded = np.linspace(wavelengths.min(), wavelengths.max(), 1000)
                    weights_expanded = np.poly1d(np.polyfit(wavelengths, weights, poly_power, w=1 / errors ** 2))(
                        wavelengths_expanded)
                    ax.plot(wavelengths_expanded, weights_expanded)
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
                    spectrum = self.countCubesToSave[iRow, iCol, :].data
                    ax.scatter(wavelengths, spectrum, label='spectrum', alpha=.7, color='green')
                    ax.set_ylim(min(spectrum) - 2 * np.nanstd(spectrum), max(spectrum) + 2 * np.nanstd(spectrum))
                    ax.set_title('p %d,%d' % (iRow, iCol))
                    ax.set_ylabel('spectrum')
                    ax.set_xlabel(r'$\lambda$ ($\AA$)')
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
        self.plotName = 'WeightsWvlSlices_{0}'.format(self.startTime)
        self._setupPlots()
        matplotlib.rcParams['font.size'] = 4
        wavelengths = self.wavelengths
        for iWvl, wvl in enumerate(wavelengths):
            if self.iPlot % self.nPlotsPerPage == 0:
                self.fig = plt.figure(figsize=(10, 10), dpi=100)
            ax = self.fig.add_subplot(self.nPlotsPerCol, self.nPlotsPerRow, self.iPlot % self.nPlotsPerPage + 1)
            ax.set_title(r'Weights %.0f $\AA$' % wvl)
            image = self.flatWeights[:, :, iWvl]
            vmax = np.nanmean(image) + 3 * np.nanstd(image)
            plt.imshow(image.T, cmap=plt.get_cmap('viridis'), origin='lower', vmax=vmax, vmin=0)
            plt.colorbar()
            if self.iPlot % self.nPlotsPerPage == self.nPlotsPerPage - 1:
                pdf = PdfPages(os.path.join(self.out_directory, 'temp.pdf'))
                pdf.savefig(self.fig)
            self.iPlot += 1
            ax = self.fig.add_subplot(self.nPlotsPerCol, self.nPlotsPerRow, self.iPlot % self.nPlotsPerPage + 1)
            ax.set_title(r'Spectrum %.0f $\AA$' % wvl)
            image = self.totalCube[:, :, iWvl]
            vmax = np.nanmean(image) + 3 * np.nanstd(image)
            plt.imshow(image.T, cmap=plt.get_cmap('viridis'), origin='lower', vmin=0, vmax=vmax)
            plt.colorbar()
            if self.sol and iWvl == len(wavelengths) - 1:
                pdf = PdfPages(os.path.join(self.out_directory, 'temp.pdf'))
                pdf.savefig(self.fig)
                pdf.close()
                self._mergePlots()
                self.saved = True
                plt.close('all')
            else:
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
        self.nPlotsPerRow = 2
        self.nPlotsPerCol = 3
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
    Opens flat file using parameters from the param file, sets wavelength binning parameters, and calculates flat
    weights for flat file.  Writes these weights to a h5 file and plots weights both by pixel
    and in wavelength-sliced images.
    """

    def __init__(self, config=None, cal_file_name='flatsol_{start}.h5'):
        """
        Reads in the param file and opens appropriate flat file.  Sets wavelength binning parameters.
        """
        super().__init__(config)
        self.startTime = self.cfg.start_time
        self.expTime = self.cfg.exposure_time
        self.h5file = self.cfg.get('h5file', os.path.join(self.cfg.paths.out, str(self.startTime) + '.h5'))
        self.flatCalFileName = self.cfg.get('flatname', os.path.join(self.cfg.paths.database,
                                                                     cal_file_name.format(start=self.startTime)))

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
        self.wavelengths = self.wvlBinEdges[: -1] + np.diff(self.wvlBinEdges)
        self.wavelengths = self.wavelengths.flatten()
        self.sol = False

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
            cubeDict = self.obs.getSpectralCube(firstSec=firstSec, integrationTime=self.intTime, applyWeight=False,
                                                applyTPFWeight=False, wvlBinEdges=self.wvlBinEdges)
            cube = cubeDict['cube'] / cubeDict['effIntTime'][:, :, None]
            # TODO:  remove when we have deadtime removal code in pipeline
            cube /= (1 - cube.sum(axis=2) * self.deadtime)[:, :, None]
            bad = np.isnan(cube)  # TODO need to update masks to note why these 0s appeared
            cube[bad] = 0

            self.spectralCubes.append(cube)
            self.cubeEffIntTimes.append(cubeDict['effIntTime'])
            msg = 'Loaded Flat Spectra for seconds {} to {}'.format(int(firstSec), int(firstSec) + int(self.intTime))
            getLogger(__name__).info(msg)

        self.spectralCubes = np.array(self.spectralCubes[0])
        # Haven't had a chance to check if the following lines break - find me if you come across this before me - Sarah
        self.cubeEffIntTimes = np.array(self.cubeEffIntTimes)
        self.countCubes = self.cubeEffIntTimes * self.spectralCubes


class LaserCalibrator(FlatCalibrator):
    def __init__(self, h5dir='', config=None, cal_file_name='flatsol_laser.h5'):
        super().__init__(config)
        self.flatCalFileName = self.cfg.get('flatname', os.path.join(self.cfg.paths.database,
                                                                     cal_file_name))
        self.beamImage = self.cfg.beammap #not sure if this and the one below are what they should be
        # self.wvlFlags = self.cfg.beammap.flags
        self.xpix = self.cfg.beammap.ncols
        self.ypix = self.cfg.beammap.nrows
        self.wavelengths = np.array(self.cfg.wavelengths, dtype=float)
        self.exposure_times = self.cfg.exposure_times
        self.start_times = self.cfg.start_times
        self.h5_directory = h5dir
        self.h5_file_names = [os.path.join(self.h5_directory, str(t) + '.h5') for t in self.start_times]
        self.dark_h5_file_names = [os.path.join(self.h5_directory, str(t) + '.h5') for t in self.dark_start]

    def loadData(self):
        getLogger(__name__).info('No need to load wavecal solution data for a Laser Flat, passing through')
        pass

    def loadFlatSpectra(self):
        cps_cube_list, int_times, mask = self.make_spectralcube()
        dark_frame = self.get_dark_frame()
        self.spectralCubes = cps_cube_list
        self.cubeEffIntTimes = int_times
        for icube, cube in enumerate(self.spectralCubes):
            dark_subtracted_cube = np.zeros_like(cube)
            for iwvl, wvl in enumerate(cube[0, 0, :]):
                dark_subtracted_cube[:, :, iwvl] = np.subtract(cube[:, :, iwvl], dark_frame)
            masked_cube = np.ma.masked_array(dark_subtracted_cube, mask=mask).data
            masked_cube[masked_cube == 0] = np.nan
            self.spectralCubes[icube] = masked_cube
        self.spectralCubes = np.array(self.spectralCubes)
        self.cubeEffIntTimes = np.array(self.cubeEffIntTimes)
        self.countCubes = self.cubeEffIntTimes * self.spectralCubes # count cubes is the counts over the integration time

    def make_spectralcube(self):
        wavelengths = self.wavelengths
        nWavs = len(self.wavelengths)
        ntimes = int(np.max(self.exposure_times) / self.intTime)
        cps_cube_list = np.zeros([ntimes, self.xpix, self.ypix, nWavs])
        mask = np.zeros([self.xpix, self.ypix, nWavs])
        int_times = np.zeros([self.xpix, self.ypix, nWavs])
        for iwvl, wvl in enumerate(wavelengths):
            time = int(self.exposure_times[iwvl] / self.intTime)
            cps_cube = np.zeros([self.xpix, self.ypix, time])
            obs = ObsFile(self.h5_file_names[iwvl])
            # mask out hot pixels before finding the mean
            hot_mask = obs.flagMask(['pixcal.hot', 'pixcal.cold'])
            mask[:, :, iwvl] = hot_mask
            counts = obs.getTemporalCube(timeslice=self.intTime)
            getLogger(__name__).info('Loaded {}nm spectral cube'.format(wvl))
            cps_cube = counts['cube']/self.intTime #TODO move this division outside of the loop
            cps_cube = np.moveaxis(cps_cube, 2, 0)
            int_times[:, :, iwvl] = self.intTime
            cps_cube_list[:time, :, :, iwvl] = cps_cube
        return cps_cube_list, int_times, mask

    def get_dark_frame(self):
        '''
        takes however many dark files that are specified in the pipe.yml and computes the counts/pixel/sec for the sum
        of all the dark obs. This creates a stictched together long dark obs from all of the smaller obs given. This
        is useful for legacy data where there may not be a specified dark observation but parts of observations where
        the filter wheel was closed.

        :return: expected dark counts for each pixel over a flat observation
        '''
        frames = np.zeros((140, 146, len(self.dark_start)))
        getLogger(__name__).info('Loading dark frames for Laser flat')
        for i, file in enumerate(self.dark_h5_file_names):
            obs = ObsFile(file)
            frame = obs.getPixelCountImage(integrationTime=self.dark_int[i])['image']
            frames[:, :, i] = frame
        total_counts = np.sum(frames, axis=2)
        counts_per_sec = total_counts / np.sum(self.dark_int)
        dark_frame = counts_per_sec
        return dark_frame



def summaryPlot(flatsol, save_plot=False):
    """ Writes a summary plot of the Flat Fielding """
    flat_cal = tables.open_file(flatsol, mode='r')
    calsoln = flat_cal.root.flatcal.calsoln.read()
    weightArrPerPixel = flat_cal.root.flatcal.weights.read()
    beamImage = flat_cal.root.header.beamMap.read()
    xpix = flat_cal.root.header.xpix.read()
    ypix = flat_cal.root.header.ypix.read()
    wavelengths = flat_cal.root.flatcal.wavelengthBins.read()
    flat_cal.close()

    meanWeightList = np.zeros((xpix, ypix))
    meanSpecList = np.zeros((xpix, ypix))

    for (iRow, iCol), res_id in np.ndenumerate(beamImage):
        use = res_id == np.asarray(calsoln['resid'])
        weights = calsoln['weight'][use]
        spectrum = calsoln['spectrum'][use]
        meanWeightList[iRow, iCol] = np.nanmean(weights)
        meanSpecList[iRow, iCol] = np.nanmean(spectrum)

    weightArrPerPixel[weightArrPerPixel == 0] = np.nan
    weightArrAveraged = np.nanmean(weightArrPerPixel, axis=(0, 1))
    weightArrStd = np.nanstd(weightArrPerPixel, axis=(0, 1))
    meanSpecList[meanSpecList == 0] = np.nan
    meanWeightList[meanWeightList == 0] = np.nan

    class Dummy(object):
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def savefig(self):
            pass

    with PdfPages(flatsol.split('.h5')[0] + '_summary.pdf') if save_plot else Dummy() as pdf:

        fig = plt.figure(figsize=(10, 10))

        ax = fig.add_subplot(2, 2, 1)
        ax.set_title('Mean Flat weight across the array')
        maxValue = np.nanmean(meanWeightList) + 1 * np.nanstd(meanWeightList)
        plt.imshow(meanWeightList.T, cmap=plt.get_cmap('viridis'), vmin=0, vmax=maxValue)
        plt.colorbar()

        ax = fig.add_subplot(2, 2, 2)
        ax.set_title('Mean Flat value across the array')
        maxValue = np.nanmean(meanSpecList) + 1 * np.nanstd(meanSpecList)
        plt.imshow(meanSpecList.T, cmap=plt.get_cmap('viridis'), vmin=0, vmax=maxValue)
        plt.colorbar()

        ax = fig.add_subplot(2, 2, 3)
        ax.scatter(wavelengths, weightArrAveraged)
        ax.set_title('Mean Weight Versus Wavelength')
        ax.set_ylabel('Mean Weight')
        ax.set_xlabel(r'$\lambda$ ($\AA$)')

        ax = fig.add_subplot(2, 2, 4)
        ax.scatter(wavelengths, weightArrStd)
        ax.set_title('Standard Deviation of Weight Versus Wavelength')
        ax.set_ylabel('Standard Deviation')
        ax.set_xlabel(r'$\lambda$ ($\AA$)')
        pdf.savefig(fig)

    if not save_plot:
        plt.show()


def plotCalibrations(flatsol, wvlCalFile, pixel):
    """
    Plot weights of each wavelength bin for every single pixel
    Makes a plot of wavelength vs weights, twilight spectrum, and wavecal solution for each pixel
    """
    wavesol = wavecal.load_solution(wvlCalFile)
    assert os.path.exists(flatsol), "{0} does not exist".format(flatsol)
    flat_cal = tables.open_file(flatsol, mode='r')
    calsoln = flat_cal.root.flatcal.calsoln.read()
    wavelengths = flat_cal.root.flatcal.wavelengthBins.read()
    beamImage = flat_cal.root.header.beamMap.read()
    matplotlib.rcParams['font.size'] = 10
    res_id = beamImage[pixel[0], pixel[1]]
    index = np.where(res_id == np.array(calsoln['resid']))
    weights = calsoln['weight'][index].flatten()
    spectrum = calsoln['spectrum'][index].flatten()
    errors = calsoln['err'][index].flatten()
    if not calsoln['bad'][index]:
        fig = plt.figure(figsize=(20, 10), dpi=100)
        ax = fig.add_subplot(1, 3, 1)
        ax.scatter(wavelengths, weights, label='weights', alpha=.7, color='red')
        ax.errorbar(wavelengths, weights, yerr=errors, label='weights', color='green', fmt='o')
        ax.set_title('p %d,%d' % (pixel[0], pixel[1]))
        ax.set_ylabel('weight')
        ax.set_xlabel(r'$\lambda$ ($\AA$)')
        ax.set_ylim(min(weights) - 2 * np.nanstd(weights),
                    max(weights) + 2 * np.nanstd(weights))
        plt.plot(wavelengths, np.poly1d(calsoln[index]['coeff'][0])(wavelengths))
        # Put a plot of twilight spectrums for this pixel
        ax = fig.add_subplot(1, 3, 2)
        ax.scatter(wavelengths, spectrum, label='spectrum', alpha=.7, color='blue')
        ax.set_title('p %d,%d' % (pixel[0], pixel[1]))
        ax.set_ylim(min(spectrum) - 2 * np.nanstd(spectrum),
                    max(spectrum) + 2 * np.nanstd(spectrum))
        ax.set_ylabel('spectrum')
        ax.set_xlabel(r'$\lambda$ ($\AA$)')
        # Plot wavecal solution
        ax = fig.add_subplot(1, 3, 3)
        my_pixel = [pixel[0], pixel[1]]
        wavesol.plot_calibration(pixel=my_pixel, axes=ax)
        ax.set_title('p %d,%d' % (pixel[0], pixel[1]))
        plt.show()
    else:
        print('Pixel Failed Wavecal')


def _run(flattner):
    getLogger(__name__).debug('Calling makeCalibration on {}'.format(flattner))
    flattner.makeCalibration()


def fetch(solution_descriptors, config=None, ncpu=np.inf, remake=False):
    cfg = mkidpipeline.config.config if config is None else config
    solutions = []
    flattners = []
    for sd in solution_descriptors:
        sf = os.path.join(cfg.paths.database, sd.id)
        if not os.path.exists(sf) or remake:
            if 'flatcal' not in cfg:
                fcfg = mkidpipeline.config.load_task_config(pkg.resource_filename(__name__, 'flatcal.yml'))
            else:
                fcfg = cfg.copy()

            fcfg.unregister('flatname')
            fcfg.unregister('h5file')

            if fcfg.flatcal.method == 'Laser':
                fcfg.register('start_times', np.array([x.start for x in sd.wavecal.data]), update=True)
                fcfg.register('exposure_times', np.array([x.duration for x in sd.wavecal.data]), update=True)
                fcfg.register('wavelengths', np.array([w for w in sd.wavecal.wavelengths]), update=True)
                flattner = LaserCalibrator(config=fcfg, h5dir=cfg.paths.out, cal_file_name=sd.id)
            elif fcfg.flatcal.method == 'White':
                fcfg.register('start_time', sd.ob.start, update=True)
                fcfg.register('exposure_time', sd.ob.duration, update=True)
                fcfg.unregister('wavesol')
                flattner = WhiteCalibrator(config=fcfg, cal_file_name=sd.id)
            else:
                raise TypeError(str(fcfg.flatcal.method) + ' is an invalid method keyword argument')


            solutions.append(sf)
            flattners.append(flattner)

    if not flattners:
        return solutions

    ncpu = mkidpipeline.config.n_cpus_available(max=min(fcfg.ncpu, ncpu))
    if ncpu == 1 or len(flattners) == 1:
        for f in flattners:
            f.makeCalibration()
    else:
        pool = mp.Pool(ncpu)
        pool.map(_run, flattners)
        pool.close()
        pool.join()

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
