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



Edited by: Sarah Steiger    Date: October 31, 2019
"""
import os
import multiprocessing as mp
import scipy.constants as c

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tables
from PyPDF2 import PdfFileMerger, PdfFileReader


from matplotlib.backends.backend_pdf import PdfPages
from mkidpipeline.steps import wavecal
from photontable import Photontable
from mkidcore.corelog import getLogger
import mkidpipeline.config
from mkidpipeline.config import H5Subset
from mkidcore.utils import query
import mkidcore.pixelflags as pixelflags


class StepConfig(mkidpipeline.config.BaseStepConfig):
    yaml_tag = u'!flatcal_cfg'
    REQUIRED_KEYS = (('rate_cutoff',  0, 'Count Rate Cutoff in inverse seconds (number)'),
                     ('trim_chunks',  1, 'number of Chunks to trim (integer)'),
                     ('chunk_time', 10, 'duration of chunks used for weights (s)'),
                     ('nchunks', 6, 'number of chunks to median combine'),
                     ('power', 1, 'power of polynomial to fit, <3 advised'),
                     ('power',  0, 'TODO'),
                     ('plots', 'summary', 'none|summary|all'))
    OPTIONAL_KEYS = tuple()

    def _vet_errors(self):
        ret = []
        try:
            assert 0 <= self.rate_cutoff <= 20000
        except:
            ret.append('rate_cutoff must be [0, 20000]')

        try:
            assert 0 <= self.trim_chunks <= 1
        except:
            ret.append(f'trim_chunks must be a float in [0,1]: {type(self.trim_chunks)}')

        return ret

# TODO need to create a calibrator factory that works with three options: wavecal, white light, and filtered or laser
#  light. In essence each needs a load_data functionand maybe a load_flat_spectra in the parlance of the current
#  structure. The individual cases can be determined by seeing if the input data has a starttime or a wavesol
#  Subclasses for special functions and a factory function for deciding which to instantiate.


class FlatCalibrator:
    def __init__(self, config=None, solution_name='flat_solution.npz'):
        self.cfg = mkidpipeline.config.load_task_config(StepConfig() if config is None else config)  #TODO
        self.data_dir = self.cfg.paths.data
        self.out_directory = self.cfg.paths.out

        self.chunk_time = self.cfg.flatcal.chunk_time

        self.dark_start = []
        self.dark_int = []

        self.xpix = self.cfg.beammap.ncols
        self.ypix = self.cfg.beammap.nrows

        self.wvl_start = self.cfg.instrument.minimum_wavelength
        self.wvl_stop = self.cfg.instrument.maximum_wavelength

        self.chunks_to_trim = self.cfg.flatcal.trim_chunks

        self.obs = None
        self.beamImage = None
        self.wvl_bin_edges = None
        self.wavelengths = None
        self.r_list = None
        self.solution_name = solution_name

        if self.cfg.flatcal.use_wavecal:
            sol = wavecal.Solution(self.cfg.flatcal.wavsol)
            r, _ = sol.find_resolving_powers(cache=True)
            self.r_list = np.nanmedian(r, axis=0)
            #TODO shouldn't this pull the wavelengths from the wavesol

        self.save_plots = self.cfg.flatcal.plots.lower() == 'all'
        self.summary_plot = self.cfg.flatcal.plots.lower() in ('all', 'summary')
        if self.save_plots:
            getLogger(__name__).warning("Comanded to save debug plots, this will add ~30 min to runtime.")

        self.spectral_cube = None
        self.eff_int_times = None
        self.spectral_cube_in_counts = None
        self.delta_weights = None
        self.combined_image = None
        self.flat_weights = None
        self.flat_weight_err = None
        self.flat_flags = None
        self.plotName = None
        self.fig = None
        self.coeff_array = np.zeros(self.xpix, self.ypix)

    def load_data(self):
        pass

    def load_flat_spectra(self):
        pass

    def run(self):
        getLogger(__name__).info("Loading Data")
        self.load_data()
        getLogger(__name__).info("Loading flat spectra")
        self.load_flat_spectra()
        getLogger(__name__).info("Calculating weights")
        self.calculate_weights()
        self.calculate_coefficients()
        sol = FlatSolution(configuration=self.cfg, flat_weights=self.flat_weights, flat_weight_err=self.flat_weight_err,
                           flat_flags=self.flat_flags, coeff_array=self.coeff_array)
        sol.save(save_name=self.solution_name)

        if self.summary_plot:
            getLogger(__name__).info('Making a summary plot')
            sol.summary_plot()

        if self.save_plots:
            getLogger(__name__).info("Writing detailed plots, go get some tea.")
            getLogger(__name__).info('Plotting Weights by Wvl Slices at WeightsWvlSlices')
            self.plotWeightsWvlSlices()
        getLogger(__name__).info('Done')

    def calculate_weights(self):
        """
        Finds the weights by calculating the counts/(average counts) for each wavelength and for each time chunk. The
        length (seconds) of the time chunks are specified in the pipe.yml.

        If specified in the pipe.yml, will also trim time chunks with weights that have the largest deviation from
        the average weight.
        """
        flat_weights = np.zeros_like(self.spectral_cube)
        delta_weights = np.zeros_like(self.spectral_cube)
        for iCube, cube in enumerate(self.spectral_cube):
            wvl_averages = np.zeros_like(self.wavelengths)
            wvl_weights = np.ones_like(cube)
            for iWvl in range(self.wavelengths.size):
                wvl_averages[iWvl] = np.nanmean(cube[:, :, iWvl])
                wvl_averages_array = np.full(np.shape(cube[:,:,iWvl]), wvl_averages[iWvl])
                wvl_weights[:,:,iWvl] = wvl_averages_array/cube[:,:,iWvl]
            wvl_weights[(wvl_weights == np.inf) | (wvl_weights == 0)] = np.nan
            flat_weights[iCube, :, :, :] = wvl_weights

            # To get uncertainty in weight:
            # Assuming negligible uncertainty in medians compared to single pixel spectra,
            # then deltaWeight=weight*deltaSpectrum/Spectrum
            # deltaWeight=weight*deltaRawCounts/RawCounts
            # with deltaRawCounts=sqrt(RawCounts)#Assuming Poisson noise
            # deltaWeight=weight/sqrt(RawCounts)
            # but 'cube' is in units cps, not raw counts so multiply by effIntTime before sqrt

            delta_weights[iCube, :, :, :] = flat_weights / np.sqrt(self.eff_int_times * cube)


        weights_mask = np.isnan(flat_weights)
        self.flat_weights = np.ma.array(flat_weights, mask=weights_mask, fill_value=1.).data
        n_cubes = self.flat_weights.shape[0]
        self.delta_weights = np.ma.array(delta_weights, mask=weights_mask).data

        # sort weights and rearrange spectral cubes the same way
        if self.chunks_to_trim and n_cubes > 1:
            sorted_idxs = np.ma.argsort(self.flat_weights, axis=0)
            identity_idxs = np.ma.indices(np.shape(self.flat_weights))
            sorted_weights = self.flat_weights[
                sorted_idxs, identity_idxs[1], identity_idxs[2], identity_idxs[3]]
            spectral_cube_in_counts = self.spectral_cube_in_counts[
                sorted_idxs, identity_idxs[1], identity_idxs[2], identity_idxs[3]]
            weight_err = self.delta_weights[
                sorted_idxs, identity_idxs[1], identity_idxs[2], identity_idxs[3]]
            sl = self.chunks_to_trim
            weights_to_use = sorted_weights[sl:-sl, :, :, :]
            cubes_to_use = spectral_cube_in_counts[sl:-sl, :, :, :]
            weight_err_to_use = weight_err[sl:-sl, :, :, :]
            self.combined_image = np.ma.sum(cubes_to_use, axis=0)
            self.flat_weights, averaging_weights = np.ma.average(weights_to_use, axis=0,
                                                                     weights=weight_err_to_use ** -2.,
                                                                     returned=True)
            self.spectral_cube_in_counts = np.ma.sum(cubes_to_use, axis=0)
        else:
            self.combined_image = np.ma.sum(self.spectral_cube_in_counts, axis=0)
            self.flat_weights, averaging_weights = np.ma.average(self.flat_weights, axis=0,
                                                                     weights=self.delta_weights ** -2.,
                                                                     returned=True)
            self.spectral_cube_in_counts = np.ma.sum(self.spectral_cube_in_counts, axis=0)

        # Uncertainty in weighted average is sqrt(1/sum(averagingWeights)), normalize weights at each wavelength bin
        self.flat_weight_err = np.sqrt(averaging_weights ** -1.)
        self.flat_flags = self.flat_weights.mask
        wvl_weight_avg = np.ma.mean(np.reshape(self.flat_weights, (-1, self.wavelengths.size)), axis=0)
        self.flat_weights = np.divide(self.flat_weights.data, wvl_weight_avg)

    def calculate_coefficients(self):
        for x in range(self.xpix):
            for y in range(self.ypix):
                fittable = (self.flat_weights[x,y] != 0) & np.isfinite(self.flat_weights[x,y] + self.flat_weight_err[x,y])
                self.coeff_array[x, y]= np.polyfit(self.wavelengths[fittable], self.flat_weights[fittable],
                                                   self.cfg.flatcal.power, w=1 / self.flat_weight_err[fittable] ** 2)
        getLogger(__name__).info('Calculated Flat coefficients')

    def make_summary(self):
        generate_summary_plot(flatsol=self.save_name, save_plot=True)

    def plotWeightsWvlSlices(self):
        """
        Plot weights in images of a single wavelength bin (wavelength-sliced images)
        """
        self.plotName = 'WeightsWvlSlices_{0}'.format('TODO')  #TODO
        self._setup_plots()
        matplotlib.rcParams['font.size'] = 4
        for iWvl, wvl in enumerate(self.wavelengths):
            if self.iPlot % self.nPlotsPerPage == 0:
                self.fig = plt.figure(figsize=(10, 10), dpi=100)
            ax = self.fig.add_subplot(self.nPlotsPerCol, self.nPlotsPerRow, self.iPlot % self.nPlotsPerPage + 1)
            ax.set_title(r'Weights %.0f $\AA$' % wvl)
            image = self.flat_weights[:, :, iWvl]
            vmax = np.nanmean(image) + 3 * np.nanstd(image)
            plt.imshow(image.T, cmap=plt.get_cmap('viridis'), origin='lower', vmax=vmax, vmin=0)
            plt.colorbar()
            if self.iPlot % self.nPlotsPerPage == self.nPlotsPerPage - 1:
                pdf = PdfPages(os.path.join(self.out_directory, 'temp.pdf'))
                pdf.savefig(self.fig)
            self.iPlot += 1
            ax = self.fig.add_subplot(self.nPlotsPerCol, self.nPlotsPerRow, self.iPlot % self.nPlotsPerPage + 1)
            ax.set_title(r'Spectrum %.0f $\AA$' % wvl)
            image = self.combined_image[:, :, iWvl]
            vmax = np.nanmean(image) + 3 * np.nanstd(image)
            plt.imshow(image.T, cmap=plt.get_cmap('viridis'), origin='lower', vmin=0, vmax=vmax)
            plt.colorbar()
            if self.iPlot % self.nPlotsPerPage == self.nPlotsPerPage - 1:
                pdf = PdfPages(os.path.join(self.out_directory, 'temp.pdf'))
                pdf.savefig(self.fig)
                pdf.close()
                self._mergePlots()
                self.saved = True
                plt.close('all')
            self.iPlot += 1
        self._closePlots()

    def _setup_plots(self):
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

    def __init__(self, h5, config=None, solution_name='flat_solution.npz'):
        """
        Reads in the param file and opens appropriate flat file.  Sets wavelength binning parameters.
        """
        super().__init__(config)
        self.exposure_time = self.cfg.exposure_time
        self.h5 = h5
        self.energies = None
        self.energyBinWidth = None
        self.solution_name = solution_name

    def load_data(self):
        getLogger(__name__).info('Loading calibration data from {}'.format(self.h5))
        self.obs = self.h5.photontable
        if not self.obs.wavelength_calibrated:
            raise RuntimeError('Photon data is not wavelength calibrated.')
        self.beamImage = self.obs.beamImage
        self.xpix = self.obs.nXPix
        self.ypix = self.obs.nYPix

        self.energies = [(c.h * c.c) / (i * 10 ** (-9) * c.e) for i in self.wavelengths]
        middle = int(len(self.wavelengths) / 2.0)
        self.energyBinWidth = self.energies[middle] / (5.0 * self.r_list[middle])
        self.wvl_bin_edges = Photontable.makeWvlBins(self.energyBinWidth, self.wvl_start, self.wvl_stop)
        self.wavelengths = (self.wvl_bin_edges[: -1] + np.diff(self.wvl_bin_edges)).flatten()

    def load_flat_spectra(self):
        """
        Reads the flat data into a spectral cube whose dimensions are determined
        by the number of x and y pixels and the number of wavelength bins.
        Each element will be the spectral cube for a time chunk

        To be used for whitelight flat data
        """
        self.spectral_cube = []
        self.eff_int_times = []
        for firstSec in range(0, self.exposure_time, self.chunk_time):  # for each time chunk
            cubeDict = self.obs.getSpectralCube(firstSec=firstSec, integrationTime=self.chunk_time, applyWeight=False,
                                                applyTPFWeight=False, wvlBinEdges=self.wvl_bin_edges)
            cube = cubeDict['cube'] / cubeDict['effIntTime'][:, :, None]
            bad = np.isnan(cube)  # TODO need to update masks to note why these 0s appeared
            cube[bad] = 0

            self.spectral_cube.append(cube)
            self.eff_int_times.append(cubeDict['effIntTime'])
            msg = 'Loaded Flat Spectra for seconds {} to {}'.format(int(firstSec), int(firstSec) + int(self.chunk_time))
            getLogger(__name__).info(msg)

        self.spectral_cube = np.array(self.spectral_cube[0])
        # Haven't had a chance to check if the following lines break - find me if you come across this before me - Sarah
        self.eff_int_times = np.array(self.eff_int_times)
        self.spectral_cube_in_counts = self.eff_int_times * self.spectral_cube


class LaserCalibrator(FlatCalibrator):
    def __init__(self, h5s, solution_name='flat_solution.npz', config=None, darks=None):
        super().__init__(config)
        self.beamImage = self.cfg.beammap
        self.xpix = self.cfg.beammap.ncols
        self.ypix = self.cfg.beammap.nrows
        self.h5s = h5s
        self.wavelengths = np.array(h5s.keys(), dtype=float)
        self.darks = darks
        self.solution_name = solution_name

    def load_flat_spectra(self):
        cps_cube_list, int_times, mask = self.make_spectralcube()
        self.spectral_cube = cps_cube_list
        self.eff_int_times = int_times
        dark_frame = self.get_dark_frame()
        for icube, cube in enumerate(self.spectral_cube):
            dark_subtracted_cube = np.zeros_like(cube)
            for iwvl, wvl in enumerate(cube[0, 0, :]):
                dark_subtracted_cube[:, :, iwvl] = np.subtract(cube[:, :, iwvl], dark_frame)
            # mask out hot and cold pixels
            masked_cube = np.ma.masked_array(dark_subtracted_cube, mask=mask).data
            self.spectral_cube[icube] = masked_cube
        self.spectral_cube = np.array(self.spectral_cube)
        self.eff_int_times = np.array(self.eff_int_times)
        # count cubes is the counts over the integration time
        self.spectral_cube_in_counts = self.eff_int_times * self.spectral_cube

    def make_spectralcube(self):
        n_wvls = len(self.wavelengths)
        n_times = self.cfg.flatcal.nchunks
        exposure_times = np.array([x.duration for x in self.h5s])
        if np.any(self.chunk_time*self.cfg.flatcal.nchunks > exposure_times):
            n_times = int((exposure_times/self.chunk_time).max())
            getLogger(__name__).info('Number of chunks * chunk time is longer than the laser exposure. Using full'
                                     'length of exposure ({} chunks)'.format(n_times))
        cps_cube_list = np.zeros([n_times, self.xpix, self.ypix, n_wvls])
        mask = np.zeros([self.xpix, self.ypix, n_wvls])
        int_times = np.zeros([self.xpix, self.ypix, n_wvls])

        if self.cfg.flatcal.use_wavecal:
            delta_list = self.wavelengths / self.r_list / 2

        startw, stopw = None, None
        for wvl, h5 in self.h5s.items():
            obs = h5.photontable
            if not obs.info['isBadPixMasked'] and not self.cfg.flatcal.use_wavecal:
                getLogger(__name__).warning('H5 File not hot pixel masked, could skew flat weights')

            w_mask = self.wavelengths==wvl

            mask[:, :, w_mask] = obs.flagMask(pixelflags.PROBLEM_FLAGS)
            if self.cfg.flatcal.use_wavecal:
                startw = wvl - delta_list[w_mask]
                stopw = wvl + delta_list[w_mask]

            counts = obs.getTemporalCube(integrationTime=self.chunk_time * self.cfg.flatcal.nchunks,
                                         timeslice=self.chunk_time, startw=startw, stopw=stopw)['cube']
            getLogger(__name__).info('Loaded {}nm spectral cube'.format(wvl))
            int_times[:, :, w_mask] = self.chunk_time
            cps_cube_list[:, :, :, w_mask] = np.moveaxis(counts/self.chunk_time, 2, 0)
        return cps_cube_list, int_times, mask

    def get_dark_frame(self):
        """
        takes however many dark files that are specified in the pipe.yml and computes the counts/pixel/sec for the sum
        of all the dark obs. This creates a stitched together long dark obs from all of the smaller obs given. This
        is useful for legacy data where there may not be a specified dark observation but parts of observations where
        the filter wheel was closed.

        :return: expected dark counts for each pixel over a flat observation
        """
        if not self.darks:
            return np.zeros_like(self.spectral_cube[0][:, :, 0])

        getLogger(__name__).info('Loading dark frames for Laser flat')
        frames = []
        itime = 0
        for dark in self.darks:
            frames.append(dark.photontable.getPixelCountImage(startTime=dark.start,
                                                              integrationTime=dark.duration)['image'])
            itime += dark.duration
        return np.sum(frames, axis=2)/itime


def generate_summary_plot(flatsol, save_plot=False):
    """ Writes a summary plot of the Flat Fielding """
    flat_cal = tables.open_file(flatsol, mode='r')
    calsoln = flat_cal.root.flatcal.calsoln.read()
    weightArrPerPixel = flat_cal.root.flatcal.weights.read()
    beamImage = flat_cal.root.header.beamMap.read()
    xpix = flat_cal.root.header.xpix.read()
    ypix = flat_cal.root.header.ypix.read()
    wavelengths = flat_cal.root.flatcal.wavelength_bins.read()
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
        meanWeightList[np.isnan(meanWeightList)] = 0
        plt.imshow(meanWeightList.T, cmap=plt.get_cmap('viridis'), vmin=0.0, vmax=maxValue)
        plt.colorbar()

        ax = fig.add_subplot(2, 2, 2)
        ax.set_title('Mean Flat value across the array')
        maxValue = np.nanmean(meanSpecList) + 1 * np.nanstd(meanSpecList)
        meanSpecList[np.isnan(meanSpecList)] = 0
        plt.imshow(meanSpecList.T, cmap=plt.get_cmap('viridis'), vmin=0.0, vmax=maxValue)
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
    wavelengths = flat_cal.root.flatcal.wavelength_bins.read()
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

class FlatSolution(object):
    yaml_tag = '!fsoln'
    def __init__(self, file_path=None, configuration=None, beam_map=None, flat_weights=None, coeff_array=None,
                 wavelengths=None, flat_weight_err=None, flat_flags=None, solution_name='flat_solution'):
        self.cfg = configuration
        self.file_path = file_path
        self.beam_map = beam_map
        self.flat_weights = flat_weights
        self.wavelengths = wavelengths
        self.save_name = solution_name
        self.flat_flags = flat_flags
        self.flat_weight_err = flat_weight_err
        self.coeff_array=coeff_array
        self._file_path = os.path.abspath(file_path) if file_path is not None else file_path
        # if we've specified a file load it without overloading previously set arguments
        if self._file_path is not None:
            self.load(self._file_path, overload=False)
        # if not finish the init
        else:
            self.name = solution_name  # use the default or specified name for saving
            self.npz = None  # no npz file so all the properties should be set

    def save(self, save_name=None):
        """Save the solution to a file. The directory is given by the configuration."""
        if save_name is None:
            save_path = os.path.join(self.cfg.out_directory, self.name)
        else:
            save_path = os.path.join(self.cfg.out_directory, save_name)
        if not save_path.endswith('.npz'):
            save_path += '.npz'

        getLogger(__name__).info("Saving solution to {}".format(save_path))
        np.savez(save_path, coeff_array=self.coeff_array, flat_weights=self.flat_weights, wavelengths=self.wavelengths,
                 flat_weight_err=self.flat_weight_err, configuration=self.cfg, beam_map=self.beam_map)
        self._file_path = save_path  # new file_path for the solution

    def load(self, file_path, overload=True, file_mode='c'):
        """
        Load a solution from a file, optionally overloading previously defined attributes.
        The data will not be pulled from the npz file until first access of the data which
        can take a while.

        """
        getLogger(__name__).info("Loading solution from {}".format(file_path))
        keys = ('coeff_array', 'configuration', 'beam_map', 'flat_weights', 'flat_weight_err', 'wavelengths')
        npz_file = np.load(file_path, allow_pickle=True, encoding='bytes', mmap_mode=file_mode)
        for key in keys:
            if key not in list(npz_file.keys()):
                raise AttributeError('{} missing from {}, solution malformed'.format(key, file_path))
        self.npz = npz_file
        if overload:  # properties grab from self.npz if set to none
            for attr in keys:
                setattr(self, attr, None)
        self._file_path = file_path  # new file_path for the solution
        self.name = os.path.splitext(os.path.basename(file_path))[0]  # new name for saving
        getLogger(__name__).info("Complete")

    def summary_plot(self):
        return None

def _run(flattner):
    getLogger(__name__).debug('Calling run on {}'.format(flattner))
    flattner.run()


def load_solution(sc, singleton_ok=True):
    """sc is a solution filename string, a FlatSolution object, or a mkidpipeline.config.MKIDFlatdataDescription"""
    global _loaded_solutions
    if not singleton_ok:
        raise NotImplementedError('Must implement solution copying')
    if isinstance(sc, FlatSolution):
        return sc
    if isinstance(sc, mkidpipeline.config.MKIDFlatdataDescription):
        sc = mkidpipeline.config.spectralcal_id(sc.id)+'.npz'
    sc = sc if os.path.isfile(sc) else os.path.join(mkidpipeline.config.config.paths.database, sc)
    try:
        return _loaded_solutions[sc]
    except KeyError:
        _loaded_solutions[sc] = FlatSolution(file_path=sc)
    return _loaded_solutions[sc]


def fetch(dataset, config=None, ncpu=np.inf, remake=False):
    solution_descriptors = dataset.flatcals
    cfg = mkidpipeline.config.config if config is None else config
    solutions = []
    flattners = []
    for sd in solution_descriptors:
        sf = os.path.join(cfg.paths.database, mkidpipeline.config.flatcal_id(sd.id) + '.npz')
        if os.path.exists(sf) and not remake:
            solutions.append(load_solution(sf))
        else:
            fcfg = mkidpipeline.config.load_task_config(StepConfig()) if 'flatcal' not in cfg else cfg.copy()
            #fcfg.register('flatcal.wavsol', sd.wavecal, update=True) #TODO whats the point of this line
            if sd.method == 'laser':
                if sd.darks:
                    flattner = LaserCalibrator(h5s=sd.h5s, config=fcfg, solution_name=sf,
                                               darks=[H5Subset(d) for d in sd.darks])
                else:
                    flattner = LaserCalibrator(h5s=sd.h5s, config=fcfg, solution_name=sf)
            else:
                flattner = WhiteCalibrator(H5Subset(sd.ob), config=fcfg, solution_name=sf)

            solutions.append(sf)
            flattners.append(flattner)

    if not flattners:
        return solutions

    ncpu = mkidpipeline.config.n_cpus_available(max=min(fcfg.ncpu, ncpu))
    if ncpu == 1 or len(flattners) == 1:
        for f in flattners:
            f.run()
    else:
        pool = mp.Pool(ncpu)
        pool.map(_run, flattners)
        pool.close()
        pool.join()

    return solutions
