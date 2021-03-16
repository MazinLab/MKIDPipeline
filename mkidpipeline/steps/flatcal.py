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
import argparse
import atexit
import os
import time
from datetime import datetime
import multiprocessing as mp
import scipy.constants as c

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tables
from PyPDF2 import PdfFileMerger, PdfFileReader

from matplotlib.backends.backend_pdf import PdfPages
from mkidpipeline.steps import wavecal
from mkidcore.headers import FlatCalSoln_Description
from mkidpipeline.hdf.photontable import Photontable
from mkidcore.corelog import getLogger
import mkidcore.corelog
import mkidpipeline.config
import pkg_resources as pkg
from mkidcore.utils import query
import mkidcore.pixelflags as pixelflags
import mkidpipeline.steps.badpix as badpix

DEFAULT_CONFIG_FILE = pkg.resource_filename('mkidpipeline.calibration.flatcal', 'flatcal.yml')


class StepConfig(mkidpipeline.config.BaseStepConfig):
    yaml_tag = u'!flatcal_cfg'
    REQUIRED_KEYS = ('count_rate_cutoff', 'chunks_to_trim', 'chunk_time', 'power', 'plots')
    OPTIONAL_KEYS = tuple()

    def _vet_errors(self):
        ret = []
        try:
            assert 0 <= self.count_rate_cutoff <= 20000
        except:
            ret.append('count_rate_cutoff must be [0, 20000]')

        try:
            assert isinstance(self.chunks_to_trim, float) and 0 <= self.chunks_to_trim <= 1
        except:
            ret.append('chunks_to_trim must be a float in [0,1]')

        return ret

# NB mkidcore.config.yaml.register_class(StepConfig) Must be called

# TODO need to create a calibrator factory that works with three options: wavecal, white light, and filtered or laser
#  light. In essence each needs a load_data functionand maybe a load_flat_spectra in the parlance of the current
#  structure. The individual cases can be determined by seeing if the input data has a starttime or a wavesol
#  Subclasses for special functions and a factory function for deciding which to instantiate.


class FlatCalibrator:
    def __init__(self, config=None):
        self.config_file = DEFAULT_CONFIG_FILE if config is None else config

        self.cfg = mkidpipeline.config.load_task_config(config)
        self.use_wavecal = self.cfg.flatcal.use_wavecal
        self.data_dir = self.cfg.paths.data
        self.out_directory = self.cfg.paths.out

        self.chunk_time = self.cfg.flatcal.chunk_time

        self.dark_start = []
        self.dark_int = []

        self.xpix = self.cfg.beammap.ncols
        self.ypix = self.cfg.beammap.nrows

        self.wvl_start = self.cfg.instrument.wvl_start
        self.wvl_stop = self.cfg.instrument.wvl_stop

        self.chunks_to_trim = self.cfg.flatcal.trim_fraction

        self.obs = None
        self.beamImage = None
        self.wvl_bin_edges = None
        self.wavelengths = None

        if self.use_wavecal:
            sol_file = self.cfg.wavcal
            sol = wavecal.Solution(sol_file[0])
            r, resid = sol.find_resolving_powers(cache=True)
            self.r_list = np.nanmedian(r, axis=0)
        else:
            self.r_list = None

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

    def makeCalibration(self):
        getLogger(__name__).info("Loading Data")
        self.load_data()
        getLogger(__name__).info("Loading flat spectra")
        self.load_flat_spectra()
        getLogger(__name__).info("Calculating weights")
        self.calculate_weights()
        getLogger(__name__).info("Writing weights")
        self.write_weights(poly_power=self.cfg.flatcal.power)

        if self.summary_plot:
            getLogger(__name__).info('Making a summary plot')
            self.make_summary()

        if self.save_plots:
            getLogger(__name__).info("Writing detailed plots, go get some tea.")
            getLogger(__name__).info('Plotting Weights by Wvl Slices at WeightsWvlSlices_{0}'.format(self.start_time))
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

    def write_weights(self, poly_power=2):
        """
        Writes an h5 file to put calculated flat cal factors in
        """
        if not os.path.exists(self.out_directory):
            os.makedirs(self.out_directory)
        try:
            flatsol = tables.open_file(self.save_name, mode='w')
        except IOError:
            getLogger(__name__).error("Couldn't create flat cal file: {} ", self.save_name)
            return
        header = flatsol.create_group(flatsol.root, 'header', 'Calibration information')
        tables.Array(header, 'beamMap', obj=self.beamImage.residmap)
        tables.Array(header, 'xpix', obj=self.xpix)
        tables.Array(header, 'ypix', obj=self.ypix)
        calgroup = flatsol.create_group(flatsol.root, 'flatcal',
                                            'Table of flat calibration weights by pixel and wavelength')
        tables.Array(calgroup, 'weights', obj=self.flat_weights.data,
                     title='Flat calibration Weights indexed by pixelRow,pixelCol,wavelengthBin')
        tables.Array(calgroup, 'spectrum', obj=self.spectral_cube_in_counts.data,
                     title='Twilight spectrum indexed by pixelRow,pixelCol,wavelengthBin')
        tables.Array(calgroup, 'flags', obj=self.flat_flags,
                     title='Flat cal flags indexed by pixelRow,pixelCol,wavelengthBin. 0 is Good')
        tables.Array(calgroup, 'wavelength_bins', obj=self.wavelengths,
                     title='Wavelength bin edges corresponding to third dimension of weights array')
        description_dict = FlatCalSoln_Description(nWvlBins=len(self.wavelengths), max_power=poly_power)
        caltable = flatsol.create_table(calgroup, 'calsoln', description_dict, title='Flat Cal Table',
                                            expectedrows=self.xpix*self.ypix)
        for iRow in range(self.xpix):
            for iCol in range(self.ypix):
                entry = caltable.row
                entry['resid'] = self.beamImage.residmap[iRow, iCol]
                entry['y'] = iRow
                entry['x'] = iCol
                entry['weight'] = self.flat_weights.data[iRow, iCol, :]
                entry['err'] = self.flat_weight_err[iRow, iCol, :]
                fittable = (entry['weight'] != 0) & np.isfinite(entry['weight']+entry['err'])
                if fittable.sum() < poly_power+1:
                    entry['bad'] = True
                else:
                    entry['coeff'] = np.polyfit(self.wavelengths[fittable], entry['weight'][fittable], poly_power,
                                                w=1 / entry['err'][fittable] ** 2)
                    entry['bad'] = self.flat_flags[iRow, iCol, :].any()
                entry['spectrum'] = self.spectral_cube_in_counts.data[iRow, iCol, :]
                entry.append()
        flatsol.close()
        getLogger(__name__).info("Wrote to {}".format(self.save_name))

    def make_summary(self):
        generate_summary_plot(flatsol=self.save_name, save_plot=True)

    def plotWeightsWvlSlices(self):
        """
        Plot weights in images of a single wavelength bin (wavelength-sliced images)
        """
        self.plotName = 'WeightsWvlSlices_{0}'.format(self.start_time)
        self._setup_plots()
        matplotlib.rcParams['font.size'] = 4
        wavelengths = self.wavelengths
        for iWvl, wvl in enumerate(wavelengths):
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

    def __init__(self, config=None, cal_file_name='flatsol_{start}.h5'):
        """
        Reads in the param file and opens appropriate flat file.  Sets wavelength binning parameters.
        """
        super().__init__(config)
        self.start_time = self.cfg.start_time
        self.exposure_time = self.cfg.exposure_time
        self.h5file = self.cfg.get('h5file', os.path.join(self.cfg.paths.out, str(self.start_time) + '.h5'))
        self.save_name = self.cfg.get('flatname', os.path.join(self.cfg.paths.database,
                                                                     cal_file_name.format(start=self.start_time)))

    def load_data(self):
        getLogger(__name__).info('Loading calibration data from {}'.format(self.h5file))
        self.obs = Photontable(self.h5file)
        if not self.obs.wavelength_calibrated:
            raise RuntimeError('Photon data is not wavelength calibrated.')
        self.beamImage = self.obs.beamImage
        self.xpix = self.obs.nXPix
        self.ypix = self.obs.nYPix

        self.energies = [(c.h * c.c) / (i * 10 ** (-9) * c.e) for i in self.wavelengths]
        middle = int(len(self.wavelengths) / 2.0)
        self.energyBinWidth = self.energies[middle] / (5.0 * self.r_list[middle])

        self.wvl_bin_edges = Photontable.makeWvlBins(self.energyBinWidth, self.wvl_start, self.wvl_stop)
        self.wavelengths = self.wvl_bin_edges[: -1] + np.diff(self.wvl_bin_edges)
        self.wavelengths = self.wavelengths.flatten()
        self.sol = False

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
    def __init__(self, h5dir='', config=None, cal_file_name='flatsol_laser.h5'):
        super().__init__(config)
        self.save_name = self.cfg.get('flatname', os.path.join(self.cfg.paths.database,
                                                                     cal_file_name))
        self.beamImage = self.cfg.beammap
        self.xpix = self.cfg.beammap.ncols
        self.ypix = self.cfg.beammap.nrows
        self.wavelengths = np.array(self.cfg.wavelengths, dtype=float)
        self.exposure_times = self.cfg.exposure_times
        self.start_times = self.cfg.start_times
        self.h5_directory = h5dir
        self.h5_file_names = [os.path.join(self.h5_directory, str(t) + '.h5') for t in self.start_times]
        self.dark_h5_file_names = []

    def load_data(self):
        """

        :return:
        """
        getLogger(__name__).info('No need to load wavecal solution data for a Laser Flat, passing through')
        pass

    def load_flat_spectra(self):
        """

        :return:
        """
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
        """

        :return:
        """
        wavelengths = self.wavelengths
        n_wvls = len(self.wavelengths)
        n_times = self.cfg.flatcal.nchunks
        if np.any(self.chunk_time*self.cfg.flatcal.nchunks > self.exposure_times):
            n_times = int((self.exposure_times/self.chunk_time).max())
            getLogger(__name__).info('Number of chunks * chunk time is longer than the laser exposure. Using full'
                                     'length of exposure ({} chunks)'.format(n_times))
        cps_cube_list = np.zeros([n_times, self.xpix, self.ypix, n_wvls])
        mask = np.zeros([self.xpix, self.ypix, n_wvls])
        int_times = np.zeros([self.xpix, self.ypix, n_wvls])
        if self.use_wavecal:
            delta_list = wavelengths / self.r_list
        for iwvl, wvl in enumerate(wavelengths):
            obs = Photontable(self.h5_file_names[iwvl])
            if not obs.info['isBadPixMasked'] and not self.use_wavecal:
                getLogger(__name__).warning('H5 File not hot pixel masked, could skew flat weights')
            bad_mask = obs.flagMask(pixelflags.PROBLEM_FLAGS)
            mask[:, :, iwvl] = bad_mask
            if self.use_wavecal:
                counts = obs.getTemporalCube(integrationTime=self.chunk_time * self.cfg.flatcal.nchunks,
                                             timeslice=self.chunk_time, startw=wvl-(delta_list[iwvl]/2.),
                                             stopw=wvl+(delta_list[iwvl]/2.))
            else:
                counts = obs.getTemporalCube(integrationTime=self.cfg.flatcal.chunk_time * self.cfg.flatcal.nchunks,
                                             timeslice=self.chunk_time)
            getLogger(__name__).info('Loaded {}nm spectral cube'.format(wvl))
            cps_cube = counts['cube']/self.chunk_time  # TODO move this division outside of the loop
            cps_cube = np.moveaxis(cps_cube, 2, 0)
            int_times[:, :, iwvl] = self.chunk_time
            cps_cube_list[:, :, :, iwvl] = cps_cube
        return cps_cube_list, int_times, mask

    def get_dark_frame(self):
        '''
        takes however many dark files that are specified in the pipe.yml and computes the counts/pixel/sec for the sum
        of all the dark obs. This creates a stitched together long dark obs from all of the smaller obs given. This
        is useful for legacy data where there may not be a specified dark observation but parts of observations where
        the filter wheel was closed.

        If self.use_wavecal is True then a dark is not subtracted off since this just  takes into account total counts
        and not energy information

        :return: expected dark counts for each pixel over a flat observation
        '''
        if not self.dark_h5_file_names:
            dark_frame = np.zeros_like(self.spectral_cube[0][:,:,0])
        else:
            self.dark_start = [self.cfg.flatcal.dark_data['start_times']]
            self.dark_int = [self.cfg.flatcal.dark_data['int_times']]
            self.dark_h5_file_names = [os.path.join(self.h5_directory, str(t) + '.h5') for t in self.dark_start]
            frames = np.zeros((140, 146, len(self.dark_start)))
            getLogger(__name__).info('Loading dark frames for Laser flat')

            for i, file in enumerate(self.dark_h5_file_names):
                obs = Photontable(file)
                frame = obs.getPixelCountImage(integrationTime=self.dark_int[i])['image']
                frames[:, :, i] = frame
            total_counts = np.sum(frames, axis=2)
            total_int_time = float(np.sum(self.dark_int))
            counts_per_sec = total_counts / total_int_time
            dark_frame = counts_per_sec
        return dark_frame



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


def _run(flattner):
    getLogger(__name__).debug('Calling makeCalibration on {}'.format(flattner))
    flattner.makeCalibration()


def fetch(dataset, config=None, ncpu=np.inf, remake=False):
    solution_descriptors = dataset.flatcals
    cfg = mkidpipeline.config.config if config is None else config
    for sd in dataset.wavecals:
        wavcal = os.path.join(cfg.paths.database, mkidpipeline.config.wavecal_id(sd.id)+'.npz')
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
                fcfg.register('start_times', np.array([x.start for x in sd.wavecal.data]),
                              update=True)
                fcfg.register('exposure_times', np.array([x.duration for x in sd.wavecal.data]), update=True)
                fcfg.register('wavelengths', np.array([w for w in sd.wavecal.wavelengths]), update=True)
                fcfg.register('wavcal', np.array([wavcal]), update=True)
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

    obsfile = Photontable(flattner.h5file, mode='write')
    if not obsfile.wavelength_calibrated:
        obsfile.applyWaveCal(wavecal.load_solution(flattner.cfg.wavesol))
        getLogger(__name__).info('Applied Wavecal {} to {}.h5'.format(flattner.cfg.wavesol, flattner.h5file))

    if args.h5only:
        exit()

    flattner.makeCalibration()
