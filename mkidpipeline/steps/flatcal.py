import os
import multiprocessing as mp
import time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import mkidpipeline.definitions as definitions
from mkidpipeline.steps import wavecal
from mkidpipeline.photontable import Photontable
from mkidcore.corelog import getLogger
import mkidpipeline.config
from mkidpipeline.config import H5Subset
from mkidcore.pixelflags import FlagSet
import warnings

_loaded_solutions = {}

"""
Flatcal has two modes, a Laser Flat (`LaserCalibrator`) and a While Light Flat (`WhiteCalibrator`). The Laser Flat uses 
the same laser exposures as the wavecal to determine what the wavelength dependent response is of each pixel. This can 
be done by either trusting that the laser frames are monochromatic, and not imposing any wavelength cut in the pipeline 
(`use_wavecal` in the pipe.yaml = False), or by using the wavecal solution to only look at photons within a window 
around the laser wavelength as determined by the energy resolution at that wavelength ('use_wavecal` = True). The latter 
is recommended, especially if there is a significant out of band background that needs to be accounted for.

The White Light Flat by contrast looks at a single white (polychromatic) flat exposure. This could be a twilight flat, 
a dome flat, or any other exposure where you expect the flux on the array to be uniform at every wavelength. In this 
case the data MUST be wavecaled and the wavelength bins are determined by the energy resolution of the detector at each 
wavelength (as is the case with the Laser Flat when `use_wavecal` = True).

The Flatcal can also break up each exposure used to determine the flat into a series of `nchunks` time chunks of 
duration `chunk_time`. If `trim_chunks` is greater than 0 then only (nchunk - 2*trim_chunks) number of time cubes will
be used in the determination of the flat weights. This is to exclude chunks of time where there may have been 
contamination of the flat.

The Flatcal determines the flat weights by first generating a spectral cube for the exposure 
(`FlatCalibrator.make_spectral_cube`). This cube will have dimensions (nchunks, nx_pixels, ny_pixels, n_wavelengths). 
If a dark is specified in the data.yaml then it is also subtracted off from the spectral cube at this point. The flat 
weights are then calculated for each time chunk at each wavelength (`FlatCalibrator.calculate_weights`) by dividing the 
flux in each pixel by the average flux for that wavelength across the whole array (excluding bad pixels as determined by 
the PROBLEM_FLAGS). If `trim_chunks` > 0 then the flat weights are sorted along the time axis and the trim_chunks number 
of time slices are removed from each end. The remaining time chunks can then be averaged together to get the final flat 
weight per wavelength for each pixel in the array. 

These flat weights as a function of wavelength are then fit by a polynomial of order `power` in 
`FlatCalibrator.calculate_coefficients`. These coefficients are saved as an '.npz' file to be applied later along with 
the `flat_weights` themselves and various other parameters to assist with application and plotting.

The Flatcal application uses the fit coefficients to determine the appropriate weight for each photon depending on 
the incident pixel of the photon and its wavelength. This weight is multiplied into the `SpecWeight` column of the 
photontable. 
"""


class StepConfig(mkidpipeline.config.BaseStepConfig):
    yaml_tag = u'!flatcal_cfg'
    REQUIRED_KEYS = (('rate_cutoff', 0, 'Count Rate Cutoff in inverse seconds (number)'),
                     ('trim_chunks', 1, 'number of Chunks to trim (integer)'),
                     ('chunk_time', 10, 'duration of chunks used for weights (s)'),
                     ('nchunks', 6, 'number of chunks to median combine'),
                     ('power', 1, 'power of polynomial to fit, <3 advised'),
                     ('use_wavecal', True, 'Use a wavelength dependant correction for wavecaled data.'),
                     ('plots', 'summary', 'none|summary|all'))

    def _verify_attributes(self):
        ret = super()._verify_attributes()
        try:
            assert 0 <= self.rate_cutoff <= 20000
        except:
            ret.append('rate_cutoff must be [0, 20000]')

        try:
            assert 0 <= self.trim_chunks <= 1
        except:
            ret.append(f'trim_chunks must be a float in [0,1]: {type(self.trim_chunks)}')

        return ret


TEST_CFGS = (StepConfig(chunk_time=30, nchunks=10, trim_chunks=0, use_wavecal=True),
             StepConfig(chunk_time=30, nchunks=10, trim_chunks=0, use_wavecal=False),
             StepConfig(chunk_time=10, nchunks=5, trim_chunks=1, use_wavecal=True),
             StepConfig(chunk_time=10, nchunks=5, trim_chunks=3, use_wavecal=True))

FLAGS = FlagSet.define(
    ('bad', 0, 'either all or less than the power of the fit polynomial number of flat weights are invalid '),
    ('not_all_weights_valid', 1, 'at least one of the wavelength weights is invalid')
)

PROBLEM_FLAGS = ('pixcal.dead', 'beammap.noDacTone', 'wavecal.bad', 'wavecal.failed_convergence',
                 'wavecal.no_histograms', 'wavecal.not_attempted', 'flatcal.bad')


class FlatCalibrator:
    def __init__(self, config=None, solution_name='flat_solution.npz'):
        self.cfg = mkidpipeline.config.PipelineConfigFactory(step_defaults=dict(flatcal=StepConfig()),
                                                             cfg=config, copy=True)

        self.wvl_start = self.cfg.instrument.minimum_wavelength
        self.wvl_stop = self.cfg.instrument.maximum_wavelength

        self.wavelengths = None
        self.solution_name = solution_name

        self.save_plots = self.cfg.flatcal.plots.lower() == 'all'
        self.summary_plot = self.cfg.flatcal.plots.lower() in ('all', 'summary')
        if self.save_plots:
            getLogger(__name__).warning("Comanded to save debug plots, this will add ~30 min to runtime.")

        self.spectral_cube = None
        self.int_time = None
        self.delta_weights = None
        self.flat_weights = None
        self.flat_weight_err = None
        self.flat_flags = None
        self.coeff_array = np.zeros((self.cfg.beammap.ncols, self.cfg.beammap.nrows, self.cfg.flatcal.power + 1))
        self.mask = None
        self.h5s = None
        self.darks = None

    def run(self):
        getLogger(__name__).info("Loading flat spectra")
        self.load_flat_spectra()
        getLogger(__name__).info("Calculating weights")
        self.calculate_weights()
        self.calculate_coefficients()
        sol = FlatSolution(configuration=self.cfg, flat_weights=self.flat_weights, flat_weight_err=self.flat_weight_err,
                           flat_flags=self.flat_flags, coeff_array=self.coeff_array, wavelengths=self.wavelengths,
                           beammap=self.cfg.beammap.residmap)
        sol.save(save_name=self.solution_name)
        if self.summary_plot:
            getLogger(__name__).info('Making a summary plot')
            sol.plot_summary(save_plot=self.summary_plot)
        if self.save_plots:
            getLogger(__name__).info('Making flat weight debug plot')
            sol.debug_plot(plot_fraction=0.1, save_plot=True)
        getLogger(__name__).info('Done')

    def make_spectral_cube(self):
        pass

    def load_flat_spectra(self):
        self.make_spectral_cube()
        dark_frames = self.get_dark_frames()
        for icube, cube in enumerate(self.spectral_cube):
            # mask out any pixels with PROBLEM_FLAGS
            self.spectral_cube[icube] = np.ma.filled(np.ma.masked_array(cube - dark_frames, mask=self.mask),
                                                     fill_value=np.nan)
        self.spectral_cube = np.array(self.spectral_cube)
        # count cubes is the counts over the integration time

    def calculate_weights(self):
        """
        Finds the weights by calculating the counts/(average counts) for each wavelength and for each time chunk. The
        length (seconds) of the time chunks are specified in the pipe.yml.

        If specified in the pipe.yml, will also trim time chunks with weights that have the largest deviation from
        the average weight.
        """
        flat_weights = np.zeros_like(self.spectral_cube)
        delta_weights = np.zeros_like(self.spectral_cube)
        weight_mask = np.zeros_like(self.spectral_cube, dtype=bool)
        self.flat_flags = np.zeros(self.spectral_cube.shape[1:3], dtype=int)
        self.spectral_cube[self.spectral_cube == 0] = np.nan
        wvl_averages_array = np.nanmean(self.spectral_cube, axis=(1, 2))
        # TODO get rid of for loop if possible
        for iCube, cube in enumerate(self.spectral_cube):
            wvl_weights = wvl_averages_array[iCube] / cube
            weight_mask[iCube] = (~np.isfinite(wvl_weights)) | (wvl_weights <= 0)
            self.mask |= weight_mask[iCube]
            flat_weights[iCube] = wvl_weights
            # Note to get uncertainty in weight:
            # Assuming negligible uncertainty in medians compared to single pixel spectra,
            # then deltaWeight=weight*deltaSpectrum/Spectrum
            # deltaWeight=weight*deltaRawCounts/RawCounts
            # with deltaRawCounts=sqrt(RawCounts)#Assuming Poisson noise
            # deltaWeight=weight/sqrt(RawCounts)
            # but 'cube' is in units cps, not raw counts so multiply by effIntTime before sqrt

            delta_weights[iCube] = wvl_weights / np.sqrt(self.int_time * cube)
        self.flat_weights = np.ma.filled(np.ma.array(flat_weights, mask=weight_mask), 1.0)
        n_cubes = self.flat_weights.shape[0]
        self.delta_weights = np.ma.filled(np.ma.array(delta_weights, mask=weight_mask), 1.0)

        # sort weights and rearrange spectral cubes the same way
        if self.cfg.flatcal.trim_chunks and n_cubes > 1:
            sort_idxs = np.ma.argsort(self.flat_weights, axis=0)
            i_idxs = np.ma.indices(self.flat_weights.shape)
            sorted_weights = self.flat_weights[sort_idxs, i_idxs[1], i_idxs[2], i_idxs[3]]
            weight_err = self.delta_weights[sort_idxs, i_idxs[1], i_idxs[2], i_idxs[3]]
            sl = self.cfg.flatcal.trim_chunks
            weights_to_use = sorted_weights[sl:-sl]
            weight_err_to_use = weight_err[sl:-sl]
            self.flat_weights, averaging_weights = np.ma.average(weights_to_use, axis=0,
                                                                 weights=weight_err_to_use ** -2.,
                                                                 returned=True)
        else:
            self.flat_weights, averaging_weights = np.ma.average(self.flat_weights, axis=0,
                                                                 weights=self.delta_weights ** -2.,
                                                                 returned=True)

        # Uncertainty in weighted average is sqrt(1/sum(averagingWeights)), normalize weights at each wavelength bin
        self.flat_weight_err = np.sqrt(averaging_weights ** -1.)
        wvl_weight_avg = np.ma.mean(np.reshape(self.flat_weights, (-1, self.wavelengths.size)), axis=0)
        self.flat_weights = np.divide(self.flat_weights.data, wvl_weight_avg.data)
        self.flat_flags |= ((np.all(self.mask, axis=2) << FLAGS.flags['bad'].bit) |
                            (np.any(self.mask, axis=2) << FLAGS.flags['not_all_weights_valid'].bit))

    def calculate_coefficients(self):
        if type(self.h5s) == dict:
            beam_image = Photontable([o for o in self.h5s.values()][0].h5).beamImage
        else:
            beam_image = self.h5s.photontable.beamImage
        for (x, y), resID in np.ndenumerate(beam_image):
            fittable = ~self.mask.astype(bool)[x, y]
            if not fittable.any() or (len(fittable[fittable == True]) <= self.cfg.flatcal.power):
                self.flat_flags[x, y] |= FLAGS.flags['bad'].bit
            else:
                self.coeff_array[x, y] = np.polyfit(self.wavelengths[fittable], self.flat_weights[x, y][fittable],
                                                    self.cfg.flatcal.power,
                                                    w=1 / self.flat_weight_err[x, y][fittable] ** 2)

        getLogger(__name__).info('Calculated Flat coefficients')

    def get_dark_frames(self, enforce_wavecal=True):
        """

        :return:
        """
        if all([dark is None for dark in self.darks]):
            return np.zeros_like(self.spectral_cube[0])
        getLogger(__name__).info('Loading dark frames for Laser flat')
        frames = np.zeros_like(self.spectral_cube[0])
        for i, dark in enumerate(self.darks):
            if dark is not None:
                wcf = dark.photontable.query_header('wavecal')
                if wcf:
                    im = dark.photontable.get_fits(start=dark.start, duration=dark.duration, cube_type='time',
                                                   rate=True, bin_width=dark.duration, wave_start=self.wvl_start,
                                                   wave_stop=self.wvl_stop)['SCIENCE']
                    frames[:, :, i] = im.data[:, :, 0]
                elif not enforce_wavecal:
                    getLogger(__name__).info(
                        f'Not using a wavelength calibrated dark for {self.wavelengths[i]} nm flat '
                        f'- if you would like the wavelength range of the dark to match that of '
                        f'the wavelength flat please apply a wavecal to your dark frame.')
                    im = dark.photontable.get_fits(start=dark.start, duration=dark.duration, cube_type='time',
                                                   rate=True, bin_width=dark.duration)['SCIENCE']
                    frames[:, :, i] = im.data[:, :, 0]
                else:
                    assert enforce_wavecal is True and not wcf
                    getLogger(__name__).info(
                        f'No wavecal solution is applied for the {self.wavelengths[i]} nm dark and '
                        f'enforce_wavecal is True - dark will not be applied')
                    pass
            else:
                pass
        return frames


class WhiteCalibrator(FlatCalibrator):
    """
    Opens flat file using parameters from the param file, sets wavelength binning parameters, and calculates flat
    weights for flat file.  Writes these weights to a h5 file and plots weights both by pixel
    and in wavelength-sliced images.
    """

    def __init__(self, h5s, config=None, solution_name='flat_solution.npz', darks=None):
        """
        Reads in the param file and opens appropriate flat file.  Sets wavelength binning parameters.
        """
        super().__init__(config)
        self.h5s = h5s
        self.solution_name = solution_name
        self.darks = darks

    def make_spectral_cube(self):
        if self.cfg.flatcal.chunk_time > self.h5s.duration:
            getLogger(__name__).warning('Chunk time is longer than the exposure. Using a single chunk')
            time_edges = np.array([0, self.h5s.duration], dtype=float)
        elif self.cfg.flatcal.chunk_time * self.cfg.flatcal.nchunks > self.h5s.duration:
            nchunks = int(self.h5s.duration / self.cfg.flatcal.chunk_time)
            time_edges = np.arange(nchunks + 1, dtype=float) * self.cfg.flatcal.chunk_time
            getLogger(__name__).warning(
                f'Number of {self.cfg.flatcal.chunk_time} s chunks requested is longer than the '
                f'exposure. Using first full {nchunks} chunks.')
        else:
            time_edges = np.arange(self.cfg.flatcal.nchunks + 1, dtype=float) * self.cfg.flatcal.chunk_time

        pt = Photontable(self.h5s.timerange.h5)
        if not pt.wavelength_calibrated:
            raise RuntimeError('Photon data is not wavelength calibrated.')

        # define wavelengths to use
        wvl_edges = pt.nominal_wavelength_bins
        self.wavelengths = wvl_edges[: -1] + np.diff(wvl_edges) / 2  # wavelength bin centers

        if not pt.query_header('pixcal'):
            getLogger(__name__).warning('H5 File not hot pixel masked, will skew flat weights')

        cps_cube_list = np.zeros((len(time_edges) - 1, self.cfg.beammap.ncols, self.cfg.beammap.nrows,
                                  len(self.wavelengths)))
        for i, (wstart, wstop) in enumerate(zip(wvl_edges[:-1], wvl_edges[1:])):
            hdul = pt.get_fits(duration=self.h5s.duration, rate=True, bin_edges=time_edges, wave_start=wstart,
                               wave_stop=wstop, cube_type='time')
            cps_cube_list[:, :, :, i] = hdul['SCIENCE'].data  # moveaxis for code compatibility
        getLogger(__name__).info(f'Loaded spectral cubes')
        self.spectral_cube = cps_cube_list  # n_times, x, y, n_wvls
        self.int_time = time_edges[1] - time_edges[0]
        self.mask = pt.flagged(PROBLEM_FLAGS)[..., None] * np.ones(self.wavelengths.size, dtype=bool)


class LaserCalibrator(FlatCalibrator):
    def __init__(self, h5s, wavesol, solution_name='flat_solution.npz', config=None, darks=None):
        super().__init__(config)
        self.h5s = h5s
        self.wavelengths = np.array([key.value for key in h5s.keys()], dtype=float)
        self.darks = darks
        self.solution_name = solution_name
        r, _ = wavecal.Solution(wavesol).find_resolving_powers(cache=True)
        self.r_list = np.nanmedian(r, axis=0)

    def make_spectral_cube(self):
        n_wvls = len(self.wavelengths)
        nchunks = self.cfg.flatcal.nchunks
        x, y = self.cfg.beammap.ncols, self.cfg.beammap.nrows
        exposure_times = np.array([x.duration for x in self.h5s.values()])
        self.mask = np.zeros((x, y, n_wvls), dtype=int)
        if np.any(self.cfg.flatcal.chunk_time > exposure_times):
            getLogger(__name__).warning('Chunk time is longer than the exposure. Using a single chunk')
            flat_duration = exposure_times
        elif np.any(self.cfg.flatcal.chunk_time * self.cfg.flatcal.nchunks > exposure_times):
            nchunks = int((exposure_times / self.cfg.flatcal.chunk_time).max())
            getLogger(__name__).warning(
                f'Number of {self.cfg.flatcal.chunk_time} s chunks requested is longer than the '
                f'exposure. Using first full {nchunks} chunks.')
            flat_duration = exposure_times
        else:
            flat_duration = np.full(n_wvls, self.cfg.flatcal.chunk_time * self.cfg.flatcal.nchunks)
        cps_cube_list = np.zeros([nchunks, x, y, n_wvls])

        if self.cfg.flatcal.use_wavecal:
            delta_list = self.wavelengths / self.r_list / 2
        wvl_start, wvl_stop = None, None

        for wvl, h5 in self.h5s.items():
            pt = h5.photontable
            if not pt.query_header('pixcal') and not self.cfg.flatcal.use_wavecal:
                getLogger(__name__).warning('H5 File not hot pixel masked, this could skew the calculated flat weights')

            w_mask = self.wavelengths == wvl.value
            w_idx = np.nonzero(w_mask)[0][0]
            if self.cfg.flatcal.use_wavecal:
                wvl_start = wvl.value - delta_list[w_idx]
                wvl_stop = wvl.value + delta_list[w_idx]

            hdul = pt.get_fits(duration=flat_duration[w_idx], rate=True, bin_width=self.cfg.flatcal.chunk_time,
                               wave_start=wvl_start, wave_stop=wvl_stop, cube_type='time')

            getLogger(__name__).info(f'Loaded {wvl.value:.1f} nm spectral cube')
            cps_cube_list[:, :, :, w_idx] = hdul['SCIENCE'].data
            self.mask[:, :, w_idx] = pt.flagged(PROBLEM_FLAGS)
        self.spectral_cube = cps_cube_list
        self.int_time = self.cfg.flatcal.chunk_time
        # self.mask = pt.flagged(PROBLEM_FLAGS)[..., None] * np.ones(self.wavelengths.size, dtype=bool)


class FlatSolution(object):
    yaml_tag = '!fsoln'

    def __init__(self, file_path=None, configuration=None, beammap=None, flat_weights=None, coeff_array=None,
                 wavelengths=None, flat_weight_err=None, flat_flags=None, solution_name='flat_solution'):
        self.cfg = configuration
        self.file_path = file_path
        self.beammap = beammap
        self.flat_weights = flat_weights
        self.wavelengths = wavelengths
        self.save_name = solution_name
        self.flat_flags = flat_flags
        self.flat_weight_err = flat_weight_err
        self.coeff_array = coeff_array
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
            save_path = os.path.join(self.cfg.paths.database, self.name)
        else:
            save_path = os.path.join(self.cfg.paths.database, save_name)
        if not save_path.endswith('.npz'):
            save_path += '.npz'

        getLogger(__name__).info("Saving solution to {}".format(save_path))
        np.savez(save_path, coeff_array=self.coeff_array, flat_weights=self.flat_weights, wavelengths=self.wavelengths,
                 flat_weight_err=self.flat_weight_err, flat_flags=self.flat_flags, configuration=self.cfg,
                 beammap=self.beammap)
        self._file_path = save_path  # new file_path for the solution

    def load(self, file_path, overload=True, file_mode='c'):
        """
        Load a solution from a file, optionally overloading previously defined attributes.
        The data will not be pulled from the npz file until first access of the data which
        can take a while.

        """
        getLogger(__name__).debug("Loading solution from {}".format(file_path))
        keys = (
        'coeff_array', 'configuration', 'beammap', 'flat_weights', 'flat_flags', 'flat_weight_err', 'wavelengths')
        npz_file = np.load(file_path, allow_pickle=True, encoding='bytes', mmap_mode=file_mode)
        for key in keys:
            if key not in list(npz_file.keys()):
                raise AttributeError(f'{key} missing from {file_path}, solution malformed')
        self.npz = npz_file
        self.coeff_array = self.npz['coeff_array']
        self.cfg = self.npz['configuration']
        self.beammap = self.npz['beammap']
        self.flat_weights = self.npz['flat_weights']
        self.flat_weight_err = self.npz['flat_weight_err']
        self.wavelengths = self.npz['wavelengths']
        self.flat_flags = self.npz['flat_flags']
        if overload:  # properties grab from self.npz if set to none
            for attr in keys:
                setattr(self, attr, None)
        self._file_path = file_path  # new file_path for the solution
        self.name = os.path.splitext(os.path.basename(file_path))[0]  # new name for saving
        getLogger(__name__).info("Complete")

    def get(self, pixel=None, res_id=None):
        if not pixel and not res_id:
            raise ValueError('Need to specify either resID or pixel coordinates')
        for pix, res in np.ndenumerate(self.beammap):
            if res == res_id or pix == pixel:  # in case of non unique resIDs
                coeffs = self.coeff_array[pix[0], pix[1]]
                return np.poly1d(coeffs)

    def plot_summary(self, save_plot=True):
        """ Writes a summary plot of the Flat Fielding """
        weight_array = self.flat_weights
        wavelengths = self.wavelengths
        weight_array[self.flat_flags] = np.nan
        mean_weight_array = np.nanmean(weight_array, axis=2)

        array_averaged_weights = np.nanmean(weight_array, axis=(0, 1))
        array_std = np.nanstd(weight_array, axis=(0, 1))

        figure = plt.figure()

        gs = gridspec.GridSpec(2, 4)
        axes = np.array([figure.add_subplot(gs[:, 0:2]), figure.add_subplot(gs[0, 2:]),
                         figure.add_subplot(gs[1, 2:])])

        axes[0].set_title('Mean Flat Weight', fontsize=8)
        max = np.nanmean(mean_weight_array) + 1 * np.nanstd(mean_weight_array)
        mean_weight_array[np.isnan(mean_weight_array)] = 0
        im = axes[0].imshow(mean_weight_array.T, cmap=plt.get_cmap('viridis'), vmin=0.0, vmax=max)

        axes[1].scatter(wavelengths, array_averaged_weights)
        axes[1].set_title('Mean Weight vs. Wavelength', fontsize=8)
        axes[1].set_ylabel('Mean Weight', fontsize=8)
        axes[1].set_xlabel(r'$\lambda$ ($nm$)', fontsize=8)

        axes[2].scatter(wavelengths, array_std)
        axes[2].set_title('Std of Weight vs. Wavelength', fontsize=8)
        axes[2].set_ylabel('Standard Deviation', fontsize=8)
        axes[2].set_xlabel(r'$\lambda$ ($nm$)', fontsize=8)

        axes[0].tick_params(labelsize=8)
        axes[1].tick_params(labelsize=8)
        axes[2].tick_params(labelsize=8)
        axes[1].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        axes[2].yaxis.set_major_formatter(plt.FormatStrFormatter('%.2f'))
        divider = make_axes_locatable(axes[0])
        cax = divider.append_axes('bottom', size='5%', pad=0.3)
        cbar = figure.colorbar(im, cax=cax, orientation='horizontal')
        cbar.ax.tick_params(labelsize='small')
        plt.subplots_adjust(wspace=0.8, hspace=0.8)
        if save_plot:
            save_path = self._file_path.split('.npz')[0] + '.pdf'
            getLogger(__name__).info(f'Saving flatcal summary plot to {save_path}')
            plt.savefig(save_path)
            plt.clf()
        else:
            plt.show()

    def debug_plot(self, plot_fraction=0.1, save_plot=False):
        for (x, y), resID in np.ndenumerate(self.beammap):
            if (x * y) % (1 / plot_fraction) == 0:
                plt.scatter(self.wavelengths, self.flat_weights[x, y])
                soln = self.get(pixel=(x, y), res_id=resID)
                plt.plot(self.wavelengths, soln(self.wavelengths))
            else:
                pass
        if save_plot:
            save_path = self._file_path.split('.npz')[0] + '_debug.pdf'
            getLogger(__name__).info(f'Saving flatcal debug plot to {save_path}')
            plt.savefig(save_path)
            plt.clf()


def _run(flattner):
    getLogger(__name__).debug('Calling run on {}'.format(flattner))
    flattner.run()


def load_solution(sc, singleton_ok=True):
    """sc is a solution filename string, a FlatSolution object, or a mkidpipeline.config.MKIDFlatcal"""
    global _loaded_solutions
    if not singleton_ok:
        raise NotImplementedError('Must implement solution copying')
    if isinstance(sc, FlatSolution):
        return sc
    if isinstance(sc, definitions.MKIDFlatcal):
        sc = sc.path
    sc = sc if os.path.isfile(sc) else os.path.join(mkidpipeline.config.config.paths.database, sc)
    try:
        return _loaded_solutions[sc]
    except KeyError:
        _loaded_solutions[sc] = FlatSolution(file_path=sc)
    return _loaded_solutions[sc]


def fetch(solution_descriptors, config=None, ncpu=None, remake=False):
    fcfg = mkidpipeline.config.PipelineConfigFactory(step_defaults=dict(flatcal=StepConfig()), cfg=config, ncpu=ncpu,
                                                     copy=True)

    solutions = {}
    if not remake:
        for sd in solution_descriptors:
            try:
                solutions[sd.id] = load_solution(sd.path)
            except IOError:
                pass
            except Exception as e:
                getLogger(__name__).info(f'Failed to load {sd} due to a {e}')

    flattners = []
    for sd in set(sd for sd in solution_descriptors if sd.id not in solutions):
        if sd.method == 'laser':
            flattner = LaserCalibrator(h5s=sd.h5s, config=fcfg, solution_name=sd.path,
                                       darks=[o.dark for o in sd.obs],
                                       wavesol=sd.data.path)
        else:
            flattner = WhiteCalibrator(H5Subset(sd.data), config=fcfg, solution_name=sd.path,
                                       darks=[o.dark for o in sd.obs])

        solutions[sd.id] = sd.path
        flattners.append(flattner)

    if not flattners:
        return solutions

    poolsize = mkidpipeline.config.n_cpus_available(max=min(fcfg.get('flatcal.ncpu', inherit=True), len(flattners)))
    if poolsize == 1:
        for f in flattners:
            f.run()
    else:
        pool = mp.Pool(poolsize)
        pool.map(_run, flattners)
        pool.close()
        pool.join()

    return solutions


def apply(o: definitions.MKIDObservation, config=None):
    """
    Applies a flat calibration to the "SpecWeight" column for each pixel.

    Weights are multiplied in and replaced; NOT reversible
    """

    if o.flatcal is None:
        getLogger(__name__).info(f"No flatcal specified for {o}, nothing to do")
        return

    fc_file = o.flatcal if os.path.exists(o.flatcal) else o.flatcal.path
    of = Photontable(o.h5)
    fcfg = of.query_header('flatcal')
    if fcfg:
        if fcfg != fc_file:
            getLogger(__name__).warning(f'{o.name} is already calibrated with a different flat ({fcfg}).')
        else:
            getLogger(__name__).info(f"{o.name} is already flat calibrated.")
        return

    tic = time.time()
    calsoln = FlatSolution(fc_file)
    getLogger(__name__).info(f'Applying {calsoln.name} to {o}')

    # Set flags for pixels that have them
    to_clear = of.flags.bitmask([f'flatcal.{flag.name}' for flag in FLAGS], unknown='ignore')
    of.enablewrite()
    of.unflag(to_clear)
    for flag in FLAGS:
        mask = (calsoln.flat_flags & flag.bitmask) > 0
        of.flag(mask * of.flags.bitmask([f'flatcal.{flag.name}'], unknown='warn'))

    n_todo = len(list(of.resonators(exclude=PROBLEM_FLAGS)))
    if not n_todo:
        getLogger(__name__).warning(f'Done. There were no unflagged pixels.')
        return

    getLogger(__name__).info(f'Applying flat weights to {n_todo} unflagged pixels ('
                             f'{100 * (n_todo / calsoln.beammap.size):.2f} % of pixels).')
    with of.needed_ram():
        counter = 0
        for pixel, resid in of.resonators(exclude=PROBLEM_FLAGS, pixel=True):
            soln = calsoln.get(pixel=pixel, res_id=resid)
            if not soln:
                counter += 1
                getLogger(__name__).debug('No flat calibration for good pixel {}'.format(resid))
                continue
            indices = of.photonTable.get_where_list('resID==resid')
            if not indices.size:
                continue

            tic2 = time.time()

            if (np.diff(indices) == 1).all():  # This takes ~300s for ALL photons combined on a 70Mphot file.
                wave = of.photonTable.read(start=indices[0], stop=indices[-1] + 1, field='wavelength')
                weights = soln(wave) * of.photonTable.read(start=indices[0], stop=indices[-1] + 1, field='weight')
                weights = weights.clip(0)  # enforce positive weights only
                of.photonTable.modify_column(start=indices[0], stop=indices[-1] + 1, column=weights, colname='weight')
            else:  # This takes 3.5s per pixel on a 70 Mphot file!!!
                # raise NotImplementedError('This code path is impractically slow at present.')
                getLogger(__name__).debug('Using modify_coordinates')
                rows = of.photonTable.read_coordinates(indices)
                rows['weight'] *= soln(rows['wavelength'])
                of.photonTable.modify_coordinates(indices, rows)
                getLogger(__name__).debug('Flat weights updated in {:.2f}s'.format(time.time() - tic2))
    getLogger(__name__).info(f'No flat calibration for '
                             f'{(counter / len(list(of.resonators(exclude=PROBLEM_FLAGS, pixel=True)))) * 100:.2f} % '
                             f'good pixels ')
    of.update_header('flatcal', calsoln.name)
    try:
        assert calsoln.name == o.flatcal.id.strip('.npz')  #DO NOT REMOVE
    except AttributeError:
        assert calsoln.name == o.flatcal.strip('.npz')
    of.update_header('E_FLTCAL', calsoln.name)
    try:
        of.update_header('flatcal.method', o.flatcal.method)
    except AttributeError:
        of.update_header('flatcal.method', 'Explicit flatcal file')
    getLogger(__name__).info('Flatcal applied in {:.2f}s'.format(time.time() - tic))
