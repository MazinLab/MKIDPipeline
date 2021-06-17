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
import time
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import gridspec
from matplotlib.backends.backend_pdf import PdfPages
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mkidpipeline.steps import wavecal
from mkidpipeline.photontable import Photontable
from mkidcore.corelog import getLogger
import mkidpipeline.config
from mkidpipeline.config import H5Subset
from mkidcore.pixelflags import FlagSet, PROBLEM_FLAGS
import matplotlib as mpl
_loaded_solutions = {}


class StepConfig(mkidpipeline.config.BaseStepConfig):
    yaml_tag = u'!flatcal_cfg'
    REQUIRED_KEYS = (('rate_cutoff', 0, 'Count Rate Cutoff in inverse seconds (number)'),
                     ('trim_chunks', 1, 'number of Chunks to trim (integer)'),
                     ('chunk_time', 10, 'duration of chunks used for weights (s)'),
                     ('nchunks', 6, 'number of chunks to median combine'),
                     ('power', 1, 'power of polynomial to fit, <3 advised'),
                     ('use_wavecal', True, 'Use a wavelength dependant correction for wavecaled data.'),
                     ('plots', 'summary', 'none|summary|all'))

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


TEST_CFGS = (StepConfig(chunk_time=30, nchunks=10, trim_chunks=0, use_wavecal=True),
             StepConfig(chunk_time=30, nchunks=10, trim_chunks=0, use_wavecal=False),
             StepConfig(chunk_time=10, nchunks=5, trim_chunks=1, use_wavecal=True))

FLAGS = FlagSet.define(
    ('inf_weight', 0, 'Spurious infinite weight was calculated - weight set to 1.0'),
    ('zero_weight', 1, 'Spurious zero weight was calculated - weight set to 1.0'),
    ('nan_weight', 2, 'Derived wavelength is below formal validity range of calibration'),
    ('negative_weight', 3, 'Derived wavelength is above formal validity range of calibration'),
)

UNFLATABLE = tuple()  # todo flags that can't be flatcaled


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
        self.eff_int_times = None
        self.delta_weights = None
        self.flat_weights = None
        self.flat_weight_err = None
        self.flat_flags = None
        self.coeff_array = np.zeros((self.cfg.beammap.ncols, self.cfg.beammap.nrows, self.cfg.flatcal.power + 1))
        self.mask = None
        self.h5s = None
        self.darks = None

    def make_spectral_cube(self):
        pass

    def load_flat_spectra(self):
        self.make_spectral_cube()
        dark_frames = self.get_dark_frames()
        for icube, cube in enumerate(self.spectral_cube):
            # mask out any pixels with PROBLEM_FLAGS
            self.spectral_cube[icube] = np.ma.masked_array(cube-dark_frames, mask=self.mask).data
        self.spectral_cube = np.array(self.spectral_cube)
        self.eff_int_times = np.array(self.eff_int_times)
        # count cubes is the counts over the integration time

    def run(self):
        getLogger(__name__).info("Loading flat spectra")
        self.load_flat_spectra()
        getLogger(__name__).info("Calculating weights")
        self.calculate_weights()
        self.calculate_coefficients()
        sol = FlatSolution(configuration=self.cfg, flat_weights=self.flat_weights, flat_weight_err=self.flat_weight_err,
                           flat_flags=self.flat_flags, coeff_array=self.coeff_array, wavelengths=self.wavelengths)
        sol.save(save_name=self.solution_name)
        if self.summary_plot:
            getLogger(__name__).info('Making a summary plot')
            sol.plot_summary(save_plot=self.save_plots)
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
        weight_mask = np.zeros_like(self.spectral_cube, dtype=bool)
        self.flat_flags = np.zeros(self.spectral_cube.shape[1:], dtype=int)
        for iCube, cube in enumerate(self.spectral_cube):
            masked_cube = cube.copy()
            masked_cube[self.mask] = np.nan
            wvl_averages_array = np.nanmean(masked_cube.reshape(-1, masked_cube.shape[-1]), axis=0)
            wvl_weights = wvl_averages_array / cube
            self.flat_flags |= ((np.isinf(wvl_weights) << FLAGS.flags['inf_weight'].bit) |
                                ((wvl_weights == 0) << FLAGS.flags['zero_weight'].bit) |
                                ((wvl_weights < 0) << FLAGS.flags['negative_weight'].bit) |
                                (np.isnan(wvl_weights) << FLAGS.flags['nan_weight'].bit))
            weight_mask[iCube] = (~np.isfinite(wvl_weights)) | (wvl_weights <= 0)
            self.mask |= weight_mask[iCube]
            flat_weights[iCube] = wvl_weights
            # To get uncertainty in weight:
            # Assuming negligible uncertainty in medians compared to single pixel spectra,
            # then deltaWeight=weight*deltaSpectrum/Spectrum
            # deltaWeight=weight*deltaRawCounts/RawCounts
            # with deltaRawCounts=sqrt(RawCounts)#Assuming Poisson noise
            # deltaWeight=weight/sqrt(RawCounts)
            # but 'cube' is in units cps, not raw counts so multiply by effIntTime before sqrt

            delta_weights[iCube] = wvl_weights / np.sqrt(self.eff_int_times * cube)
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

    def calculate_coefficients(self):
        if type(self.h5s) == dict:
            beam_image = Photontable([o for o in self.h5s.values()][0].h5).beamImage
        else:
            beam_image = Photontable(self.h5s.h5).beamImage
        for (x, y), resID in np.ndenumerate(beam_image):
            fittable = ~self.mask[x, y]
            if not fittable.any():
                pass
            else:
                # TODO will raise RankWarning, find out why and if it can be ignored
                self.coeff_array[x, y] = np.polyfit(self.wavelengths[fittable], self.flat_weights[x, y][fittable],
                                                    self.cfg.flatcal.power,
                                                    w=1 / self.flat_weight_err[x, y][fittable] ** 2)
        getLogger(__name__).info('Calculated Flat coefficients')

    def get_dark_frames(self):
        """

        :return:
        """
        if all([dark is None for dark in self.darks]):
            return np.zeros_like(self.spectral_cube[0])
        getLogger(__name__).info('Loading dark frames for Laser flat')
        frames = np.zeros_like(self.spectral_cube[0])
        for i, dark in enumerate(self.darks):
            if dark is not None:
                # TODO dark frames need more thought, when should they be queried on wavelength? are they even wavecaled?
                im = dark.photontable.get_fits(start=dark.start, duration=dark.duration, cube_type='time',
                                               rate=True, bin_width=dark.duration, wave_start=self.wvl_start,
                                               wave_stop=self.wvl_stop)['SCIENCE']
                frames[:, :, i] = im.data[:, :, 0]
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
        exposure_time = self.h5s.duration
        if self.cfg.flatcal.chunk_time > exposure_time:
            getLogger(__name__).warning('Chunk time is longer than the exposure. Using a single chunk')
            time_edges = np.array([self.h5s.start, self.h5s.start + self.h5s.duration])
        elif self.cfg.flatcal.chunk_time * self.cfg.flatcal.nchunks > exposure_time:
            nchunks = int(exposure_time / self.cfg.flatcal.chunk_time)
            time_edges = self.h5s.start + np.arange(nchunks + 1) * self.cfg.flatcal.chunk_time
            getLogger(__name__).warning(
                f'Number of {self.cfg.flatcal.chunk_time} s chunks requested is longer than the '
                f'exposure. Using first full {nchunks} chunks.')
        else:
            time_edges = np.self.h5s.start + np.arange(self.cfg.flatcal.nchunks + 1) * self.cfg.flatcal.chunk_time

        pt = Photontable(self.h5s.timerange.h5)
        if not pt.wavelength_calibrated:
            raise RuntimeError('Photon data is not wavelength calibrated.')

        # define wavelengths to use
        edges = pt.nominal_wavelength_bins
        self.wavelengths = edges[: -1] + np.diff(edges)  # wavelength bin centers

        if not pt.query_header('pixcal'):
            getLogger(__name__).warning('H5 File not hot pixel masked, will skew flat weights')

        cps_cube_list = []
        for wstart, wstop in zip(edges[:-1], edges[1:]):
            hdul = pt.get_fits(rate=True, bin_edges=time_edges, wave_start=wstart, wave_stop=wstop, cube_type='time')
            cps_cube_list.append(np.moveaxis(hdul['SCIENCE'].data, 2, 0))  # moveaxis for code compatibility
        getLogger(__name__).info(f'Loaded spectral cubes')
        self.spectral_cube = np.array(cps_cube_list)  # n_times, x, y, n_wvls
        # TODO if the rest of the algorithm doesn't take good care of this then including it here for future expansion
        #  is silly and it should be considered for removal
        self.eff_int_times = np.full(self.spectral_cube.shape[1:], fill_value=time_edges[1] - time_edges[0])
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
        mask = np.zeros([x, y, n_wvls], dtype=bool)
        int_times = np.zeros([x, y, n_wvls])

        if self.cfg.flatcal.use_wavecal:
            delta_list = self.wavelengths / self.r_list / 2
        wvl_start, wvl_stop = None, None

        for wvl, h5 in self.h5s.items():
            obs = h5.photontable
            if not obs.query_header('pixcal') and not self.cfg.flatcal.use_wavecal:
                getLogger(__name__).warning('H5 File not hot pixel masked, this could skew the calculated flat weights')

            w_mask = self.wavelengths == wvl.value
            w_idx = np.nonzero(w_mask)[0][0]
            mask[:, :, w_idx] = obs.flagged(PROBLEM_FLAGS)
            if self.cfg.flatcal.use_wavecal:
                wvl_start = wvl.value - delta_list[w_idx]
                wvl_stop = wvl.value + delta_list[w_idx]

            hdul = obs.get_fits(duration=flat_duration[w_idx], rate=True, bin_width=self.cfg.flatcal.chunk_time,
                                wave_start=wvl_start, wave_stop=wvl_stop, cube_type='time')

            getLogger(__name__).info(f'Loaded {wvl.value:.1f} nm spectral cube')
            int_times[:, :, w_idx] = self.cfg.flatcal.chunk_time
            cps_cube_list[:, :, :, w_idx] = np.moveaxis(hdul['SCIENCE'].data, 2, 0)
        self.spectral_cube = cps_cube_list
        self.eff_int_times = int_times
        self.mask = mask


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
                 beam_map=self.beam_map)
        self._file_path = save_path  # new file_path for the solution


    def load(self, file_path, overload=True, file_mode='c'):
        """
        Load a solution from a file, optionally overloading previously defined attributes.
        The data will not be pulled from the npz file until first access of the data which
        can take a while.

        """
        getLogger(__name__).info("Loading solution from {}".format(file_path))
        keys = ('coeff_array', 'configuration', 'beam_map', 'flat_weights', 'flat_flags', 'flat_weight_err', 'wavelengths')
        npz_file = np.load(file_path, allow_pickle=True, encoding='bytes', mmap_mode=file_mode)
        for key in keys:
            if key not in list(npz_file.keys()):
                raise AttributeError(f'{key} missing from {file_path}, solution malformed')
        self.npz = npz_file
        self.coeff_array = self.npz['coeff_array']
        self.cfg = self.npz['configuration']
        self.beam_map = self.npz['beam_map']
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
        for pix, res in self.cfg.beammap:
            if res == res_id or pix == pixel:  # in case of non unique resIDs
                coeffs = self.coeff_array[pixel[0], pixel[1]]
                return np.poly1d(coeffs)


    def plot_summary(self, save_plot=False):
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
        axes[1].set_xlabel(r'$\lambda$ ($\AA$)', fontsize=8)

        axes[2].scatter(wavelengths, array_std)
        axes[2].set_title('Std of Weight vs. Wavelength', fontsize=8)
        axes[2].set_ylabel('Standard Deviation', fontsize=8)
        axes[2].set_xlabel(r'$\lambda$ ($\AA$)', fontsize=8)

        axes[0].tick_params(labelsize=8)
        axes[1].tick_params(labelsize=8)
        axes[2].tick_params(labelsize=8)
        divider = make_axes_locatable(axes[0])
        cax = divider.append_axes('bottom', size='5%', pad=0.3)
        cbar = figure.colorbar(im, cax=cax, orientation='horizontal')
        cbar.ax.tick_params(labelsize='small')
        plt.subplots_adjust(wspace=0.8, hspace=0.8)
        if save_plot:
            plt.savefig(self._file_path.split('.npz')[0] + '.pdf')
        else:
            plt.show()


def _run(flattner):
    getLogger(__name__).debug('Calling run on {}'.format(flattner))
    flattner.run()


def load_solution(sc, singleton_ok=True):
    """sc is a solution filename string, a FlatSolution object, or a mkidpipeline.config.MKIDFlatcalDescription"""
    global _loaded_solutions
    if not singleton_ok:
        raise NotImplementedError('Must implement solution copying')
    if isinstance(sc, FlatSolution):
        return sc
    if isinstance(sc, mkidpipeline.config.MKIDFlatcalDescription):
        sc = sc.path
    sc = sc if os.path.isfile(sc) else os.path.join(mkidpipeline.config.config.paths.database, sc)
    try:
        return _loaded_solutions[sc]
    except KeyError:
        _loaded_solutions[sc] = FlatSolution(file_path=sc)
    return _loaded_solutions[sc]


def fetch(dataset, config=None, ncpu=None, remake=False):
    solution_descriptors = getattr(dataset, 'flatcals', dataset)

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


def apply(o: mkidpipeline.config.MKIDObservation, config=None):
    """
    Applies a flat calibration to the "SpecWeight" column for each pixel.

    Weights are multiplied in and replaced; NOT reversible
    """

    # cfg = mkidpipeline.config.PipelineConfigFactory(step_defaults=dict(flatcal=StepConfig()), cfg=config, copy=True)

    if o.flatcal is None:
        getLogger(__name__).info(f"No flatcal specified for {o}, nothing to do")
        return

    of = o.photontable
    fcf = of.query_header('flatcal')
    if fcf:
        if fcf != o.flatcal.path:
            getLogger(__name__).warning(f'{o} is already calibrated with a different flat ({fcf}).')
        else:
            getLogger(__name__).info(f"{o} is already flat calibrated.")
        return

    tic = time.time()
    calsoln = FlatSolution(o.flatcal.path)
    getLogger(__name__).info(f'Applying {calsoln} to {o}')

    # Set flags for pixels that have them
    to_clear = of.flags.bitmask([f'flatcal.{flag.name}' for flag in FLAGS], unknown='ignore')
    of.unflag(to_clear)
    for flag in FLAGS:
        mask = (calsoln.flat_flags & flag.bitmask).sum(2) > 0
        of.flag(mask * of.flags.bitmask([f'flatcal.{flag.name}'], unknown='warn'))

    for pixel, resID in of.resonators(exclude=UNFLATABLE, pixel=True):
        soln = calsoln.get(pixel=pixel, res_id=resID)
        if not soln:
            getLogger(__name__).warning('No flat calibration for good pixel {}'.format(resID))
            continue

        indices = of.photonTable.get_where_list('resID==resID')
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

    of.update_header('flatcal', calsoln.file_path)
    of.update_header('FLATCAL.ID', calsoln.id)  # TODO ensure is pulled over from definition/is consistent
    of.update_header('FLATCAL.TYPE', calsoln.type)  # TODO add type parameter to calsoln
    getLogger(__name__).info('Flatcal applied in {:.2f}s'.format(time.time() - tic))
