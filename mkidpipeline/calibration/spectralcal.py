"""
Author: Seth Meeker        Date:November 21, 2012

Edited for MEC by: Sarah Steiger    Date: April 1, 2020

Opens MKID observation of a spectrophotometric standard star and associated wavelength cal file,
reads in all photons and converts to energies.
Bins photons to generate a spectrum, then divides this into the known spectrum (published)
of the object to create a sensitivity curve.
This curve is then written out to the obs file as spectral weights

Flags are associated with each pixel - see mkidcore/pixelFlags.py for descriptions.

Assumes h5 files are wavelength calibrated, flatcalibrated and linearity corrected (deadtime corrected)
"""

import sys,os
import tables
import numpy as np
from mkidcore import pixelflags
from configparser import ConfigParser
from mkidpipeline.hdf.photontable import ObsFile
from mkidpipeline.utils.utils import rebin, gaussianConvolution, fitBlackbody
from mkidpipeline.utils import mkidstandards
from mkidcore.corelog import getLogger
import mkidcore.corelog
import scipy.constants as c
from mkidcore.objects import Beammap
import matplotlib.pyplot as plt

class Configuration(object):
    """Configuration class for the spectrophotometric calibration analysis."""
    yaml_tag = u'!spectralcalconfig'

    def __init__(self, configfile=None, start_time=tuple(), intTime=30, beammap=None,
                 h5dir='', outdir='', bindir='', savedir='', h5_file_names='', sky_file_names='', object_name='',
                 photometry='', method='', energy_bin_width=0.01, wvlStart=850, wvlStop=1375, energy_start=1.46,
                 energy_stop=0.9, nWvlBins=50, dead_time=0.000001, collecting_area=0, plots=True):

        # parse arguments
        self.configuration_path = configfile
        self.h5_directory = h5dir
        self.out_directory = outdir
        self.bin_directory = bindir
        self.save_directory = savedir
        self.object_name = object_name
        self.start_time = list(map(int, start_time))
        self.h5_file_names = list(map(str, h5_file_names))
        self.sky_file_names = list(map(str, sky_file_names))
        self.dead_time = float(dead_time)
        self.intTime = float(intTime)
        self.h5_directory = h5dir
        self.energyBinWidth = float(energy_bin_width)
        self.wvlStart = float(wvlStart)
        self.wvlStop = float(wvlStop)
        self.energyStart = float(energy_start)
        self.energyStop = float(energy_stop)
        self.nWvlBins = float(nWvlBins)
        self.obs = None
        self.sky_obs=None
        self.photometry = str(photometry)
        self.std_wvls = None
        self.std_flux = None
        self.binned_std_spectrum = np.array()
        self.wvl_bin_edges = np.array()
        self.plots = bool(plots)
        self.collecting_area = float(collecting_area)
        if self.configuration_path is not None:
            # load in the configuration file
            cfg = mkidcore.config.load(self.configuration_path)
            self.ncpu = cfg.ncpu
            self.beammap = cfg.beammap
            self.xpix = self.beammap.ncols
            self.ypix = self.beammap.nrows
            self.beam_map_path = self.beammap.file
            self.wvlStart = cfg.instrument.wvl_start * 10.0 # in angstroms
            self.wvlStop = cfg.instrument.wvl_stop * 10.0 # in angstroms
            self.energyStart = (c.h * c.c) / (self.wvlStart * 10**(-10) * c.e)
            self.energyStop = (c.h * c.c) / (self.wvlStop * 10**(-10) * c.e)
            self.nWvlBins = (self.energyStop - self.energyStart) / self.energyBinWidth
            self.h5_directory = cfg.paths.out
            self.out_directory = cfg.paths.database
            self.bin_directory = cfg.paths.data
            self.save_directory = cfg.paths.database
            self.start_time = list(map(int, cfg.spectralcal.start))
            self.dead_time = cfg.instrument.deadtime
            self.object_name = cfg.spectralcal.object_name
            self.energyBinWidth = cfg.instrument.energy_bin_width
            self.h5_directory = cfg.paths.out
            self.photometry = cfg.spectralcal.photometry_type
            self.collecting_area = cfg.spectralcal.collecting_area
            self.plots = cfg.spectralcal.plots
            self.intTime = cfg.spectralcal.duration if cfg.spectralcal.duration else (cfg.spectralcal.start - cfg.spectralcal.stop)
            try:
                self.h5_file_names = [os.path.join(self.h5_directory, cfg.h5_file_names)]
                self.sky_file_names = [os.path.join(self.h5_directory, cfg.sky_file_names)]
            except KeyError:
                self.h5_file_names = [os.path.join(self.h5_directory, str(cfg.h5_file_names) + '.h5')]
                self.sky_file_names = [os.path.join(self.h5_directory, str(cfg.sky_file_names) + '.h5')]
            try:
                self.templar_configuration_path = cfg.templar.file
            except KeyError:
                if self.beammap.frequencies is None:
                    getLogger(__name__).warning('Beammap loaded without frequencies and no templar config specified.')
            self.obs = ObsFile(self.h5_file_name)
        else: #TODO figure out what needs to go in the else
            self.beammap = beammap if beammap is not None else Beammap(default='MEC')
            self.xpix = self.beammap.ncols
            self.ypix = self.beammap.nrows
            self.beam_map_path = beammap.file
            self.h5_file_name = [os.path.join(self.h5_directory, str(self.start_time) + '.h5')]

    @classmethod
    def to_yaml(cls, representer, node):
        d = node.__dict__.copy()
        d.pop('config')
        return representer.represent_mapping(cls.yaml_tag, d)

    @classmethod
    def from_yaml(cls, constructor, node):
        d = mkidcore.config.extract_from_node(constructor, 'configuration_path', node)
        return cls(d['configuration_path'])

    def hdf_exist(self):
        """Check if all hdf5 files specified exist."""
        return all(map(os.path.isfile, self.h5_file_name))

    def write(self, file_):
        """Save the configuration to a file"""
        with open(file_, 'w') as f:
            mkidcore.config.yaml.dump(self, f)


class SpectralCalibrator(object):
    """
    Opens flux file, prepares standard spectrum, and calculates flux factors for the file.
    Method is provided in param file. If 'relative' is selected, an obs file with standard star defocused over
    the entire array is expected, with accompanying sky file to do sky subtraction.
    If any other method is provided, 'absolute' will be done by default, wherein a point source is assumed
    to be present. The obs file is then broken into spectral frames with photometry (psf or aper) performed
    on each frame to generate the observed spectrum.
    """
    def __init__(self, configuration, solution_name='solution.npz', _shared_tables=None, fit_array=None, main=True):

        self.cfg = Configuration(configuration) if not isinstance(configuration, Configuration) else configuration
        self.solution_name = solution_name
        self.solution = SpectralSolution(configuration=self.cfg, fit_array=fit_array, beam_map=self.cfg.beammap.residmap,
                                         beam_map_flags=self.cfg.beammap.flagmap, solution_name=self.solution_name)
        self.flux_spectrum = np.zeros((self.cfg.xpix, self.cfg.ypix))
        self.flux_spectra = np.zeros((self.cfg.xpix, self.cfg.ypix))
        self.flux_effTime = np.zeros((self.cfg.xpix, self.cfg.ypix))
        self.wvl_bin_edges = None
        self.sky_spectra = np.zeros((self.cfg.xpix, self.cfg.ypix))
        self.sky_spectrum = np.zeros((self.cfg.xpix, self.cfg.ypix))
        self.sky_effTime = np.zeros((self.cfg.xpix, self.cfg.ypix))

    def run(self, verbose=False, parallel=None, save=True, plot=None):
        """
        Compute the spectrophotometric calibration for the data specified in the configuration
        object. This method runs load_relative_spectrum() and load_sky_spectrum() or load_absolute_sepctrum(),
        load_standard_spectrum(), and calculate_specweights() sequentially.

        Args:
            verbose: a boolean specifying whether to print a progress bar to the stdout.
            parallel: a boolean specifying whether to use more than one core in the
                      computation.
            save: a boolean specifying if the result will be saved.
            plot: a boolean specifying if a summary plot for the computation will be
                  saved.
        """
        getLogger(__name__).info("Loading Spectrum from MEC")
        try:
            if self.photometry == 'relative':
                getLogger(__name__).info('Performing relative photometry')
                self._run("load_relative_spectrum", parallel=parallel, verbose=verbose)
                getLogger(__name__).info('Flux spectrum loaded')
                self._run("load_sky_spectrum", parallel=parallel, verbose=verbose)
                getLogger(__name__).info('Sky spectrum loaded ')
            elif self.photometry == 'PSF' or self.photometry == 'aperture':
                getLogger(__name__).info('Extracting point source spectrum ')
                self._run("load_absolute_spectrum", parallel=parallel, verbose=verbose)
            if self._shared_tables is not None:
                del self._shared_tables
                self._shared_tables = None
            getLogger(__name__).info("Loading Standard Spectrum")
            self._run("load_standard_spectrum", parallel=parallel, verbose=verbose)
            getLogger(__name__).info("Calculating Spectrophotometric Weights")
            self._run("calculate_specweights", parallel=parallel, verbose=verbose)
            if save:
                self.solution.save(save_name=self.solution_name if isinstance(self.solution_name, str) else None)
            if plot or (plot is None and self.cfg.summary_plot):
                save_name = self.solution_name.rpartition(".")[0] + ".pdf"
                self.solution.plot_summary(save_name=save_name)
        except KeyboardInterrupt:
            getLogger(__name__).info("Keyboard shutdown requested ... exiting")

    def _run(self, method, pixels=None, wavelengths=None, verbose=False, parallel=True):
        getattr(self, method)(pixels=pixels, wavelengths=wavelengths, verbose=verbose)


    def load_absolute_spectrum(self):
        '''
         Extract the MEC measured spectrum of the spectrophotometric standard by breaking data into spectral cube
         and performing photometry (aperture or psf) on each spectral frame
         '''
        getLogger(__name__).info('performing {} photometry on MEC spectrum').format(self.cfg.photometry)
        cube_dict = self.cfg.obs.getSpectralCube(firstSec=self.cfg.start_time, integrationTime=self.cfg.intTime,
                                                 applyWeight=True)
        cube = np.array(cube_dict['cube'], dtype=np.double)
        effIntTime = cube_dict['effIntTime']
        self.wvl_bin_edges = cube['WvlBinEdges']
        # add third dimension to effIntTime for broadcasting
        effIntTime = np.reshape(effIntTime, np.shape(effIntTime) + (1,))
        # put cube into counts/s in each pixel
        cube /= effIntTime
        light_curve = lightcurve.lightcurve() #TODO figure out what this is supposed to do
        self.flux_spectrum = np.empty(self.cfg.nWvlBins, dtype=float)
        self.sky_spectrum = np.zeros(self.cfg.nWvlBins, dtype=float)
        for i in np.arange(self.cfg.nWvlBins):
            frame = cube[:, :, i]
            if self.cfg.photometry == 'aperture':
                flux_dict = light_curve.perform_photometry(self.cfg.photometry, frame, [[self.centroid_col, self.centroid_row]],
                                             expTime=None, aper_radius=self.aperture, annulus_inner=self.annulus_inner,
                                             annulus_outer=self.annulus_outer, interpolation="linear")
                self.flux_spectrum[i] = flux_dict['flux']
                self.sky_spectrum[i] = flux_dict['sky_flux']
            else:
                # default or if 'PSF' is specified
                flux_dict = light_curve.perform_photometry(self.cfg.photometry, frame, [[self.centroid_col, self.centroid_row]],
                                             expTime=None, aper_radius=self.aperture)
                self.flux_spectrum[i] = flux_dict['flux']
        self.flux_spectrum = self.flux_spectrum / self.cfg.energyBinWidth / self.cfg.collecting_area  # spectrum now in counts/s/Angs/cm^2
        self.sky_spectrum = self.sky_spectrum / self.cfg.energyBinWidth / self.cfg.collecting_area
        return self.flux_spectrum, self.sky_spectrum

    def load_relative_spectrum(self):
        '''
        loads the relative spectrum
        :return:
        '''
        for (x, y), res_id in np.ndenumerate(self.cfg.obs.beamImage):
            flux_dict = self.cfg.obs.getPixelSpectrum(x, y, energyBinWidth=self.cfg.energyBinWidth,
                                                      applyWeight=True, firstSec=0, integrationTime=-1)
            self.flux_spectra[x, y], self.flux_effTime[x, y] = flux_dict['spectrum'], flux_dict['effIntTime']
        self.flux_spectra = self.flux_spectra / self.cfg.energyBinWidth / self.flux_effTime
        self.flux_spectrum = self.calculate_median(self.flux_spectra)  # find median of subtracted spectra across whole array
        return self.flux_spectrum

    def load_sky_spectrum(self):
        '''
        loads the sky spectrum
        :return:
        '''
        for (x, y), res_id in np.ndenumerate(self.cfg.sky_obs.beamImage):
            flux_dict = self.cfg.sky_obs.getPixelSpectrum(x, y, energyBinWidth=self.cfg.energyBinWidth,
                                                          applyWeight=True, firstSec=0, integrationTime=-1)
            self.sky_spectra[x, y], self.sky_effTime[x, y] = flux_dict['spectrum'], flux_dict['effIntTime']
        self.sky_spectra = self.sky_spectra / self.cfg.energyBinWidth / self.sky_effTime
        self.sky_spectrum = self.calculate_median(self.sky_spectra)  # find median of subtracted spectra across whole array
        return self.sky_spectrum

    # TODO allow a model standard spectra to be uploaded
    def load_standard_spectrum(self):
        standard = mkidstandards.MKIDStandards()
        std_data = standard.load(self.cfg.object_name)
        std_data = standard.countsToErgs(std_data)  # convert standard star spectrum to ergs/s/Angs/cm^2 for BB fitting and cleaning
        self.std_wvls = np.array(std_data[:, 0])
        self.std_flux = np.array(std_data[:, 1])  # standard star object spectrum in ergs/s/Angs/cm^2

        convX_rev, convY_rev = self.extend_spectrum(self.std_wvls, self.std_flux)
        convX = convX_rev[::-1]  # convolved spectrum comes back sorted backwards, from long wvls to low which screws up rebinning
        convY = convY_rev[::-1]
        # rebin cleaned spectrum to flat cal's wvlBinEdges
        rebin_std_data = rebin(convX, convY, self.wvl_bin_edges)
        rebin_std_wvls = np.array(rebin_std_data[:, 0])
        rebin_std_flux = np.array(rebin_std_data[:, 1])
        if self.cfg.plots:
            #plot final resampled spectrum
            plt.plot(convX, convY*1E15,color='blue')
            plt.step(rebin_std_wvls, rebin_std_flux*1E15, color='black', where='mid')
            plt.legend(['%s Spectrum'%self.cfg.object_name, 'Blackbody Fit', 'Gaussian Convolved Spectrum',
                        'Rebinned Spectrum'], 'upper right', numpoints=1)
            plt.xlabel("Wavelength (\r{A})")
            plt.ylabel("Flux (10$^{-15}$ ergs s$^{-1}$ cm$^{-2}$ \r{A}$^{-1}$)")
            plt.ylim(0.9*min(rebin_std_flux)*1E15, 1.1*max(rebin_std_flux)*1E15)
            plt.savefig(self.cfg.save_directory + 'FluxCal_StdSpectrum_%s.eps'%self.cfg.object_name, format='eps')
        # convert standard spectrum back into counts/s/angstrom/cm^2
        rebin_star_data = standard.ergsToCounts(rebin_std_data)
        self.binned_std_spectrum = np.array(rebin_star_data[:, 1])

    def extend_spectrum(self, x, y):
        '''
        BB Fit to extend standard spectrum to  1500 nm
        '''

        fraction = 1.0 / 3.0
        nirX = np.arange(int(x[(1.0 - fraction) * len(x)]), 20000)
        T, nirY = fitBlackbody(x, y, fraction=fraction, newWvls=nirX, tempGuess=5600)

        extended_wvl = np.concatenate((x, nirX[nirX > max(x)]))
        extended_flux = np.concatenate((y, nirY[nirX > max(x)]))

        getLogger(__name__).info('Loading in wavecal solution to get median energy resolution R')
        sol_file = [f for f in os.listdir(self.cfg.save_directory) if f.endswith('.npz') and f.startswith('wavcal')]  # TODO have a better way to pull out the wavecal solution file
        sol = mkidpipeline.calibration.wavecal.Solution(str(self.cfg.save_directory) + "/" + sol_file[0])
        r_list, resid = sol.find_resolving_powers()
        r = np.median(np.nanmedian(r_list, axis=0))

        # Gaussian convolution to smooth std spectrum to MKIDs median resolution
        newX, newY = gaussianConvolution(extended_wvl, extended_flux, xEnMin=0.005, xEnMax=6.0, xdE=0.001,
                                         fluxUnits="lambda", r=r, plots=False)
        return newX, newY

    def calculate_specweights(self):
        """
        Calculate the sensitivity spectrum: the weighting factors that correct the flat calibrated spectra to the real spectra

        For relative calibration:
        First subtract sky spectrum from MKID observed spectrum. Then take median of this spectrum as it should be identical
        across the array, assuming the flat cal has done its job. Then divide this into the known spectrum of the object.

        For absolute calibration:
        self.flux_spectra already has sky subtraction included. Simply divide this spectrum into the known standard spectrum.
        """
        if self.cfg.photometry == 'relative':
            self.subtracted_spectrum = self.flux_spectrum - self.sky_spectrum
            norm_wvl = 5500  # Angstroms. Choose an arbitrary wvl to normalize the relative correction at
            ind = np.where(self.cfg.wvl_bin_edges >= norm_wvl)[0][0] - 1
            self.subtracted_spectrum = self.subtracted_spectrum / (self.subtracted_spectrum[ind])  # normalize
            # normalize treated Std spectrum while we are at it
            self.binned_spectrum = self.binned_spectrum / (self.binned_spectrum[ind])

        # Calculate FluxCal factors
        self.flux_factors = self.binned_spectrum / self.subtracted_spectrum

        self.flux_flags = np.empty(np.shape(self.flux_factors), dtype='int')
        self.flux_flags.fill(pixelflags.speccal['good'])  # Initialise flag array filled with 'good' flags

        self.flux_flags[self.flux_factors == np.inf] = pixelflags.speccal['infWeight']
        self.flux_factors[self.flux_factors == np.inf] = 1.0
        self.flux_flags[np.isnan(self.flux_factors)] = pixelflags.speccal['nanWeight']
        self.flux_factors[np.isnan(self.flux_factors)] = 1.0
        self.flux_flags[self.flux_factors <= 0] = pixelflags.speccal['LEzeroWeight']
        self.flux_factors[self.flux_factors <= 0] = 1.0

    def calculate_median(self, spectra):
        spectra2d = np.reshape(spectra,[self.cfg.xpix*self.cfg.ypix, self.cfg.nWvlBins])
        wvl_median = np.zeros(self.cfg.nWvlBins, dtype=float)
        for iWvl, wvl in enumerate(self.cfg.nWvlBins):
            spectrum = spectra2d[:, iWvl]
            good_spectrum = spectrum[spectrum != 0]#dead pixels need to be taken out before calculating medians
            wvl_median[iWvl] = np.median(good_spectrum)
        return wvl_median


class SpectralSolution(object):
    def __init__(self, file_path=None, fit_array=None, configuration=None, beam_map=None,
                 beam_map_flags=None, solution_name='spectral_solution'):
        # default parameters
        self._parse = True
        # load in arguments
        self._file_path = os.path.abspath(file_path) if file_path is not None else file_path
        self.fit_array = fit_array
        self.beam_map = beam_map.astype(int) if isinstance(beam_map, np.ndarray) else beam_map
        self.beam_map_flags = beam_map_flags
        self.cfg = configuration
        # if we've specified a file load it without overloading previously set arguments
        if self._file_path is not None:
            self.load(self._file_path, overload=False)
        # if not finish the init
        else:
            self.name = solution_name  # use the default or specified name for saving
            self.npz = None  # no npz file so all the properties should be set
            self._finish_init()

    def save(self, save_name=None):
        """Save the solution to a file. The directory is given by the configuration."""
        if save_name is None:
            save_path = os.path.join(self.cfg.out_directory, self.name)
        else:
            save_path = os.path.join(self.cfg.out_directory, save_name)
        if not save_path.endswith('.npz'):
            save_path += '.npz'
        # make sure the configuration is pickleable if created from __main__
        if self.cfg.__class__.__module__ == "__main__":
            from mkidpipeline.calibration.spectralcal import Configuration
            self.cfg = Configuration(self.cfg.configuration_path)

        getLogger(__name__).info("Saving spectrophotometric solution to {}".format(save_path))
        np.savez(save_path, fit_array=self.weight_array, configuration=self.cfg, beam_map=self.beam_map,
                 beam_map_flags=self.beam_map_flags)
        self._file_path = save_path  # new file_path for the solution

    def plot_summary(self):
        #TODO write this function
        pass

def load_solution(sc, singleton_ok=True):
    """sc is a solution filename string, a SpectralSolution object, or a mkidpipeline.config.MKIDSpectraldataDescription"""
    global _loaded_solutions
    if not singleton_ok:
        raise NotImplementedError('Must implement solution copying')
    if isinstance(sc, SpectralSolution):
        return sc
    if isinstance(sc, mkidpipeline.config.MKIDSpectraldataDescription):
        sc = mkidpipeline.config.spectralcal_id(sc.id)+'.npz'
    sc = sc if os.path.isfile(sc) else os.path.join(mkidpipeline.config.config.paths.database, sc)
    try:
        return _loaded_solutions[sc]
    except KeyError:
        _loaded_solutions[sc] = SpectralSolution(file_path=sc)
    return _loaded_solutions[sc]


def fetch(solution_descriptors, config=None, ncpu=None, remake=False, **kwargs):
    cfg = mkidpipeline.config.config if config is None else config
    solutions = []
    for sd in solution_descriptors:
        sf = os.path.join(cfg.paths.database, mkidpipeline.config.spectralcal_id(sd.id)+'.npz')
        if os.path.exists(sf) and not remake:
            solutions.append(load_solution(sf))
        else:
            if 'spectralcal' not in cfg:
                scfg = mkidpipeline.config.load_task_config(pkg.resource_filename(__name__, 'spectralcal.yml'))
            else:
                scfg = cfg.copy()
            scfg.register('start_time', [x.start for x in sd.data], update=True)
            scfg.register('exposure_time', [x.duration for x in sd.data], update=True)
            if ncpu is not None:
                scfg.update('ncpu', ncpu)
            cal = SpectralCalibrator(scfg, solution_name=sf)
            cal.run(**kwargs)
            # solutions.append(load_solution(sf))  # don't need to reload from file
            solutions.append(cal.solution)  # TODO: solutions.append(load_solution(cal.solution))

    return solutions
