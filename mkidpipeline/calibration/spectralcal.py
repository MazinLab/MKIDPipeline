"""
Author: Seth Meeker        Date:November 21, 2012

Edited for MEC by: Sarah Steiger    Date: April 1, 2020

Opens MKID observation of a spectrophotometric standard star and associated wavelength cal file,
reads in all photons and converts to energies.
Bins photons to generate a spectrum, then divides this into the known spectrum (published)
of the object to create a sensitivity curve.
This curve is then written out to the obs file as spectral weights

Flags are associated with each pixel - see mkidcore/pixelFlags.py for descriptions.

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

    def __init__(self, configfile=None, start_time=tuple(), intTime=tuple(), beammap=None,
                 h5dir='', outdir='', bindir='', savedir='', h5_file_name='', object_name='', photometry='',
                 energy_bin_width=tuple(), wvlStart=tuple(), wvlStop=tuple(), energy_start=tuple(), energy_stop=tuple(),
                 nWvlBins=tuple(), dead_time=tuple(), plots=True):

        # parse arguments
        self.configuration_path = configfile
        self.h5_directory = h5dir
        self.out_directory = outdir
        self.bin_directory = bindir
        self.save_directory = savedir
        self.object_name = object_name
        self.start_time = list(map(int, start_time))
        self.h5_file_name = list(map(str, h5_file_name))
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
        self.flux_spectrum = None
        self.sky_spectrum = None
        self.photometry = str(photometry)
        self.std_wvls = np.array()
        self.std_flux = np.array()
        self.flux_spectrum = np.array()
        self.sky_spectrum = np.array()
        self.binned_std_spectrum = np.array()
        self.wvl_bin_edges = np.array()
        self.plots = bool(plots)
        if self.configuration_path is not None:
            # load in the configuration file
            cfg = mkidcore.config.load(self.configuration_path)
            self.ncpu = cfg.ncpu
            self.beammap = cfg.beammap
            self.xpix = self.beammap.ncols
            self.ypix = self.beammap.nrows
            self.beam_map_path = cfg.beammap.file
            self.wvlStart = cfg.instrument.wvl_start * 10.0 # in angstroms
            self.wvlStop = cfg.instrument.wvl_stop * 10.0 # in angstroms
            self.energyStart = (c.h * c.c) / (self.wvlStart * 10**(-10) * c.e)
            self.energyStop = (c.h * c.c) / (self.wvlStop * 10**(-10) * c.e)
            self.nWvlBins = (self.energyStop - self.energyStart) / self.energyBinWidth
            self.h5_directory = cfg.paths.out
            self.out_directory = cfg.paths.database
            self.bin_directory = cfg.paths.data
            self.save_directory = self.cfg.paths.database
            self.start_time = list(map(int, cfg.spectralcal.start))
            self.dead_time = cfg.instrument.deadtime
            self.object_name = cfg.spectralcal.object_name
            self.energyBinWidth = cfg.instrument.energy_bin_width
            self.h5_directory = cfg.paths.out
            self.photometry = cfg.spectralcal.photometry_type
            self.plots = cfg.spectralcal.plots
            self.intTime = cfg.spectralcal.duration if cfg.spectralcal.duration else (cfg.spectralcal.start - cfg.spectralcal.stop)
            try:
                self.h5_file_name = [os.path.join(self.h5_directory, cfg.h5_file_name)]
            except KeyError:
                self.h5_file_name = [os.path.join(self.h5_directory, str(cfg.h5_file_name) + '.h5')]
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
        getLogger(__name__).info("Computing Factors for FlatCal")

    def run(self, verbose=False, parallel=None, save=True, plot=None):
        """
        Compute the spectrophotometric calibration for the data specified in the configuration
        object. This method runs load_MEC_spectrum(), load_standard_spectrum(), and
        calculate_specweightd() sequentially.

        Args:
            verbose: a boolean specifying whether to print a progress bar to the stdout.
            parallel: a boolean specifying whether to use more than one core in the
                      computation.
            save: a boolean specifying if the result will be saved.
            plot: a boolean specifying if a summary plot for the computation will be
                  saved.
        """
        try:
            getLogger(__name__).info("Loading Spectrum from MEC")
            self._run("load_MEC_spectrum", parallel=parallel, verbose=verbose)
            if self._shared_tables is not None:
                del self._shared_tables
                self._shared_tables = None
            getLogger(__name__).info("Loading Standard Spectrum")
            self._run("load_standard_spectrum", parallel=parallel, verbose=verbose)
            getLogger(__name__).info("Calculating Spectrophotometric Weights")
            self._run("calculate_specweights", parallel=parallel, verbose=verbose)
            if save:
                self.solution.save(save_name=save if isinstance(save, str) else None)
            if plot or (plot is None and self.cfg.summary_plot):
                save_name = self.solution_name.rpartition(".")[0] + ".pdf"
                self.solution.plot_summary(save_name=save_name)
        except KeyboardInterrupt:
            getLogger(__name__).info("Keyboard shutdown requested ... exiting")

    def _run(self, method, pixels=None, wavelengths=None, verbose=False, parallel=True):
        getattr(self, method)(pixels=pixels, wavelengths=wavelengths, verbose=verbose)


    def load_MEC_spectrum(self):
        '''
         Extract the MEC measured spectrum of the spectrophotometric standard by breaking data into spectral cube
         and performing photometry (aperture or psf) on each spectral frame
         '''
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

        for i in np.arange(self.nWvlBins):
            frame = cube[:, :, i]
            if self.photometry == 'aperture':
                flux_dict = light_curve.perform_photometry(self.photometry, frame, [[self.centroid_col, self.centroid_row]],
                                             expTime=None, aper_radius=self.aperture, annulus_inner=self.annulus_inner,
                                             annulus_outer=self.annulus_outer, interpolation="linear")
                self.flux_spectrum[i] = flux_dict['flux']
                self.sky_spectrum[i] = flux_dict['sky_flux']
            else:
                flux_dict = light_curve.perform_photometry(self.photometry, frame, [[self.centroid_col, self.centroid_row]],
                                             expTime=None, aper_radius=self.aperture)
                self.flux_spectrum[i] = flux_dict['flux']

        self.flux_spectrum = self.flux_spectrum / self.bin_widths / self.collecting_area  # spectrum now in counts/s/Angs/cm^2
        self.sky_spectrum = self.sky_spectrum / self.bin_widths / self.collecting_area

        return self.flux_spectrum, self.sky_spectrum

    def load_relativespectrum(self):
        self.flux_spectra = [[[] for i in np.arange(self.nCol)] for j in np.arange(self.nRow)]
        self.flux_effTime = [[[] for i in np.arange(self.nCol)] for j in np.arange(self.nRow)]
        for iRow in np.arange(self.nRow):
            for iCol in np.arange(self.nCol):
                count = self.flux_file.getPixelCount(iRow, iCol)
                flux_dict = self.flux_file.getPixelSpectrum(iRow, iCol, weighted=True, firstSec=0, integrationTime=-1)
                self.flux_spectra[iRow][iCol], self.flux_effTime[iRow][iCol] = flux_dict['spectrum'], flux_dict['effIntTime']
        self.flux_spectra = np.array(self.flux_spectra)
        self.flux_effTime = np.array(self.flux_effTime)
        deadtime_corr = self.get_deadtimecorrection(self.flux_file)
        self.flux_spectra = self.flux_spectra / self.bin_widths / self.flux_effTime * deadtime_corr
        self.flux_spectrum = self.calculate_median(self.flux_spectra)  # find median of subtracted spectra across whole array
        return self.flux_spectrum

    def load_skyspectrum(self):
        self.sky_spectra = [[[] for i in np.arange(self.nCol)] for j in np.arange(self.nRow)]
        self.sky_effTime = [[[] for i in np.arange(self.nCol)] for j in np.arange(self.nRow)]
        for iRow in np.arange(self.nRow):
            for iCol in np.arange(self.nCol):
                count = self.sky_file.getPixelCount(iRow, iCol)
                sky_dict = self.sky_file.getPixelSpectrum(iRow, iCol, weighted=True, firstSec=0, integrationTime=-1)
                self.sky_spectra[iRow][iCol], self.sky_effTime[iRow][iCol] = sky_dict['spectrum'], sky_dict['effIntTime']
        self.sky_spectra = np.array(self.sky_spectra)
        self.sky_effTime = np.array(self.sky_effTime)
        deadtime_corr = self.get_deadtimecorrection(self.sky_file)
        self.sky_spectra = self.sky_spectra / self.binWidths / self.sky_effTime * deadtime_corr
        self.sky_spectrum = self.calculate_median(self.sky_spectra)  # find median of subtracted spectra across whole array
        return self.sky_spectrum

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
        self.subtracted_spectrum = self.flux_spectrum - self.sky_spectrum
        self.subtracted_spectrum = np.array(self.subtracted_spectrum,
                                           dtype=float)  # cast as floats so division does not fail later

        if self.method == 'relative':
            norm_wvl = 5500  # Angstroms. Choose an arbitrary wvl to normalize the relative correction at
            ind = np.where(self.wvlBinEdges >= norm_wvl)[0][0] - 1
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
        spectra2d = np.reshape(spectra,[self.nRow*self.nCol,self.nWvlBins])
        wvl_median = np.empty(self.nWvlBins,dtype=float)
        for iWvl in np.arange(self.nWvlBins):
            spectrum = spectra2d[:,iWvl]
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
