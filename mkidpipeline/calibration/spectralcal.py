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
from mkidpipeline.utils.photometry import perform_photometry
from mkidpipeline.utils.standardspectra import StandardSpectrum
from mkidcore.corelog import getLogger
import mkidcore.corelog
import scipy.constants as c
import mkidpipeline
from mkidcore.objects import Beammap
import matplotlib.pyplot as plt
# from mkidpipeline.utils.photometry import *


class Configuration(object):
    """Configuration class for the spectrophotometric calibration analysis."""
    yaml_tag = u'!spectralcalconfig'

    def __init__(self, configfile=None, start_times=tuple(), sky_start_times=tuple(), intTimes=tuple(), beammap=None,
                 h5dir='', outdir='', bindir='', savedir='', h5_file_names='', sky_file_names='', object_name='', ra=None,
                 dec=None, photometry='', method='', energy_bin_width=0.01, wvlStart=850, wvlStop=1375, energy_start=1.46,
                 energy_stop=0.9, nWvlBins=50, dead_time=0.000001, collecting_area=0, plots=True, url_path=''):
        # parse arguments
        self.configuration_path = configfile
        self.h5_directory = h5dir
        self.out_directory = outdir
        self.bin_directory = bindir
        self.save_directory = savedir
        self.url_path=url_path
        self.object_name = object_name
        self.ra = ra
        self.dec = dec
        self.start_times = list(map(int, start_times))
        self.sky_start_times = list(map(int, sky_start_times))
        self.h5_file_names = list(map(str, h5_file_names))
        self.sky_file_names = list(map(str, sky_file_names))
        self.dead_time = float(dead_time)
        self.intTimes = list(map(float, intTimes))
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
        self.binned_std_spectrum = None
        self.wvl_bin_edges = None
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
            self.url_path = cfg.spectralcal.standard_url
            self.start_times = list(map(int, cfg.start_times))
            self.sky_start_times = list(map(int, cfg.sky_start_times))
            self.dead_time = cfg.instrument.deadtime
            self.object_name = cfg.object_name
            self.ra = cfg.ra if cfg.ra else None
            self.dec = cfg.dec if cfg.dec else None
            self.energyBinWidth = cfg.instrument.energy_bin_width
            self.photometry = cfg.spectralcal.photometry_type
            self.collecting_area = cfg.spectralcal.collecting_area
            self.save_plots = cfg.spectralcal.save_plots
            self.intTimes = cfg.exposure_times
            try:
                self.h5_file_names = [os.path.join(self.h5_directory, f) for f in self.start_times]
                self.sky_file_names = [os.path.join(self.h5_directory, f) for f in self.sky_start_times]
            except TypeError:
                self.h5_file_names = [os.path.join(self.h5_directory, str(t) + '.h5') for t in self.start_times]
                self.sky_file_names = [os.path.join(self.h5_directory, str(t) + '.h5') for t in self.sky_start_times]
            try:
                self.templar_configuration_path = cfg.templar.file
            except KeyError:
                if self.beammap.frequencies is None:
                    getLogger(__name__).warning('Beammap loaded without frequencies and no templar config specified.')
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
    def __init__(self, configuration, solution_name='solution.npz', _shared_tables=None, weight_array=None, main=True):

        self.cfg = Configuration(configuration) if not isinstance(configuration, Configuration) else configuration
        self.solution_name = solution_name
        self.solution = SpectralSolution(configuration=self.cfg, weight_array=weight_array, beam_map=self.cfg.beammap.residmap,
                                         beam_map_flags=self.cfg.beammap.flagmap, solution_name=self.solution_name)
        self.obs = [ObsFile(f) for f in self.cfg.h5_file_names]
        self.sky_obs = [ObsFile(f) for f in self.cfg.sky_file_names]
        self.flux_spectrum = np.zeros((self.cfg.xpix, self.cfg.ypix))
        self.flux_spectra = np.zeros((self.cfg.xpix, self.cfg.ypix))
        self.flux_effTime = np.zeros((self.cfg.xpix, self.cfg.ypix))
        self.wvl_bin_edges = self.makeWvlBins()
        self.sky_spectra = np.zeros((self.cfg.xpix, self.cfg.ypix))
        self.sky_spectrum = np.zeros((self.cfg.xpix, self.cfg.ypix))
        self.sky_effTime = np.zeros((self.cfg.xpix, self.cfg.ypix))
        self.std_spectrum = None
        self.std_wvls = None
        self.std_flux = None
        self.rebin_std_wvls = None
        self.rebin_std_flux = None

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
            if self.cfg.photometry == 'relative':
                getLogger(__name__).info('Performing relative photometry')
                self.load_relative_spectrum()
                getLogger(__name__).info('Flux spectrum loaded')
                self.load_sky_spectrum()
                getLogger(__name__).info('Sky spectrum loaded ')
            elif self.cfg.photometry == 'PSF' or self.cfg.photometry == 'aperture':
                getLogger(__name__).info('Extracting point source spectrum ')
                self.load_absolute_spectrum()
            getLogger(__name__).info("Loading Standard Spectrum")
            self.load_standard_spectrum()
            getLogger(__name__).info("Calculating Spectrophotometric Weights")
            self.calculate_specweights()
            if save:
                self.solution.save(save_name=self.solution_name if isinstance(self.solution_name, str) else None)
            if plot or (plot is None and self.cfg.summary_plot):
                save_name = self.solution_name.rpartition(".")[0] + ".pdf"
                self.solution.plot_summary(save_name=save_name)
        except KeyboardInterrupt:
            getLogger(__name__).info("Keyboard shutdown requested ... exiting")

    def makeWvlBins(self):
        """
        returns an array of wavelength bin edges, with a fixed energy bin width
        withing the limits given in wvlStart and wvlStop

        Returns:
            an array of wavelength bin edges that can be used with numpy.histogram(bins=wvlBinEdges)
        """

        # Calculate upper and lower energy limits from wavelengths
        # Note that start and stop switch when going to energy

        nWvlBins = int((self.cfg.energyStart - self.cfg.energyStop) / self.cfg.energyBinWidth)
        # Construct energy bin edges
        energyBins = np.linspace(self.cfg.energyStart, self.cfg.energyStop, nWvlBins + 1)
        # Convert back to wavelength
        wvlBinEdges = np.array(c.h * c.c * 1.e10 / (energyBins * c.e))
        return wvlBinEdges

    def load_absolute_spectrum(self):
        '''
         Extract the MEC measured spectrum of the spectrophotometric standard by breaking data into spectral cube
         and performing photometry (aperture or psf) on each spectral frame
         '''
        getLogger(__name__).info('performing {} photometry on MEC spectrum').format(self.cfg.photometry)
        cube_dict = self.cfg.obs.getSpectralCube(firstSec=self.cfg.start_time, integrationTime=self.cfg.intTime,
                                                 applyWeight=True, energyBinWidth=self.cfg.energyBinWidth)
        cube = np.array(cube_dict['cube'], dtype=np.double)
        effIntTime = cube_dict['effIntTime']
        self.wvl_bin_edges = cube['WvlBinEdges']
        # add third dimension to effIntTime for broadcasting
        effIntTime = np.reshape(effIntTime, np.shape(effIntTime) + (1,))
        # put cube into counts/s in each pixel
        cube /= effIntTime
        self.flux_spectrum = np.empty(self.cfg.nWvlBins, dtype=float)
        self.sky_spectrum = np.zeros(self.cfg.nWvlBins, dtype=float)
        for i in np.arange(self.cfg.nWvlBins):
            # perform photometry on every wavelength bin
            frame = cube[:, :, i]
            if self.cfg.photometry == 'aperture':
                flux_dict = perform_photometry(self.cfg.photometry, frame, [[self.centroid_col, self.centroid_row]],
                                             expTime=None, aper_radius=self.aperture, annulus_inner=self.annulus_inner,
                                             annulus_outer=self.annulus_outer, interpolation="linear")
                self.flux_spectrum[i] = flux_dict['flux']
                self.sky_spectrum[i] = flux_dict['sky_flux']
            else:
                # default or if 'PSF' is specified
                flux_dict = perform_photometry(self.cfg.photometry, frame, [[self.centroid_col, self.centroid_row]],
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

    def load_standard_spectrum(self):
        standard = StandardSpectrum(save_path=self.cfg.save_directory, url_path=self.cfg.url_path,
                                    object_name=self.cfg.object_name[0], object_ra=self.cfg.ra[0],
                                    object_dec=self.cfg.dec[0])
        self.std_wvls, self.std_flux = standard.get()  # standard star object spectrum in ergs/s/Angs/cm^2
        ind = np.where(self.std_wvls > self.cfg.wvlStart)
        #TODO confirm spectra is in the correct units
        convX_rev, convY_rev = self.extend_spectrum(self.std_wvls[ind], self.std_flux[ind])
        convX = convX_rev[::-1]  # convolved spectrum comes back sorted backwards, from long wvls to low which screws up rebinning
        convY = convY_rev[::-1]
        # rebin cleaned spectrum to flat cal's wvlBinEdges
        rebin_std_data = rebin(convX, convY, self.wvl_bin_edges[self.wvl_bin_edges<convX.max()])
        # convert standard spectrum back into counts/s/angstrom/cm^2
        rebin_std_data = standard.ergs_to_counts(rebin_std_data)
        self.rebin_std_flux = np.array(rebin_std_data[:, 1])
        self.rebin_std_wvls = np.array(rebin_std_data[:, 0])

    def extend_spectrum(self, x, y):
        '''
        BB Fit to extend standard spectrum to 1500 nm and to convolve it with a gaussian kernel
        '''

        fraction = 1.0 / 3.0
        nirX = np.arange(int(x[int((1.0 - fraction) * len(x))]), self.cfg.wvlStop)
        T, nirY = fitBlackbody(x, y, fraction=fraction, newWvls=nirX, tempGuess=5600)

        extended_wvl = np.concatenate((x, nirX[nirX > max(x)]))
        extended_flux = np.concatenate((y, nirY[nirX > max(x)]))

        getLogger(__name__).info('Loading in wavecal solution to get median energy resolution R')
        sol_file = [f for f in os.listdir(self.cfg.save_directory) if f.endswith('.npz') and f.startswith('wavcal')]  # TODO have a better way to pull out the wavecal solution file
        sol = mkidpipeline.calibration.wavecal.Solution(str(self.cfg.save_directory) + "/" + sol_file[0])
        r_list, resid = sol.find_resolving_powers()
        r = np.median(np.nanmedian(r_list, axis=0))
        # Gaussian convolution to smooth std spectrum to MKIDs median resolution
        newX, newY = gaussianConvolution(extended_wvl, extended_flux, xEnMin=self.cfg.energyStop,
                                         xEnMax=self.cfg.energyStart, fluxUnits="lambda", r=r, plots=False)
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
    def __init__(self, file_path=None, weight_array=None, configuration=None, beam_map=None,
                 beam_map_flags=None, solution_name='spectral_solution'):
        # default parameters
        self._parse = True
        # load in arguments
        self._file_path = os.path.abspath(file_path) if file_path is not None else file_path
        self.weight_array = weight_array
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
        np.savez(save_path, weight_array=self.weight_array, configuration=self.cfg, beam_map=self.beam_map,
                 beam_map_flags=self.beam_map_flags)
        self._file_path = save_path  # new file_path for the solution

    def plot_summary(self, save_name='summary_plot.pdf'):
        #TODO write this function
        pass

def load_solution(sc, singleton_ok=True):
    """sc is a solution filename string, a SpectralSolution object, or a mkidpipeline.config.MKIDSpectralReference"""
    global _loaded_solutions
    if not singleton_ok:
        raise NotImplementedError('Must implement solution copying')
    if isinstance(sc, SpectralSolution):
        return sc
    if isinstance(sc, mkidpipeline.config.MKIDSpectralReference):
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
            scfg.register('start_times', [x.start for x in sd.data], update=True)
            scfg.register('sky_start_times', [x.start for x in sd.sky_data], update=True)
            scfg.register('exposure_times', [x.duration for x in sd.data], update=True)
            scfg.register('ra', [x.ra for x in sd.data], update=True)
            scfg.register('dec', [x.dec for x in sd.data], update=True)
            scfg.register('object_name', [x.name for x in sd.data], update=True)
            if ncpu is not None:
                scfg.update('ncpu', ncpu)
            cal = SpectralCalibrator(scfg, solution_name=sf)
            cal.run(**kwargs)
            # solutions.append(load_solution(sf))  # don't need to reload from file
            solutions.append(cal.solution)  # TODO: solutions.append(load_solution(cal.solution))

    return solutions
