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
import numpy as np
from mkidcore import pixelflags
from mkidpipeline.hdf.photontable import Photontable
from mkidpipeline.utils.utils import rebin, gaussianConvolution, fitBlackbody
from mkidcore.corelog import getLogger
import mkidcore.corelog
import scipy.constants as c
from astropy import units as u
import mkidpipeline
import matplotlib.pyplot as plt
import urllib.request as request
from urllib.error import URLError
import shutil
from contextlib import closing
from astroquery.sdss import SDSS
import astropy.coordinates as coord
from specutils import Spectrum1D
from photutils.detection import IRAFStarFinder
from photutils.psf import IntegratedGaussianPRF, DAOGroup
from photutils.background import MMMBackground, MADStdBackgroundRMS
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import gaussian_sigma_to_fwhm
import scipy.ndimage as ndimage
from photutils import aperture_photometry
from photutils import CircularAperture
from scipy.interpolate import griddata
import pkg_resources as pkg
from astropy.table import Table
import matplotlib.gridspec as gridspec

_loaded_solutions = {}


class Configuration(object):
    """Configuration class for the spectrophotometric calibration analysis."""
    yaml_tag = u'!spectralcalconfig'

    def __init__(self, configfile=None, start_times=tuple(), sky_start_times=tuple(), intTimes=tuple(),
                 h5dir='', savedir='', h5_file_names='', sky_file_names='', object_name='', ra=None, dec=None,
                 photometry='', energy_bin_width=0.01, wvlStart=850, wvlStop=1375, collecting_area=5 * 10**5, summary_plot=True,
                 url_path='', aperture_radius=3, obj_pos=None, sky_pos=None, interpolation='linear'):
        # parse arguments
        self.configuration_path = configfile
        self.h5_directory = h5dir
        self.save_directory = savedir
        self.url_path = url_path
        self.object_name = object_name
        self.ra = ra
        self.dec = dec
        self.start_times = list(map(int, start_times))
        self.sky_start_times = list(map(int, sky_start_times))
        self.h5_file_names = list(map(str, h5_file_names))
        self.sky_file_names = list(map(str, sky_file_names))
        self.intTimes = list(map(float, intTimes))
        self.h5_directory = h5dir
        self.energyBinWidth = float(energy_bin_width)
        self.wvlStart = float(wvlStart)
        self.wvlStop = float(wvlStop)
        self.photometry = str(photometry)
        self.wvl_bin_edges = None
        self.wvl_bin_widths = None
        self.summary_plot = bool(summary_plot)
        self.collecting_area = float(collecting_area)
        self.aperture_radius = aperture_radius
        self.obj_pos = obj_pos
        self.sky_pos = sky_pos
        self.nwvlbins = None
        self.interpolation = interpolation
        self.data = None
        if self.configuration_path is not None:
            # load in the configuration file
            cfg = mkidcore.config.load(self.configuration_path)
            self.ncpu = cfg.ncpu
            self.wvlStart = cfg.instrument.wvl_start * 10.0 # in angstroms
            self.wvlStop = cfg.instrument.wvl_stop * 10.0 # in angstroms
            self.energyStart = (c.h * c.c) / (self.wvlStart * 10**(-10) * c.e)
            self.energyStop = (c.h * c.c) / (self.wvlStop * 10**(-10) * c.e)
            self.energyBinWidth = cfg.instrument.energy_bin_width
            self.nwvlbins = int((self.energyStart - self.energyStop) / self.energyBinWidth)
            self.wvl_bin_widths = np.zeros(self.nwvlbins)
            self.h5_directory = cfg.paths.out
            self.save_directory = cfg.paths.database
            self.url_path = cfg.spectralcal.standard_url
            self.start_times = list(map(int, cfg.start_times))
            self.sky_start_times = list(map(int, cfg.sky_start_times))
            self.object_name = cfg.object_name
            self.ra = cfg.ra if cfg.ra else None
            self.dec = cfg.dec if cfg.dec else None
            self.photometry = cfg.spectralcal.photometry_type
            self.summary_plot = cfg.spectralcal.summary_plot
            self.intTimes = cfg.exposure_times
            self.aperture_radius = float(cfg.spectralcal.aperture_radius)
            self.collecting_area = 5 * 10**5 #TODO check this conversion
            self.data = cfg.data
            self.obj_pos = tuple(float(s) for s in cfg.spectralcal.object_position.strip("()").split(",")) \
                if cfg.spectralcal.object_position else None
            self.sky_pos = tuple(float(s) for s in cfg.spectralcal.sky_position.strip("()").split(","))
            self.interpolation = cfg.spectralcal.interpolation
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
            pass

    @classmethod
    def to_yaml(cls, representer, node):
        d = node.__dict__.copy()
        d.pop('config')
        return representer.represent_mapping(cls.yaml_tag, d)

    @classmethod
    def from_yaml(cls, constructor, node):
        d = mkidcore.config.extract_from_node(constructor, 'configuration_path', node)
        return cls(d['configuration_path'])

    def write(self, file_):
        """Save the configuration to a file"""
        with open(file_, 'w') as f:
            mkidcore.config.yaml.dump(self, f)


class StandardSpectrum:
    """
    replaces the MKIDStandards class from the ARCONS pipeline for MEC.
    """
    def __init__(self, save_path='', url_path=None, object_name=None, object_ra=None, object_dec=None, coords=None,
                 reference_wavelength=5500):
        self.save_dir = save_path
        self.ref_wvl = reference_wavelength
        self.object_name = object_name
        self.ra = object_ra
        self.dec = object_dec
        self.url_path = url_path
        self.coords = coords # SkyCoord object
        self.spectrum_file = None
        self.k = 5.03411259e7

    def get(self):
        """
        function which creates a spectrum directory, populates it with the spectrum file either pulled from the ESO
        catalog, SDSS catalog, a URL, or a specified path to a .txt file and returns the wavelength and flux column in the appropriate units
        :return: wavelengths (Angstroms), flux (erg/s/cm^2/A)
        """
        self.save_dir = create_spectra_directory(save_dir=self.save_dir)
        self.coords = get_coords(object_name=self.object_name, ra=self.ra, dec=self.dec)
        data = self.fetch_spectra()
        return data[:, 0], data[:, 1]

    def fetch_spectra(self):
        '''
        called from get(), searches either a URL, ESO catalog or uses astroquery.SDSS to search the SDSS catalog. Puts
        the retrieved spectrum in a '/spectrum/' folder in self.save_dir
        :return:
        '''
        if self.url_path is not None:
            if self.url_path.endswith('.txt'):
                data = np.loadtxt(self.url_path)
            else:
                self.spectrum_file = fetch_spectra_URL(object_name=self.object_name, url_path=self.url_path,
                                                       save_dir=self.save_dir)
                data = np.loadtxt(self.spectrum_file)
            return data
        else:
            self.spectrum_file = fetch_spectra_ESO(object_name=self.object_name, save_dir=self.save_dir)
            if not self.spectrum_file:
                self.spectrum_file = fetch_spectra_SDSS(object_name=self.object_name, save_dir=self.save_dir,
                                                        coords=self.coords)
                try:
                    data = np.loadtxt(self.spectrum_file)
                    return data
                except ValueError:
                    getLogger(__name__).warning(
                        'Could not find standard spectrum for this object in SDSS or ESO catalog')
                    return self.spectrum_file
            data = np.loadtxt(self.spectrum_file)
            # to convert to the appropriate units if ESO spectra
            data[:, 1] = data[:, 1] * 10**(-16)
            return data

    def counts_to_ergs(self, a):
        '''
        converts units of the spectra from counts to ergs
        :return:
        '''
        a[:, 1] /= (a[:, 0] * self.k)
        return a

    def ergs_to_counts(self, a):
        '''
        converts units of the spectra from ergs to counts
        :return:
        '''
        a[:, 1] *= (a[:, 0] * self.k)
        return a


class SpectralCalibrator(object):
    """
    Opens flux file, prepares standard spectrum, and calculates flux factors for the file.
    Method is provided in param file. If 'relative' is selected, an obs file with standard star defocused over
    the entire array is expected, with accompanying sky file to do sky subtraction.
    If any other method is provided, 'absolute' will be done by default, wherein a point source is assumed
    to be present. The obs file is then broken into spectral frames with photometry (psf or aper) performed
    on each frame to generate the observed spectrum.
    """
    def __init__(self, configuration=None, h5_file_names=None, sky_h5_file_names=None, interpolation='linear',
                 solution_name='solution.npz', curve=None, aperture_radius=3, sky_pos=None, obj_pos=None):
        self.flux_spectrum = None
        self.flux_spectra = None
        self.flux_effTime = None
        self.wvl_bin_edges = None
        self.sky_spectra = None
        self.sky_spectrum = None
        self.sky_effTime = None
        self.std_spectrum = None
        self.std_wvls = None
        self.std_flux = None
        self.rebin_std_wvls = None
        self.rebin_std_flux = None
        self.obs = None
        self.sky_obs = None
        self.interpolation = interpolation
        self.aperture_radius = aperture_radius
        self.obj_pos = obj_pos
        self.sky_pos = sky_pos
        self.wvl_bin_widths = None
        self.curve = curve
        self.image = None
        self.bb_flux = None
        self.bb_wvls = None
        self.conv_wvls = None
        self.conv_flux = None
        self.data = None
        if h5_file_names:
            self.obs = [Photontable(f) for f in h5_file_names]
        if sky_h5_file_names:
            self.sky_obs = [Photontable(f) for f in sky_h5_file_names]
        if configuration:
            self.cfg = Configuration(configuration) if not isinstance(configuration, Configuration) else configuration
            self.obj_pos = self.cfg.obj_pos
            self.sky_pos = self.cfg.sky_pos
            self.aperture_radius = self.cfg.aperture_radius
            self.solution_name = solution_name
            self.interpolation = self.cfg.interpolation
            self.solution = ResponseCurve(configuration=self.cfg, curve=curve,
                                          solution_name=self.solution_name)
            self.obs = [Photontable(f) for f in self.cfg.h5_file_names]
            self.sky_obs = [Photontable(f) for f in self.cfg.sky_file_names]
            self.wvl_bin_widths = self.cfg.wvl_bin_widths
            self.data = self.cfg.data

    def run(self, save=True, plot=None):
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
            # if self.cfg.photometry == 'relative':
            #     getLogger(__name__).info('Performing relative photometry')
            #     self.load_relative_spectrum()
            #     getLogger(__name__).info('Flux spectrum loaded')
            #     self.load_sky_spectrum()
            #     getLogger(__name__).info('Sky spectrum loaded ')
            getLogger(__name__).info('Extracting point source spectrum ')
            self.load_absolute_spectrum()
            getLogger(__name__).info("Loading Standard Spectrum")
            self.load_standard_spectrum()
            getLogger(__name__).info("Calculating Spectrophotometric Response Curve")
            self.calculate_response_curve()
            self.solution = ResponseCurve(configuration=self.cfg, curve=self.curve,
                                          solution_name=self.solution_name)
            if save:
                self.solution.save(save_name=self.solution_name if isinstance(self.solution_name, str) else None)
            if plot or (plot is None and self.cfg.summary_plot):
                save_name = self.solution_name.rpartition(".")[0] + ".pdf"
                self.plot_summary(save_name=save_name)
        except KeyboardInterrupt:
            getLogger(__name__).info("Keyboard shutdown requested ... exiting")

    def load_absolute_spectrum(self):
        """
         Extract the MEC measured spectrum of the spectrophotometric standard by breaking data into spectral cubes
         and performing photometry (aperture or psf) on each spectral frame
         """
        getLogger(__name__).info('performing {} photometry on MEC spectrum'.format(self.cfg.photometry))
        if len(self.obs) == 1:
            cube_dict = self.obs[0].getSpectralCube(integrationTime=self.cfg.intTimes[0],
                                                    applyWeight=True, wvlStart=self.cfg.wvlStart/10,
                                                    wvlStop=self.cfg.wvlStop/10, energyBinWidth=self.cfg.energyBinWidth,
                                                    exclude_flags=pixelflags.PROBLEM_FLAGS)
            cube = np.array(cube_dict['cube'], dtype=np.double)
            effIntTime = cube_dict['effIntTime']
            self.wvl_bin_edges = cube_dict['wvlBinEdges'] * 10  # get this into units of Angstroms
            # add third dimension to effIntTime for broadcasting
            effIntTime = np.reshape(effIntTime, np.shape(effIntTime) + (1,))
            # put cube into counts/s in each pixel
            cube /= effIntTime
        else:
            cube = []
            ref_obs = self.obs[0].getSpectralCube(integrationTime=self.cfg.intTimes[0],
                                                  applyWeight=True, wvlStart=self.cfg.wvlStart/10,
                                                  wvlStop=self.cfg.wvlStop/10, energyBinWidth=self.cfg.energyBinWidth,
                                                  exclude_flags=pixelflags.PROBLEM_FLAGS)
            self.wvl_bin_edges = ref_obs['wvlBinEdges'] * 10
            for wvl in range(len(self.wvl_bin_edges) - 1):
                getLogger(__name__).info(
                    'using wavelength range {} - {}'.format(self.wvl_bin_edges[wvl]/10, self.wvl_bin_edges[wvl+1]/10))
                drizzled = mkidpipeline.drizzler.form(self.data, mode='spatial', wvlMin=self.wvl_bin_edges[wvl]/10,
                                                      wvlMax=self.wvl_bin_edges[wvl+1]/10, pixfrac=0.5, wcs_timestep=1,
                                                      exp_timestep=1, exclude_flags=pixelflags.PROBLEM_FLAGS,
                                                      usecache=False, ncpu=1, derotate=True, align_start_pa=False,
                                                      whitelight=False, debug_dither_plot=False)
                getLogger(__name__).info(('finished image {}/ {}'.format(wvl+1.0, len(self.wvl_bin_edges) - 1)))
                cube.append(drizzled.cps)
            cube = np.moveaxis(cube, 0, -1)
        self.image = np.sum(cube, axis=2)
        n_wvl_bins = len(self.wvl_bin_edges) - 1
        for i, edge in enumerate(self.wvl_bin_edges):
            try:
                self.wvl_bin_widths[i] = self.wvl_bin_edges[i + 1] - self.wvl_bin_edges[i]
            except IndexError:
                pass
        self.flux_spectrum = np.empty(n_wvl_bins, dtype=float)
        self.sky_spectrum = np.empty(n_wvl_bins, dtype=float)
        if self.obj_pos is None:
            getLogger(__name__).info('No coordinate specified for the object. Performing a PSF fit '
                                     'to find the location')
            x, y, flux = psf_photometry(self.image, 3.0)
            self.obj_pos = (x[0], y[0])
            getLogger(__name__).info('Found the object at {}'.format(self.obj_pos))
        for i in np.arange(n_wvl_bins):
            # perform photometry on every wavelength bin
            frame = cube[:, :, i]
            if self.cfg.photometry == 'aperture':
                if self.interpolation is not None:
                    frame = interpolateImage(frame, method=self.interpolation)
                obj_flux, sky_flux = aper_photometry(frame, self.obj_pos, self.sky_pos, self.aperture_radius)
                self.flux_spectrum[i] = obj_flux
                self.sky_spectrum[i] = sky_flux
            else:
                # default or if 'PSF' is specified
                x, y, flux = psf_photometry(cube[:, :, i], 4.0, x0=self.cfg.obj_pos[0], y0=self.cfg.obj_pos[1])
                self.flux_spectrum[i] = flux
        self.flux_spectrum = self.flux_spectrum / self.wvl_bin_widths / self.cfg.collecting_area  # spectrum now in counts/s/Angs/cm^2
        self.sky_spectrum = self.sky_spectrum / self.wvl_bin_widths / self.cfg.collecting_area
        return self.flux_spectrum, self.sky_spectrum

    def load_standard_spectrum(self):
        standard = StandardSpectrum(save_path=self.cfg.save_directory, url_path=self.cfg.url_path,
                                    object_name=self.cfg.object_name[0], object_ra=self.cfg.ra[0],
                                    object_dec=self.cfg.dec[0])
        self.std_wvls, self.std_flux = standard.get()  # standard star object spectrum in ergs/s/Angs/cm^2
        ind = np.where(self.std_wvls > self.cfg.wvlStart)
        convX_rev, convY_rev = self.extend_spectrum(self.std_wvls[ind], self.std_flux[ind])
        # convolved spectrum comes back sorted backwards, from long wvls to low which screws up rebinning
        self.conv_wvls = convX_rev[::-1]
        self.conv_flux = convY_rev[::-1]

        # rebin cleaned spectrum to flat cal's wvlBinEdges
        rebin_std_data = rebin(self.conv_wvls, self.conv_flux, self.wvl_bin_edges)

        # convert standard spectrum back into counts/s/angstrom/cm^2
        rebin_std_data = standard.ergs_to_counts(rebin_std_data)
        self.rebin_std_flux = np.array(rebin_std_data[:, 1])
        self.rebin_std_wvls = np.array(rebin_std_data[:, 0])

    def extend_spectrum(self, x, y):
        """
        BB Fit to extend standard spectrum to 1500 nm and to convolve it with a gaussian kernel
        """

        fraction = 1.0 / 3.0
        nirX = np.arange(int(x[int((1.0 - fraction) * len(x))]), self.cfg.wvlStop)
        T, nirY = fitBlackbody(x, y, fraction=fraction, newWvls=nirX)
        if np.any(x >= self.cfg.wvlStop):
            self.bb_wvls = x
            self.bb_flux = y
        else:
            self.bb_wvls = np.concatenate((x, nirX[nirX > max(x)]))
            self.bb_flux = np.concatenate((y, nirY[nirX > max(x)]))

        getLogger(__name__).info('Loading in wavecal solution to get median energy resolution R')
        sol_file = [f for f in os.listdir(self.cfg.save_directory) if f.endswith('.npz') and f.startswith('wavcal')]  # TODO have a better way to pull out the wavecal solution file
        sol = mkidpipeline.calibration.wavecal.Solution(str(self.cfg.save_directory) + "/" + sol_file[0])
        r_list, resid = sol.find_resolving_powers(cache=True)
        r = np.median(np.nanmedian(r_list, axis=0))

        # Gaussian convolution to smooth std spectrum to MKIDs median resolution
        newX, newY = gaussianConvolution(self.bb_wvls, self.bb_flux, xEnMin=self.cfg.energyStop,
                                         xEnMax=self.cfg.energyStart, fluxUnits="lambda", r=r, plots=False)
        return newX, newY

    def calculate_response_curve(self):
        """
        Divide the MEC Spectrum by the rebinned and gaussian conivlved standard spectrum
        """
        curve_x = self.rebin_std_wvls
        curve_y = self.rebin_std_flux/self.flux_spectrum
        self.curve = np.vstack((curve_x, curve_y))
        return self.curve

    def plot_summary(self, save_name='summary_plot.pdf'):
        figure = plt.figure()
        gs = gridspec.GridSpec(2, 2)
        axes_list = np.array([figure.add_subplot(gs[0, 0]), figure.add_subplot(gs[0, 1]),
                              figure.add_subplot(gs[1, 0]), figure.add_subplot(gs[1, 1])])
        axes_list[0].imshow(self.image)
        circle = plt.Circle((self.obj_pos[0], self.obj_pos[1]), self.aperture_radius, fill=False, color='r')
        axes_list[0].add_artist(circle)
        axes_list[0].set_title('MKID Instrument Image of Standard', size=8)

        axes_list[1].step(self.std_wvls, self.std_flux, where='mid', label='{} Spectrum'.format(self.cfg.object_name[0]))
        axes_list[1].step(self.bb_wvls, self.bb_flux, where='mid', label='BB fit')
        axes_list[1].step(self.conv_wvls, self.conv_flux, where='mid', label='Convolved Spectrum')
        axes_list[1].set_xlabel('Wavelength (A)')
        axes_list[1].set_ylabel('Flux (erg/s/cm^2)')
        axes_list[1].legend(loc='upper right', prop={'size': 6})

        axes_list[2].step(self.rebin_std_wvls, self.rebin_std_flux, where='mid',
                          label='Rebinned Standard Histogram of Object')
        axes_list[2].step(self.rebin_std_wvls, self.flux_spectrum, where='mid',
                          label='MKID Histogram of Object')
        axes_list[2].set_title('Object Histograms', size=8)
        axes_list[2].legend(loc='upper right', prop={'size': 6})
        axes_list[2].set_xlabel('Wavelength (A)')
        axes_list[2].set_ylabel('counts/s/cm^2/A')

        axes_list[3].plot(self.curve[0], self.curve[1])
        axes_list[3].set_title('Response Curve', size=8)
        plt.tight_layout()
        plt.savefig(save_name)
        return axes_list


class ResponseCurve(object):
    def __init__(self, file_path=None, curve=None, configuration=None, solution_name='spectral_solution'):
        # default parameters
        self._parse = True
        # load in arguments
        self._file_path = os.path.abspath(file_path) if file_path is not None else file_path
        self.curve = curve
        self.cfg = configuration
        # if we've specified a file load it without overloading previously set arguments
        if self._file_path is not None:
            self.load(self._file_path)
        # if not finish the init
        else:
            self.name = solution_name  # use the default or specified name for saving
            self.npz = None  # no npz file so all the properties should be set

    def save(self, save_name=None):
        """Save the solution to a file. The directory is given by the configuration."""
        if save_name is None:
            save_path = os.path.join(self.cfg.save_directory, self.name)
        else:
            save_path = os.path.join(self.cfg.save_directory, save_name)
        if not save_path.endswith('.npz'):
            save_path += '.npz'
        getLogger(__name__).info("Saving spectrophotometric response curve to {}".format(save_path))
        np.savez(save_path, curve=self.curve, configuration=self.cfg)
        self._file_path = save_path  # new file_path for the solution

    def load(self, file_path, file_mode='c'):
        """
        loads in a response curve from a saved npz file and sets rleevant attributes
        """
        getLogger(__name__).info("Loading solution from {}".format(file_path))
        keys = ('curve', 'configuration')
        npz_file = np.load(file_path, allow_pickle=True, encoding='bytes', mmap_mode=file_mode)
        for key in keys:
            if key not in list(npz_file.keys()):
                raise AttributeError('{} missing from {}, solution malformed'.format(key, file_path))
        self.npz = npz_file
        self.curve = self.npz['curve']
        self.cfg = self.npz['configuration']
        self._file_path = file_path  # new file_path for the solution
        self.name = os.path.splitext(os.path.basename(file_path))[0]  # new name for saving
        getLogger(__name__).info("Complete")


def name_to_ESO_extension(object_name):
    """
    converts an input object name string to the standard filename format for the ESO standards catalog on their
    ftp server
    :return:
    """
    extension = ''
    for char in object_name:
        if char.isupper():
            extension = extension + char.lower()
        elif char == '+':
            extension = extension
        elif char == '-':
            extension = extension + '_'
        else:
            extension = extension + char
    return 'f{}.dat'.format(extension)


def fetch_spectra_ESO(object_name, save_dir):
    """
    fetches a standard spectrum from the ESO catalog and downloads it to self.savedir if it exists. Requires
    self.object_name to not be None
    :return:
    """
    getLogger(__name__).info('Looking for {} spectrum in ESO catalog'.format(object_name))
    ext = name_to_ESO_extension(object_name)
    path = 'ftp://ftp.eso.org/pub/stecf/standards/'
    folders = np.array(['ctiostan/', 'hststan/', 'okestan/', 'wdstan/', 'Xshooter/'])
    spectrum_file = None
    if os.path.exists(save_dir + ext):
        getLogger(__name__).info('Spectrum already loaded, will not be reloaded')
        spectrum_file = save_dir + ext
        return spectrum_file
    for folder in folders:
        try:
            with closing(request.urlopen(path + folder + ext)) as r:
                with open(save_dir + ext, 'wb') as f:
                    shutil.copyfileobj(r, f)
            spectrum_file = save_dir + ext
        except URLError:
            pass
    return spectrum_file


def fetch_spectra_SDSS(object_name, save_dir, coords):
    """
    saves a textfile in self.save_dir where the first column is the wavelength in angstroms and the second
    column is flux in erg cm-2 s-1 AA-1
    :return: the path to the saved spectrum file
    """
    if os.path.exists(save_dir + object_name + 'spectrum.dat'):
        getLogger(__name__).info('Spectrum already loaded, will not be reloaded')
        spectrum_file = save_dir + object_name + 'spectrum.dat'
        return spectrum_file
    getLogger(__name__).info('Looking for {} spectrum in SDSS catalog'.format(object_name))
    result = SDSS.query_region(coords, spectro=True)
    if not result:
        getLogger(__name__).warning('Could not find spectrum for {} at {},{} in SDSS catalog'.format(object_name, coords[0], coords[1]))
        spectrum_file = None
        return spectrum_file
    spec = SDSS.get_spectra(matches=result)
    data = spec[0][1].data
    lamb = 10**data['loglam'] * u.AA
    flux = data['flux'] * 10 ** -17 * u.Unit('erg cm-2 s-1 AA-1')
    spectrum = Spectrum1D(spectral_axis=lamb, flux=flux)
    res = np.array([spectrum.spectral_axis, spectrum.flux])
    res = res.T
    spectrum_file = save_dir + object_name + 'spectrum.dat'
    np.savetxt(spectrum_file, res, fmt='%1.4e')
    getLogger(__name__).info('Spectrum loaded for {} from SDSS catalog'.format(object_name))
    return spectrum_file


def aper_photometry(image, obj_position, sky_position, radius):
    """
    wrapper for aperture_photometry
    :param image: 2D array pn which aperture photometry will be performed
    :param obj_position: tuple, (x, Y) coordinate of the object
    :param sky_position: tuple (x, y ) coordinate of where you wouldl ike the background reference flux to be found
    :param radius: radius of the aperture
    :return: object flux, sky flux in the units of image
    """
    positions = np.array([obj_position, sky_position])
    aperture = CircularAperture(positions, r=radius)
    photometry_table = aperture_photometry(image, aperture)
    object_flux = photometry_table[0]['aperture_sum']
    sky_flux = photometry_table[1]['aperture_sum']
    return object_flux, sky_flux


def psf_photometry(img, sigma_psf, x0=None, y0=None):
    """
    Function for performing PSF photometry using astropy modules

    :param img: 2D array on which to perform photometry
    :param sigma_psf: standard deviation of the fitted PSF
    :param x0: x centroid location to fit PSF
    :param y0: y centroid location to fit PSF, if x0 and y0 are None will find the brightest source in img and fit that
    :return: x0, y0, and total flux
    """
    image = ndimage.gaussian_filter(img, sigma=3, order=0)
    bkgrms = MADStdBackgroundRMS()
    std = bkgrms(image)
    iraffind = IRAFStarFinder(threshold=3.5 * std, fwhm=sigma_psf * gaussian_sigma_to_fwhm, minsep_fwhm=0.01,
                              roundhi=5.0, roundlo=-5.0, sharplo=0.0, sharphi=2.0)
    daogroup = DAOGroup(2.0 * sigma_psf * gaussian_sigma_to_fwhm)
    mmm_bkg = MMMBackground()
    fitter = LevMarLSQFitter()
    psf_model = IntegratedGaussianPRF(sigma=sigma_psf)
    if x0 and y0:
        from photutils.psf import BasicPSFPhotometry
        psf_model.x_0.fixed = True
        psf_model.y_0.fixed = True
        pos = Table(names=['x_0', 'y_0'], data=[x0, y0])
        photometry = BasicPSFPhotometry(group_maker=daogroup,bkg_estimator=mmm_bkg, psf_model=psf_model,
                                        fitter=LevMarLSQFitter(), fitshape=(11,11))
        res = photometry(image=image, init_guesses=pos)
        return res['x_0'], res['y_0'], res['flux_0']
    from photutils.psf import IterativelySubtractedPSFPhotometry
    photometry = IterativelySubtractedPSFPhotometry(finder=iraffind, group_maker=daogroup, bkg_estimator=mmm_bkg,
                                                    psf_model=psf_model, fitter=fitter, niters=1, fitshape=(11, 11))
    res = photometry(image=image)
    ind = np.where(res['flux_0'] == res['flux_0'].max())
    return res['x_0'][ind], res['y_0'][ind], res['flux_0'][ind]


def fetch_spectra_URL(object_name, url_path, save_dir):
    """
    grabs the spectrum from a given URL and saves it in self.savedir
    :return: the file path to the saved spectrum
    """
    if os.path.exists(save_dir + object_name + 'spectrum.dat'):
        getLogger(__name__).info('Spectrum already loaded, will not be reloaded')
        spectrum_file = save_dir + object_name + 'spectrum.dat'
        return spectrum_file
    if not url_path:
        getLogger(__name__).warning('No URL path specified')
        pass
    else:
        with closing(request.urlopen(url_path)) as r:
            with open(save_dir + object_name + 'spectrum.dat', 'wb') as f:
                shutil.copyfileobj(r, f)
        spectrum_file = save_dir + object_name + 'spectrum.dat'
        return spectrum_file


def interpolateImage(inputArray, method='linear'):
    """
    Seth 11/13/14
    2D interpolation to smooth over missing pixels using built-in scipy methods
    INPUTS:
        inputArray - 2D input array of values
        method - method of interpolation. Options are scipy.interpolate.griddata methods:
                 'linear' (default), 'cubic', or 'nearest'
    OUTPUTS:
        the interpolated image with same shape as input array
    """

    finalshape = np.shape(inputArray)

    dataPoints = np.where(np.logical_or(np.isnan(inputArray), inputArray == 0) == False)  # data points for interp are only pixels with counts
    data = inputArray[dataPoints]
    dataPoints = np.array((dataPoints[0], dataPoints[1]), dtype=np.int).transpose()  # griddata expects them in this order

    interpPoints = np.where(inputArray != np.nan)  # should include all points as interpolation points
    interpPoints = np.array((interpPoints[0], interpPoints[1]), dtype=np.int).transpose()

    interpolatedFrame = griddata(dataPoints, data, interpPoints, method)
    interpolatedFrame = np.reshape(interpolatedFrame, finalshape)  # reshape interpolated frame into original shape

    return interpolatedFrame


def get_coords(object_name, ra, dec):
    """
    finds the SkyCoord object given a specified ra and dec or object_name
    :return: SkyCoord object
    """
    if not object_name:
        coords = coord.SkyCoord(ra, dec, unit=('hourangle', 'deg'))
    else:
        coords = coord.SkyCoord.from_name(object_name)
    if not coords:
        getLogger(__name__).error('No coordinates found for spectrophotometric calibration object')
    return coords


def create_spectra_directory(save_dir):
    """
    creates a spectrum directory in the save directory to put the spectra. If not called then the spectrum will
    just be saved in save_path
    """
    if os.path.exists(save_dir + '/spectra/'):
        getLogger(__name__).info('Spectrum directory already exists in {}, not going to make a new one'.format(save_dir))
    else:
        os.mkdir(save_dir + '/spectra/')
    return save_dir + '/spectra/'


def load_solution(sc, singleton_ok=True):
    """sc is a solution filename string, a ResponseCurve object, or a mkidpipeline.config.MKIDSpectralReference"""
    global _loaded_solutions
    if not singleton_ok:
        raise NotImplementedError('Must implement solution copying')
    if isinstance(sc, ResponseCurve):
        return sc
    if isinstance(sc, mkidpipeline.config.MKIDSpectralReference):
        sc = mkidpipeline.config.spectralcal_id(sc.id)+'.npz'
    sc = sc if os.path.isfile(sc) else os.path.join(mkidpipeline.config.config.paths.database, sc)
    try:
        return _loaded_solutions[sc]
    except KeyError:
        _loaded_solutions[sc] = ResponseCurve(file_path=sc)
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
            try:
                scfg.register('start_times', [x.start for x in sd.data], update=True)
            except AttributeError:
                scfg.register('start_times', [x.start for x in sd.data[0].obs], update=True)
            if sd.sky_data is not None:
                scfg.register('sky_start_times', [x.start for x in sd.sky_data],
                              update=True)
            else:
                scfg.register('sky_start_times', [], update=True)
            try:
                scfg.register('exposure_times', [x.duration for x in sd.data], update=True)
                scfg.register('ra', [x.ra for x in sd.data], update=True)
                scfg.register('dec', [x.dec for x in sd.data], update=True)
                scfg.register('object_name', [x.target for x in sd.data], update=True)
                scfg.register('data', sd.data, update=True)
            except AttributeError:
                scfg.register('exposure_times', [x.duration for x in sd.data[0].obs], update=True)
                scfg.register('ra', [x.ra for x in sd.data[0].obs], update=True)
                scfg.register('dec', [x.dec for x in sd.data[0].obs], update=True)
                scfg.register('object_name', [x.target for x in sd.data[0].obs], update=True)
                scfg.register('data', sd.data[0], update=True)
            if ncpu is not None:
                scfg.update('ncpu', ncpu)
            cal = SpectralCalibrator(scfg, solution_name=sf)
            cal.run(**kwargs)
            # solutions.append(load_solution(sf))  # don't need to reload from file
            solutions.append(cal.solution)  # TODO: solutions.append(load_solution(cal.solution))

    return solutions
