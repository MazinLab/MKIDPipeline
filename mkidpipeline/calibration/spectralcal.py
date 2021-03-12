"""
Author: Sarah Steiger    Date: April 1, 2020

Loads a standard spectrum from and convolves and rebind it ot match the MKID energy resolution and bin size. Then
generates an MKID spectrum of the object by performing photometry (aperture or PSF) on the MKID image. Finally
divides the flux values for the standard by the MKID flux values for each bin to get a calibration curve.

Assumes h5 files are wavelength calibrated, and they should also first be flatcalibrated and linearity corrected
(deadtime corrected)
"""

import sys,os
import numpy as np
from mkidcore import pixelflags
from mkidpipeline.hdf.photontable import Photontable
from mkidpipeline.utils.speccal_utils import rebin, gaussianConvolution, fitBlackbody
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
from mkidpipeline.utils.photometry import *
from scipy.interpolate import griddata
import pkg_resources as pkg
import matplotlib.gridspec as gridspec
_loaded_solutions = {}


class Configuration(object):
    """Configuration class for the spectrophotometric calibration analysis."""
    yaml_tag = u'!spectralcalconfig'

    def __init__(self, configfile=None, start_times=tuple(), h5dir='', savedir='', h5_file_names='', object_name='',
                 ra=None, dec=None, photometry='', energy_bin_width=0.01, wvl_start=850, wvl_stop=1375,
                 wvl_bin_edges=None, summary_plot=True, std_path='', aperture_radius=3, obj_pos=None,
                 use_satellite_spots=True, interpolation='linear'):
        # parse arguments
        self.configuration_path = configfile
        self.h5_directory = h5dir
        self.save_directory = savedir
        self.url_path = std_path
        self.object_name = object_name
        self.ra = ra
        self.dec = dec
        self.start_times = list(map(int, start_times))
        self.h5_file_names = list(map(str, h5_file_names))
        self.h5_directory = h5dir
        self.energyBinWidth = float(energy_bin_width)
        self.wvlStart = float(wvl_start)
        self.wvlStop = float(wvl_stop)
        self.wvl_bin_edges = wvl_bin_edges
        self.photometry = str(photometry)
        self.summary_plot = bool(summary_plot)
        self.aperture_radius = aperture_radius
        self.obj_pos = obj_pos
        self.interpolation = interpolation
        self.data = None
        self.use_satellite_spots = use_satellite_spots
        if self.configuration_path is not None:
            # load in the configuration file
            cfg = mkidcore.config.load(self.configuration_path)
            self.use_satellite_spots = cfg.use_satellite_spots
            self.ncpu = cfg.ncpu
            self.wvlStart = cfg.instrument.wvl_start * 10.0 # in angstroms
            self.wvlStop = cfg.instrument.wvl_stop * 10.0 # in angstroms
            self.wvl_bin_edges = cfg.wvl_bin_edges
            self.energyStart = (c.h * c.c) / (self.wvlStart * 10**(-10) * c.e)
            self.energyStop = (c.h * c.c) / (self.wvlStop * 10**(-10) * c.e)
            sol = mkidpipeline.calibration.wavecal.Solution(cfg.wavcal)
            r, resid = sol.find_resolving_powers(cache=True)
            self.r_list = np.nanmedian(r, axis=0)
            self.energyBinWidth = ((self.energyStart + self.energyStop)/2)/(np.median(self.r_list) * 5.0)
            if self.wvl_bin_edges is None:
                nwvlbins = int((self.energyStart - self.energyStop) / self.energyBinWidth)
            else:
                nwvlbins= len(self.wvl_bin_edges) - 1
            self.wvl_bin_widths = np.zeros(nwvlbins)
            self.wvl_bin_centers = np.zeros(nwvlbins)
            self.h5_directory = cfg.paths.out
            self.save_directory = cfg.paths.database
            self.std_path = cfg.standard_path
            self.start_times = list(map(int, cfg.start_times))
            self.object_name = cfg.object_name
            self.ra = cfg.ra if cfg.ra else None
            self.dec = cfg.dec if cfg.dec else None
            self.photometry = cfg.spectralcal.photometry_type
            self.summary_plot = cfg.spectralcal.summary_plot
            self.aperture_radius = cfg.aperture_radius
            self.data = cfg.data
            self.obj_pos = tuple(float(s) for s in cfg.obj_pos.strip("()").split(",")) \
                if cfg.obj_pos else None
            self.interpolation = cfg.spectralcal.interpolation
            try:
                self.h5_file_names = [os.path.join(self.h5_directory, f) for f in self.start_times]
            except TypeError:
                self.h5_file_names = [os.path.join(self.h5_directory, str(t) + '.h5') for t in self.start_times]
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
    def __init__(self, save_path='', std_path=None, object_name=None, object_ra=None, object_dec=None, coords=None):
        self.save_dir = save_path
        self.object_name = object_name
        self.ra = object_ra
        self.dec = object_dec
        self.std_path = std_path
        self.coords = coords # SkyCoord object
        self.spectrum_file = None
        self.k = 5.03411259e7

    def get(self):
        """
        function which creates a spectrum directory, populates it with the spectrum file either pulled from the ESO
        catalog, SDSS catalog, a URL, or a specified path to a .txt file and returns the wavelength and flux column in the appropriate units
        :return: wavelengths (Angstroms), flux (erg/s/cm^2/A)
        """
        self.coords = get_coords(object_name=self.object_name, ra=self.ra, dec=self.dec)
        data = self.fetch_spectra()
        return data[:, 0], data[:, 1]

    def fetch_spectra(self):
        '''
        called from get(), searches either a URL, ESO catalog or uses astroquery.SDSS to search the SDSS catalog. Puts
        the retrieved spectrum in a '/spectrum/' folder in self.save_dir
        :return:
        '''
        if self.std_path is not None:
            try:
                data = np.loadtxt(self.std_path)
            except OSError:
                self.spectrum_file = fetch_spectra_URL(object_name=self.object_name, url_path=self.std_path,
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
                    getLogger(__name__).error(
                        'Could not find standard spectrum for this object, please find a spectrum and point to it in '
                        'the standard_path in your pipe.yml')
                    sys.exit()
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

    """
    def __init__(self, configuration=None, h5_file_names=None, solution_name='solution.npz', interpolation=None,
                 use_satellite_spots=True, obj_pos=None):

        self.interpolation = interpolation
        self.use_satellite_spots = use_satellite_spots
        self.obj_pos = obj_pos

        self.flux_spectra = None
        self.flux_effTime = None
        self.wvl_bin_edges = None
        self.std_spectrum = None
        self.std_wvls = None
        self.std_flux = None
        self.rebin_std_wvls = None
        self.rebin_std_flux = None
        self.obs = None
        self.wvl_bin_widths = None
        self.curve = None
        self.image = None
        self.bb_flux = None
        self.bb_wvls = None
        self.conv_wvls = None
        self.conv_flux = None
        self.data = None
        self.wvl_bin_centers = None
        self.flux_spectrum = None
        self.cube = None
        self.aperture_radii = None
        self.contrast = None

        if h5_file_names:
            self.obs = [Photontable(f) for f in h5_file_names]
        if configuration:
            self.cfg = Configuration(configuration) if not isinstance(configuration, Configuration) else configuration
            self.obj_pos = self.cfg.obj_pos
            self.solution_name = solution_name
            self.use_satellite_spots = self.cfg.use_satellite_spots
            self.interpolation = self.cfg.interpolation
            self.obs = [Photontable(f) for f in self.cfg.h5_file_names]
            self.wvl_bin_widths = self.cfg.wvl_bin_widths
            self.wvl_bin_edges = self.cfg.wvl_bin_edges
            self.data = self.cfg.data
            self.wvl_bin_centers = self.cfg.wvl_bin_centers
            self.contrast = np.zeros_like(self.wvl_bin_centers)
            self.aperture_radii = np.zeros_like(self.wvl_bin_centers)
            self.platescale = self.data.wcscal.platescale
            self.solution = ResponseCurve(configuration=self.cfg, curve=self.curve, wvl_bin_widths=self.wvl_bin_widths,
                                          wvl_bin_centers= self.wvl_bin_centers, cube=self.cube,
                                          solution_name=self.solution_name)

    def run(self, save=True, plot=None):
        """

        :param save:
        :param plot:
        :return:
        """
        try:
            getLogger(__name__).info("Loading Spectrum from MEC")
            self.load_absolute_spectrum()
            getLogger(__name__).info("Loading Standard Spectrum")
            self.load_standard_spectrum()
            getLogger(__name__).info("Calculating Spectrophotometric Response Curve")
            self.calculate_response_curve()
            self.solution = ResponseCurve(configuration=self.cfg, curve=self.curve, wvl_bin_widths=self.wvl_bin_widths,
                                          wvl_bin_centers=self.wvl_bin_centers, cube=self.cube,
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
            if self.wvl_bin_edges is None:
                ref_obs = self.obs[0].getSpectralCube(integrationTime=self.cfg.intTimes[0],
                                                      applyWeight=True, wvlStart=self.cfg.wvlStart/10,
                                                      wvlStop=self.cfg.wvlStop/10, energyBinWidth=self.cfg.energyBinWidth,
                                                      exclude_flags=pixelflags.PROBLEM_FLAGS)
                self.wvl_bin_edges = ref_obs['wvlBinEdges'] * 10
            for wvl in range(len(self.wvl_bin_edges) - 1):
                if self.use_satellite_spots:
                    derotate=False
                else:
                    derotate=True
                getLogger(__name__).info('using wavelength range {} - {}'.format(self.wvl_bin_edges[wvl] / 10,
                                                            self.wvl_bin_edges[wvl + 1] / 10))
                drizzled = mkidpipeline.drizzler.form(self.data, mode='spatial', wvlMin=self.wvl_bin_edges[wvl] / 10,
                                                      wvlMax=self.wvl_bin_edges[wvl + 1] / 10, pixfrac=0.5,
                                                      wcs_timestep=1, exp_timestep=1,
                                                      exclude_flags=pixelflags.PROBLEM_FLAGS, usecache=False, ncpu=1,
                                                      derotate=derotate, align_start_pa=False, whitelight=False,
                                                      debug_dither_plot=False)
                getLogger(__name__).info(('finished image {}/ {}'.format(wvl + 1.0, len(self.wvl_bin_edges) - 1)))
                cube.append(drizzled.cps)
            self.cube = np.array(cube)
        self.image = np.sum(self.cube, axis=0)
        n_wvl_bins = len(self.wvl_bin_edges) - 1

        # define useful quantities
        for i, edge in enumerate(self.wvl_bin_edges):
            try:
                self.wvl_bin_widths[i] = self.wvl_bin_edges[i + 1] - self.wvl_bin_edges[i]
            except IndexError:
                pass
        for i, edge in enumerate(self.wvl_bin_edges):
            try:
                self.wvl_bin_centers[i] = (self.wvl_bin_edges[i + 1] + self.wvl_bin_edges[i])/2.0
            except IndexError:
                pass

        self.flux_spectrum = np.zeros(n_wvl_bins)

        if self.use_satellite_spots:
            fluxes = mec_measure_satellite_spot_flux(self.cube, wvl_start=self.wvl_bin_edges[:-1],
                                                     wvl_stop=self.wvl_bin_edges[1:])
            self.flux_spectrum = np.nanmean(fluxes, axis=1)
        else:
            if self.obj_pos is None:
                getLogger(__name__).info('No coordinate specified for the object. Performing a PSF fit '
                                         'to find the location')
                x, y, flux = astropy_psf_photometry(cube[:,:,0], 5.0)
                ind = np.where(flux == flux.max())
                self.obj_pos = (x.data.data[ind][0], y.data.data[ind][0])
                getLogger(__name__).info('Found the object at {}'.format(self.obj_pos))
            for i in np.arange(n_wvl_bins):
                # perform photometry on every wavelength bin
                frame = cube[:, :, i]
                if self.interpolation is not None:
                    frame = interpolate_image(frame, method=self.interpolation)
                rad = get_aperture_radius(self.wvl_bin_centers[i], self.platescale)
                self.aperture_radii[i] = rad
                obj_flux = aper_photometry(frame, self.obj_pos, rad)
                self.flux_spectrum[i] = obj_flux
        return self.flux_spectrum

    def load_standard_spectrum(self):
        standard = StandardSpectrum(save_path=self.cfg.save_directory, std_path=self.cfg.std_path,
                                    object_name=self.cfg.object_name[0], object_ra=self.cfg.ra[0],
                                    object_dec=self.cfg.dec[0])
        self.std_wvls, self.std_flux = standard.get()  # standard star object spectrum in ergs/s/Angs/cm^2
        conv_wvls_rev, conv_flux_rev = self.extend_spectrum(self.std_wvls, self.std_flux)
        # convolved spectrum comes back sorted backwards, from long wvls to low which screws up rebinning
        self.conv_wvls = conv_wvls_rev[::-1]
        self.conv_flux = conv_flux_rev[::-1]

        # rebin cleaned spectrum to flat cal's wvlBinEdges
        rebin_std_data = rebin(self.conv_wvls, self.conv_flux, self.wvl_bin_edges)
        if self.use_satellite_spots:
            for i, wvl in enumerate(self.wvl_bin_centers):
                self.contrast[i] = satellite_spot_contrast(wvl)
                rebin_std_data[i,1] = rebin_std_data[i,1] * self.contrast[i]
        self.rebin_std_flux = np.array(rebin_std_data[:, 1])
        self.rebin_std_wvls = np.array(rebin_std_data[:, 0])

    def extend_spectrum(self, x, y):
        """
        BB Fit to extend standard spectrum to 1500 nm and to convolve it with a gaussian kernel corresponding to the
        energy resolution of the detector. If spectrum spans whole MKID range will just convolve with the gaussian
        """
        r = np.median(np.nanmedian(self.cfg.r_list, axis=0))
        if np.round(x[-1]) < self.cfg.wvlStop:
            fraction = 1.0 / 3.0
            nirX = np.arange(int(x[int((1.0 - fraction) * len(x))]), self.cfg.wvlStop)
            T, nirY = fitBlackbody(x, y, fraction=fraction, newWvls=nirX)
            if np.any(x >= self.cfg.wvlStop):
                self.bb_wvls = x
                self.bb_flux = y
            else:
                self.bb_wvls = np.concatenate((x, nirX[nirX > max(x)]))
                self.bb_flux = np.concatenate((y, nirY[nirX > max(x)]))
            # Gaussian convolution to smooth std spectrum to MKIDs median resolution
            newX, newY = gaussianConvolution(self.bb_wvls, self.bb_flux, xEnMin=self.cfg.energyStop,
                                             xEnMax=self.cfg.energyStart, fluxUnits="lambda", r=r, plots=False)
        else:
            getLogger(__name__).info('Standard Spectrum spans whole energy range - no need to perform blackbody fit')
            # Gaussian convolution to smooth std spectrum to MKIDs median resolution
            std_stop = (c.h * c.c) / (self.std_wvls[0] * 10**(-10) * c.e)
            std_start = (c.h * c.c) / (self.std_wvls[-1] * 10 ** (-10) * c.e)
            newX, newY = gaussianConvolution(x, y, xEnMin=std_start, xEnMax=std_stop, fluxUnits="lambda", r=r,
                                             plots=False)
        return newX, newY

    def calculate_response_curve(self):
        """
        Divide the MEC Spectrum by the rebinned and gaussian convolved standard spectrum
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
        axes_list[0].set_title('MKID Instrument Image of Standard', size=8)

        std_idx = np.where(np.logical_and(self.cfg.wvlStart < self.std_wvls, self.std_wvls < self.cfg.wvlStop))
        conv_idx = np.where(np.logical_and(self.cfg.wvlStart < self.conv_wvls, self.conv_wvls < self.cfg.wvlStop))

        axes_list[1].step(self.std_wvls[std_idx], self.std_flux[std_idx], where='mid', label='{} Spectrum'.format(self.cfg.object_name[0]))
        if self.bb_flux:
            axes_list[1].step(self.bb_wvls, self.bb_flux, where='mid', label='BB fit')
        axes_list[1].step(self.conv_wvls[conv_idx], self.conv_flux[conv_idx], where='mid', label='Convolved Spectrum')
        axes_list[1].set_xlabel('Wavelength (A)')
        axes_list[1].set_ylabel('Flux (erg/s/cm^2)')
        axes_list[1].legend(loc='upper right', prop={'size': 6})


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
    def __init__(self, file_path=None, curve=None, configuration=None, wvl_bin_widths=None, wvl_bin_centers=None,
                 cube=None, errors=None, solution_name='spectral_solution'):
        # default parameters
        self._parse = True
        # load in arguments
        self._file_path = os.path.abspath(file_path) if file_path is not None else file_path
        self.curve = curve
        self.cfg = configuration
        self.wvl_bin_widths = wvl_bin_widths
        self.wvl_bin_centers = wvl_bin_centers
        self.cube = cube
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
        np.savez(save_path, curve=self.curve, wvl_bin_widths=self.wvl_bin_widths, wvl_bin_centers=self.wvl_bin_centers,
                 cube=self.cube, configuration=self.cfg)
        self._file_path = save_path  # new file_path for the solution

    def load(self, file_path, file_mode='c'):
        """
        loads in a response curve from a saved npz file and sets relevant attributes
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
        self.wvl_bin_widths = self.npz['wvl_bin_widths']
        self.wvl_bin_centers = self.npz['wvl_bin_centers']
        self.cube = self.npz['cube']
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
        getLogger(__name__).warning('Could not find spectrum for {} at {},{} in SDSS catalog'.format(object_name, coords.ra, coords.dec))
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

def interpolate_image(input_array, method='linear'):
    """
    Seth 11/13/14
    2D interpolation to smooth over missing pixels using built-in scipy methods
    :param input_array: N x M array
    :param method:
    :return: N x M interpolated image array
    """
    final_shape = np.shape(input_array)
    # data points for interp are only pixels with counts
    data_points = np.where(np.logical_or(np.isnan(input_array), input_array == 0) == False)
    data = input_array[data_points]
    # griddata expects them in this order
    data_points = np.array((data_points[0], data_points[1]), dtype=np.int).transpose()
    # should include all points as interpolation points
    interp_points = np.where(input_array != np.nan)
    interp_points = np.array((interp_points[0], interp_points[1]), dtype=np.int).transpose()

    interpolated_frame = griddata(data_points, data, interp_points, method)
    # reshape interpolated frame into original shape
    interpolated_frame = np.reshape(interpolated_frame, final_shape)

    return interpolated_frame

def get_coords(object_name, ra, dec):
    """
    finds the SkyCoord object given a specified ra and dec or object_name
    :return: SkyCoord object
    """
    if ra and dec:
        coords = coord.SkyCoord(ra, dec, unit=('hourangle', 'deg'))
    else:
        try:
            coords = coord.SkyCoord.from_name(object_name)
        except TimeoutError:
            coords=None
    if not coords:
        getLogger(__name__).error('No coordinates found for spectrophotometric calibration object')
    return coords

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

def satellite_spot_contrast(lam):
    """

    :param lam: wavelength in angstroms
    :return: contrast
    """
    getLogger(__name__).info('Using satellite spot contrast for a 25 nm astrogrid')
    ref = 1.55*10**4
    contrast = 2.72e-3*(ref / lam)**2 # 2.72e-3 number from Currie et. al. 2018b
    return contrast

def fetch(dataset, config=None, ncpu=None, remake=False, **kwargs):
    solution_descriptors = dataset.spectralcals
    cfg = mkidpipeline.config.config if config is None else config
    for sd in dataset.wavecals:
        wavcal = os.path.join(cfg.paths.database, mkidpipeline.config.wavecal_id(sd.id)+'.npz')
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
            try:
                scfg.register('exposure_times', [x.duration for x in sd.data], update=True)
                scfg.register('ra', [x.ra for x in sd.data], update=True)
                scfg.register('dec', [x.dec for x in sd.data], update=True)
                scfg.register('object_name', [x.target for x in sd.data], update=True)
                scfg.register('data', sd.data, update=True)
                scfg.register('obj_pos', sd.object_position, update=True)
                scfg.register('aperture_radius', sd.aperture_radius, update=True)
                scfg.register('wvl_bin_edges', sd.wvl_bin_edges, update=True)
                scfg.register('use_satellite_spots', sd.use_satellite_spots, update=True)
                scfg.register('standard_path', sd.standard_path, update=True)
                scfg.register('wavcal', wavcal, update=True)
            except AttributeError:
                scfg.register('exposure_times', [x.duration for x in sd.data[0].obs], update=True)
                scfg.register('ra', [x.ra for x in sd.data[0].obs], update=True)
                scfg.register('dec', [x.dec for x in sd.data[0].obs], update=True)
                scfg.register('object_name', [x.target for x in sd.data[0].obs], update=True)
                scfg.register('data', sd.data[0], update=True)
                scfg.register('obj_pos', sd.object_position, update=True)
                scfg.register('aperture_radius', sd.aperture_radius, update=True)
                scfg.register('use_satellite_spots', sd.use_satellite_spots, update=True)
                scfg.register('wvl_bin_edges', sd.wvl_bin_edges, update=True)
                scfg.register('standard_path', sd.standard_path, update=True)
                scfg.register('wavcal', wavcal, update=True)
            if ncpu is not None:
                scfg.update('ncpu', ncpu)
            cal = SpectralCalibrator(scfg, solution_name=sf)
            cal.run(**kwargs)
            # solutions.append(load_solution(sf))  # don't need to reload from file
            solutions.append(cal.solution)  # TODO: solutions.append(load_solution(cal.solution))

    return solutions
