"""
The speccal takes in an MKID wavelength cube in arbitrary 'counts' units and converts it to a wavelength cube with the
same dimensions but in units of ergs/s/cm^2/angstroms.

The speccal first loads a standard spectrum which can be specified by a two column text file in the data.yaml, or it is
pulled directly from SDSS or ESO databases based on the object name from the fits header. If a spectrum is not defined
in the data.yaml and the object is not present in either database then the speccal will throw an error. Once the
standard spectrum is loaded, it is convolved and rebinned to match the energy resolution (wavelength bin size) of the
MKID detector. If bin edges are specified in the pipe.yaml then those bins will be used, or else bins are used that
nyquist sample the energy resolution of the detector as is determined by the wavecal.

An MKID spectrum of the object is then found by performing photometry (aperture or PSF) on the MKID image. This
photometry can be performed at the location of an astrophysical point source, or the satellite spots can be used to get
the spectrum of an object in the image located behind a coronagraph.

Once the standard and MKID spectra are determined, they can be divided to generate a calibration curve. This calibration
curve is saved to an npz file in the form of an interpolated univariate spline in the ResponseCurve object. The
configuration used to generate the solution, the wavelength bin edges used, and the original uncalibrated MKID cube are
also all saved in the ResponseCurve.

The speccal is applied by calling the 'apply' function on an MKID wavelength cube and will return a calibrated cube of
the same dimensions.

Speccal assumes that the h5 files are wavelength calibrated, and they should also first be flat calibrated and linearity
corrected (deadtime corrected).
"""
import sys
import os
import urllib.request as request
from urllib.error import URLError
import shutil
import numpy as np
from contextlib import closing
import scipy.constants as c
from astropy import units as u
from astroquery.sdss import SDSS
import astropy.coordinates as coord
from specutils import Spectrum1D
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import mkidpipeline.definitions as definitions
from mkidcore.corelog import getLogger
import mkidcore.config
from astropy.wcs import WCS
import mkidpipeline.config
from mkidpipeline.utils.resampling import rebin
from mkidpipeline.utils.fitting import fit_blackbody
from mkidpipeline.utils.smoothing import gaussian_convolution
from mkidpipeline.utils.interpolating import interpolate_image
from mkidpipeline.utils.photometry import get_aperture_radius, aper_photometry, astropy_psf_photometry, \
    mec_measure_satellite_spot_flux
from mkidpipeline.steps.drizzler import form
from mkidpipeline.photontable import Photontable
from scipy.interpolate import InterpolatedUnivariateSpline
from astropy.io import fits
import scipy.integrate
from mkidpipeline.utils.resampling import rebin
_loaded_solutions = {}


PROBLEM_FLAGS = ('pixcal.hot', 'pixcal.cold', 'pixcal.dead', 'beammap.noDacTone', 'wavecal.bad',
                 'wavecal.failed_validation', 'wavecal.failed_convergence', 'wavecal.not_monotonic',
                 'wavecal.not_enough_histogram_fits', 'wavecal.no_histograms',
                 'wavecal.not_attempted')


class StepConfig(mkidpipeline.config.BaseStepConfig):
    yaml_tag = u'!speccal_cfg'
    REQUIRED_KEYS = (('photometry_type', 'aperture', 'aperture | psf'),
                     ('plots', 'summary', 'summary | none'),
                     ('interpolation', 'linear', ' linear | cubic | nearest'),
                     ('wvl_bin_edges', [], 'list of wavelength bin edges to use for determining the solution'
                                           ' (in Angstroms). Defaults to nyquist sampling the energy resolution'),
                     ('fit_order', 1, 'order of the univariate spline to fit the soectrophotometric repsonse curve -'
                                      'must be shorter than the length of the wvl_bin_edges if specified'))

    def _verify_attribues(self):
        """Returns a list missing keys from the pipeline config"""
        errors = super(StepConfig, self)._verify_attribues()
        if 'fit_order' in self and 'wvl_bin_edges' in self:
            n_edge = len(self.wvl_bin_edges)
            if n_edge and self.fit_order>=n_edge:
                errors.append('Speccal fit order must be less than the number of bin edges')
        return errors


class StandardSpectrum:
    """
    Handles retrieving and storing relevant information about the spectrum which is being used as the standard
    """
    def __init__(self, save_path='', std_path=None, name=None, ra=None, dec=None, coords=None):
        self.save_dir = save_path
        self.object_name = name
        self.ra = ra
        self.dec = dec
        self.std_path = std_path
        self.coords = coords  # SkyCoord
        self.spectrum_file = None
        self.k = 5.03411259e7

    def fetch_spectra(self):
        """
        searches either a URL, ESO catalog or uses astroquery.SDSS to search the SDSS catalog. Puts
        the retrieved spectrum in a '/spectrum/' folder in self.save_dir
        :return:
        """
        conversion = 1
        filename = None
        if os.path.isfile(self.std_path):
            filename = self.std_path
        elif self.std_path:
            filename = fetch_spectra_URL(object_name=self.object_name, url_path=self.std_path, save_dir=self.save_dir)
        else:
            try:
                filename = fetch_spectra_ESO(object_name=self.object_name, save_dir=self.save_dir)
                conversion = 1e-16          # to convert to the appropriate units if ESO spectra
            except:
                getLogger(__name__).info('ESO query failed, falling back to SDSS')
                try:
                    coords = self.coords or get_coords(object_name=self.object_name, ra=self.ra, dec=self.dec)
                    filename = fetch_spectra_SDSS(object_name=self.object_name, save_dir=self.save_dir, coords=coords)
                except ConnectionError:
                    getLogger(__name__).info('SDSS query failed')

        self.spectrum_file = filename

        try:
            data = np.loadtxt(self.spectrum_file)
            data[:, 1] *= conversion
        except (ValueError, IndexError, OSError):
            getLogger(__name__).error(f'Could not find standard spectrum for {self.object_name} with path')
            raise

        return data

    def photons_to_ergs(self, a):
        """
        converts units of the spectra from counts to ergs
        :return:
        """
        a[:, 1] /= (a[:, 0] * self.k)
        return a

    def ergs_to_photons(self, a):
        """
        converts units of the spectra from ergs to counts
        :return:
        """
        a[:, 1] *= (a[:, 0] * self.k)
        return a


class SpectralCalibrator:
    """

    """
    def __init__(self, configuration=None, solution_name='solution.npz', interpolation=None, aperture=None,
                 data=None, use_satellite_spots=True, save_path=None, platescale=.0104, std_path='',
                 photometry_type='aperture', summary_plot=True, fit_order=None, ncpu=1):
        self.interpolation = interpolation
        self.use_satellite_spots = use_satellite_spots
        self.solution_name = solution_name
        self.data = data
        self.aperture = aperture
        self.save_path = save_path
        self.platescale = platescale*u.arcsec
        self.std_path = std_path
        self.photometry = photometry_type
        self.summary_plot = summary_plot
        self.cfg = configuration
        self.std_path = self.data.spectrum
        self.ncpu = ncpu
        self.fit_order = fit_order
        # various spectra
        self.std_wvls = None
        self.std_flux = None
        self.rebin_std = None
        self.bb = None
        self.mkid = None
        self.conv = None
        self.curve = None
        self.cube = None
        self.contrast = None
        self.wcs = None

        if configuration is not None:
            # load in the configuration file
            cfg = mkidcore.config.load(configuration)
            self.save_path = cfg.paths.database
            if len(cfg.speccal.wvl_bin_edges) != 0:
                self.wvl_bin_edges = cfg.speccal.wvl_bin_edges*u.nm
            else:
                pt = Photontable(self.data.obs[0].h5)
                self.wvl_bin_edges = pt.nyquist_wavelengths()*u.nm
            self.platescale = [v.platescale for v in self.data.wcscal.values()][0]
            self.solution = ResponseCurve(configuration=cfg, curve=self.curve, wvl_bin_edges=self.wvl_bin_edges,
                                          cube=self.cube, solution_name=self.solution_name)
            self.photometry = cfg.speccal.photometry_type
            self.contrast = np.zeros(len(self.wvl_bin_edges) - 1)
            self.plots = cfg.speccal.plots
            self.interpolation = cfg.speccal.interpolation
            self.fit_order = cfg.speccal.fit_order
            sol = mkidpipeline.steps.wavecal.Solution(cfg.wave_sol)
            r, resid = sol.find_resolving_powers(cache=True)
            r_list = np.nanmedian(r, axis=0)
            self.r = np.median(r_list, axis=0)
        else:
            pt = Photontable(self.data.obs[0].h5)
            self.wvl_bin_edges = pt.nyquist_wavelengths()*u.nm
            self.r = pt.query_header('energy_resolution')
        self.energy_start = (c.h * c.c)/(self.wvl_bin_edges[0].to(u.m) * c.e).value
        self.energy_stop = (c.h * c.c)/(self.wvl_bin_edges[-1].to(u.m) * c.e).value
        self.aperture_radius = np.zeros(len(self.wvl_bin_edges) - 1) if self.wvl_bin_edges else None

    def run(self, save=True, plot=None):
        getLogger(__name__).info("Loading Spectrum from MEC")
        self.load_absolute_spectrum()
        getLogger(__name__).info("Loading Standard Spectrum")
        self.load_standard_spectrum()
        getLogger(__name__).info("Calculating Spectrophotometric Response Curve")
        self.calculate_response_curve()
        self.solution = ResponseCurve(configuration=self.cfg, curve=self.curve[0], wvl_bin_edges=self.wvl_bin_edges,
                                      cube=self.cube, solution_name=self.solution_name)
        if save:
            self.solution.save(save_name=self.solution_name if isinstance(self.solution_name, str) else None)
        if plot or (plot is None and self.plots == 'summary'):
            save_name = self.solution_name.rpartition(".")[0] + ".pdf"
            self.plot_summary(save_name=save_name)

    def load_absolute_spectrum(self):
        """
         Extract the MEC measured spectrum of the spectrophotometric standard by breaking data into spectral cubes
         and performing photometry (aperture or psf) on each spectral frame
         """
        getLogger(__name__).info('performing {} photometry on MEC spectrum'.format(self.photometry))
        pt = Photontable(self.data.obs[0].h5)
        self.wcs = pt.get_wcs(wcs_timestep=pt.duration, derotate=not self.use_satellite_spots)[0]
        if len(self.data.obs) == 1:
            pt = Photontable(self.data.obs[0].h5)
            hdul = pt.get_fits(weight=True, rate=True, cube_type='wave',
                               bin_edges=self.wvl_bin_edges, bin_type='energy')
            cube = hdul['SCIENCE'].data
        else:
            cube = []
            for wvl in range(len(self.wvl_bin_edges) - 1):
                getLogger(__name__).info('using wavelength range {} - {}'
                                         .format(self.wvl_bin_edges[wvl].to(u.nm).value,
                                                 self.wvl_bin_edges[wvl + 1].to(u.nm).value))
                drizzled = form(self.data, mode='drizzle', wave_start=self.wvl_bin_edges[wvl],
                                wave_stop=self.wvl_bin_edges[wvl + 1], pixfrac=0.5,
                                wcs_timestep=1, exclude_flags=PROBLEM_FLAGS, usecache=False,
                                duration=min([o.duration for o in self.data.obs]), ncpu=self.ncpu,
                                derotate=not self.use_satellite_spots)
                getLogger(__name__).info(('finished image {}/ {}'.format(wvl + 1.0, len(self.wvl_bin_edges) - 1)))
                cube.append(drizzled.cps)
        self.cube = np.array(cube, dtype=np.double)
        n_wvl_bins = len(self.wvl_bin_edges) - 1

        wvl_bin_centers = [(a.value + b.value) / 2 for a, b in zip(self.wvl_bin_edges, self.wvl_bin_edges[1::])]

        self.mkid = np.zeros((2, n_wvl_bins))
        self.mkid[0] = wvl_bin_centers
        if self.use_satellite_spots:
            phot_cube = self.cube.copy()
            fluxes = mec_measure_satellite_spot_flux(phot_cube,
                                                     wvl_start=self.wvl_bin_edges[:-1].to(u.Angstrom).value,
                                                     wvl_stop=self.wvl_bin_edges[1:].to(u.Angstrom).value,
                                                     platescale=self.platescale.value,
                                                     wcs=self.wcs)
            self.mkid[1] = np.nanmean(fluxes, axis=1)
        else:
            cube = self.cube.copy()
            try:
                x, y, r = self.aperture
            except ValueError:
                getLogger(__name__).warning('Aperture for the speccal must be in the format (x/RA, y/DEC, r) OR '
                                            'satellite, instead got {self.aperture}')
            for i in np.arange(n_wvl_bins):
                frame = cube[i, :, :]
                if self.interpolation is not None:
                    frame = interpolate_image(frame, method=self.interpolation)
                if not r:
                    rad = get_aperture_radius(wvl_bin_centers[i], self.platescale)
                else:
                    rad = r
                self.aperture_radius[i] = rad
                # TODO if x in ra and dec convert to x,y coordinates
                wcs = drizzled.wcs
                x, y = wcs.all_world2pix(x, y)
                obj_flux = aper_photometry(frame, (x, y), rad)
                self.mkid[1][i] = obj_flux
        return self.mkid

    def load_standard_spectrum(self):
        # creat a spectrum directory, populate with the spectrum file either pulled from the ESO catalog, SDSS
        # catalog, a URL, or a specified path to a .txt file and returns the wavelength and flux column in the
        # appropriate units
        c = self.data.obs[0].skycoord
        standard = StandardSpectrum(save_path=self.save_path, std_path=self.std_path,
                                    name=self.data.obs[0].metadata['OBJECT'],
                                    ra=c.ra if c else None, dec=c.dec if c else None)

        data = standard.fetch_spectra()
        self.std_wvls, std_flux = data[:, 0]*u.Angstrom, data[:, 1]

        self.std_flux = std_flux * u.erg/u.s/u.Angstrom/u.cm**2
        conv_wvls_rev, conv_flux_rev = self.extend_and_convolve(self.std_wvls.value, self.std_flux.value)
        # convolved spectrum comes back sorted backwards, from long wvls to low which screws up rebinning
        self.conv = np.vstack((conv_wvls_rev[::-1], conv_flux_rev[::-1]))

        # rebin cleaned spectrum to flat cal's wvlBinEdges
        rebin_std_data = rebin(self.conv[0], self.conv[1], self.wvl_bin_edges.to(u.Angstrom).value)
        wvl_bin_centers = [(a.value + b.value) / 2 for a, b in zip(self.wvl_bin_edges.to(u.Angstrom), self.wvl_bin_edges[1::].to(u.Angstrom))]

        if self.use_satellite_spots:
            for i, wvl in enumerate(wvl_bin_centers):
                self.contrast[i] = satellite_spot_contrast(wvl)
                rebin_std_data[i, 1] = rebin_std_data[i, 1] * self.contrast[i]
        self.rebin_std = np.vstack((np.array(rebin_std_data[:, 0]), np.array(rebin_std_data[:, 1])))

    def extend_and_convolve(self, x, y):
        """
        BB Fit to extend standard spectrum to 1500 nm and to convolve it with a gaussian kernel corresponding to the
        energy resolution of the detector. If spectrum spans whole MKID range will just convolve with the gaussian
        """
        if np.round(x[-1]) < self.wvl_bin_edges.to(u.Angstrom).value[-1]:
            fraction = 1.0 / 3.0
            nirX = np.arange(int(x[int((1.0 - fraction) * len(x))]), self.wvl_bin_edges.to(u.Angstrom).value[-1])
            T, nirY = fit_blackbody(x, y, fraction=fraction, new_wvls=nirX)
            if np.any(x >= self.wvl_bin_edges[-1]):
                self.bb = np.vstack((x, y))
            else:
                wvls = np.concatenate((x, nirX[nirX > max(x)]))
                flux = np.concatenate((y, nirY[nirX > max(x)]))
                self.bb = np.vstack((wvls, flux))
            # Gaussian convolution to smooth std spectrum to MKIDs median resolution
            new_x, new_y = gaussian_convolution(self.bb[0], self.bb[1], x_en_min=self.energy_stop.value,
                                                x_en_max=self.energy_start.value, flux_units="lambda", r=self.r, plots=False)
        else:
            getLogger(__name__).info('Standard Spectrum spans whole energy range - no need to perform blackbody fit')
            # Gaussian convolution to smooth std spectrum to MKIDs median resolution
            std_stop = (c.h * c.c) / (self.std_wvls[0].to(u.m) * c.e)
            std_start = (c.h * c.c) / (self.std_wvls[-1].to(u.m) * c.e)
            new_x, new_y = gaussian_convolution(x, y, x_en_min=std_start.value, x_en_max=std_stop.value,
                                                flux_units="lambda", r=self.r, plots=False)
        return new_x, new_y

    def calculate_response_curve(self):
        """
        Divide the MEC Spectrum by the rebinned and gaussian convolved standard spectrum
        """
        curve_x = self.rebin_std[0]
        curve_y = self.rebin_std[1] / self.mkid[1]
        if len(curve_x) == 1:
            curve_x = [self.wvl_bin_edges.to(u.Angstrom).value[0], self.wvl_bin_edges.to(u.Angstrom).value[1]]
            curve_y = [curve_y[0] for i in range(2)]
            spl = InterpolatedUnivariateSpline(curve_x, curve_y, w=None, k=1) #TODO figure out weights
        else:
            spl = InterpolatedUnivariateSpline(curve_x, curve_y, w=None, k=self.fit_order) #TODO figure out weights
        self.curve = spl, np.vstack((curve_x, curve_y))
        return self.curve

    def plot_summary(self, save_name='summary_plot.pdf'):
        figure = plt.figure()
        gs = gridspec.GridSpec(2, 2)
        axes_list = np.array([figure.add_subplot(gs[0, 0]), figure.add_subplot(gs[0, 1]),
                              figure.add_subplot(gs[1, 0]), figure.add_subplot(gs[1, 1])])
        axes_list[0].imshow(np.sum(self.cube, axis=0))
        axes_list[0].set_title('MKID Instrument Image of Standard', size=8)

        std_idx = np.where(np.logical_and(self.wvl_bin_edges.to(u.nm)[0] < self.std_wvls.to(u.nm), self.std_wvls.to(u.nm)
                                          < self.wvl_bin_edges.to(u.nm)[-1]))
        conv_idx = np.where(np.logical_and(self.wvl_bin_edges.to(u.Angstrom).value[0] < self.conv[0], self.conv[0]
                                           < self.wvl_bin_edges.to(u.Angstrom).value[-1]))

        axes_list[1].plot(self.std_wvls.value[std_idx], self.std_flux.value[std_idx], label='Standard Spectrum')
        if self.bb:
            axes_list[1].step(self.bb[0], self.bb[1], where='mid', label='BB fit')
        axes_list[1].plot(self.conv[0][conv_idx], self.conv[1][conv_idx], label='Convolved Spectrum')
        axes_list[1].step(self.rebin_std[0], self.rebin_std[1]/self.contrast, where='mid',
                          label='Rebinned Standard')
        axes_list[1].set_xlabel('Wavelength (A)')
        axes_list[1].set_ylabel('Flux (erg/s/cm^2)')
        axes_list[1].legend(loc='upper right', prop={'size': 6})

        axes_list[2].step(self.mkid[0], self.mkid[1], where='mid',
                          label='MKID Histogram of Object')
        axes_list[2].set_title('Object Histograms', size=8)
        axes_list[2].legend(loc='upper right', prop={'size': 6})
        axes_list[2].set_xlabel('Wavelength (A)')
        axes_list[2].set_ylabel('counts/s/cm^2/A')

        spl, curve = self.curve
        print(spl(curve[0]))
        axes_list[3].step(curve[0], curve[1], label='MKID Spectrum/Reference Spectrum')
        axes_list[3].plot(curve[0], spl(curve[0]), label='Spline Fit')
        axes_list[3].set_title('Response Curve', size=8)

        axes_list[0].tick_params(labelsize=8)
        axes_list[1].tick_params(labelsize=8)
        axes_list[2].tick_params(labelsize=8)
        axes_list[3].tick_params(labelsize=8)
        plt.tight_layout()
        plt.savefig(save_name)
        return axes_list


class ResponseCurve:
    def __init__(self, file_path=None, curve=None, configuration=None, wvl_bin_edges=None, cube=None,
                 solution_name='spectral_solution'):
        # default parameters
        self._parse = True
        # load in arguments
        self._file_path = os.path.abspath(file_path) if file_path is not None else file_path
        self.curve = curve
        self.cfg = configuration
        self.wvl_bin_edges = wvl_bin_edges#*u.Angstrom
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
            save_path = os.path.join(self.cfg.paths.database, self.name)
        else:
            save_path = os.path.join(self.cfg.paths.database, save_name)
        if not save_path.endswith('.npz'):
            save_path += '.npz'
        getLogger(__name__).info("Saving spectrophotometric response curve to {}".format(save_path))
        np.savez(save_path, curve=self.curve, wvl_bin_edges=self.wvl_bin_edges.to(u.Angstrom), cube=self.cube, configuration=self.cfg)
        self._file_path = save_path  # new file_path for the solution

    def load(self, file_path, file_mode='c'):
        """
        loads in a response curve from a saved npz file and sets relevant attributes
        """
        getLogger(__name__).debug("Loading solution from {}".format(file_path))
        keys = ('curve', 'configuration')
        npz_file = np.load(file_path, allow_pickle=True, encoding='bytes', mmap_mode=file_mode)
        for key in keys:
            if key not in list(npz_file.keys()):
                raise AttributeError('{} missing from {}, solution malformed'.format(key, file_path))
        self.npz = npz_file
        self.curve = self.npz['curve']
        self.cfg = self.npz['configuration']
        self.wvl_bin_edges = self.npz['wvl_bin_edges']
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
        getLogger(__name__).warning(
            'Could not find spectrum for {} at {},{} in SDSS catalog'.format(object_name, coords.ra, coords.dec))
        spectrum_file = None
        return spectrum_file
    spec = SDSS.get_spectra(matches=result)
    data = spec[0][1].data
    lamb = 10 ** data['loglam'] * u.AA
    flux = data['flux'] * 1e-17 * u.Unit('erg cm-2 s-1 AA-1')
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
            coords = None
    if not coords:
        getLogger(__name__).error('No coordinates found for spectrophotometric calibration object')
    return coords


def satellite_spot_contrast(lam, ref_contrast=2.72e-3, ref_wvl=1.55*1e4):  # 2.72e-3 number from Currie et. al. 2018b
    """

    :param lam:
    :param ref_contrast:
    :param ref_wvl:
    :return:
    """
    contrast = ref_contrast * (ref_wvl / lam) ** 2
    return contrast


def load_solution(sc, singleton_ok=True):
    """sc is a solution filename string, a ResponseCurve object, or a mkidpipeline.config.MKIDSpeccalDescription"""
    global _loaded_solutions
    if not singleton_ok:
        raise NotImplementedError('Must implement solution copying')
    if isinstance(sc, ResponseCurve):
        return sc
    if isinstance(sc, definitions.MKIDSpeccalDescription):
        sc = sc.path
    sc = sc if os.path.isfile(sc) else os.path.join(mkidpipeline.config.config.paths.database, sc)
    try:
        return _loaded_solutions[sc]
    except KeyError:
        _loaded_solutions[sc] = ResponseCurve(file_path=sc)
    return _loaded_solutions[sc]


def fetch(solution_descriptors, config=None, ncpu=None, remake=False, **kwargs):
    try:
        solution_descriptors = solution_descriptors.speccals
    except AttributeError:
        pass

    scfg = mkidpipeline.config.PipelineConfigFactory(step_defaults=dict(speccal=StepConfig()), cfg=config, ncpu=ncpu,
                                                    copy=True)
    solutions = {}
    if not remake:
        for sd in solution_descriptors:
            try:
                if not os.path.exists(sd.path):
                    raise IOError
                solutions[sd.id] = load_solution(sd.path)
            except IOError:
                pass
            except Exception as e:
                getLogger(__name__).info(f'Failed to load {sd} due to a {e} error')
                raise

    for sd in set(sd for sd in solution_descriptors if sd.id not in solutions):
        wavcal = [list(sd.data.wavecal.values())[0].path for sd in solution_descriptors][0]
        scfg.register('wave_sol', wavcal, update=True)
        cal = SpectralCalibrator(scfg, solution_name=sd.path, data=sd.data,
                                 use_satellite_spots=True if sd.aperture == 'satellite' else False,
                                 aperture=sd.aperture if not sd.aperture == 'satellite' else None,
                                 std_path=sd.data.spectrum, ncpu=ncpu if ncpu is not None else 1)
        try:
            cal.run(**kwargs)
            solutions[sd.id] = cal.solution
            getLogger(__name__).info(f'{sd} made.')
        except IOError as e:
            solutions[sd.id] = None
            getLogger(__name__).error(f'Unable to fetch solution for {sd}: {e}')
    return solutions


def apply(fits_file, wvl_bins, solution='', overwrite=False):
    # TODO get wvl bins from fits header
    """
    applies a sepccal solution to a spectral cube in a fits file format.
    :param fits_file: file to be calibrated
    :param wvl_bins: wavelenght bin edges of the spectral cube in the fits file (in Angstroms)
    :param solution: file path the speccal solution to use
    :param overwrite: if True will overwrite the spectral cube in the fits file with the calibrated cube. if False
    will save a new fits file in the same location with and added '_calibrated'
    :return: flux calibrated spectral cube
    """
    ff = fits.open(fits_file)
    hdr = fits.getheader(fits_file, 0)
    if hdr['E_SPECAL'] != 'none':
        getLogger(__name__).info(f'{fits_file} already flux calibrated, skipping calibration.')
        flux_calibrated_cube = ff[1].data
        return flux_calibrated_cube
    s_cube = ff[1].data
    if len(s_cube.shape) == 2:
        getLogger(__name__).warning(f'Cube only has spatial dimensions. Make sure that you have intended to run a '
                                    f'spectral calibration for photometry in 1 spectral band only')
        s_cube = np.array([s_cube])
    elif s_cube.shape[0] != (len(wvl_bins) - 1):
        getLogger(__name__).warning(f'Mismatch between shape of spectral cube wavelength axis '
                                    f'({s_cube.shape[0]}) and number of specified wavelength bins '
                                    f'({len(wvl_bins) - 1})')
        return
    flux_calibrated_cube = np.zeros_like(s_cube)
    soln = ResponseCurve(solution)
    curve = ResponseCurve(solution).curve.flatten()[0]
    soln_wvl_bins = soln.wvl_bin_edges
    soln_wvl_centers = np.array([(soln_wvl_bins[i]+soln_wvl_bins[i+1])/2 for i in range(len(soln_wvl_bins) -1)])
    wvl_bin_centers = np.array([(wvl_bins[i]+wvl_bins[i+1])/2 for i in range(len(wvl_bins) -1)])
    if min(wvl_bin_centers) < min(soln_wvl_centers) or max(wvl_bin_centers) > max(soln_wvl_centers):
        getLogger(__name__).warning(f'Spectral cube wavelengths {wvl_bin_centers} exceed the bounds of the wavelengths '
                                    f'used to make the calibration {soln_wvl_centers}. Will not calibrate points outside'
                                    f'of {soln_wvl_centers[0]} - {soln_wvl_centers[-1]}')
    if not wvl_bins:
        raise ValueError('wavelength bins for the spectral cube must be specified')
    if np.all(soln_wvl_centers == wvl_bin_centers):
        for wvl, cube in enumerate(s_cube):
            flux_calibrated_cube[wvl] = s_cube[wvl] * curve(wvl_bin_centers[wvl])
    elif len(soln_wvl_centers) > len(wvl_bin_centers):
        getLogger(__name__).info(f'Calibrating spectral cube that has a lower resolution than Speccal solution '
                                 f'was made with - speccal solution will be rebinned to match spectral cube '
                                 f'binnning')
        resampled = rebin(soln_wvl_centers * 10, curve(soln_wvl_centers * 10), wvl_bins)
        resampled_wvls = resampled[:, 0]
        resampled_curve = resampled[:, 1]
        if not np.all(np.abs(resampled_wvls-wvl_bin_centers) < 1e-2*np.mean(np.diff(wvl_bins))):
            getLogger(__name__).warning('Resampled speccal solution bins do not match wavelength bins of the given fits '
                                        'file within tolerence!')
        for wvl, cube in enumerate(s_cube):
            flux_calibrated_cube[wvl] = s_cube[wvl] * resampled_curve[wvl]
    else:
        #if speccal solution is lower resolution get weighted average to find specweight for each wvl
        getLogger(__name__).warning(f'Calibrating spectral cube that has a higher resolution than Speccal solution '
                                    f'was made with - you may want to generate a new speccal solution or decrease '
                                    f'the resolution of your spectral cube')
        for wvl, cube in enumerate(s_cube):
            flux_calibrated_cube[wvl] = s_cube[wvl] * np.full(s_cube[wvl].shape, curve(wvl_bin_centers[wvl]))
    ff[1].data = flux_calibrated_cube
    fits.setval(fits_file, 'E_SPECAL', value=str(soln.name))
    if overwrite:
        ff.writeto(fits_file)
    else:
        ff.writeto(fits_file[:-5] + '_calibrated.fits')
    getLogger(__name__).info(f'Finished calibrating {fits_file}')
    return flux_calibrated_cube
