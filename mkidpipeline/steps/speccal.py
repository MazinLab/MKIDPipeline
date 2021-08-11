"""
Loads a standard spectrum from and convolves and rebind it ot match the MKID energy resolution and bin size. Then
generates an MKID spectrum of the object by performing photometry (aperture or PSF) on the MKID image. Finally
divides the flux values for the standard by the MKID flux values for each bin to get a calibration curve.

Assumes h5 files are wavelength calibrated, and they should also first be flatcalibrated and linearity corrected
(deadtime corrected)
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
                                           ' (in Angstroms). Defaults to nyquist sampling the energy resolution')) #TODO


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
        self.coords = coords  # SkyCoord object
        self.spectrum_file = None
        self.k = 5.03411259e7

    def get(self):
        """
        creates a spectrum directory, populates it with the spectrum file either pulled from the ESO catalog, SDSS
        catalog, a URL, or a specified path to a .txt file and returns the wavelength and flux column in the
        appropriate units
        :return: wavelengths (Angstroms), flux (erg/s/cm^2/A)
        """
        self.coords = get_coords(object_name=self.object_name, ra=self.ra, dec=self.dec)
        data = self.fetch_spectra()
        return data[:, 0], data[:, 1]

    def fetch_spectra(self):
        """
        called from get(), searches either a URL, ESO catalog or uses astroquery.SDSS to search the SDSS catalog. Puts
        the retrieved spectrum in a '/spectrum/' folder in self.save_dir
        :return:
        """
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
            data[:, 1] = data[:, 1] * 1e-16
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
    def __init__(self, configuration=None, solution_name='solution.npz', interpolation=None, aperture=None,
                 data=None, use_satellite_spots=True, save_path=None, platescale=.0104, std_path='',
                 photometry_type='aperture', summary_plot=True, ncpu=1):

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
        # various spectra
        self.std = None
        self.rebin_std = None
        self.bb = None
        self.mkid = None
        self.conv = None
        self.curve = None
        self.cube = None
        self.contrast = None

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
        try:
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
            if plot or (plot is None and self.plots is 'summary'):
                save_name = self.solution_name.rpartition(".")[0] + ".pdf"
                self.plot_summary(save_name=save_name)
        except KeyboardInterrupt:
            getLogger(__name__).info("Keyboard shutdown requested ... exiting")

    def load_absolute_spectrum(self):
        """
         Extract the MEC measured spectrum of the spectrophotometric standard by breaking data into spectral cubes
         and performing photometry (aperture or psf) on each spectral frame
         """
        getLogger(__name__).info('performing {} photometry on MEC spectrum'.format(self.photometry))
        if len(self.data.obs) == 1:
            hdul = Photontable(self.data.obs[0].h5).get_fits(weight=True, rate=True, cube_type='wave',
                                                             bin_edges=self.wvl_bin_edges, bin_type='energy')
            cube = np.array(hdul['SCIENCE'].data, dtype=np.double)
        else:
            cube = []
            for wvl in range(len(self.wvl_bin_edges) - 1):
                getLogger(__name__).info('using wavelength range {} - {}'
                                         .format(self.wvl_bin_edges[wvl].to(u.nm).value,
                                                 self.wvl_bin_edges[wvl + 1].to(u.nm).value))
                drizzled = form(self.data, mode='spatial', wave_start=self.wvl_bin_edges[wvl].to(u.nm).value,
                                wave_stop=self.wvl_bin_edges[wvl + 1].to(u.nm).value, pixfrac=0.5,
                                wcs_timestep=1, exclude_flags=PROBLEM_FLAGS, usecache=False,
                                duration=min([o.duration for o in self.data.obs]), ncpu=self.ncpu,
                                derotate=not self.use_satellite_spots)
                getLogger(__name__).info(('finished image {}/ {}'.format(wvl + 1.0, len(self.wvl_bin_edges) - 1)))
                cube.append(drizzled.cps)
            self.cube = np.array(cube)
        n_wvl_bins = len(self.wvl_bin_edges) - 1

        wvl_bin_centers = [(a + b) / 2 for a, b in zip(self.wvl_bin_edges, self.wvl_bin_edges[1::])]

        self.mkid = np.zeros((n_wvl_bins, n_wvl_bins))
        self.mkid[0] = wvl_bin_centers
        if self.use_satellite_spots:
            fluxes = mec_measure_satellite_spot_flux(self.cube, wvl_start=self.wvl_bin_edges[:-1],
                                                     wvl_stop=self.wvl_bin_edges[1:], platescale=self.platescale)
            self.mkid[1] = np.nanmean(fluxes, axis=1)
        else:
            try:
                x, y, r = self.aperture
            except ValueError:
                getLogger(__name__).warning('Aperture for the speccal must be in the format (x/RA, y/DEC, r) OR '
                                            'satellite, instead got {self.aperture}')
            for i in np.arange(n_wvl_bins):
                frame = cube[:, :, i]
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
        standard = StandardSpectrum(save_path=self.save_path, std_path=self.std_path,
                                    name=self.data.obs[0].header['OBJECT'], ra=self.data.obs[0].header['RA'],
                                    dec=self.data.obs[0].header['DEC'])
        std_wvls, std_flux = standard.get()  # standard star object spectrum in ergs/s/Angs/cm^2
        self.std = np.hstack(std_wvls*u.Angstrom, std_flux*u.erg*(1/u.s)*(1/u.Angstrom)*(1/u.cm**2))
        conv_wvls_rev, conv_flux_rev = self.extend_and_convolve(self.std[0], self.std[1])
        # convolved spectrum comes back sorted backwards, from long wvls to low which screws up rebinning
        self.conv = np.hstack((conv_wvls_rev[::-1], conv_flux_rev[::-1]))

        # rebin cleaned spectrum to flat cal's wvlBinEdges
        rebin_std_data = rebin(self.conv[0], self.conv[1], self.wvl_bin_edges)
        wvl_bin_centers = [(a + b) / 2 for a, b in zip(self.wvl_bin_edges, self.wvl_bin_edges[1::])]

        if self.use_satellite_spots:
            for i, wvl in enumerate(wvl_bin_centers):
                self.contrast[i] = satellite_spot_contrast(wvl)
                rebin_std_data[i, 1] = rebin_std_data[i, 1] * self.contrast[i]
        self.rebin_std = np.hstack(np.array(rebin_std_data[:, 0]), np.array(rebin_std_data[:, 1]))

    def extend_and_convolve(self, x, y):
        """
        BB Fit to extend standard spectrum to 1500 nm and to convolve it with a gaussian kernel corresponding to the
        energy resolution of the detector. If spectrum spans whole MKID range will just convolve with the gaussian
        """
        if np.round(x[-1]) < self.wvl_bin_edges[-1]:
            fraction = 1.0 / 3.0
            nirX = np.arange(int(x[int((1.0 - fraction) * len(x))]), self.wvl_bin_edges[-1])
            T, nirY = fit_blackbody(x, y, fraction=fraction, new_wvls=nirX)
            if np.any(x >= self.wvl_bin_edges[-1]):
                self.bb = np.hstack((x, y))
            else:
                wvls = np.concatenate((x, nirX[nirX > max(x)]))
                flux = np.concatenate((y, nirY[nirX > max(x)]))
                self.bb = np.hstack((wvls, flux))
            # Gaussian convolution to smooth std spectrum to MKIDs median resolution
            new_x, new_y = gaussian_convolution(self.bb[0], self.bb[1], x_en_min=self.energy_stop,
                                                x_en_max=self.energy_start, flux_units="lambda", r=self.r, plots=False)
        else:
            getLogger(__name__).info('Standard Spectrum spans whole energy range - no need to perform blackbody fit')
            # Gaussian convolution to smooth std spectrum to MKIDs median resolution
            std_stop = (c.h * c.c) / (self.std[0][0].to(u.m) * c.e)
            std_start = (c.h * c.c) / (self.std[0][-1].to(u.m) * c.e)
            new_x, new_y = gaussian_convolution(x, y, x_en_min=std_start, x_en_max=std_stop, flux_units="lambda", r=self.r,
                                                plots=False)
        return new_x, new_y

    def calculate_response_curve(self):
        """
        Divide the MEC Spectrum by the rebinned and gaussian convolved standard spectrum
        """
        curve_x = self.rebin_std[0]
        curve_y = self.rebin_std[1] / self.mkid
        spl = InterpolatedUnivariateSpline(curve_x, curve_y, w=None) #TODO figure out weights
        self.curve = spl, np.vstack((curve_x, curve_y))
        return self.curve

    def plot_summary(self, save_name='summary_plot.pdf'):
        figure = plt.figure()
        gs = gridspec.GridSpec(2, 2)
        axes_list = np.array([figure.add_subplot(gs[0, 0]), figure.add_subplot(gs[0, 1]),
                              figure.add_subplot(gs[1, 0]), figure.add_subplot(gs[1, 1])])
        axes_list[0].imshow(np.sum(self.cube, axis=0))
        axes_list[0].set_title('MKID Instrument Image of Standard', size=8)

        std_idx = np.where(np.logical_and(self.wvl_bin_edges[0] < self.std[0], self.std[0] < self.wvl_bin_edges[-1]))
        conv_idx = np.where(np.logical_and(self.wvl_bin_edges[0] < self.conv[0], self.conv[0] < self.wvl_bin_edges[-1]))

        axes_list[1].step(self.std[0][std_idx], self.std[1][std_idx], where='mid',
                          label='Standard Spectrum')
        if self.bb:
            axes_list[1].step(self.bb[0], self.bb[1], where='mid', label='BB fit')
        axes_list[1].step(self.conv[0][conv_idx], self.conv[1][conv_idx], where='mid', label='Convolved Spectrum')
        axes_list[1].set_xlabel('Wavelength (A)')
        axes_list[1].set_ylabel('Flux (erg/s/cm^2)')
        axes_list[1].legend(loc='upper right', prop={'size': 6})

        axes_list[2].step(self.rebin_std[0], self.mkid, where='mid',
                          label='MKID Histogram of Object')
        axes_list[2].set_title('Object Histograms', size=8)
        axes_list[2].legend(loc='upper right', prop={'size': 6})
        axes_list[2].set_xlabel('Wavelength (A)')
        axes_list[2].set_ylabel('counts/s/cm^2/A')

        spl, x, y = self.curve
        axes_list[3].plot(x, y)
        axes_list[3].plot(x, spl(x))
        axes_list[3].set_title('Response Curve', size=8)
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
        self.curve = curve # TODO should be spline (can call ResponseCurve.curve(xs)) to get flux caliibrated values
        self.cfg = configuration
        self.wvl_bin_edges = wvl_bin_edges
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
        np.savez(save_path, curve=self.curve, wvl_bin_edges=self.wvl_bin_edges, cube=self.cube, configuration=self.cfg)
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
    if isinstance(sc, mkidpipeline.config.MKIDSpeccalDescription):
        sc = sc.path
    sc = sc if os.path.isfile(sc) else os.path.join(mkidpipeline.config.config.paths.database, sc)
    try:
        return _loaded_solutions[sc]
    except KeyError:
        _loaded_solutions[sc] = ResponseCurve(file_path=sc)
    return _loaded_solutions[sc]


def fetch(dataset, config=None, ncpu=None, remake=False, **kwargs):
    solution_descriptors = dataset.speccals

    cfg = mkidpipeline.config.PipelineConfigFactory(step_defaults=dict(speccal=StepConfig()), cfg=config, ncpu=ncpu,
                                                    copy=True)
    for sd in dataset.wavecals:
        wavcal = sd.path

    solutions = []
    for sd in solution_descriptors:
        sf = sd.path
        if os.path.exists(sf) and not remake:
            solutions.append(load_solution(sf))
        else:
            if 'speccal' not in cfg:
                scfg = mkidpipeline.config.load_task_config(StepConfig())
            else:
                scfg = cfg.copy()
            try:
                scfg.register('wave_sol', wavcal, update=True)
            except AttributeError:
                scfg.register('wave_sol', wavcal, update=True)
            cal = SpectralCalibrator(scfg, solution_name=sf, data=sd.data,
                                     use_satellite_spots=True if sd.aperture=='satellite' else False,
                                     aperture = sd.aperture if not sd.aperture == 'satellite' else None,
                                     std_path=sd.data.spectrum, ncpu=ncpu if ncpu else 1)
            cal.run(**kwargs)
            # solutions.append(load_solution(sf))  # don't need to reload from file
            solutions.append(cal.solution)  # TODO: solutions.append(load_solution(cal.solution))
    return solutions


def apply(s_cube, wvl_bins=None, config=None, solution=''):
    config = mkidpipeline.config.PipelineConfigFactory(step_defaults=dict(pixcal=StepConfig()), cfg=config, copy=True)
    flux_calibrated_cube = np.zeros_like(s_cube)
    soln = ResponseCurve(solution).curve
    soln_wvl_bins = ResponseCurve(solution).wvl_bin_edges
    if not wvl_bins:
        raise ValueError('wavelength bins for the spectral cube must be specified')
    if soln_wvl_bins == wvl_bins:
        for (wvl, x, y), idx in np.ndenumerate(s_cube):
            flux_calibrated_cube[wvl, x, y] = s_cube[wvl, x, y] * soln(s_cube[wvl, x, y]) # TODO make sure this is correct
        return flux_calibrated_cube
    elif len(soln_wvl_bins) > len(wvl_bins):
        # TODO integrate over solution bins
        return flux_calibrated_cube
    else:
        # TODO throw warning that you are oversampling the energy resolution of the solution
        #  - might want to make a new solution
        return flux_calibrated_cube