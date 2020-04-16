'''
Author: Sarah Steiger   Date: April 10, 2020

Contains the SpectrumManager class for handling the loading and saving of standard spectrum for the
Spectrophotometric calibration in the MKIDPipeline.

List of ESO standard spectra: https://www.eso.org/sci/observing/tools/standards/spectra/stanlis.html

Adapted from the MKIDStandard class in the ARCONS pipeline
'''

import numpy as np
from astroquery.sdss import SDSS
import astropy.coordinates as coord
import os
from mkidcore.corelog import getLogger
import urllib.request as request
from urllib.error import URLError
import shutil
from contextlib import closing
from scipy.constants import *
from specutils import Spectrum1D
import ftplib

class StandardSpectrum:
    '''
    replaces the MKIDStandards class from the ARCONS pipeline for MEC.
    '''
    def __init__(self, save_path='', url_path=None, object_name=None, object_ra=None, object_dec=None, coords=None,
                 reference_wavelength=5500):
        self.save_dir = save_path
        self.ref_wvl = reference_wavelength
        self.object_name = object_name
        self.ra = object_ra
        self.dec = object_dec
        self.url_path = url_path
        self.coords = coords #SkyCoord object
        self.spectrum_file = None
        self.k = ((1.0*10**-10)/(1.0*10**7))/h/c

    def get(self):
        self.create_directory()
        self.get_coords()
        self.fetch_spectra()
        data = self.load_spectra()
        return data[:, 0], data[:, 1]

    def create_directory(self):
        '''
        creates a spectrum directory in the save directory to put the spectra. If not called then the spectrum will
        just be saved in save_path
        '''
        if os.path.exists(self.save_dir + '/spectra/'):
            getLogger(__name__).info('Spectrum directory already exists in {}, not going to make a new one'.format(self.save_dir))
        else:
            os.mkdir(self.save_dir + '/spectra/')
        self.savedir = self.save_dir + '/spectra/'

    def get_coords(self):
        '''
        finds the SkyCoord object given a specified ra and dec or object_name
        :return: SkyCoord object
        '''
        if self.object_name is None and self.coords is None:
            getLogger(__name__).error('Need to specify either an object name or coordinate')
        if not self.object_name:
            self.coords = coord.SkyCoord(self.ra, self.dec, unit=('hourangle', 'deg'))
        else:
            self.coords = coord.SkyCoord.from_name(self.object_name)
        if not self.coords:
            getLogger(__name__).error('No coordinates found for spectrophotometric calibration object')
        return self.coords

    def fetch_spectra(self):
        '''

        :return:
        '''
        if self.url_path is not None:
            self.spectrum_file = self.fetch_spectra_URL()
            return self.spectrum_file
        else:
            self.spectrum_file = self.fetch_spectra_ESO()
            if not self.spectrum_file:
                self.spectrum_file = self.fetch_spectra_SDSS()
            if not self.spectrum_file:
                getLogger(__name__).warning('Could not find standard spectrum for this object in SDSS or ESO catalog')
            return self.spectrum_file

    def name_to_ESO_extension(self):
        '''
        converts an input object name string to the standard filename format for the ESO standards catalog on their
        ftp server
        :return:
        '''
        extension = ''
        for char in self.object_name:
            if char.isupper():
                extension = extension + char.lower()
            elif char == '+':
                extension = extension
            elif char == '-':
                extension = extension + '_'
            else:
                extension = extension + char
        return 'f{}.dat'.format(extension)

    def fetch_spectra_ESO(self):
        '''
        fetches a standard spectrum from the ESO catalog and downloads it to self.savedir if it exist. Requires
        self.object_name to not be None
        :return:
        '''
        getLogger(__name__).info('Looking for {} spectrum in ESO catalog'.format(self.object_name))
        ext = self.name_to_ESO_extension()
        path = 'ftp://ftp.eso.org/pub/stecf/standards/'
        folders = np.array(['ctiostan/', 'hststan/', 'okestan/', 'wdstan/', 'Xshooter/'])
        if os.path.exists(self.save_dir + ext):
            getLogger(__name__).info('Spectrum already loaded, will not be reloaded')
            self.spectrum_file = self.save_dir + ext
            return self.spectrum_file
        for folder in folders:
            try:
                with closing(request.urlopen(path + folder + ext)) as r:
                    with open(self.save_dir + ext, 'wb') as f:
                        shutil.copyfileobj(r, f)
                self.spectrum_file = self.save_dir + ext
            except URLError:
                pass
        if self.spectrum_file is not None:
            getLogger(__name__).info('Spectrum loaded for {} from ESO catalog'.format(self.object_name))
            return self.spectrum_file

    def fetch_spectra_SDSS(self):
        '''
        saves a textfile in self.save_dir where the first column is the wavelength in angstroms and the second
        column is fluc in erg cm-2 s-1 AA-1
        :return: the path to the saved spectrum file
        '''
        if os.path.exists(self.save_dir + self.object_name + 'spectrum.dat'):
            getLogger(__name__).info('Spectrum already loaded, will not be reloaded')
            self.spectrum_file = self.save_dir + self.object_name + 'spectrum.dat'
            return self.spectrum_file
        getLogger(__name__).info('Looking for {} spectrum in SDSS catalog'.format(self.object_name))
        result = SDSS.query_region(self.coords, spectro=True)
        if not result:
            getLogger(__name__).warning('Could not find spectrum for {} at ({},{}) in SDSS catalog')\
                .format(self.object_name, self.coords[0], self.coords[1])
        spec = SDSS.get_spectra(matches=result)
        data = spec[0][1].data
        lamb = 10**data['loglam'] * u.AA
        flux = data['flux'] * 10 ** -17 * u.Unit('erg cm-2 s-1 AA-1')
        spectrum = Spectrum1D(spectral_axis=lamb, flux=flux)
        res = np.array([spectrum.spectral_axis, spectrum.flux])
        res = res.T
        np.savetxt(self.save_dir + self.object_name + 'spectrum.dat', res, fmt='%1.4e')
        self.spectrum_file = self.save_dir + self.object_name + 'spectrum.dat'
        if self.spectrum_file is not None:
            getLogger(__name__).info('Spectrum loaded for {} from SDSS catalog'.format(self.object_name))
            return self.spectrum_file

    def fetch_spectra_URL(self):
        '''
        grabs the spectrum from a given URL and saves it in self.savedir
        :return: the file path to the saved spectrum
        '''
        if os.path.exists(self.save_dir + self.object_name + 'spectrum.dat'):
            getLogger(__name__).info('Spectrum already loaded, will not be reloaded')
            self.spectrum_file = self.save_dir + self.object_name + 'spectrum.dat'
            return self.spectrum_file
        if not self.url_path:
            getLogger(__name__).warning('No URL path specified')
            pass
        else:
            with closing(request.urlopen(self.url_path)) as r:
                with open(self.save_dir + self.object_name + 'spectrum.dat', 'wb') as f:
                    shutil.copyfileobj(r, f)
            self.spectrum_file = self.save_dir + self.object_name + 'spectrum.dat'
            return self.spectrum_file

    def load_spectra(self):
        '''
        get spectra in a numpy array that can be used by the rest of the spectrophotometric cal
        :return:
        '''
        if not self.spectrum_file:
            getLogger(__name__).error('Need to fetch the required h5 file first with self.fetch_spectra')
            return
        else:
            array = np.loadtxt(self.spectrum_file)
            return array

    def get_reference_flux(self, a):
        '''

        :return:
        '''
        x = a[:, 0]
        y = a[:, 1]
        index = np.searchsorted(x, self.ref_wvl)
        if index < 0:
            index = 0
        if index > x.size - 1:
            index = x.size - 1
        return y[index]

    def normalize_flux(self, a):
        '''

        :param a:
        :return:
        '''
        reference_flux = self.get_reference_flux(a)
        a[:, 1] /= reference_flux
        return a

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