"""
Author: Sarah Steiger                 Date: 11/24/2020

Photometry utility functions
"""
import numpy as np
from astropy.table import Table
from photutils.detection import IRAFStarFinder
from photutils.psf import IntegratedGaussianPRF, DAOGroup
from photutils.background import MMMBackground, MADStdBackgroundRMS
from astropy.modeling.fitting import LevMarLSQFitter
from astropy.stats import gaussian_sigma_to_fwhm
import scipy.ndimage as ndimage
from photutils import aperture_photometry
from photutils import CircularAperture
from photutils import CircularAnnulus
from photutils.psf import IterativelySubtractedPSFPhotometry
from photutils.psf import BasicPSFPhotometry


def aper_photometry(image, obj_position, radius):
    """
    performs aperture photometry on an image. Sums all of the photons in a circular aperture around the target and
    subtracts off the sky background. Sky background is determined by looking at the average counts in an annulus of
    outer radius (radius + 0.5*radius) around the target aperture

    :param image: 2D array, image on which to perform aperture photometry
    :param obj_position: tuple (in pixels)
    :param radius: in pixels, radius of circular aperture in which to sum target photons
    :return: sky subtracted object flux
    """
    position = np.array([obj_position])
    circ_aperture = CircularAperture(position, r=radius)
    annulus_aperture = CircularAnnulus(position, r_in=radius, r_out=radius + (0.5 * radius))
    apers = [circ_aperture, annulus_aperture]
    photometry_table = aperture_photometry(image, apers)
    object_flux = photometry_table['aperture_sum_0']
    bkgd_mean = photometry_table['aperture_sum_1'] / annulus_aperture.area
    sky_flux = bkgd_mean * circ_aperture.area
    return object_flux - sky_flux


def psf_photometry(img, sigma_psf, aperture=3, x0=None, y0=None, filter=True, sigma_filter=1):
    """
    performs PSF photometry on an image. If x0 and y0 are None will attempt to locate the target by searching for the
    brightest PSF in the field

    :param img: 2D array, image on which to perform PSF photometry
    :param sigma_psf: float, standard deviation of the PSF
    :param aperture: int, size of the paerture (pixels)
    :param x0: x position of the target (pixels)
    :param y0: y position of the target (pixels)
    :param filter: If True will apply a gaussian filter to the image with standard deviation sigma_filter before
     performing PSF photometry
    :param sigma_filter: standard deviation of gaussian filter to apply to the image
    :return: x0 column of photometry table, y0 column of photometry table, flux column of photometry table
    """
    if filter:
        image = ndimage.gaussian_filter(img, sigma=sigma_filter, order=0)
    else:
        image = img
    bkgrms = MADStdBackgroundRMS()
    std = bkgrms(image[image != 0])
    iraffind = IRAFStarFinder(threshold=2 * std, fwhm=sigma_psf * gaussian_sigma_to_fwhm, minsep_fwhm=0.01,
                              roundhi=5.0, roundlo=-5.0, sharplo=0.0, sharphi=2.0)
    daogroup = DAOGroup(2.0 * sigma_psf * gaussian_sigma_to_fwhm)
    mmm_bkg = MMMBackground()
    fitter = LevMarLSQFitter()
    psf_model = IntegratedGaussianPRF(sigma=sigma_psf)
    if x0 and y0:
        psf_model.x_0.fixed = True
        psf_model.y_0.fixed = True
        pos = Table(names=['x_0', 'y_0'], data=[x0, y0])
        photometry = BasicPSFPhotometry(group_maker=daogroup, bkg_estimator=mmm_bkg, psf_model=psf_model,
                                        fitter=LevMarLSQFitter(), fitshape=(11, 11))
        res = photometry(image=image, init_guesses=pos)
        return res['x_0'], res['y_0'], res['flux_0']
    photometry = IterativelySubtractedPSFPhotometry(finder=iraffind, group_maker=daogroup, bkg_estimator=mmm_bkg,
                                                    psf_model=psf_model, fitter=fitter, niters=10, fitshape=(11, 11),
                                                    aperture_radius=aperture)
    res = photometry(image=image)
    return res['x_0'], res['y_0'], res['flux_0']
