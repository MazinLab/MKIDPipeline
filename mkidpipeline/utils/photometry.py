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
from photutils.psf import BasicPSFPhotometry
from astropy.modeling import fitting
from astropy.modeling.models import *
import matplotlib.pyplot as plt
from scipy.interpolate import griddata


def get_aperture_radius(lam, platescale):
    """
    function to get the diffraction limited aperture radius for MEC
    :param lam: wavelength in angstroms
    :param platescale: platescale in arcseconds/pix
    :return: radius (in pixels)
    """
    D = 8.2 * 1e10
    theta_rad = 1.22 * (lam/D)
    a = 4.8481368e-6
    theta_mas = theta_rad * (1/a)
    r = 0.5 * theta_mas * (1/platescale)
    return r

def aper_photometry(image, obj_position, radius, box_size=10, bkgd_subtraction_type='plane'):
    """
    performs aperture photometry on an image. Sums all of the photons in a circular aperture around the target and
    subtracts off the sky background. Sky background is determined by looking at the average counts in an annulus of
    outer radius (radius + 0.5*radius) around the target aperture

    :param image: 2D array, image on which to perform aperture photometry
    :param obj_position: tuple (in pixels)
    :param radius: in pixels, radius of circular aperture in which to sum target photons
    :param bkgd_subtraction_type: 'plane' will perform a plane fit, 'annulus' will find the average counts in an
     annulus surrounding the aperture
    :return: sky subtracted object flux in counts/sec
    """
    position = np.array(obj_position)
    if bkgd_subtraction_type == 'plane':
        xp = obj_position[1]
        yp = obj_position[0]
        crop_img = image[int(xp) - box_size:int(xp) + (box_size+1), int(yp) - box_size:int(yp) + (box_size+1)]
        p_back_init = Polynomial2D(degree=1)
        fit_p = fitting.LevMarLSQFitter()
        x, y = np.meshgrid(np.arange(np.shape(crop_img)[0]), np.arange(np.shape(crop_img)[0]))
        p = fit_p(p_back_init, x, y, crop_img)
        background = p(x, y)
        image[int(xp) - box_size:int(xp) + (box_size+1), int(yp) - box_size:int(yp) + (box_size+1)] -= background
        circ_aperture = CircularAperture(position, r=radius)
        photometry_table = aperture_photometry(image, circ_aperture)
        object_flux = photometry_table['aperture_sum']
        return object_flux
    elif bkgd_subtraction_type == 'annulus':
        circ_aperture = CircularAperture(position, r=radius)
        annulus_aperture = CircularAnnulus(position, r_in=radius, r_out=radius+0.5*radius)
        apers = [circ_aperture, annulus_aperture]
        photometry_table = aperture_photometry(image, apers)
        object_flux = photometry_table['aperture_sum_0']
        bkgd_mean = photometry_table['aperture_sum_1'] / annulus_aperture.area
        sky_flux = bkgd_mean * circ_aperture.area
        return object_flux - sky_flux
    else:
        raise KeyError('invalid background subtraction type given ({}), must be either plane or annulus'
                       .format(bkgd_subtraction_type))


def astropy_psf_photometry(img, sigma_psf, aperture=3, x0=None, y0=None, filter=True, sigma_filter=1):
    """
    performs PSF photometry on an image. If x0 and y0 are None will attempt to locate the target by searching for the
    brightest PSF in the field

    :param img: 2D array, image on which to perform PSF photometry
    :param sigma_psf: float, standard deviation of the PSF
    :param aperture: int, size of the aperture (pixels)
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
        pos = Table(names=['x_0', 'y_0'], data=[x0, y0])
        photometry = BasicPSFPhotometry(group_maker=daogroup, bkg_estimator=mmm_bkg, psf_model=psf_model,
                                        fitter=LevMarLSQFitter(), fitshape=(11, 11))
        res = photometry(image=image, init_guesses=pos)
        return res['x_fit'], res['y_fit'], res['flux_0']
    photometry = BasicPSFPhotometry(finder=iraffind, group_maker=daogroup, bkg_estimator=mmm_bkg,
                                                    psf_model=psf_model, fitter=fitter, fitshape=(11, 11),
                                                    aperture_radius=aperture)
    res = photometry(image=image)
    return res['x_0'], res['y_0'], res['flux_0']


def gaussian_fit_psf_photometry(image, obj_pos, box_size=15, bkgd_poly_deg=1, x_std = 0.5, y_std=2, theta=np.pi/4,
                             interpolate=False):
    """
    performs PSF photometry by fitting a plane background model and a gaussian
    :param image: 2D array image
    :param obj_pos: location of the satellite spot (in pixels)
    :param box_size: box size in which to perform the fit
    :param bkgd_poly_deg: degree polynomial with which to it the background
    :param x_std: standard deviation of the gaussian with which to fit the satellite spot - x direction
    :param y_std: standard deviation of the gaussian with which to fit the satellite spot - y direction
    :param theta: angle off 'up' that the spot makes
    :param interpolate: is True will interpolate the image before performing the fit - recommended for non drizzled data
    :return: total satellite spot counts (in counts/sec)
    """

    x_pos = int(obj_pos[1])
    y_pos = int(obj_pos[0])
    frame = image[x_pos-box_size:x_pos+box_size, y_pos-box_size:y_pos+box_size]
    if len(frame[frame==0]) > (len(frame.flatten())/2):
        print('More than half of the frame values are 0 - returning None for the flux')
        return None, None
    else:
        if interpolate:
            frame = interpolateImage(frame)
        x, y = np.meshgrid(np.arange(np.shape(frame)[0]), np.arange(np.shape(frame)[0]))
        p_spot_init = Gaussian2D(x_mean=box_size, y_mean=box_size, x_stddev=x_std, y_stddev=y_std, theta=theta)
        p_back_init = Polynomial2D(degree=bkgd_poly_deg)
        p_init = p_spot_init + p_back_init
        fit_p = fitting.LevMarLSQFitter()
        p = fit_p(p_init, x, y, frame)
        signal = p[0](x,y)
        return np.sum(signal)

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


def mec_measure_satellite_spot_flux(cube, aperradii=None, wvl_start=None, wvl_stop=None, wcs=None,
                                    platescale=0.0104, D=8.2):
    """
    performs aperture photometry using an adaptation of the racetrack aperture from the polarimetry mode of the
    GPI pipeline (http://docs.planetimager.org/pipeline/usage/tutorial_polphotometry.html)

    :param cube: [wvl, xdim, ydim] cube on which to perform photometry
    :param aperradii: radius of the aperture - if 'None' will use the diffraction limited aperture for each wvl
    :param wvl_start: array, start wavelengths in angstroms
    :param wvl_stop: array, stop wavelengths in angstroms
    :param platescale: platescale in arcsec/pix
    :param D: telescope diameter in meters
    :return: background subtracted flux of the satellite spot in counts/sec
    """
    flux = np.zeros((len(cube), 4))
    if len(cube) != len(wvl_start) or len(cube) != len(wvl_stop):
        raise ValueError('cube must have same wavelength dimensions wvl start and wvl stop')
    for i, img in enumerate(cube):
        imgspare = img.copy()
        dim = np.shape(cube[0, :, :])
        starx = dim[0] / 2
        stary = dim[1] / 2
        lambdamin = wvl_start[i]
        lambdamax = wvl_stop[i]
        landa = lambdamin + (lambdamax - lambdamin)/2.0
        if aperradii is None:
            aperradii = get_aperture_radius(landa, platescale)
        R_spot = np.zeros(3)
        R_spot[0] = (206265 / platescale) * 15.91 * lambdamin / (D*1e10)
        R_spot[1] = (206265 / platescale) * 15.91 * landa / (D*1e10)
        R_spot[2] = (206265 / platescale) * 15.91 * lambdamax / (D*1e10)
        halflength = R_spot[2] - R_spot[1]

        wcs_rot = wcs.wcs.pc
         # ROT_ANG = [42.73, 136, 226.9, 311.7]
        ROT_ANG=[45, 45, 45, 45]
        ROT_ANG = np.deg2rad(ROT_ANG)
        xs0 = R_spot[1] * np.cos(ROT_ANG[0])
        ys0 = R_spot[1] * np.sin(ROT_ANG[0])
        xs1 = R_spot[1] * np.cos(ROT_ANG[1])
        ys1 = -R_spot[1] * np.sin(ROT_ANG[1])
        xs2 = -R_spot[1] * np.cos(ROT_ANG[2])
        ys2 = -R_spot[1] * np.sin(ROT_ANG[2])
        xs3 = -R_spot[1] * np.cos(ROT_ANG[3])
        ys3 = R_spot[1] * np.sin(ROT_ANG[3])

        spot_posx = np.array([xs0, xs1, xs2, xs3])
        spot_posy = np.array([ys0, ys1, ys2, ys3])
        for idx, coord in enumerate(spot_posx):
            new_x, new_y = wcs_rot.dot(np.array([spot_posx[idx], spot_posy[idx]]))
            spot_posx[idx] = new_x + starx
            spot_posy[idx] = new_y + stary
        print(spot_posx, spot_posy)
        spot_xsep = [spot_posx[0] - starx, spot_posx[1] - starx, spot_posx[2] - starx, spot_posx[3] - starx]
        spot_ysep = [spot_posy[0] - stary, spot_posy[1] - stary, spot_posy[2] - stary, spot_posy[3] - stary]

        spot_rotang = [np.arctan(spot_ysep[0]/spot_xsep[0]), np.arctan(spot_ysep[1]/spot_xsep[1]),
                                 np.arctan(spot_ysep[2]/spot_xsep[2]) - np.pi, np.pi + np.arctan(spot_ysep[3]/spot_xsep[3])]
        for j in range(4):
            flux[i, j], imgspare = racetrack_aper(img, imgspare, spot_posx[j], spot_posy[j], spot_rotang[j],
                                                  aperradii, halflength)
    return flux

def racetrack_aper(img, imgspare, x_guess, y_guess, rotang, aper_radii, halflength, box_size=20):
    """

    :param img: 2D numpy array image
    :param imgspare: copy of img for debugging
    :param x_guess: x location of aperture (in pixels)
    :param y_guess: y location of aperture (in pixels)
    :param rotang: rotation angle of satellite spot
    :param aper_radii: aperture radius (in pixels)
    :param halflength: halflength of the satellite spot
    :param box_size: size of box to use around each aperture to calculate the background (in pixels)
    :return: background subtracted flux, debug image
    """
    rotang *= -1
    where_nan = np.where(np.isnan(img))
    img[where_nan] = 0
    spot_halflen = halflength
    dims = np.shape(img)

    crop_img = img[int(y_guess) - box_size:int(y_guess) + (box_size+1),
               int(x_guess) - box_size:int(x_guess) + (box_size+1)]
    xpos = x_guess
    ypos = y_guess
    xcoord, ycoord = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]))
    xppos = np.cos(rotang) * xpos - np.sin(rotang) * ypos
    yppos = np.sin(rotang) * xpos + np.cos(rotang) * ypos

    xpcoord = np.cos(rotang) * xcoord - np.sin(rotang) * ycoord
    ypcoord = np.sin(rotang) * xcoord + np.cos(rotang) * ycoord

    source_mid = (ypcoord > yppos-aper_radii) & (ypcoord < yppos + aper_radii) & (xpcoord > xppos - spot_halflen)\
                 & (xpcoord < xppos + spot_halflen)
    source_bot = (xpcoord < xppos - spot_halflen) & \
                 ((ypcoord-yppos)**2 + (xpcoord-(xppos-spot_halflen))**2 < aper_radii**2)
    source_top = (xpcoord > xppos + spot_halflen) & \
                 ((ypcoord - yppos)**2 + (xpcoord - (xppos + spot_halflen))**2 < aper_radii**2)
    source = np.where(source_mid | source_bot | source_top)

    # subtract off background flux
    p_back_init = Polynomial2D(degree=1)
    fit_p = fitting.LevMarLSQFitter()
    x, y = np.meshgrid(np.arange(np.shape(crop_img)[0]), np.arange(np.shape(crop_img)[0]))
    p = fit_p(p_back_init, x, y, crop_img)
    background = p(x, y)
    img[int(y_guess) - box_size:int(y_guess) + (box_size + 1),
    int(x_guess) - box_size:int(x_guess) + (box_size + 1)] -= background
    flux = np.sum(img[source])

    imgspare[source] = 10000
    return flux, imgspare

