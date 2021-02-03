from mkidpipeline.utils.photometry import *
import numpy as np
import matplotlib.pyplot as plt


def signal_to_noise(image, num_apertures, center, dist, pa, radius, save_path=None):
    """
    function to calculate the signal to noise of a target using aperture photometry. This code takes a series of
    apertures in a circle at a constant separation from the host star. The background subtracted flux in the aperture
    containing the companion is deeemed the 'signal' and divided by the standard deviation of the background
    subtracted flux in the remaining aperture (the 'noise')

    :param image: 2D image array
    :param num_apertures: total number of apertures to use
    :param center: center location of the image (tuple, in pixels)
    :param dist: radial distance from center of the image at which you want to perform the calculation
    :param pa: position angle of target of interest (in degrees)
    :param radius: aperture radius (in pixels)
    :param save_path: if not None will save an image of the apertures used to perform the calculation
    :return: signal, noise
    """
    aperture_values = np.zeros(num_apertures)
    fig, ax = plt.subplots()
    ax.imshow(image)
    angular_size = 2*np.arctan(radius/dist)
    for i in range(num_apertures):
        aperture_center = get_aperture_center(pa, i*angular_size, center, dist)
        im = image.copy()
        aperture_values[i] = aper_photometry(im, aperture_center, radius)
        circle1 = plt.Circle((aperture_center[0], aperture_center[1]), radius=radius, fill=False, color='r')
        circle2= plt.Circle((aperture_center[0], aperture_center[1]), radius=radius+0.5*radius, fill=False, color='g')
        ax.add_artist(circle1)
        ax.add_artist(circle2)
    max = np.max(aperture_values)
    max_idx = np.where(aperture_values==max)
    signal = max
    aperture_values[max_idx] = np.nan
    aperture_values[max_idx] = np.nan
    noise = np.nanstd(aperture_values)
    if save_path:
        plt.savefig(save_path + 'sn_debug_image.pdf')
    return signal, noise

def get_aperture_center(pa, angular_size, obj_center, dist):
    """
    returns the pixel coordinate of the center of an aperture a given angular size away from a target at a position as
    defined by pa
    :param pa: position angle of the target of interest
    :param angular_size: angular distance away from the pa you want the aperture to be
    :param obj_center: pixel location fo the target of interest
    :param dist: radial distance of the apertures from image center
    :return: center (tuple in pixels)
    """
    theta = np.deg2rad(pa) + angular_size
    delta_x = dist*np.cos(theta)
    delta_y = dist*np.sin(theta)
    center = (obj_center[0] + delta_x, obj_center[1] + delta_y)
    return center

def get_num_apertures(aperture_radius, dist):
    """
    utility function to get the total number of apertures for a given size and radial distance from the center of the
    image
    :param aperture_radius: in pixels
    :param dist: radial distance from center of image (in pixels)
    :return:
    """
    angular_size = np.arctan((aperture_radius*2)/dist)
    angular_size = np.rad2deg(angular_size)
    return int(360/angular_size), angular_size

def pixel_lambda_over_d(lam, n):
    """
    get the pixel radius from the center for a certain n*lam/D
    :param lam: wavelength (angstroms)
    :param n: integer number
    :return: radius (in pixels)
    """
    b = 206265
    lam_over_d = (lam/8.2e10) * b * 1000
    pix = (n*lam_over_d)/10.4
    return pix

def get_aperture_radius(lam):
    """
    get diffraction limited aperture size
    :param lam: wavelength (in angstroms!)
    :return: radius of the aperture in pixels equal to 2x diffraction limit
    """
    D = 8.2 *(10**10)
    theta_rad = 1.22 * (lam/D)
    a = 4.8481368e-9
    theta_mas = theta_rad * (1/a)
    r = 0.5*theta_mas * (1/10.4)
    return r
