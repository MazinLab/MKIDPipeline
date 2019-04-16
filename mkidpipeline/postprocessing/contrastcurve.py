import glob
import numpy as np
import pyximport;

pyximport.install()
from mkidpipeline import badpix as bp
from mkidpipeline.hdf.photontable import ObsFile as obs
from mkidpipeline.utils.plottingTools import plot_array as pa

from astropy.convolution import AiryDisk2DKernel, Gaussian2DKernel, convolve
from scipy.optimize import curve_fit
import skimage.transform as tf
from scipy import ndimage
from astropy.modeling.functional_models import AiryDisk2D
from mkidcore.instruments import CONEX2PIXEL

from astropy.io import fits
from astropy.coordinates import SkyCoord
import astropy.units as units
from scipy.interpolate import griddata
from scipy.ndimage.filters import median_filter
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from astropy.io import fits

from photutils import DAOStarFinder, centroids, centroid_2dg, centroid_1dg, centroid_com, CircularAperture, \
    aperture_photometry


def ditherp_2_pixel(positions):
    """ A function to convert the connex offset to pixel displacement"""
    positions = np.asarray(positions)
    pix = np.asarray(CONEX2PIXEL(positions[:, 0], positions[:, 1])) - np.array(CONEX2PIXEL(0, 0)).reshape(2, 1)
    return pix


def func(x, slope, intercept):
    return x * slope + intercept


def upsample_image(image, npix):
    """
    Upsamples the array so that the rotation and shift can be done to subpixel precision
    each pixel will be converted in a square of nPix*nPix
    """
    upsampled = image.repeat(npix, axis=0).repeat(npix, axis=1)
    return upsampled


def median_stack(stack):
    return np.nanmedian(stack, axis=0)


def mean_stack(stack):
    return np.nanmean(stack, axis=0)


def negatives_to_nans(image):
    image = image.astype(float)
    image[image < 0] = np.nan
    return image


def dist(yc, xc, y1, x1):
    """
    Return the Euclidean distance between two points.
    """
    return np.sqrt((yc - y1) ** 2 + (xc - x1) ** 2)


def embed_image(image, framesize=1, pad_value=-1):
    """
    Gets a numpy array and -1-pads it. The frame size gives the dimension of the frame in units of the
    biggest dimension of the array (if image.shape gives (2,4), then 4 rows of -1s will be added before and
    after the array and 4 columns of -1s will be added before and after. The final size of the array will be (10,12))
    It is padding with -1 and not 0s to distinguish the added pixels from valid pixels that have no photons. Masked pixels
    (dead or hot) are nan
    It returns a numpy array
    """
    frame_pixsize = int(max(image.shape) * framesize)
    padded_array = np.pad(image, frame_pixsize, 'constant', constant_values=pad_value)
    return padded_array


def rotate_shift_image(image, degree, xshift, yshift):
    """
    Rotates the image counterclockwise and shifts it in x and y
    When shifting, the pixel that exit one side of the array get in from the other side. Make sure that
    the padding is large enough so that only -1s roll and not real pixels
    """
    ###makes sure that the shifts are integers
    xshift = int(round(xshift))
    yshift = int(round(yshift))
    rotated_image = ndimage.rotate(image, degree, order=0, cval=-1, reshape=False)
    rotated_image = negatives_to_nans(rotated_image)

    xshifted_image = np.roll(rotated_image, xshift, axis=1)
    rotated_shifted = np.roll(xshifted_image, yshift, axis=0)
    return rotated_shifted


def interpolate_image(image, method='linear'):
    '''
    2D interpolation to smooth over missing pixels using built-in scipy methods

    INPUTS:
        image - 2D input array of values
        method - method of interpolation. Options are scipy.interpolate.griddata methods:
                 'linear' (default), 'cubic', or 'nearest'

    OUTPUTS:
        the interpolated image with same shape as input array
    '''

    finalshape = np.shape(image)

    datapoints = np.where(
        np.logical_or(np.isnan(image), image == 0) == False)  # data points for interp are only pixels with counts
    data = image[datapoints]
    datapoints = np.array((datapoints[0], datapoints[1]),
                          dtype=np.int).transpose()  # griddata expects them in this order

    interppoints = np.where(image != np.nan)  # should include all points as interpolation points
    interppoints = np.array((interppoints[0], interppoints[1]), dtype=np.int).transpose()

    interpolated_frame = griddata(datapoints, data, interppoints, method)
    interpolated_frame = np.reshape(interpolated_frame, finalshape)  # reshape interpolated frame into original shape

    return interpolated_frame


def align_stack_image(output_dir, output_filename, int_time, xcon, ycon):
    """
    output_dir='/mnt/data0/isabel/microcastle/51Eri/51Eriout/dither3/'
    output_filename='51EriDither3'
    int_time=60
    xcon = [-0.1, -0.1, -0.1, -0.1, -0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2, 0.2, 0.35, 0.35, 0.35, 0.35, 0.35, 0.5, 0.5, 0.5, 0.5, 0.5]
    ycon = [-0.5, -0.25, 0.0, 0.25, 0.5, -0.5, -0.25, 0.0, 0.25, 0.5, -0.5, -0.25, 0.0, 0.25, 0.5, -0.5, -0.25, 0.0, 0.25, 0.5, -0.5, -0.25, 0.0, 0.25, 0.5]
    """

    obsfile_list = glob.glob(output_dir + '15*.h5')

    npos = len(obsfile_list)
    wvlStart = 900  # True for parasiting
    wvlStop = 1140

    numpyfxnlist = []
    for i in range(npos):
        obsfile = obs(obsfile_list[i], mode='write')
        img = obsfile.getPixelCountImage(firstSec=0, integrationTime=int_time, applyWeight=True, flagToUse=0,
                                         wvlStart=wvlStart, wvlStop=wvlStop)
        print(
        'Running getPixelCountImage on ', 0, 'seconds to ', int_time, 'seconds of data from wavelength ', wvlStart,
        'to ', wvlStop)
        obsfile.file.close()
        saveim = np.transpose(img['image'])

        print('applying HPM to ', i)
        dead_mask = saveim == 0
        fluxbox = bp.hpm_flux_threshold(saveim, fwhm=4, dead_mask=dead_mask)
        outfile = output_dir + output_filename + 'HPMasked%i.npy' % i
        np.save(outfile, fluxbox['image'])
        numpyfxnlist.append(outfile)

    rough_shiftsx = []
    rough_shiftsy = []
    centroidsx = []
    centroidsy = []

    pad_fraction = .8

    xopt, xcov = curve_fit(func, np.array([-0.035, 0.23, 0.495]), np.array([125.12754088, 106.53992258, 92.55050812]), sigma=np.array([0.33597449, 0.82476065, 1.6125932 ]))
    yopt, ycov = curve_fit(func, np.array([-0.76, -0.38, 0., 0.38]), np.array([36.53739721, 61.29792346, 90.77367552, 115.0451042 ]), sigma=np.array([1., 0.13307094, 1.26640359, 0.38438538]))

    xpos = np.array(xcon) * xopt[0] + xopt[1]
    ypos = np.array(ycon) * yopt[0] + yopt[1]

    # starting point centroid guess for first frame is where all subsequent frames will be aligned to
    refpointx = xpos[0]
    refpointy = ypos[0]
    print('xpos', xpos[0], 'ypos', ypos[0])

    # determine the coarse x and y shifts that subsequent frames must be moved to align with first frame
    dxs = (np.zeros(len(xpos)) + refpointx) - xpos
    dys = (np.zeros(len(ypos)) + refpointy) - ypos

    # load dithered science frames
    dither_frames = []
    for i in range(npos):
        image = np.load(numpyfxnlist[i]) / int_time  # divide by eff int time
        image[image == 0] = ['nan']
        rough_shiftsx.append(dxs[i])
        rough_shiftsy.append(dys[i])
        centroidsx.append(refpointx - dxs[i])
        centroidsy.append(refpointy - dys[i])
        padded_frame = embed_image(image, framesize=pad_fraction)
        shifted_frame = rotate_shift_image(padded_frame, 0, dxs[i], dys[i])

        dither_frames.append(shifted_frame)

    final_image = median_stack(np.array(dither_frames))
    dead_mask = np.isnan(final_image)
    reHPM = bp.hpm_flux_threshold(final_image, fwhm=4, dead_mask=dead_mask)
    outfilestack = output_dir + output_filename + '_medianstacked_exptimeNORM'
    np.save(outfilestack, reHPM['image'])
    pa(reHPM['image'])


def flux_estimator(datafileFLUX, xcentroid_flux, ycentroid_flux, sat_spot_bool=False, ND_filter_bool=True,
                   sat_spotcorr=5.5, ND_filtercorr=1):
    data = np.load(datafileFLUX)
    fig, axs = plt.subplots(1, 1)
    axs.imshow(data, origin='lower', interpolation='nearest')

    fwhm_guess = 8.0
    marker = '+'
    ms, mew = 30, 2.
    box_size = fwhm_guess * 5

    mask = np.isnan(data)

    axs.plot(xcentroid_flux, ycentroid_flux, color='red', marker=marker, ms=ms, mew=mew)  # check how we did
    positions = centroids.centroid_sources(data, xcentroid_flux, ycentroid_flux, box_size=int(box_size),
                                           centroid_func=centroid_com, mask=mask)
    axs.plot(positions[0], positions[1], color='blue', marker=marker, ms=ms, mew=mew)  # check how the fit did
    plt.title('Estimate in red, fit in blue')
    plt.show()

    xpos = positions[0]
    ypos = positions[1]

    positions_ap = [(xpos[0], ypos[0])]
    apertures = CircularAperture(positions_ap, r=8.)
    phot_table = aperture_photometry(data, apertures, mask=mask)

    fig, axs = plt.subplots(1, 1)
    axs.imshow(data, origin='lower', interpolation='nearest')
    apertures.plot(color='white', lw=1)
    plt.show()

    norm = phot_table['aperture_sum'].data

    if sat_spot_bool:
        norm = norm * sat_spotcorr
        factor = (1 / sat_spotcorr)

    if ND_filter_bool:
        factor = 10 ** (-ND_filtercorr)
        norm = norm / factor

    return {'norm': norm, 'xpos': xpos, 'ypos': ypos, 'factor': factor}


def prepare_forCC(datafile, outfile_name, interp_bool=True, smooth_bool=True, xcenter=242, ycenter=180, box_size=100):
    data = np.load(datafile)

    if interp_bool:
        datainterp = interpolate_image(data)
        if smooth_bool:
            proc_data = median_filter(datainterp, size=3)

    elif smooth_bool:
        proc_data = median_filter(data, size=3)

    else:
        proc_data = data

    actualcenterx = int(data.shape[1] / 2)
    actualcentery = int(data.shape[0] / 2)
    roll = [actualcenterx - xcenter, actualcentery - ycenter]
    print(actualcenterx, actualcentery)
    proc_data_centered = np.roll(proc_data, roll[1], 0)
    proc_data_centered = np.roll(proc_data_centered, roll[0], 1)

    data_cropped = proc_data_centered[actualcentery - box_size:actualcentery + box_size,
                   actualcenterx - box_size:actualcenterx + box_size]
    pa(data_cropped)

    np.save(outfile_name, data_cropped)


def make_CoronagraphicProfile(datafileCC, unocculted=False, unoccultedfile='/mnt/data0/isabel/microcastle/51EriUnocculted.npy',
                       badpix_bool=False, normalize=1, plot_bool=False, fwhm_est=8, nlod=12, **fluxestkwargs):
    normalize = 390908.2702642
    if unocculted:
        normdict = flux_estimator(unoccultedfile, **fluxestkwargs)
        norm = normdict['norm']
        factor = normdict['factor']
    else:
        norm = normalize

    speckles = np.load(datafileCC)
    lod = fwhm_est

    sep = np.arange(nlod + 1)

    # pixel coords of center of images.  Assume images are already centered
    centerx = int(speckles.shape[1] / 2)
    centery = int(speckles.shape[0] / 2)

    dead_mask = np.isnan(speckles)

    positions_ap1 = []
    for i in np.arange(nlod) + 1:
        positions_ap1.append([centerx, centery - i * lod])
    apertures1 = CircularAperture(positions_ap1, r=lod / 2)
    phot_table1 = aperture_photometry(speckles, apertures1, mask=dead_mask)
    print(phot_table1)

    fig, axs = plt.subplots(1, 1)
    axs.imshow(speckles, origin='lower', interpolation='nearest')
    apertures1.plot(color='white', lw=1)
    plt.show()

    fig, ax1 = plt.subplots()

    ax1.plot(sep[1:], phot_table1['aperture_sum'] / normalize, linewidth=2, label=r'Coronagraphic PSF Profile')
    ax1.plot(sep[1:], np.sqrt(phot_table1['aperture_sum']) / norm, linestyle='-.', linewidth=2,
             label=r'1-$\sigma$ Photon noise')

    ax1.set_xlabel(r'Separation ($\lambda$/D)', fontsize=14)
    ax1.set_ylabel(r'Normalized Azimuthally Averaged Intensity', fontsize=14)
    ax1.set_xlim(0, nlod)

    ax1.set_ylim(2e-5, 1)
    ax1.set_yscale('log')
    ax1.legend()
    plt.show()


def make_CC(datafileCC, unocculted=False, unoccultedfile='/mnt/data0/isabel/microcastle/51EriUnocculted.npy',
            badpix_bool=False, normalize=1, fwhm_est=8, nlod=12, plot=False, mec_hack=False, **fluxestkwargs):
    normalize = 390908.2702642
    if unocculted:
        normdict = flux_estimator(unoccultedfile, **fluxestkwargs)
        norm = normdict['norm']
        factor = normdict['factor']
    else:
        norm = normalize

    speckles = np.load(datafileCC)
    lod = fwhm_est
    sep_full = np.arange(nlod + 1)

    # pixel coords of center of images.  Assume images are already centered
    centerx = int(speckles.shape[1] / 2)
    centery = int(speckles.shape[0] / 2)

    dead_mask = np.isnan(speckles)

    spMeans0 = [0]
    spMeans = [0]
    spStds = [0]
    spSNRs = [0]

    for i in np.arange(nlod) + 1:
        sourcex = centerx
        sourcey = centery - i * lod
        sep = dist(centery, centerx, sourcey, sourcex)

        angle = np.arcsin(lod / 2. / sep) * 2
        number_apertures = int(np.floor((2) * np.pi / angle))
        yy = np.zeros((number_apertures))
        xx = np.zeros((number_apertures))
        cosangle = np.cos(angle)
        sinangle = np.sin(angle)
        xx[0] = sourcex - centerx
        yy[0] = sourcey - centery
        for j in range(number_apertures - 1):
            xx[j + 1] = cosangle * xx[j] + sinangle * yy[j]
            yy[j + 1] = cosangle * yy[j] - sinangle * xx[j]

        xx[:] += centerx
        yy[:] += centery
        rad = lod / 2.
        apertures = CircularAperture((xx, yy), r=rad)  # Coordinates (X,Y)
        fluxes = aperture_photometry(speckles, apertures, method='exact')
        fluxes = np.array(fluxes['aperture_sum'])

        f_source = fluxes[0].copy()
        fluxes = fluxes[1:]
        n2 = fluxes.shape[0]
        ##snr = (f_source - np.nanmean(fluxes))/(np.nanstd(fluxes)*np.sqrt(1+(1/n2)))
        snr = (normalize - np.nanmean(fluxes)) / (np.nanstd(fluxes) * np.sqrt(1 + (1 / n2)))

        if plot:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.imshow(speckles, origin='lower', interpolation='nearest', alpha=0.5,
                      cmap='gray')
            for i in range(xx.shape[0]):
                # Circle takes coordinates as (X,Y)
                aper = plt.Circle((xx[i], yy[i]), radius=lod / 2., color='r', fill=False, alpha=0.8)
                ax.add_patch(aper)
                cent = plt.Circle((xx[i], yy[i]), radius=0.8, color='r', fill=True, alpha=0.5)
                ax.add_patch(cent)
                aper_source = plt.Circle((sourcex, sourcey), radius=0.7, color='b', fill=True, alpha=0.5)
                ax.add_patch(aper_source)
            ax.grid(False)
            plt.show()

        spMeans0.append(f_source)
        spMeans.append(np.nanmean(fluxes))
        spStds.append(np.nanstd(fluxes))
        spSNRs.append(snr)

    spMeans0 = np.array(spMeans0)
    print('spMeans0', spMeans0)
    spMeans = np.array(spMeans)
    print('spMeans', spMeans)
    spStds = np.array(spStds)
    print('spStds', spStds)
    spSNRs = np.array(spSNRs)
    print('spSNRs', 1 / spSNRs)

    fig, ax1 = plt.subplots()

    ax1.plot(sep_full[1:], spMeans[1:] / norm, linewidth=2, label=r'Azimuthally Averaged Mean Coronagraphic Intensity')
    ax1.plot(sep_full[1:], spStds[1:] / norm, linestyle='-.', linewidth=2, label=r'Azimuthal Standard Deviation')
    ax1.plot(sep_full[1:], np.sqrt(spMeans[1:]) / norm, linestyle='-.', linewidth=2,
             label=r'Square Root of the Azimuthally Averaged Mean Coronagraphic Intensity')

    if unocculted:
        ax1.plot(sep, psfMeans / norm, linewidth=2, label=r'Unocculted PSF Profile')

    ax1.set_xlabel(r'Separation ($\lambda$/D)', fontsize=14)
    ax1.set_ylabel(r'Normalized by Unocculted PSF Intensity', fontsize=14)
    ax1.set_xlim(0, nlod)

    ax1.set_ylim(2e-5, 1)
    ax1.set_yscale('log')
    ax1.legend()
    plt.show()