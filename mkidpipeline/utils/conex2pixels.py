#!/usr/bin/env python3
import os
import numpy as np
import skimage.transform as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from photutils import DAOStarFinder, centroids, centroid_2dg
from scipy.stats import chisquare
from astropy import stats
from mkidpipeline.definitions import MKIDDither


#Example (with a configured pipeline)
# ditherfile = '/mnt/data0/isabel/highcontrastimaging/Jan2019Run/20190112/51Eri/51EriDither1/51Eri_wavecalib/51eri_ditheruseable.cfg'
# xform_con2pix, xform_pix2con = get_transforms(ditherfile, fwhm_guess=3.0, wvl_start=900, wvl_stop=1200, plot=True)


def get_transforms(ditherfile, wvl_start=None, wvl_stop=None, fwhm_guess=3.0, fit_power=1, plot=False):
    dither = MKIDDither(name=os.path.basename(ditherfile), data=ditherfile)

    box_size = fwhm_guess * 10

    debug_images = []
    pixel_positions = []
    source_est = []

    for o in dither.obs:
        obs = o.photontable
        data = obs.get_fits(weight=False, wave_start=wvl_start, wave_stop=wvl_stop, rate=False)['SCIENCE'].data.T
        debug_images.append(data)
        mean, median, std = stats.sigma_clipped_stats(data, sigma=3.0, mask_value=0)
        mask = data == 0

        sources = DAOStarFinder(fwhm=fwhm_guess, threshold=5. * std)(data - median, mask=mask)
        source = sources[sources['flux'].argmax()]
        source_est.append((source['xcentroid'], source['ycentroid']))

        position = centroids.centroid_sources(data, source['xcentroid'], source['ycentroid'], box_size=int(box_size),
                                              centroid_func=centroid_2dg, mask=mask)
        pixel_positions.append((position[0][0], position[1][0]))

    pixel_positions, conex_positions = np.array(pixel_positions), np.array(dither.pos)
    xform_con2pix = tf.estimate_transform('polynomial', conex_positions, pixel_positions, order=fit_power)
    xform_pix2con = tf.estimate_transform('polynomial', pixel_positions, conex_positions, order=fit_power)

    if plot:
        n = int(round(np.sqrt(len(dither.obs))+.5))
        fig, axs = plt.subplots(n, n, figsize=(20, 15))
        marker = '+'
        ms, mew = 30, 2.
        for index, image in enumerate(debug_images[:-1]):
            axs.flat[index].imshow(image, origin='lower', interpolation='nearest', cmap='viridis')
            axs.flat[index].add_patch(Rectangle((source_est[index][0] - (box_size / 2),
                                                 source_est[index][1] - (box_size / 2)), box_size, box_size,
                                                linewidth=1, edgecolor='r', fill=None))
            axs.flat[index].plot(source_est[index][0], source_est[index][1], color='r', marker=marker, ms=ms, mew=mew)
            axs.flat[index].plot(pixel_positions[index][0], pixel_positions[index][1], color='b', marker=marker,
                                 ms=ms, mew=mew)
            axs.flat[index].set_title(f'Dither {index} Est. (r) Cent. (b)')
        plt.show()
        _plotresiduals(xform_con2pix, conex_positions, pixel_positions, n)

    return xform_con2pix, xform_pix2con


def _plotresiduals(xform_con2pix, conex_positions, pixel_positions, n):
    positions_fit = xform_con2pix(conex_positions)
    centroid_x = np.hsplit(pixel_positions, 2)[0]
    centroid_y = np.hsplit(pixel_positions, 2)[1]
    fit_x = np.hsplit(positions_fit, 2)[0]
    fit_y = np.hsplit(positions_fit, 2)[1]

    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 15))
    axs[0][0].scatter(centroid_x, centroid_y, facecolor='red')
    axs[0][0].scatter(fit_x, fit_y, facecolors='blue')
    axs[0][0].set_title('Pixel Positions (red) and Fit (blue)')

    centroid_vector = np.sqrt(centroid_x ** 2 + centroid_y ** 2)
    fit_vector = np.sqrt(fit_x ** 2 + fit_y ** 2)
    residuals_vector = centroid_vector - fit_vector
    chisq_vector = chisquare(fit_vector, centroid_vector)
    axs[0][1].scatter(np.arange(n), residuals_vector)
    axs[0][1].set_title(f'Residuals Total, fit has chisq: {chisq_vector[0][0]:.3f}')

    residuals_x = centroid_x - fit_x
    axs[1][0].scatter(np.arange(n), residuals_x)
    axs[1][0].set_title('Residuals X')
    residuals_y = centroid_y - fit_y
    axs[1][1].scatter(np.arange(n), residuals_y)
    axs[1][1].set_title('Residuals Y')
    plt.show()
