import os
import numpy as np
import skimage.transform as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from photutils import DAOStarFinder, centroids, centroid_2dg
from scipy.stats import chisquare
from astropy import stats
from mkidpipeline.config import MKIDDitheredObservation
from photontable import Photontable


def get_transforms(ditherfile, datadir, wvl_start=None, wvl_stop=None, fwhm_guess=3.0, fit_power=1, CONEX_ERROR=0.0001,
                   plot=False):
    dither = MKIDDitheredObservation(os.path.basename(ditherfile), ditherfile, None, None)
    obs_files = [os.path.join(datadir, '{}.h5'.format(o.start)) for o in dither.obs]

    box_size = fwhm_guess * 10

    debug_images = []
    pixel_positions = []
    source_est = []

    for file in obs_files:
        obs = Photontable(file)
        data = obs.get_fits(applyWeight=False,  wvlStart=wvl_start, wvlStop=wvl_stop, countRate=False)['SCIENCE'].data
        data = np.transpose(data)
        debug_images.append(data)
        mean, median, std = stats.sigma_clipped_stats(data, sigma=3.0, mask_value=0)
        mask = np.zeros_like(data, dtype=bool)
        mask[data == 0] = True

        sources = DAOStarFinder(fwhm=fwhm_guess, threshold=5. * std)(data - median, mask=mask)
        source = sources[sources['flux'].argmax()]
        source_est.append((source['xcentroid'], source['ycentroid']))

        position = centroids.centroid_sources(data, source['xcentroid'], source['ycentroid'], box_size=int(box_size),
                                              centroid_func=centroid_2dg, mask=mask)
        pixel_positions.append((position[0][0], position[1][0]))

    pixel_positions = np.array(pixel_positions)
    conex_positions = np.array(dither.pos)

    xform_con2pix = tf.estimate_transform('polynomial', conex_positions, pixel_positions, order=fit_power)
    xform_pix2con = tf.estimate_transform('polynomial', pixel_positions, conex_positions, order=fit_power)

    if plot:
        axis = int(round(np.sqrt(len(obs_files)), 0))
        fig, axs = plt.subplots(axis, axis, figsize=(20, 15))
        i = 0
        j = 0
        for index, image in enumerate(debug_images[:-1]):
            axs[i, j].imshow(image, origin='lower', interpolation='nearest', cmap='viridis')
            axs[i, j].add_patch(
                Rectangle((source_est[index][0] - (box_size / 2), source_est[index][1] - (box_size / 2)), box_size,
                          box_size,
                          linewidth=1, edgecolor='r', fill=None))
            marker = '+'
            ms, mew = 30, 2.
            axs[i, j].plot(source_est[index][0], source_est[index][1], color='red', marker=marker, ms=ms, mew=mew)
            axs[i, j].plot(pixel_positions[index][0], pixel_positions[index][1], color='blue', marker=marker, ms=ms,
                           mew=mew)
            axs[i, j].set_title('Red + = Estimate, Blue + = Centroid for Dither Pos %i' % index)
            if (index + 1) % axis == 0 and (index + 1) != len(obs_files):
                i += 1
                j = 0
            elif index != len(obs_files) - 1:
                j += 1
        plt.show()

        _plotresiduals(xform_con2pix, conex_positions, pixel_positions)

    return xform_con2pix, xform_pix2con


def _plotresiduals(xform_con2pix, conex_positions, pixel_positions):
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
    axs[0][1].scatter(np.arange(len(obs_files)), residuals_vector)
    axs[0][1].set_title('Residuals Total, fit has chisquare of %s' % str(round(chisq_vector[0][0], 3)))

    residuals_x = centroid_x - fit_x
    chisq_x = chisquare(fit_x, centroid_x)
    axs[1][0].scatter(np.arange(len(obs_files)), residuals_x)
    axs[1][0].set_title('Residuals X')
    residuals_y = centroid_y - fit_y
    chisq_y = chisquare(fit_y, centroid_y)
    axs[1][1].scatter(np.arange(len(obs_files)), residuals_y)
    axs[1][1].set_title('Residuals Y')
    plt.show()


if __name__ == '__main__':
    ditherfile = '/mnt/data0/isabel/highcontrastimaging/Jan2019Run/20190112/51Eri/51EriDither1/51Eri_wavecalib/51eri_ditheruseable.cfg'
    datadir = '/mnt/data0/isabel/highcontrastimaging/Jan2019Run/20190112/51Eri/51EriDither1/51Eri_wavecalib/'

    xform_con2pix, xform_pix2con = get_transforms(ditherfile, datadir, fwhm_guess=3.0, wvl_start=900, wvl_stop=1200,
                                                  fit_power=1)
