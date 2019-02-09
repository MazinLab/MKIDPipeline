import os
import numpy as np
import skimage.transform as tf
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from photutils import DAOStarFinder, centroids, centroid_2dg, centroid_1dg, centroid_com
from astropy import stats
from mkidpipeline.config import MKIDObservingDither
from mkidpipeline.hdf.photontable import ObsFile

def get_con2pix(ditherfile, datadir, wvl_start=None, wvl_stop=None, fwhm_guess=3.0, fit_power=1, CONEX_ERROR=0.0001):

    dither=MKIDObservingDither(os.path.basename(ditherfile), ditherfile, None, None)
    obs_files = [os.path.join(datadir, '{}.h5'.format(o.start)) for o in dither.obs]

    box_size=fwhm_guess*10

    debug_images=[]
    pixel_positions=[]
    source_est=[]

    for file in obs_files:
        obs = ObsFile(file)
        data = obs.getPixelCountImage(applyWeight=False, flagToUse = 0, wvlStart=wvl_start,
                                      wvlStop=wvl_stop)['image']
        data = np.transpose(data)
        debug_images.append(data)
        mean, median, std = stats.sigma_clipped_stats(data, sigma=3.0, mask_value=0)
        mask = np.zeros_like(data, dtype=bool)
        mask[data == 0] = True

        sources = DAOStarFinder(fwhm=fwhm_guess, threshold=5.*std)(data - median, mask=mask)
        source = sources[sources['flux'].argmax()]
        source_est.append((source['xcentroid'], source['ycentroid']))

        position = centroids.centroid_sources(data, source['xcentroid'], source['ycentroid'], box_size=int(box_size), centroid_func=centroid_2dg, mask=mask)
        pixel_positions.append((position[0][0],position[1][0]))

    pixel_positions=np.array(pixel_positions)
    conex_positions = np.array(dither.pos)
    conex_positon_errors = np.full_like(conex_positions, fill_value=CONEX_ERROR)

    xform = tf.estimate_transform('polynomial', conex_positions, pixel_positions, order=fit_power)

    for index, image in enumerate(debug_images):
        fig, ax = plt.subplots(1, 1)
        ax.imshow(image, origin='lower', interpolation='nearest', cmap='viridis')
        ax.add_patch(Rectangle((source_est[index][0]- (box_size / 2), source_est[index][1] - (box_size / 2)), box_size, box_size,
                      linewidth=1, edgecolor='r', fill=None))
        marker = '+'
        ms, mew = 30, 2.
        plt.plot(source_est[index][0], source_est[index][1], color='red', marker=marker, ms=ms, mew=mew)
        plt.plot(pixel_positions[index][0], pixel_positions[index][1], color='blue', marker=marker, ms=ms, mew=mew)
        ax.set_title('Red + = Estimate, Blue + = Centroid for Dither Pos %i'%index)
        plt.show()

    positions_fit=xform(conex_positions)
    plt.scatter(np.hsplit(pixel_positions,2)[0], np.hsplit(pixel_positions,2)[1])
    plt.scatter(np.hsplit(positions_fit,2)[0], np.hsplit(positions_fit,2)[1])
    plt.show()
    residuals_x=np.hsplit(pixel_positions,2)[0]-np.hsplit(positions_fit,2)[0]
    residuals_y=np.hsplit(pixel_positions,2)[1]-np.hsplit(positions_fit,2)[1]
    plt.scatter(np.arange(len(obs_files)), residuals_x)
    plt.show()
    plt.scatter(np.arange(len(obs_files)), residuals_y)
    plt.show()
    return xform

if __name__ == '__main__':

    ditherfile = '/mnt/data0/isabel/highcontrastimaging/Jan2019Run/20190112/51Eri/51EriDither1/51Eri_wavecalib/51eri_ditheruseable.cfg'
    datadir = '/mnt/data0/isabel/highcontrastimaging/Jan2019Run/20190112/51Eri/51EriDither1/51Eri_wavecalib/'

    xform=get_con2pix(ditherfile, datadir, fwhm_guess=3.0, wvl_start=900, wvl_stop=1200, fit_power=2)