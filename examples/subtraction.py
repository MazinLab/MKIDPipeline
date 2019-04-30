# TODO move this functionality to an analysis/postprocessing package rather than an example?

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib
from astropy.io import fits
from scipy.ndimage import rotate, zoom
import argparse
# from vip_hci import pca
from mkidpipeline.imaging.drizzler import form
import mkidpipeline
import mkidcore.corelog as pipelinelog

def rot_array(img, pivot,angle):
    padX = [img.shape[1] - pivot[0], pivot[0]]
    padY = [img.shape[0] - pivot[1], pivot[1]]
    imgP = np.pad(img, [padY, padX], 'constant')
    imgR = rotate(imgP, angle, reshape=False)
    return imgR[padY[0] : -padY[1], padX[0] : -padX[1]]


def clipped_zoom(img, zoom_factor, **kwargs):
    """ Courtesy of
    https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions """

    h, w = img.shape[:2]
    # For multichannel images we don't want to apply the zoom factor to the RGB
    # dimension, so instead we create a tuple of zoom factors, one per array
    # dimension, with 1's for any trailing dimensions after the width and height.
    zoom_tuple = (zoom_factor,) * 2 + (1,) * (img.ndim - 2)

    # Zooming out
    if zoom_factor < 1:
        # Bounding box of the zoomed-out image within the output array
        zh = int(np.round(h * zoom_factor))
        zw = int(np.round(w * zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2
        # Zero-padding
        out = np.zeros_like(img)
        out[top:top + zh, left:left + zw] = zoom(img, zoom_tuple, **kwargs)

    # Zooming in
    elif zoom_factor > 1:
        # Bounding box of the zoomed-in region within the input array
        zh = int(np.round(h / zoom_factor))
        zw = int(np.round(w / zoom_factor))
        top = (h - zh) // 2
        left = (w - zw) // 2
        out = zoom(img[top:top + zh, left:left + zw], zoom_tuple, **kwargs)
        # `out` might still be slightly larger than `img` due to rounding, so
        # trim off any extra pixels at the edges
        trim_top = ((out.shape[0] - h) // 2)
        trim_left = ((out.shape[1] - w) // 2)
        if trim_top < 0 or trim_left < 0:
            temp = np.zeros_like(img)
            temp[:out.shape[0], :out.shape[1]] = out
            out = temp
        else:
            out = out[trim_top:trim_top + h, trim_left:trim_left + w]

    # If zoom_factor == 1, just return the input array
    else:
        out = img
    return out


def ADI():
    """

    This function needs to be updated!!!

    First median collapsing a time cube of all the raw dither images to a get a static PSF of the virtual grid. For
    each dither derotate the static PSF and isolate the relevant area. Subtract that static reference from the
    median of the derorated dither

    :return:
    """

    obsdir = 'Singles/KappaAnd_dither+lasercal/wavecal_files'
    ditherlog = 'KAnd_1545626974_dither.log'
    target = '* Kap And'

    wvlMin = 850
    wvlMax = 1100
    startt = 0
    intt = 100#60
    pixfrac = .5

    # get static psf of virtual grid
    tess, drizwcs = form(4, rotate=0, target=target, ditherlog=ditherlog, obsdir=obsdir, wvlMin=wvlMin,
                         wvlMax=wvlMax, startt=startt, intt=intt, pixfrac=pixfrac, driz_sci=True)
    y = np.ma.masked_where(tess[:, 0] == 0, tess[:, 0])
    static_map = np.ma.median(y, axis=0).filled(0)

    pretty_plot(static_map, drizwcs.wcs.cdelt[0], drizwcs.wcs.crval, vmin=0, vmax=1000)
    write_fits(static_map, target+'_rot.fits')

    datadir = '/mnt/data0/isabel/mec/'
    ditherdesc = get_ditherdesc(target, datadir, ditherlog, rotate=1)

    # derotate the static psf for each dither time
    derot_static = np.zeros((len(ditherdesc.description.obs), static_map.shape[0], static_map.shape[1]))
    for ia, ha in enumerate(ditherdesc.dithHAs):
        # TODO use wcs and drizzle instead of rot_array
        starxy = drizwcs.all_world2pix([[ditherdesc.cenRA, ditherdesc.cenDec]], 1)[0].astype(np.int)
        derot_static[ia] = rot_array(static_map, starxy, -np.rad2deg(ha))

    # create derotated time cube
    tess, drizwcs = form(4, rotate=1, target=target, ditherlog=ditherlog, obsdir=obsdir, wvlMin=wvlMin,
                         wvlMax=wvlMax, startt=startt, intt=intt, pixfrac=pixfrac, driz_sci=True)

    tcube = tess[:,0]
    for i, image in enumerate(tcube):
        # shrink the reference psf to the relevant area
        derot_static[i][image == 0] = 0

        #subtract the reference
        image -= derot_static[i]

    # sum collapse the differential
    diff = np.sum(tcube, axis=0)
    pretty_plot(diff, drizwcs.wcs.cdelt[0], drizwcs.wcs.crval, vmin=100, vmax=1000)
    plt.imshow(diff, origin='lower')
    plt.show()


def SDI():
    """

    Median collapse a tesseract along the time dimension to produce a spectral cube with minimal hot pixels. Then
    radially scale the channels and median collapse to produce the reference PSF. Scale and bubtract that PSF from the
    spectral cube and median collpase to form an image. Needs to be verified

    """

    parser = argparse.ArgumentParser(description='Photon Drizzling Utility')
    parser.add_argument('cfg', type=str, help='The configuration file')
    args = parser.parse_args()
    cfg = mkidpipeline.config.load_task_config(args.cfg)

    fitsname = 'SDI.fits'

    nwvlbins = 5
    wvlMin = 850
    wvlMax = 1100
    wsamples = np.linspace(wvlMin, wvlMax, nwvlbins + 1)
    scale_list = wsamples[::-1] * 2. / (wvlMax + wvlMin)

    # main function of drizzler
    drizzle = form(cfg.dither, mode='temporal', ConnexOrigin2COR=cfg.drizzler.connexorigin2cor,
                   pixfrac=cfg.drizzler.pixfrac, cor_coords=cfg.drizzler.cor_coords, wvlMin=wvlMin, wvlMax=wvlMax,
                   device_orientation=cfg.drizzler.device_orientation, nwvlbins=nwvlbins, ntimebins=1, derotate=True,
                   fitsname=fitsname)

    # Get median spectral cube
    mask_tess = np.ma.masked_where(drizzle.data == 0, drizzle.data)
    medDither = np.ma.median(mask_tess, axis=0).filled(0)

    # # Inspect the spectral cube
    for i in range(nwvlbins):
        show = True if i == nwvlbins - 1 else False
        plt.imshow(medDither[i])#, drizwcs.wcs.cdelt[0], drizwcs.wcs.crval, vmin=1, vmax=10, show=show)


    fits.writeto(cfg.dither.name + '_med.fits', medDither, drizwcs.to_header(), overwrite=True)

    # Using PCA doesn't appear to work well
    # SDI = pca.pca(medDither, angle_list=np.zeros((medDither.shape[0])), scale_list=scale_list)

    # Do it manually
    scale_cube = np.zeros_like(medDither)
    for i in range(nwvlbins):
        scale_cube[i] = clipped_zoom(medDither[i], scale_list[i])
        show = True if i == nwvlbins - 1 else False
        pretty_plot(scale_cube[i], drizwcs.wcs.cdelt[0], drizwcs.wcs.crval, vmin=1, vmax=10, show=show)

    ref = np.median(scale_cube, axis=0)
    SDI = medDither - ref

    pretty_plot(SDI, drizwcs.wcs.cdelt[0], drizwcs.wcs.crval, vmin=1, vmax=10)

    fits.writeto(cfg.dither.name + '_SDI.fits', SDI, drizwcs.to_header(), overwrite=True)


if __name__ == '__main__':
    SDI()
