import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mkidpipeline.imaging.drizzler import form, pretty_plot, smooth, write_fits

if __name__ == '__main__':
    # target = 'HD 34700'
    # ditherlog = 'HD34700_1547278116_dither.log'
    target = 'Trapezium'
    ditherlog = 'Trapezium_1547374552_dither.log'
    wvlMin = 850
    wvlMax = 1100
    startt = 0
    intt = 30
    pixfrac = .5

    # image, drizwcs = form(2, target=target, ditherlog=ditherlog, wvlMin=wvlMin,
    #                   wvlMax=wvlMax, startt=startt, intt=intt, pixfrac=pixfrac)
    #
    # pretty_plot(image, drizwcs.wcs.cdelt[0], drizwcs.wcs.crval, vmin=0, vmax=400)
    # write_fits(image, target+'_mean.fits')

    tess, drizwcs = form(4, rotate=1, target=target, ditherlog=ditherlog, wvlMin=wvlMin,
                     wvlMax=wvlMax, startt=startt, intt=intt, pixfrac=pixfrac)

    y = np.ma.masked_where(tess[:, 0] == 0, tess[:, 0])
    static_map = np.ma.median(y, axis=0).filled(0)

    pretty_plot(smooth(static_map), drizwcs.wcs.cdelt[0], drizwcs.wcs.crval, vmin=0, vmax=10)
    write_fits(image, target+'_med.fits')