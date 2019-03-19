'''First median collapsing a time cube of all the raw dither images to a get a static PSF of the virtual grid.
For each dither derotate the static PSF and isolate the relevant area.
Subtract that static reference from the median of the derorated dither '''

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy.ndimage import rotate
from mkidpipeline.imaging.drizzler import form, pretty_plot, get_ditherdesc, write_fits

def rot_array(img, pivot,angle):
    padX = [img.shape[1] - pivot[0], pivot[0]]
    padY = [img.shape[0] - pivot[1], pivot[1]]
    imgP = np.pad(img, [padY, padX], 'constant')
    imgR = rotate(imgP, angle, reshape=False)
    return imgR[padY[0] : -padY[1], padX[0] : -padX[1]]

if __name__ == '__main__':
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

