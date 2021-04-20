import math
import matplotlib.pylab as plt
import numpy as np
from numpy import linalg
try:
    from skimage.transform import rotate as imrotate
except ImportError:
    from scipy.misc import imrotate
from astropy import wcs
from astropy.io import fits
from astropy.coordinates import Angle
from mkidcore.corelog import getLogger
from matplotlib.colors import LogNorm
import astropy.units as u


def fit_rigid_rotation(x, y, ra, dec, x0=0, y0=0):
    """
    calculate the rigid rotation from row,col positions to ra,dec positions

    return dictionary of theta,tx,ty, such that

    ra  = c*dx - s*dx + dra
    dec = s*dy + c*dy + ddec
    
    with c = scale*cos(theta) and s = scale*sin(theta)
         dx = x-x0 and dy = y-y0

    ra,dec are input in decimal degrees

    The scale and rotation of the transform are recovered from the cd matrix;
      rm = w.wcs.cd
      wScale = math.sqrt(rm[0,0]**2+rm[0,1]**2) # degrees per pixel
      wTheta = math.atan2(rm[1,0],rm[0,0])      # radians
    """
    assert (len(x) == len(y) == len(ra) == len(dec)), "all inputs must be same length"
    assert (len(x) > 1), "need at least two points"

    dx = x - x0
    dy = y - y0
    a = np.zeros((2 * len(x), 4))
    b = np.zeros(2 * len(x))
    for i in range(len(x)):
        a[2 * i, 0] = -dy[i]
        a[2 * i, 1] = dx[i]
        a[2 * i, 2] = 1
        b[2 * i] = ra[i]

        a[2 * i + 1, 0] = dx[i]
        a[2 * i + 1, 1] = dy[i]
        a[2 * i + 1, 3] = 1
        b[2 * i + 1] = dec[i]
    answer, residuals, rank, s = linalg.lstsq(a, b)

    # put the fit parameters into the WCS structure
    sst = answer[0]  # scaled sin theta
    sct = answer[1]  # scaled cos theta
    dra = answer[2]
    ddec = answer[3]
    scale = math.sqrt(sst ** 2 + sct ** 2)
    theta = math.degrees(math.atan2(sst, sct))
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [x0, y0]  # reference pixel position
    w.wcs.crval = [dra, ddec]  # reference sky position
    w.wcs.cd = [[sct, -sst], [sst, sct]]  # scaled rotation matrix
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w


def get_device_orientation(coords, fits_filename='Theta1 Orionis B_mean.fits', separation=0.938, pa=253):
    """
    Given the position angle and offset of secondary calculate its RA and dec then
    continually update the FITS with different rotation matricies to tune for device orientation

    Default pa and offset for Trap come from https://arxiv.org/pdf/1308.4155.pdf figures 7 and 11

    B1 vs B2B3 barycenter separation is 0.938 and the position angle is 253 degrees

    :param coords:
    :param fits_filename:
    :param separation:
    :param pa:
    :return:
    """

    angle_from_east = 270 - pa

    companion_ra_arcsec = np.cos(np.deg2rad(angle_from_east)) * separation
    companion_ra_offset = (companion_ra_arcsec * u.arcsec).to(u.deg).value
    companion_ra = coords.ra.deg + companion_ra_offset

    companion_dec_arcsec = np.sin(np.deg2rad(angle_from_east)) * separation
    companion_dec_offset = (companion_dec_arcsec * u.arcsec).to(u.deg).value
    # minus sign here since reference object is below central star
    companion_dec = coords.dec.deg - companion_dec_offset

    getLogger(__name__).info('Target RA {} and dec {}'.format(Angle(companion_ra * u.deg).hms,
                                                              Angle(companion_dec * u.deg).dms))

    update = True
    device_orientation = 0
    hdu1 = fits.open(fits_filename)[1]

    field = hdu1.data
    while update:

        getLogger(__name__).info('Close this figure')
        ax1 = plt.subplot(111, projection=wcs.WCS(hdu1.header))
        ax1.imshow(field, norm=LogNorm(), origin='lower', vmin=1)
        plt.show()

        user_input = input(' *** INPUT REQUIRED *** \nEnter new angle (deg) or F to end: ')
        if user_input == 'F':
            update = False
        else:
            device_orientation += float(user_input)

        getLogger(__name__).warning('Using untested migration from scipy.misc.imrotate to skimage.transform.rotate. '
                                    'Verify results and remove this log message.')
        field = imrotate(hdu1.data, device_orientation, interp='bilinear')

    getLogger(__name__).info('Using position angle {} deg for device'.format(device_orientation))

    return np.deg2rad(device_orientation)

