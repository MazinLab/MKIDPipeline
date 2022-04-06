import numpy as np
from numpy import linalg
try:
    from skimage.transform import rotate as imrotate
except ImportError:
    from scipy.misc import imrotate
from astropy import wcs


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
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [x0, y0]  # reference pixel position
    w.wcs.crval = [dra, ddec]  # reference sky position
    w.wcs.cd = [[sct, -sst], [sst, sct]]  # scaled rotation matrix
    w.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    return w
