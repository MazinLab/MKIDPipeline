#fits reading test
import numpy as np
import pyfits
from arrayPopup import plotArray

def readFITS(filename):
    f = pyfits.open(filename)
    scidata = np.array(f[0].data)
    return scidata

if __name__ == "__main__":
    fname = '/mnt/data0/ProcessedData/seth/imageStacks/PAL2016b/SAO42642_8Dithers_3xSamp_allHPM_20161122.fits'
    im = readFITS(fname)
    plotArray(im,title='loaded fits image',origin='upper',vmin=0)
