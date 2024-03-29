# DS9 notes (Isabel, 7/16/2018)

I'm going to be working on some tools for working with FITS files (specifically, converting from .img OR .bin OR .h5 files to FITS files).  SAO DS9 is an astronomical imaging application equipped to handle FITS files (http://ds9.si.edu/site/Home.html).  It is well-supported and can run on almost any platform (MAC/Windows/Linux).  It has a lot of functionality in terms of importing coordinate systems, manipulating images (rotations, pan, zoom).  You can also load source catalogs from 2MASS, SDSS.  

### To install DS9:
1. Install ds9 via the recommended SAO and or conda approach for your OS
2. Add to your .bashrc file:  `export XPA_METHOD=local`
3. Now from the command line:  `ds9&` You should see a new window open up.  You can load FITS files directly into the GUI and go from there.

Here's a quick tutorial if you are unfamiliar with ds9 and you want to see more of its functionality:  https://astrobites.org/2011/03/09/how-to-use-sao-ds9-to-examine-astronomical-images/

DS9 is extremely scriptable!

I personally am using pyds9 as a python interface to ds9. The pyds9 module uses a Python interface to XPA to communicate with DS9. This includes not only sending data and settings to DS9, but also retrieving them into Python. 

Installation instructions here:  https://github.com/ericmandel/pyds9  Pyds9 is supported (well, at least as of a few months ago, its developers are still answering questions and addressing issues). Check conda though before blindly following!

### Quick fun activity with Pyds9
```python
import numpy as np
from astropy.io import fits
from astropy.utils.data import download_file
from astropy.io.fits import getdata
import pyds9
image_file = download_file('http://data.astropy.org/tutorials/FITS-images/HorseHead.fits', cache=True )
(im_int, hdr) = getdata(image_file, header=True) #image is numpy array
im = im_int.astype(np.float64)  # convert data from int to float
im +=0.01 # im is a numpy array, so we can do math on it.
d = pyds9.DS9('foo1')  # start ds9.  'd' is the way to call ds9
d.set_np2arr(im) # sending ndarray im directly to ds9
d.set("colorbar no")   # example of manipulating the ds9 window
d.set("scale zscale")  # example of manipulating the ds9 window
d.set("zoom to 0.6 0.6")
```