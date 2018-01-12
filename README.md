# DarknessPipeline
Data reduction pipeline for DARKNESS, an MKID IFU for high contrast imaging

Installation
============

Install anaconda with python 3.6.  You will also need the packages pyintervals, astropy, pytables

IMPORTANT: FIXING PYQT4 BACKEND PROBLEMS
Upon re-installing python, the Qt backend is automatically set to PySide, which breaks Matt’s array pop-up gui (and possibly other GUIs that have not been tested yet). To fix this, the matplotlib rcParams file can be permanently edited to make PyQt4 your backend. Do the following. (instructions borrowed from matplotlib site: http://matplotlib.org/users/customizing.html#the-matplotlibrc-file)
 
To find your rcParams file, try:

ipython> import matplotlib

ipython> matplotlib.matplotlib_fname()

'/home/foo/.config/matplotlib/matplotlibrc'
 
Then find the line in your rc file that looks like:

#backend.qt4 : PyQt4        # PyQt4 | PySide
 
And make sure it is uncommented and set to PyQt4. With Canopy’s default install it will likely be PySide.


Pipeline Quick Start Guide
==========================

Selecting the .bin files you want to work with
----------------------------------------------

The first step is going to the observing log and finding out what files you want to work with. Once you have the data, you have two choices:

1.  If the data is taken with a single dither position, you can directly convert to a HDF5 (.h5) file using RawDataProcessinf/Bin2HDF.  Bin2HDF uses a config file that looks like this:

/home/bmazin/HR8799/rawdata
1507183126
301
/home/bmazin/HR8799/h5/finalMap_20170924.txt
1

The first line is the path of the .bin files.
The second line is the start time (and filename) of the data.
The third line is the duration in seconds to put into the .h5 file. Beware, filesize can grow quickly - 300 seconds of data from the 2017b run comes in at about 2.5 GB.
The fourth line is the location of the beam map file.
The fifth line is flag for specifying the data is beam mapped. It should almost always be 1.


Creating HDF5 files from the .bin files
----------------------------------------------


Wavelength Calibrating
----------------------------------------------


Flatfielding
----------------------------------------------


Creating Image Cubes
----------------------------------------------



Making Speckle Statistics Maps
----------------------------------------------
