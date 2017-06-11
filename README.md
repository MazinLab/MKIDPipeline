# DarknessPipeline
Data reduction pipeline for DARKNESS, an MKID IFU for high contrast imaging

IMPORTANT: FIXING PYQT4 BACKEND PROBLEMS
Upon re-installing python, the Qt backend is automatically set to PySide, which breaks Matt’s array pop-up gui (and possibly other GUIs that have not been tested yet). To fix this, the matplotlib rcParams file can be permanently edited to make PyQt4 your backend. Do the following. (instructions borrowed from matplotlib site)
 
To find your rcParams file, try:
ipython> import matplotlib
ipython> matplotlib.matplotlib_fname()
'/home/foo/.config/matplotlib/matplotlibrc'
 
Then find the line in your rc file that looks like:
#backend.qt4 : PyQt4        # PyQt4 | PySide
 
And make sure it is uncommented and set to PyQt4. With Canopy’s default install it will likely be PySide.
