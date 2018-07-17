# DarknessPipeline
Data reduction pipeline for DARKNESS, a MKID IFU for high contrast imaging.

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

The first step is going to the observing log and finding out what files you want to work with. Look for good seeing conditions and note the times of the nearest laser cals.  Copy the .bin files to your local computer if possible.  Never alter or delete the .bin files on dark!

You can look at the raw data using python darkBinViewer.py in /QuickLook.


Creating HDF5 files from the .bin files
----------------------------------------------

Once you have the data, you have two choices:

1.  If the data is taken with a single dither position, you can directly convert to a HDF5 (.h5) file using /RawDataProcessing/Bin2HDF.  Bin2HDF uses a config file that looks like this:

    ```
    80 125
    /home/bmazin/HR8799/rawdata
    1507183126
    301
    /home/bmazin/HR8799/h5/finalMap_20170924.txt
    1
    /home/bmazin/HR8799/h5
    ```
The first line has the dimensions of the array (80 xpix by 125 ypix for DARKNESS)
The secon line is the path of the .bin files.
The third line is the start time (and filename) of the data.
The fourth line is the duration in seconds to put into the .h5 file. Beware, filesize can grow quickly - 300 seconds of data from the 2017b run comes in at about 2.5 GB.
The fifth line is the location of the beam map file.
The fifth line is flag for specifying the data is beam mapped. It should almost always be 1. The file format is picky. If Bin2HDF fails, make sure there are no extra spaces in the configuration file, all the files exist and that you have permissions to access all of them and their directories.
The sixth line is the output directory for the h5 file.

2. If the data is a dither stack taken with python ditherScript.py, find the associated .cfg file that ditherScript outputs.  Then run pythion Dither2HDF.py in /RawDataProcessing.  For example, python Dither2H5.py ditherStack_1507183126.cfg 0.  The 0 at the end is how much time to clip from each dither.  This is usually going to be 0 as ditherScript.py already exludes the time while the image is moving.

Dither2HDF creates a seperate .h5 file for each dither position.

Once you have .h5 files, you can look at 1 second raw images in hdfview.


Wavelength Calibration
----------------------------------------------
The wavelength calibration code is in the Calibration/WavelengthCal/ folder. There are three files that are important.

    WaveCal.py -- main file that contains all of the calibration code
    plotWaveCal.py -- plotting functions for viewing the results of the calibration
    ./Params/default.cfg -- default config file, used for reference

Additionally, an .h5 file must be generated from the .bin files for each wavelength in the calibration. These file names will be referenced in the configuration file. See the section "Creating HDF5 files from the .bin files" for more info.
#### Additional required packages
Some non-standard python packages are needed for the wavelength calibration to work. They are lmfit, progressbar, and PyPDF2. They can be downloaded with the following commands:

    conda install -c conda-forge lmfit  
    conda install -c conda-forge progressbar2  
    conda install -c conda-forge pypdf2

Additionally, Latex needs to be installed and configured to work with matplotlib. This only affects the plotting routines. The solution file will be generated even if Latex is not set up correctly. If installing Latex out-of-the-box still makes the WaveCal code error, there are a couple of things to try.
+ Look at the error. Is there a required latex package that can't be loaded? If so, download a more comprehensive latex distribution.
+ Check that dvipng is installed and up to date.
+ Check that Ghostscript is up to date.
+ If none of the above work, go to https://matplotlib.org/users/usetex.html for more detailed information.

#### Configuration file
The default.cfg file is a reference configuration file and has detailed comments in it describing each parameter. The configuration file is setup using python syntax. (i.e. if a parameter is a string, it must be surrounded by quotation marks.) The most important parameters will be described here.

    Data Section:
    directory      -- full path to the folder containing the .h5 files for each laser
                      wavelength (string)
    wavelengths    -- wavelengths in nanometers being used in the calibration (list of
                      numbers)
    file_names     -- .h5 file names for the wavelengths above using the same ordering
                      (list of strings)

    Fit Section:
    parallel       -- determines if multiple processes will be used to compute the
                      solution in parallel. This option greatly decreases the computation
                      time if your computer has 4 or more cores. The code might freeze if
                      this option is used on a slower computer. (True or False)

    Output Section:
    out_directory  -- full path to the folder where the output data will be saved. This
                      includes the calsol.h5 solution file, log files, and summary plots.
                      (string)
    summary_plot   -- determines whether or not to save a summary plot as a pdf after the
                      computation. It is useful for characterizing a device (True or
                      False)
    templar_config -- If summary_plot = True, the templar config file used for taking the
                      data can be used to make an energy resolution vs frequency plot.
                      This is optional. Use the full path to the file (including the file
                      name). (string)

#### Running from the command line
Before running the wavelength calibration, generate the .h5 files for the laser data and make your configuration file. The calibration can be run from the command line using the syntax

    python /path/to/WaveCal.py /other/path/to/my_config.cfg
'/path/to/' and '/other/path/to/' are the full or relative paths to the WaveCal.py and my_config.cfg files respectively. 'my_config.cfg' is your custom configuration file. If no configuration file is specified, the default configuration file will be used. Never commit changes to the default configuration file to the repository.

The solution .h5 file will be saved in the output directory as calsol_timestamp.h5, where timestamp is the utc time stamp for the start time of the wavelength calibration.
#### Running from a script or a Python shell
The calibration can also be run from a script or a python shell. This option allows the flexibility to compute the solution for a group of selected pixels. It is particularly useful for debugging and speeding up the computation time. The following lines of code demonstrate the process:

    from DarknessPipeline.Calibration.WavelengthCal import WaveCal as W
    w = W.WaveCal(config_file='/path/to/my_config.cfg')
    w.makeCalibration(pixels=my_pixels)
'/path/to/my_config.cfg' is the path and filename of your configuration file as a string. my_pixels is a list of length 2 lists containing the (row, column) of each pixel you want included in the calculation. (e.g. [[1, 2], [3, 4], [10, 50]]) If not included, the calculation will be done on all of the pixels.
#### Plotting the results of the calibration
Plots of the results of a wavelength calibration can be made with functions in plotWaveCal.py. Six main types of plots are available.

    plotEnergySolution()  -- plots of the phase to energy calibration solution for a
                             particular pixel
    plotHistogramFits()   -- plots of the phase height histogram fits for a particular
                             pixel
    plotRHistogram()      -- a histogram plot of computed energy resolutions for the array
                             at different wavelengths
    plotCenterHist()      -- a histogram plot of the gaussian centers for different
                             wavelengths
    plotRvsF()            -- a scatter plot of energy resolutions as a function of
                             resonance frequency
    plotFitParameters()   -- a summary plot of the distributions of each fit parameter
                             over the different wavelengths
    plotResolutionImage() -- plots an image of the array color coded by the energy
                             resolution
    plotSummary()         -- a summary plot of the wavelength calibration solution.
                             Usually generated automatically after running the
                             calibration code. It includes plotRHistogram, plotRvsF,
                             plotCenterHist and other summary statistics
The following sample script shows how to generate each plot. View the docstrings for more detail on the options for each plot.

    from DarknessPipeline.Calibration.WavelengthCal import plotWaveCal as p

    # path to your wavecal solution file
    file_name = '/path/to/calsol_timestamp.h5'

    ### plot info about a single pixel
    # pixel you care about
    my_pixel = [12, 15]
    p.plotEnergySolution(file_name, pixel=my_pixel)
    p.plotHistogramFits(file_name, pixel=my_pixel)
    # could use keyword res_id=my_res_id instead of pixel

    ### plot array overview information
    # pick which wavelengths you want plotted in the histograms (optional)
    my_mask = [True, True, True, True]
    p.plotRHistogram(file_name, mask=my_mask)
    p.plotCenterHist(file_name, mask=my_mask)
    p.plotFitParameters(file_name)
    p.plotResolutionImage(file_name)
    # your templar configuration file
    templar_config = '/path/to/templarconf.cfg'
    p.plotRvsF(file_name, templar_config)
    p.plotSummary(file_name, templar_config)

#### Applying the wavelength calibration
After the wavelength calibration .h5 solution file is made, it can be applied to an obs file by using this code snippet

    # path to your wavecal solution file
    file_name = '/path/to/calsol_timestamp.h5'
    obs.applyWaveCal(file_name)
where obs is your obs file object. The method applyWaveCal() will change all of the phase heights to wavelengths in nanometers. For pixels where no calibration is available, the phase heights are not changed and a flag is applied to mark the pixel as uncalibrated. Warning, the wavelength calibration can not be undone after applied and permanently alters the .h5 file. Make a backup .h5 file if you are testing different calibrations.

Flat Fielding
----------------------------------------------
The flatfield calibration code is in the Calibration/FlatCal/ folder. These two files run the FlatCal:

    FlatCal.py -- main file that contains the calibration code and the FlatCal plotting functions
    ./Params/default.cfg -- default config file, used for reference.  

The FlatCal can be run on a single flat h5 file or several flat h5 files.  You will make use of different parameters in the config file depending on whether your flat h5 files are in the ScienceData directory or your own working directory (see the section on how to use the config file).

FlatCal will default to the reference config file.  I recommend copying it over to your working directory, renaming it, and editing it as appropriate for your specific calibration

#### Before running the FlatCal:
     Generate an .h5 file from the .bin files for the desired timespan of the flatfield. It may be named with a timestamp or descriptive name (e.g. 20171004JBandEndofNightFlat.h5)
     See the section "Creating HDF5 files from the .bin files" for more info.

     Make a wavecal solution file and apply it to the flat h5 file.  See Wavelength Calibration for more details on how to do that.

#### Configuration file
The default.cfg file is a reference configuration file and has detailed comments in it describing each parameter. The configuration file is setup using python syntax. (i.e. if a parameter is a string, it must be surrounded by quotation marks.) The most important parameters will be described here.  There are is a function in FlatCal which will check each parameter to confirm it is the correct format (string, int, etc).
NOTE that some of these parameters are conditional and some are required.  See notes below and default.cfg for which ones are required, which are conditional.

    Data Section:  This section provides the location and description of the FlatCal data being processed.
----------------------------------------------------------------------------------------------------------------------------------------------
    Important:  Fill these four parameters out ONLY if your flat h5 files are in the dark ScienceData directory AND are named their starting timestamps.  
    If it has been copied to your personal data reduction directory, leave them as empty strings, ''
    If these are not empty strings, the code will search in '/mnt/data0/ScienceData/Run/Date for flatObsTstamps.h5 and generate a FlatCalSoln file:
    '/mnt/data0/ScienceData/Run/Date/flatCalTstamp_calSoln.h5'

    run                      -- e.g. PAL2017b, which observing run is the flatcal file from? (string)  CONDITIONAL
    date                     -- e.g. 20171005, which night is the flatcal file from (string)   CONDITIONAL
    flatCalTstamp            -- Timestamp which will be prefix of FlatCalSolution file (string)   CONDITIONAL
    flatObsTstamps           -- List of starting timestamps for the flat calibration, one for each Flat h5 file used
                                (list of strings, even if there is just one file being used) [] CONDITIONAL
------------------------------------------------------------------------------------------------------------------------------------------------
    wvlDate                  -- Wavelength Sunset Date (string).  Leave '' if wavecal is already applied     CONDITIONAL
    wvlCalFile               -- Wavecal Solution Directory Path + File (To be used in plotting)   REQUIRED
------------------------------------------------------------------------------------------------------------------------------------------------   
    Important:  Fill these two parameters out IF your flat h5 file has been copied to your personal data reduction directory

    flatPath                 -- Path to your flat h5 file (string) CONDITIONAL
    calSolnPath              -- Output Cal Soln path (string).  Include the path and a basename for what you want the Flat Cal Solution files to be titled
                                (e.g '/mnt/data0/isabel/DeltaAnd/Flats/DeltaAndFlatSoln.h5')
                                You will get EXPTIME/INTTIME number of files titled DeltaAndFlatSoln#.h5     CONDITIONAL
------------------------------------------------------------------------------------------------------------------------------------------------
    intTime                  -- Integration time to break up larger h5 files into in seconds (number) (5 sec recommended)   REQUIRED
    expTime                  -- Total exposure time (number)   REQUIRED

    Instrument Section:  This section provides information about the specific wavelength and energy parameters for this instrument   
----------------------------------------------------------------------------------------------------------------------------------------------
    deadtime                 -- (number) REQUIRED
    energyBinWidth           -- Energy Bin width in eV (number)  REQUIRED
    wvlStart                 -- Starting wavelength in nanometers (number)  REQUIRED
    wvlStop                  -- Final wavelength in nanometers (number)  REQUIRED

    Calibration Section:  This section provides the parameters specific to the flat calibration function   
----------------------------------------------------------------------------------------------------------------------------------------------
    countRateCutoff          -- Count rate cutoff in seconds (number)  REQUIRED
    fractionOfChunksToTrim   -- Fraction of Chunks to trim (integer)  REQUIRED
    timeMaskFileName         -- Time mask file name (string)  CONDITIONAL
    timeSpacingCut           -- Time spacing cut (never used)  (string)  REQUIRED 'None'


#### Running from the command line
Before running the Flat calibration, generate your flat h5 file and run the wavecal on it.
The Flat calibration can be run from the command line using the syntax:

    python /path/to/FlatCal.py /other/path/to/my_config.cfg
'/path/to/' and '/other/path/to/' are the full or relative paths to the FlatCal.py and my_config.cfg files respectively. 'my_config.cfg' is your custom configuration file. If no configuration file is specified, the default configuration file will be used. Never commit changes to the default configuration file to the repository.

The FlatCal will output a number of calsoln h5 files (EXPTIME/INTTIME) all with the name '.../calSolPath+[index].h5' where index spans EXPTIME/INTTIME.
It will also output three plot pdfs for each h5 calsoln file: '.../calSolPath+[index]__mask.pdf', '.../calSolPath+[index]__wvlSlices.pdf', '.../calSolPath+[index].pdf'

#### Running from a script or a Python shell
The calibration can also be run from a script or a python shell.

    from Calibration.FlatCal import FlatCal as F
    f = F.FlatCal(config_file='/path/to/my_config.cfg')
    f.loadFlatSpectra()
    f.checkCountRates()
    f.calculateWeights()

#### Plotting the results of the calibration
Plots of the results of a flatfield calibration will be done automatically in FlatCal.py

    plotWeightsByPixelWvlCompare() -- Plot weights of each wavelength bin for every single pixel
                                      Makes a plot of wavelength vs weights, twilight spectrum, and wavecal solution for each pixel
                                      '.../calSolPath+[index].pdf'

    plotWeightsWvlSlices()         -- Plot weights in images of a single wavelength bin (wavelength-sliced images)
                                      '.../calSolPath+[index]__wvlSlices.pdf'     

    plotMaskWvlSlices()            -- Plot mask in images of a single wavelength bin (wavelength-sliced images)
                                      '.../calSolPath+[index]_WavelengthCompare.pdf'

#### Applying the flatfield calibration
After the flatfield calibration .h5 solution files are made, they can be applied to an obs file by using this code

    ObsFN = '/path/to/obsfile.h5'

    #Path to the location of the FlatCal solution files should contain the base filename of these solution files
    #e.g '/mnt/data0/isabel/DeltaAnd/Flats/DeltaAndFlatSoln.h5')
    #Code will grab all files titled DeltaAndFlatSoln#.h5 from that directory

    calSolnPath='/path/to/calSolutionFile.h5'

    obsfile=obs(ObsFN, mode='write')
    obsfilecal=obs.applyFlatCal(obsfile,calSolnPath,verbose=True)

Weights are multiplied in and replaced; if "weights" are the contents of the "SpecWeight" column, weights = weights*weightArr. NOT reversible unless the original contents (or weightArr) is saved.
Will write plots of flatcal solution (5 second increments over a single flat exposure) with average weights overplotted to a pdf for pixels which have a successful FlatCal.
Written to the calSolnPath+'FlatCalSolnPlotsPerPixel.pdf'

Creating Image Cubes
----------------------------------------------


QuickLook
----------------------------------------------
#### Using quickLook.py
    Before you start:
        -create HDF5 file from .bin files as described above
        -apply wave cal to HDF5 as described above
        -(optional) set $MKID_DATA_DIR to the path where your HDF5 is located

    -In the command line, run
        >> python quickLook.py
    -Go to File>Open and select your HDF5 file, click OK.
        -The Beam Flag Image is automatically displayed, showing you what pixels are good and bad.
         See pipelineFlags.py for explanation of the flag values.
    -Specify your desired start/stop times and wavelengths.
    -To view an image, select the "Raw Counts" radio button, and click "Plot Image"
    -To view a single pixel's timestream, intensity histogram, or spectrum
        -click on the desired pixel. This is now your "Active Pixel", and is shown
         the bottom of the window.
        -Go to "Plot" menu and click on the type you want.
        -Selecting a new Active Pixel will update the subplots.



Making Speckle Statistics Maps
----------------------------------------------
