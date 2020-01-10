"""
binparse.py
Author: Kristina Davis
Oct 2018

This code is somewhat based on darkBinViewer.py from Seth+group.
It is meant to be a module to parse individual or a series of .bin files and
construct imaging or spectral datacubes, where image cubes are sequenced in
time (pix value is total # counts per time, all wavelengths) and spectral
data cubes are sequenced by phase (pix value is total # counts per phase bin).




"""

import numpy as np
import os
from mkidcore.hdf.mkidbin import parse
import matplotlib.pyplot as plt
from mkidcore.corelog import getLogger
import mkidcore.corelog
import argparse
import time

#####################################################################################################
#####################################################################################################

##################
# Module Functions
##################


def obs_list(basepath, tstampStart, tstampEnd):
    """
    Make a list of files to parse given a start and end time

    Inputs:
     basepath: path to the obs files eg: /mnt/data0/ScienceData/PAL218a/052918/
     tstampStart: second (UTC) that you want to start the file
     tstampEnd: second (UTC) that you want to end the file

    :return: a list of file names
    """
    return [os.path.join(basepath, str(int(msec)) + '.bin') for msec in range(tstampStart, tstampEnd + 1)]


def _makephasecube(xs, ys, phs, img_shape, range, d_phase, verbose=False):
    """See documentation in ParseBin.phasecube() for best info"""
    # Declaring Data Cube
    phs_bins = np.linspace(range[0], range[1], int((range[1]-range[0])/d_phase))  # use linspace to avoid floating errors, as with eg. arange
    phs_cube = np.zeros((img_shape[0], img_shape[1], phs_bins.shape[0]))

    # Looping to Create Data Cube
    tic = time.time()
    for i, value in enumerate(phs_bins):
        mask = (value < phs) & (phs < value+d_phase)
        np.add.at(phs_cube[:, :, i], (ys[mask], xs[mask]), 1)

    toc = time.time()
    if verbose:
        getLogger('binparse').debug("Time to make phase cube is {:4.2f} seconds".format(toc - tic))

    return phs_cube


def _makeimage(xs, ys, img_shape, verbose=False):
    """See documentation in ParseBin.image() for best info"""
    tic = time.time()
    ret = np.zeros(tuple(img_shape))
    np.add.at(ret, (ys, xs), 1)
    toc = time.time()
    if verbose:
        getLogger('binparse').debug("Time to make image is {:4.2f} seconds".format( toc - tic))
    return ret

#####################################################################################################
#####################################################################################################

################
# Classes
################


class ParsedBin(object):
    """
    Parse a .bin File and return photon list

    input: files = list of obs file names, full path specified
            could be returned by obs_list in module definitions
           pix_shape = dimensions of pixels in the MKID array as a tuple eg. (125,80)
            Default is None, so you don't need to specify if all you want is
            a photon list. If pix_shape=None, then you can't call .image
            or .phasecube
           Verbose = True if you want to print the timing stats, otherwise default
            is False

    output: a numpy object
        Attributes:
        .obs_files = input file name(s)
        .pix_shape = None or tuple of array size (shape of pixels on MKID board)
        .tstamp = list of photon arrival times (info from header+photon .5ms time)
        .phase = list of phases (not yet wavelength cal-ed)
        .baseline = list of baselines
        .roach = list of roach #s
        .x = list of x coordinates
        .y = list of y coordinates
        .obs_nphotons = number of photons in each of the .bin files sent into the list
        .tot_photons = total number of photons in the list

    """
    def __init__(self, files, pix_shape=None, verbose=False):
        # NB it isn't possible to automatically figure out the dimensions of the image as all pixels in an extremal row
        # or column might be dark.

        # Saving List of File Names
        self.obs_files = files
        self.pix_shape = pix_shape
        self.nXPix = pix_shape[1]  # keeping notation consistent with the beammap class
        self.nYPix = pix_shape[0]

        self.vb = verbose

        # Allocating Memory
        self._icube = None
        self._pcube = None
        self._pcube_meta = None

        self.x = np.empty(0, dtype=np.int)
        self.y = np.empty(0, dtype=np.int)
        self.tstamp = np.empty(0, dtype=np.uint64)
        self.baseline = np.empty(0, dtype=np.uint64)
        self.phase = np.empty(0, dtype=np.float32)
        self.roach = np.empty(0, dtype=np.int)
        self.obs_nphotons = np.empty(0, dtype=np.int)
        tic = time.time()

        # Parsing the Files and Appending to Photon List
        # Calls the file parsebin.pyx. If errors, first try making sure parsebin.pyx has been compiled
        #  Documentation in the .pyx file can help do this

        for f in files:
            try:
                # Calling new parse file
                parsef = parse(f)

                self.x = np.append(self.x, parsef.x)
                self.y = np.append(self.y, parsef.y)
                self.tstamp = np.append(self.tstamp, parsef.tstamp)
                self.baseline = np.append(self.baseline, parsef.baseline)
                self.phase = np.append(self.phase, parsef.phase)
                self.roach = np.append(self.roach, parsef.roach)

                self.obs_nphotons = np.append(self.obs_nphotons,parsef.x.shape)

            except (IOError, ValueError) as e:
                getLogger('binparse').error('Could not open file', exc_info=True)

        # Finding Total Number of Photons in the List
        self.tot_photons = int(sum(self.obs_nphotons))
        if self.tot_photons != self.x.shape[0]:
            raise RuntimeWarning('Error calculating total photons: Check for fake photons')

        toc = time.time()
        if verbose:
            msg = ("Parsing {} photons in {} "
                   "files took {:4.2f} seconds").format(self.tot_photons,len(files), toc - tic)
            getLogger('binparse').debug(msg)

    ###################################
    # Make Image
    ###################################
    def image(self):
        """
        makes a single image from the data in the photon list of the given range of .bin files

        Inputs: None (just self)

        Returns:
            2D numpy array of 16 bit integers.
            The array is a greyscaled image where the pixel value
            is equal to the number of photon hits per
            pixel in the given .bin file(s).
        """
        if self.pix_shape is None:
            raise ValueError("Cannot make image if pix_shape unspecified at class instance")

        if self._icube is None:
            self._icube = _makeimage(self.x, self.y, self.pix_shape, self.vb)
        return self._icube


    def getPixelCountImage(self, **kwargs):
        """
        This is a dummy function to emulate a similar method in photontable.py
        """
        return self.image()

    ###################################
    # Make Phase Cube
    ###################################
    def phasecube(self, range, dp):
        """
        makes a data cube of a single .bin file with phase as 3rd axis

        This code calls a function _makephasecube defined above, mostly for speeding up the parsing
         if you don't initially want to make the phase cube. _makephasecube won't be called until
         ParseBin.phasecube(range,dx) is called

        Inputs:
            range = tuple of range of phases to bin, in degrees eg. (-360,0)
            dp = size of a single phase bin for axis3 of the data cube, in units of deg eg. 5

        Output:
            .shape = 3D shape of the cube
            .bin_size = size of the phase bin in degrees (phasebin size of axis 3)
            returns: data cube of shape (img_shape[0],img_shape[1],(range[1]-range[0])/d_phase)
                the value of each x,y,z location is the number of photon hits
                per pixel per phase bin in the entire photon list in the obs range

        """
        if self.pix_shape is None:
            raise ValueError("Cannot make phase cube if pix_shape unspecified at class instance")

        if self._pcube is None or self._pcube_meta != (range, dp):
            self._pcube = _makephasecube(self.x, self.y, self.phase, self.pix_shape, range, dp, self.vb)
            self._pcube_meta = (range, dp)

        return self._pcube


    def getPixelPhotonList(self,xCoord = None, yCoord = None, **kwargs):
        """
        Emulates the method of the same name in photontable.py
        """
        if xCoord is None or yCoord is None:
            print('x and/or y coordinate not specified')
            return
        tstamps = self.tstamp[np.logical_and(self.x == xCoord, self.y == yCoord)]
        tstamps -= np.amin(tstamps) # remove offset so that smallest timestamp is at zero
        phase = self.phase[np.logical_and(self.x == xCoord, self.y == yCoord)]

        # the datatype of the structured numpy array returned by getPixelPhotonList in ObsFile is:
        # dtype = [('ResID', '<u4'), ('Time', '<u4'), ('Wavelength', '<f4'), ('SpecWeight', '<f4'),('NoiseWeight', '<f4')])

        # bin files only have timestamps and wavelengths, the xy coords are already specified for this method

        # note on dtype for Time: if we remove the initial offset at the beginning of the file, then
        # dtype can be <u4 (uint32). If we don't remove it, then it needs to be <u8 (uint64).

        wtype = np.dtype([('Time', '<u4'), ('Wavelength', '<f4')])
        w = np.empty(len(tstamps),wtype)
        w['Time'] = tstamps
        w['Wavelength'] = phase
        return w


    ###################################
    # Reshape
    ###################################
    def reshape(self, newshape):
        """
        Allows us to reshape the array to new pix_shape without re-parsing the data.
        After you use it, you would need to re-make the .image and .phasecubes. This
        is another way of saying this changes the class instance but not the method
        output.
        """
        if newshape[0] < self.x.max() or newshape[1] < self.x.max():
            raise ValueError('Bad shape')
        self._pcube = None
        self._icube = None
        self.pix_shape = newshape

#####################################################################################################
#####################################################################################################
# Main
#####################################################################################################
#####################################################################################################

if __name__ == "__main__":

    # Setting format of the logger
    mkidcore.corelog.create_log('binparse', console=True, propagate=False,
                       fmt='%(levelname)s %(message)s', level=mkidcore.corelog.DEBUG)

    # Assigning Keyword Arguments
    parser = argparse.ArgumentParser(description='MKID Python Binfile Parser')
    parser.add_argument('basepath', type=str, help='Full path to obs files')
    parser.add_argument('starttime', type=int, help='Starting timestamp')
    parser.add_argument('endtime', type=int, help='Ending timestamp (inclusive)')
    args = parser.parse_args()

    # Make Obs list
    fnames = obs_list(args.basepath, args.starttime, args.endtime)

    # Test Settings
    img_shape = (125, 80) # mec (146,140)
    phase_binsize = 2
    phs_range = (-250, -200)

    # Testing the Code
    test1 = ParsedBin(fnames, img_shape, verbose=True)
    timg = test1.image()
    test_cube = test1.phasecube(phs_range, phase_binsize)

    # Plot Test Image

    fig, ax = plt.subplots()
    im = ax.imshow(timg)
    plt.show()

    # Plotting Test Phase Cube
    fig, ax = plt.subplots()
    phsbin_midpts = np.linspace(phs_range[0], phs_range[1], int((phs_range[1]-phs_range[0]) / phase_binsize)) + (phase_binsize) / 2
    spec = test_cube[30, 62, :]
    plt.bar(phsbin_midpts, spec, width=phase_binsize)
    plt.show()

