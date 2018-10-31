"""
binParse.py
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
import sys
from mkidpipeline.hdf.parsebin import parse
import matplotlib.pyplot as plt

from time import time

#####################################################################################################
#####################################################################################################

################
# Definitions
################


def obs_list(basepath, tstampStart, tstampEnd):
    '''
    Make a list of files to parse given a start and end time

    Inputs:
     basepath: path to the obs files eg: /mnt/data0/ScienceData/PAL218a/052918/
     tstampStart: second (UTC) that you want to start the file
     tstampEnd: second (UTC) that you want to end the file

    :return: a list of file names
    '''
    obs_list = []
    msec_list = np.arange(tstampStart, tstampEnd + 1)  # millisecond list

    for nbin in enumerate(msec_list):
        obs_list.append(os.path.join(basepath, str(nbin[1])+'.bin'))

    return obs_list


def _makephasecube(xs, ys, phs, img_shape, range, d_phase, vb):
    """See documentation in ParseBin.phasecube() for best info"""
    # Declaring Data Cube
    phs_bins = np.linspace(range[0], range[1], int((range[1]-range[0])/d_phase))  # use linspace to avoid floating errors, as with eg. arange
    phs_cube = np.zeros((img_shape[0], img_shape[1], phs_bins.shape[0]))

    # Looping to Create Data Cube
    tic = time()
    for i,value in enumerate(phs_bins):
        this_phsbin = np.logical_and(phs>value, phs<value+d_phase).nonzero()
        for x,y in zip(xs[this_phsbin[0]], ys[this_phsbin[0]]):
            phs_cube[y, x, i] += 1
    toc = time()
    if vb is True:
        print("Time to make phase cube is {:4.2f} seconds".format(toc - tic))
    return phs_cube

def _makeimage(xs, ys, img_shape, vb):
    """See documentation in ParseBin.image() for best info"""
    tic = time()
    ret = np.zeros((img_shape[0], img_shape[1]))
    for x, y in zip(xs, ys):
        ret[y, x] += 1
    toc = time()
    if vb is True:
        print("Time to make image is {:4.2f} seconds".format( toc - tic))
    return ret

#####################################################################################################
#####################################################################################################
################
# Classes
################


class ParseBin:
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
        .roach_num = list of roach #s
        .x = list of x coordinates
        .y = list of y coordinates
        .nphotons = number of photons in each of the .bin files sent into the list
        .total_photons = total number of photons in the list

    """
    def __init__(self, files, pix_shape=None, Verbose=False):
        # Saving List of File Names
        self.obs_files = files
        self.pix_shape = pix_shape
        self.vb = Verbose

        self.x = np.empty(0, dtype=np.int)
        self.y = np.empty(0, dtype=np.int)
        self.tstamp = np.empty(0, dtype=np.uint64)
        self.baseline = np.empty(0, dtype=np.uint64)
        self.phase = np.empty(0, dtype=np.float32)
        self.roach = np.empty(0, dtype=np.int)
        self.nphotons = np.empty(0, dtype=np.int)
        tic = time()

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

                self.nphotons = np.append(self.nphotons,parsef.x.shape)


            except (IOError, ValueError) as e:
                # import mkidcore.corelog as clog
                # clog.getLogger('BinViewer').error('Help', exc_info=True)
                print(e)

        # Finding Total Number of Photons in the List
        self.total_photons = int(sum(self.nphotons))
        if self.total_photons != self.x.shape[0]:
            raise ValueError('Error calculating total photons: Check for fake photons')

        toc = time()
        if Verbose is True:
            print("Parsing {} photons in {} files took {:4.2f} seconds".format(self.total_photons,len(files), toc - tic))

    ###################################
    # Make Image
    ###################################
    def image(self):
        """
        makes a single image from the data in the photon list of the given range of .bin files

        Inputs:
            shape is a tuple with the size of the image eg. [125,80]

        Returns:
            2D numpy array of 16 bit integers.
            The array is a greyscaled image where the pixel value
            is equal to the number of photon hits per
            pixel in the given .bin file(s).
        """
        if self.pix_shape is None:
            raise ValueError("Cannot make image if pix_shape unspecified at class instance")

        return _makeimage(self.x, self.y, self.pix_shape, self.vb)

    ###################################
    # Make Phase Cube
    ###################################

    def phasecube(self, range, dx):
        """
        makes a data cube of a single .bin file with phase as 3rd axis

        This code calls a function _makephasecube defined above, mostly for speeding up the parsing
         if you don't initially want to make the phase cube. _makephasecube won't be called until
         ParseBin.phasecube(range,dx) is called

        Inputs:
            range = tuple of range of phases to bin, in degrees eg. (-360,0)
            d_phase = size of a single phase bin for axis3 of the data cube, in units of deg eg. 5

        Output:
            .shape = 3D shape of the cube
            .bin_size = size of the phase bin in degrees (phasebin size of axis 3)
            returns: data cube of shape (img_shape[0],img_shape[1],(range[1]-range[0])/d_phase)
                the value of each x,y,z location is the number of photon hits
                per pixel per phase bin in the entire photon list in the obs range

        """
        if self.pix_shape is None:
            raise ValueError("Cannot make phase cube if pix_shape unspecified at class instance")

        return _makephasecube(self.x, self.y, self.phase, self.pix_shape, range, dx, self.vb)

    ###################################
    # Reshape
    ###################################
    def reshape(self, newshape):
        """
        Allows us to reshape the array to new pix_shape without re-parsing the data.
        After you use it, you would need to re-make the .image and .phasecubes. This
        is another way of saying this changes the class instance but not the method
        output
        """
        if newshape[0] < self.x.max() or newshape[1] < self.x.max():
            raise ValueError('Bad shape')
        self.pix_shape = newshape

#####################################################################################################
#####################################################################################################


#####################################################################################################
#####################################################################################################


    def plot_phs_spec(self,x,y):
        import matplotlib.pyplot as plt

        phsbin_midpts = np.linspace(-360,0,int(360/self.dx))+(self.dx)/2
        spec = self.cube[x,y,:]
        print(str(spec))
        plt.bar(phsbin_midpts,spec,width=self.dx)
        plt.show()




#####################################################################################################
#####################################################################################################

#####################################################################################################
#####################################################################################################

if __name__ == "__main__":
    kwargs = {}
    if len(sys.argv) != 4:
        print('Usage: {} basepath tstampStart tstampEnd'.format(sys.argv[0]))
        exit(0)
    else:
        kwargs['basepath'] = str(sys.argv[1])
        kwargs['tstampStart'] = int(sys.argv[2])
        kwargs['tstampEnd'] = int(sys.argv[3])


    fnames = obs_list(**kwargs)
    img_shape = (125, 80)

    test1 = ParseBin(fnames, img_shape, Verbose=True)
    timg = test1.image()
    fig, ax = plt.subplots()
    im = ax.imshow(timg)
    plt.show()

    phase_binsize = 2
    phs_range = (-250,-200)
    test_cube = test1.phasecube(phs_range,phase_binsize)

    fig, ax = plt.subplots()
    phsbin_midpts = np.linspace(phs_range[0], phs_range[1], int((phs_range[1]-phs_range[0]) / phase_binsize)) + (phase_binsize) / 2
    spec = test_cube[30, 62, :]
    plt.bar(phsbin_midpts, spec, width=phase_binsize)
    plt.show()

