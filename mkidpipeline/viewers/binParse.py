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


# Defining the Size of an Image
# needs to be updated if changed between runs/instruments
#image_shape = {'nrows':125,'ncols':80}

#####################################################################################################
#####################################################################################################

################
# Definitions
################

def obs_list(basepath,tstampStart,tstampEnd):
    '''
    Make a list of files to parse given a start and end time

    Inputs:
     basepath: path to the obs files eg: /mnt/data0/ScienceData/PAL218a/052918/
     tstampStart: second (UTC) that you want to start the file
     tstampEnd: second (UTC) that you want to end the file

    :return: a list of file names
    '''
    obs_list = []
    msec_list = np.arange(tstampStart, tstampEnd + 1) # millisecond list

    for nbin in enumerate(msec_list):
        obs_list.append(os.path.join(basepath,str(nbin[1])+'.bin'))

    return obs_list

def _makephasecube(xs,ys,phs, img_shape,range,d_phase):
    """
    makes a data cube of a single .bin file with phase as 3rd axis, broken up by bin range

    We assume that the range of data is from -360 0, which seems to be the case from parsed data
    The unfortunate part about this is that for the most part we ignore other data when making
     the phase cube. So we don't keep any info from the baseline, roachnum, etc.


    Inputs:
            phase_dx- size of a single phase bin for axis3 of the data cube, in units of deg

    Output: data cube with attributes:
    """
    phs_bins = np.linspace(range[0],range[1],int((range[1]-range[0])/d_phase))
    phs_cube = np.zeros((img_shape[0], img_shape[1], phs_bins.shape[0]))
    stcube = time()

    for i,value in enumerate(phs_bins):
        this_phsbin = np.logical_and(phs>value,phs<value+d_phase).nonzero()
        for x,y in zip(xs[this_phsbin[0]],ys[this_phsbin[0]]):
            phs_cube[y,x,i] += 1
    etcube = time()
    print("Making phase cube of single bin file took {} seconds".format(etcube-stcube))
    return phs_cube

#####################################################################################################
#####################################################################################################
################
# Classes
################

class ParseBin:
    """
    Parse a .bin File and return photon list

    input: list of obs file names, full path specified
            could be returned by obs_list in module definitions

    output: a numpy object
        Attributes:
        .obs_files = input file name
        .tstamp = list of photon arrival times (info from header+photon .5ms time)
        .phase = list of phases (not yet wavelength cal-ed)
        .baseline = list of baselines
        .roach_num = list of roach #s
        .x = list of x coordinates
        .y = list of y coordinates

    """
    def __init__(self,files):
        self.obs_files = files

        self.x = np.empty(0, dtype=np.int)
        self.y = np.empty(0, dtype=np.int)
        self.tstamp = np.empty(0, dtype=np.uint64)
        self.baseline = np.empty(0, dtype=np.uint64)
        self.phase = np.empty(0, dtype=np.float32)
        self.roach = np.empty(0, dtype=np.int)
        self.nphotons = np.empty(0, dtype=np.int)
        tic = time()

        for f in files:
            try:
                # Calling new parse file
                parsef = parse(f)

                self.x = np.append(self.x,parsef.x)
                self.y = np.append(self.y,parsef.y)
                self.tstamp = np.append(self.tstamp,parsef.tstamp)
                self.baseline = np.append(self.baseline,parsef.baseline)
                self.phase = np.append(self.phase,parsef.phase)
                self.roach = np.append(self.roach,parsef.roach)

                self.nphotons = np.append(self.nphotons,parsef.x.shape)

            except (IOError, ValueError) as e:
                # import mkidcore.corelog as clog
                # clog.getLogger('BinViewer').error('Help', exc_info=True)
                print(e)

        self.total_photons = int(sum(self.nphotons))
        toc = time()
        print("Parsing {} photons in {} files took {} seconds".format(self.total_photons,len(files), toc - tic))

    ###################################
    # Make Image
    ###################################
    def image(self,shape):
        """
        makes a single image from the data in the single bin file
        calls ParseSingleBin

        This code makes an image of the parsed file. The image is a greyscaled
        image where the pixel value is equal to the number of photon hits per
        pixel in the given .bin file.

        shape is a tuple with the size of the image [x,y]
        """
        self.shape = shape

        tic = time()
        ret = np.zeros(shape, dtype=np.uint16)
        for x, y in zip(self.x, self.y):
            ret[y,x] += 1
        toc = time()
        print("Time to make image with {} photons is {} seconds".format(self.total_photons,toc-tic))
        return ret

    ###################################
    # Make Image
    ###################################

    def phasecube(self, shape,range, dx):
        self.shape = (shape,(range[1]-range[0])/dx)
        self.phs_size = dx
        return _makephasecube(self.x, self.y, self.phase, shape,range, dx)


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

    def plot_phs_img(self,plotbin):
        import matplotlib.pyplot as plt

        try:
            phs_range = np.linspace(-360, 0, int(360 / self.dx))

            phs_img = self.cube[:,:,self.dx]

            fig, ax = plt.subplots()
            im = ax.imshow(phs_img)
            plt.show()

        except ValueError as e:
            print(e)
            print("input phase bin out of range\n")
            print("Number of phase bins ={}".format(phsbin_middles.shape))


#####################################################################################################
#####################################################################################################




    def plot_phs_spec(self,x,y):
        import matplotlib.pyplot as plt

        spec = self.cube[x,y,:]
        fig, ax = plt.subplots()
        im = ax.imshow(spec)
        plt.show()

    def plot_phs_img(self,dx):
        import matplotlib.pyplot as plt

        phs_img = self.cube[:,:,dx]
        fig, ax = plt.subplots()
        im = ax.imshow(phs_img)
        plt.show()

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
    test1 = ParseBin(fnames)
    img_shape = [125,80]
    #timg = test1.image(img_shape)
    #fig, ax = plt.subplots()
    #im = ax.imshow(timg)
    #plt.show()

    phase_binsize = 2
    phs_range = [-250,-200]
    test_cube = test1.phasecube(img_shape,phs_range,phase_binsize)

    fig, ax = plt.subplots()
    phsbin_midpts = np.linspace(phs_range[0], phs_range[1], int((phs_range[1]-phs_range[0]) / phase_binsize)) + (phase_binsize) / 2
    spec = test_cube[60, 62, :]
    plt.bar(phsbin_midpts, spec, width=phase_binsize)
    plt.show()

