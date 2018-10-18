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
import matplotlib.pyplot as plt

from time import time


# Defining the Size of an Image
# needs to be updated if changed between runs/instruments
image_shape = {'nrows':125,'ncols':80}

#####################################################################################################
#####################################################################################################
class ParseSingleBin:
    """
    Parse a Single .bin File
    Assume basePath points to data directory

    input: a single file, full path name specified

    output: a numpy object
        Attributes:
        .name = input file name
        .tstamp = list of photon arrival times (info from header+photon .5ms time)
        .phase = list of phases (not yet wavelength cal-ed)
        .baseline = list of baselines
        .roach_num = list of roach #s
        .x = list of x coordinates
        .y = list of y coordinates


    """
    def __init__(self,binf):
        from mkidpipeline.hdf.parsebin import parse

        try:
            # Calling new parse file
            pstartT = time()
            parsef = parse(binf, 10)
            pendT = time()
            print("Parsing file {} took {} seconds".format(binf, pendT - pstartT))

            self.x_location = parsef.x
            self.y_location = parsef.y
            self.tstamp = parsef.tstamp
            self.baseline = parsef.baseline
            self.phase = parsef.phase
            self.roach_num = parsef.roach
            self.fname = binf
            self.fsize = parsef.x.shape

        except (IOError, ValueError) as e:
            # import mkidcore.corelog as clog
            # clog.getLogger('BinViewer').error('Help', exc_info=True)
            print(e)

#####################################################################################################
#####################################################################################################

class SingleBinImage:
    """
    makes a single image from the data in the single bin file
    calls ParseSingleBin

    This code makes an image of the parsed file. The image is a greyscaled
    image where the pixel value is equal to the number of photon hits per
    pixel in the given .bin file.

    There is an option to plot the image as a callable method. It does not
    automatically plot itself.


    Inputs: full bin filename
            image_shape- as a dict, with format:
                image_shape = {'nrows':125,'ncols':80}


    Output:
        Attributes:
            .fname = string, full file name from input
            .nphotons = number of photons in the file (length of parsed file)
            .img_shape = as a dict, with format:
                        image_shape = {'nrows':xx,'ncols':yy}
            .img  = 2D image file
                    no phase information is saved in the image.

        Methods:
        Plotting methods are written to be called on the object after it has been created, ie:
                test_img = SingleBinImage(binf, img_shape)
                test_img.plot_img()
        .plot_img()
            Plots a 2D image of the bin file, where the value at each x,y position is the
             total number of photon counts during the timespan of the .bin file

    """
    def __init__(self,binf,image_shape):
        parsed = ParseSingleBin(binf)
        self.fname = parsed.fname
        self.nphotons = parsed.fsize
        self.img_shape = image_shape

        self.image = np.zeros((image_shape['nrows'], image_shape['ncols']), dtype=np.uint16)
        for x, y in np.array(list(zip(parsed.x_location, parsed.y_location))):
            self.image[y, x] += 1

    def plot_img(self):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        im = ax.imshow(self.image)
        plt.show()

#####################################################################################################
#####################################################################################################

class SingleBinPhaseCube:
    """
    makes a data cube of a single .bin file with phase as 3rd axis, broken up by bin range
    Calls ParseSingleBin

    We assume that the range of data is from -360 0, which seems to be the case from parsed data
    The unfortunate part about this is that for the most part we ignore other data when making
     the phase cube. So we don't keep any info from the baseline, roachnum, etc.

    There is an option to plot the image as a callable method. It does not
     automatically plot itself.

    Inputs: single bin file-full path name
            image_shape- as a dict, with format:
                image_shape = {'nrows':125,'ncols':80}
            phase_dx- size of a single phase bin for axis3 of the data cube, in units of deg

    Output: data cube with attributes:
        .fname = string, full file name from input
        .nphotons = number of photons in the file (length of parsed file)
        .img_shape =image_shape- as a dict, with format:
                    image_shape = {'nrows':125,'ncols':80}
        .dx = the phasebin size, in degrees
        .cube

        Methods
        Plotting methods are written to be called on the object after it has been created, ie:
                cube1 = SingleBinPhaseCube(binf, img_shape,dx)
                cube1.plot_sometype(inputs)
        .plot_phase_spec(x,y)
            Plots a histogram of the phase spectra at given x,y position
        .plot_img(dx)
            Plots 2D image of phase dx, where dx is a bin number, not the actual phase range

    """
    def __init__(self,binf,image_shape,phase_dx):
        parsed = ParseSingleBin(binf)
        self.fname = parsed.fname
        self.nphotons = parsed.fsize
        self.img_shape = image_shape
        self.dx = phase_dx

        phs_range = np.linspace(-360,0,int(360/phase_dx))
        phs_cube = np.zeros((image_shape['nrows'], image_shape['ncols'], phs_range.shape[0]))
        stcube = time()

        for i,value in enumerate(phs_range):
            this_phsbin = np.logical_and(parsed.phase>value,parsed.phase<value+phase_dx).nonzero()
            for x,y in np.array(list(zip(parsed.x_location[this_phsbin[0]],parsed.y_location[this_phsbin[0]]))):
                phs_cube[y,x,i] += 1
        etcube = time()
        print("Making phase cube of single bin file took {} seconds".format(etcube-stcube))
        self.cube = phs_cube

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
class ParseStack:
    """
    load & parse a stack of .bin files
    Assume basePath points to data directory

    input: the **kwargs from running the code
        basepath run date tstampStart tstampEnd


    output: a big-ass photon list
        .fname = list of input file names
        .stacked_fsize = list of file sizes (# of photons) of each .bin file
        .tstamp = list of photon arrival times (info from header+photon .5ms time)
        .phase = list of phases (not yet wavelength cal-ed)
        .baseline = list of baselines
        .roach_num = list of roach #s
        .x = list of x coordinates
        .y = list of y coordinates


    """
    def __init__(self,basepath,run,date,tstampStart,tstampEnd):
        run_path = os.path.join(basepath, str(run), str(date))
        #run_path = os.path.join(run_path, str(date))
        msec_list = np.arange(tstampStart, tstampEnd + 1) # millisecond list

        self.x_location = []
        self.y_location = []
        self.tstamp = []
        self.baseline = []
        self.phase = []
        self.roach_num =[]
        self.fname = []
        self.stacked_fsize = [] # a list of the sizes of the bin files in the stack.
                                # Useful if you wanted to break the stack into individual images again

        for nbin in enumerate(msec_list):
            print(nbin)
            this_bin = os.path.join(run_path,str(nbin[1])+'.bin')
            print(this_bin)
            new = ParseSingleBin(this_bin)

            self.x_location = np.append(self.x_location,new.x_location)
            self.y_location = np.append(self.y_location,new.y_location)
            self.tstamp = np.append(self.tstamp,new.tstamp)
            self.phase = np.append(self.phase,new.phase)
            self.baseline = np.append(self.baseline,new.baseline)
            self.roach_num = np.append(self.roach_num,new.roach_num)
            self.fname = np.append(self.fname,new.fname)
            self.stacked_fsize = np.append(self.stacked_fsize,new.fsize)

#####################################################################################################
#####################################################################################################
class BinStackPhaseCube:
    def __init__(self,image_shape,phase_dx,basepath,run,date,tstampStart,tstampEnd):
        # Parse the Stack
        parsed = ParseStack(basepath,run,date,tstampStart,tstampEnd)

        # Assigning info from ParseStack
        self.fnames = parsed.fname
        self.stack_sizes = parsed.stacked_fsize
        self.img_shape = image_shape

        # Total Number of Photons in Stack
        nphotons = 0
        for i in range(1,parsed.stacked_fsize.shape[0]):
            nphotons = nphotons+parsed.stacked_fsize[i]
        self.nphotons = nphotons

        # Creating Data Cube
        phs_range = np.linspace(-360,0,int(360/phase_dx))
        phs_cube = np.zeros((image_shape['nrows'], image_shape['ncols'], phs_range.shape[0]))
        stcube = time()

        for i,value in enumerate(phs_range):
            this_phsbin = np.logical_and(parsed.phase>value,parsed.phase<value+phase_dx).nonzero()
            for x,y in np.array(list(zip(parsed.x_location[this_phsbin[0]],parsed.y_location[this_phsbin[0]]))):
                phs_cube[y,x,i] += 1
        etcube = time()
        print("Making phase cube of {} stacked files took {} seconds".format((tstampEnd-tstampStart),(etcube-stcube)))
        self.cube = phs_cube


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
    if len(sys.argv) != 6:
        print('Usage: {} basepath run date tstampStart tstampEnd'.format(sys.argv[0]))
        exit(0)
    else:
        kwargs['basepath'] = str(sys.argv[1])
        kwargs['run'] = str(sys.argv[2])
        kwargs['date'] = int(sys.argv[3])
        kwargs['tstampStart'] = int(sys.argv[4])
        kwargs['tstampEnd'] = int(sys.argv[5])


    data_path = kwargs['basepath'] + '/' + kwargs['run'] + '/' + str(kwargs['date']) + '/'
    first_file = data_path + str(kwargs['tstampStart']) + '.bin'
    test_single_parse = ParseSingleBin(first_file)
    #test_parse = ParseStack(**kwargs)

    test_img = SingleBinImage(first_file,image_shape)
    test_img.plot_img()
    #SingleBinImage.plot_img(test_img)

    test_cube = SingleBinPhaseCube(first_file,image_shape,5)
    test_cube.plot_phs_img(70)
    test_cube.plot_phs_spec(50,20)

    #testParse = ParseSingleBin(firstFile)
    #test_cube_stack = BinStackPhaseCube(image_shape,5,**kwargs)