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

class ParseSingleBin:
    """
    Parse a Single .bin File
    Assume basePath points to data directory

    input: a single file, full path name specified

    output: a numpy object with attributes:
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


class SingleBinImage:
    """
    makes a single image from the data in the single bin file
    calls ParseSingleBin


    Inputs: full bin filename
            image_shape

    Output: 2D image file
        no phase information is saved in the image.

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


class SingleBinPhaseCube:
    def __init__(self,binf,image_shape,phase_dx):
        parsed = ParseSingleBin(binf)
        self.fname = parsed.fname
        self.nphotons = parsed.fsize
        self.img_shape = image_shape

        phs_range = np.arange(-np.pi,np.pi,phase_dx)
        self.phs_cube = np.zeros((image_shape['nrows'], image_shape['ncols'], phs_range.shape))
        stcube = time()
        '''
        for x,y in zip(parsed.x_location,parsed.y_location,parsed.phase):
            for z in enumerate(phs_range):
                #if z[1]
                self.phs_cube[x,y,z[0]] += 1
        etcube = time()
        print("Making phase cube of single bin file took {} seconds".format(etcube-stcube))
        '''

class ParseStack:
    """
    load & parse a stack of .bin files
    Assume basePath points to data directory

    input: a start time and stop time for a specified path


    output: ?

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
        self.fsize = []

        for nbin in enumerate(msec_list):
            print(nbin)
            this_bin = os.path.join(run_path,str(nbin[1])+'.bin')
            print(this_bin)
            new = ParseSingleBin(this_bin)

            #self.x_location = np.append(self.x_location,new.x_location)
            self.y_location = np.append(self.y_location,new.y_location)
            self.tstamp = np.append(self.tstamp,new.tstamp)
            self.phase = np.append(self.phase,new.phase)
            self.baseline = np.append(self.baseline,new.baseline)
            self.roach_num = np.append(self.roach_num,new.roach_num)
            self.fname = np.append(self.fname,new.fname)
            self.fsize = np.append(self.fsize,new.fsize)




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
    SingleBinImage.plot_img(test_img)

    #test_cube = SingleBinPhaseCube(first_file,image_shape,5)

    #testParse = ParseSingleBin(firstFile)
