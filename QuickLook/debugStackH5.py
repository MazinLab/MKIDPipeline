import sys, os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import tables

def makeDebugPlots(h5Path):

    if (not os.path.exists(h5Path)):
        msg='file does not exist: %s'%h5Path
        if verbose:
            print msg
        raise Exception(msg)

    h5file = tables.openFile(h5Path, mode='r')
    
    paramDict = h5file.getNode('/imageStack/params').read()
    print paramDict['startTimes'][0]

    '''
    imageStack = np.array(h5file.getNode('/imageStack/rawImgs').read())
    timeStamps = np.array(h5file.getNode('/imageStack/timestamps').read())
    ditherArray = np.array(h5file.getNode('/imageStack/dithers').read())
    hotArray = np.array(h5file.getNode('/imageStack/hpms').read())
    coldArray = np.array(h5file.getNode('/imageStack/cpms').read())
    deadArray = np.array(h5file.getNode('/imageStack/dpms').read())
    aperArray = np.array(h5file.getNode('/imageStack/ams').read())
    roughXArray = np.array(h5file.getNode('/imageStack/roughX').read())
    roughYArray = np.array(h5file.getNode('/imageStack/roughY').read())
    fineXArray = np.array(h5file.getNode('/imageStack/fineX').read())
    fineYArray = np.array(h5file.getNode('/imageStack/fineY').read())
    centXArray = np.array(h5file.getNode('/imageStack/centX').read())
    centYArray = np.array(h5file.getNode('/imageStack/centY').read())
    dark = np.array(h5file.getNode('/imageStack/dark').read())
    flat = np.array(h5file.getNode('/imageStack/flat').read())
    final = np.array(h5file.getNode('/imageStack/finalImg').read())
    
    h5file.close()

    print "Loaded H5 file..."

    plt.figure()
    plt.plot(timeStamps,'o')
    plt.title('Timestamps')
    plt.show()

    plt.figure()
    plt.plot(timeStamps, centXArray,'bo')
    plt.title('Centroid X,Y')

    plt.plot(timeStamps, centYArray,'ro')
    plt.show()
    '''

if __name__ == "__main__":
    kwargs = {}
    if len(sys.argv) != 2:
        print 'Usage: {} full/path/to/stack.h5'.format(sys.argv[0])
        exit(0)
    else:
        kwargs['h5Path'] = str(sys.argv[1])

    makeDebugPlots(**kwargs)
