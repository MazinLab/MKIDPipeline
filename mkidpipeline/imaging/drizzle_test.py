import numpy as np
import matplotlib.pylab as plt
import RADecImage as rdi
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.convolution import AiryDisk2DKernel
import sys, os
sys.path.append(os.environ['MEDIS_DIR'])
sys.path.append(os.path.join(os.environ['MEDIS_DIR'],'Detector'))
import Detector.pipeline as pipe
import random
from scipy import misc
from scipy.ndimage import rotate
from imaging import drizzle

# from .distribution import Distribution

class Distribution(object):
    """
    draws samples from a one dimensional probability distribution,
    by means of inversion of a discrete inverstion of a cumulative density function

    the pdf can be sorted first to prevent numerical error in the cumulative sum
    this is set as default; for big density functions with high contrast,
    it is absolutely necessary, and for small density functions,
    the overhead is minimal

    a call to this distibution object returns indices into density array
    """
    def __init__(self, pdf, sort = True, interpolation = True, transform = lambda x: x):
        self.shape          = pdf.shape
        self.pdf            = pdf.ravel()
        self.sort           = sort
        self.interpolation  = interpolation
        self.transform      = transform

        #a pdf can not be negative
        assert(np.all(pdf>=0))

        #sort the pdf by magnitude
        if self.sort:
            self.sortindex = np.argsort(self.pdf, axis=None)
            self.pdf = self.pdf[self.sortindex]
        #construct the cumulative distribution function
        self.cdf = np.cumsum(self.pdf)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def sum(self):
        """cached sum of all pdf values; the pdf need not sum to one, and is imlpicitly normalized"""
        return self.cdf[-1]

    def __call__(self, N):
        """draw """
        #pick numbers which are uniformly random over the cumulative distribution function
        # print N, self.ndim, self.sum
        choice = np.random.uniform(high = self.sum, size = N)
        #find the indices corresponding to this point on the CDF
        index = np.searchsorted(self.cdf, choice)
        #if necessary, map the indices back to their original ordering
        if self.sort:
            index = self.sortindex[index]
        #map back to multi-dimensional indexing
        index = np.unravel_index(index, self.shape)
        index = np.vstack(index)

        #is this a discrete or piecewise continuous distribution?
        if self.interpolation:
            index = np.float_(index)
            # print index[0][:50], np.random.uniform(size=index.shape[1])[:50], (index[0] + np.random.uniform(size=index.shape[1]))[:50], index.shape, type(index[0])
            index[0] += np.random.uniform(size=index.shape[1])
            # print index[0][:50]

        return self.transform(index)

def sample_cube(datacube, num_events):
    print 'creating photon data from reference cube'
    dist = Distribution(datacube, interpolation=True)

    photons = dist(num_events)
    return photons

def arange_into_cube(packets, size):
    # print 'Sorting packets into xy grid (no phase or time sorting)'
    cube = [[[] for i in range(size[0])] for j in range(size[1])]
    # dprint(np.shape(cube))
    # plt.hist(packets[:,1], bins=100)
    # plt.show()
    for ip, p in enumerate(packets):
        x = np.int_(p[1])
        y = np.int_(p[2])
        # cube[x][y].append([p[0],p[1]])
        cube[x][y].append(p[0])
        # if len(packets)>=1e7 and ip%10000==0: misc.progressBar(value = ip, endvalue=len(packets))
    # print cube[x][y]
    # cube = time_sort(cube)
    return cube

def create_bad_pix(array_size, pix_yield=0.8, plot=False):
    amount = int(array_size[0]*array_size[1]*(1.-pix_yield))

    bad_ind = random.sample(list(range(array_size[0]*array_size[1])), amount)

    # bad_y = random.sample(y, amount)
    bad_y = np.int_(np.floor(bad_ind/array_size[1]))
    bad_x = bad_ind%array_size[1]

    mask = np.ones((array_size))

    mask[bad_x, bad_y]=0
    if plot:
        plt.imshow(image)
        plt.show()

    return mask

def rot(image, xy, angle):
    im_rot = rotate(image,angle)
    org_center = (np.array(image.shape[:2][::-1])-1)/2.
    rot_center = (np.array(im_rot.shape[:2][::-1])-1)/2.
    org = xy-org_center
    a = np.deg2rad(angle)
    new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),
            -org[0]*np.sin(a) + org[1]*np.cos(a) ])
    return im_rot, new+rot_center

span=50
airydisk_2D_kernel = np.array(AiryDisk2DKernel(4, x_size=span, y_size=span))

# plt.imshow(airydisk_2D_kernel, interpolation='none', origin='lower', norm=LogNorm())
# plt.xlabel('x [pixels]')
# plt.ylabel('y [pixels]')
# plt.colorbar()
# plt.show()

sky_span =250

sky = np.zeros((sky_span,sky_span))
array_size = np.array([146,146]) # has to be numpy array # and equal for now!
# array_mask = create_bad_pix(array_size, pix_yield=0.8)

n_sources = 6
dist = 40
start = 0
for nx in range(n_sources):
    for ny in range(n_sources):
        sx1 = start + nx*dist
        sx2 = sx1+span
        sy1 = start + ny*dist
        sy2 = sy1+span

        ix1 = 0
        ix2 = ix1+span
        iy1 = 0
        iy2 = iy1+span

        # print sx1, sx2, ix1, ix2
        sky[sx1:sx2, sy1:sy2] += airydisk_2D_kernel[ix1:ix2,iy1:iy2]

# plt.imshow(sky, interpolation='none', origin='lower', norm=LogNorm())
# plt.show()

positions = 5
sky_sample = np.zeros((positions, array_size[0], array_size[1]))
for ir, r in enumerate(np.linspace(0,90,positions)):
    sky_pos, _ = rot(sky, np.array([sky_span//2,sky_span//2]), r)
    sky_pos[sky_pos<=0] = 0
    print sky_pos.shape

    # plt.imshow(sky_pos, interpolation='none', origin='lower', norm=LogNorm())
    # plt.show()

    sample_ind_x = (sky_pos.shape[0]-array_size[0])/2.
    sample_ind_y = (sky_pos.shape[1]-array_size[1])/2.

    print sample_ind_x, sky_pos.shape[0], array_size[0]
    this_sky_sample = sky_pos[np.int(np.floor(sample_ind_x)):np.int(np.floor(-sample_ind_x)),
                                    np.int(np.floor(sample_ind_y)):np.int(np.floor(-sample_ind_y))]

    # sky_sample *= array_mask

    # plt.imshow(this_sky_sample, interpolation='none', origin='lower', norm=LogNorm())
    # plt.show()

    sky_sample[ir] = this_sky_sample

photons = sample_cube(sky_sample, int(1e4))
# print photons, photons.shape
packets = np.transpose(photons)
print packets[:5]

# np.save('fakedata.npy', packets)
# packets = np.load('fakedata.npy')

cube = arange_into_cube(packets, (array_size[0], array_size[1]))
image = pipe.make_intensity_map(cube, (array_size[0], array_size[1]))

# plt.imshow(image, norm=LogNorm())
# plt.show()

dithLogFilename = 'quickRotTest2.log'

ditherDict = drizzle.loadDitherLog(dithLogFilename)
# con2pix = drizzle.getCon2Pix(files[0], files[1], ditherDict, filename = dir+'con2pix.txt')
ditherDict = drizzle.getPixOff(ditherDict,con2pix=None)
print ditherDict

# Initialise empty image centered on Crab Pulsar
virtualImage = rdi.RADecImage(nPixRA=250, nPixDec=250, vPlateScale=0.25,
                              cenRA=0., cenDec=89*np.pi/180,
                              ditherDict=ditherDict)
packets[:,0] *= 5375
# packets[:,0] += ditherDict['startTimes'][0]

print packets[:5]
[virtualImage.nDPixRow, virtualImage.nDPixCol] = array_size
virtualImage.photWeights = None
virtualImage.detExpTimes = None

for ix in range(ditherDict['nSteps']):

    effExposure = packets[np.where((packets[:,0] > ditherDict['relStartTimes'][ix])
                                & (packets[:,0] < ditherDict['relEndTimes'][ix]))]
    effExposure = effExposure.T
    # print effExposure[:,5], effExposure.shape

    thisImage, thisGridDec, thisGridRA = np.histogram2d(effExposure[1], effExposure[2], bins=146)
    # plt.imshow(thisImage)
    # plt.show()
    virtualImage.stackExposure(effExposure, ditherInd=ix, doStack=True)