import numpy as np
import matplotlib.pylab as plt
# import RADecImage as rdi
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from astropy.convolution import AiryDisk2DKernel
import sys, os
from astropy.coordinates import EarthLocation, SkyCoord
import astropy.units as u
from astropy import time
from astroplan import Observer
import random
from scipy import misc
from scipy.ndimage import rotate
from imaging import drizzler

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
    print('creating photon data from reference cube')
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

def make_intensity_map(cube, size, plot=False):
    # print 'Making a map of photon locations for all phases and times'
    int_map = np.zeros((size[1],size[0]))
    print(len(cube), len(cube[0]), int_map.shape)
    for x in range(size[1]):
        for y in range(size[0]):
            int_map[x,y]=len(cube[x][y])

    if plot:
        plt.figure()
        plt.imshow(np.log10(int_map), origin='lower', interpolation='none')
    return int_map

# def rot(image, xy, angle):
#     im_rot = rotate(image,angle)
#     org_center = (np.array(image.shape[:2][::-1])-1)/2.
#     rot_center = (np.array(im_rot.shape[:2][::-1])-1)/2.
#     org = xy-org_center
#     a = np.deg2rad(angle)
#     new = np.array([org[0]*np.cos(a) + org[1]*np.sin(a),
#             -org[0]*np.sin(a) + org[1]*np.cos(a) ])
#     return im_rot, new+rot_center

def rot(img, pivot,angle):
    padX = [img.shape[1] - pivot[0], pivot[0]]
    padY = [img.shape[0] - pivot[1], pivot[1]]
    imgP = np.pad(img, [padY, padX], 'constant')
    imgR = rotate(imgP, angle, reshape=False)
    return imgR[padY[0] : -padY[1], padX[0] : -padX[1]]

span=25
airydisk_2D_kernel = np.array(AiryDisk2DKernel(5, x_size=span, y_size=span))

# plt.imshow(airydisk_2D_kernel, interpolation='none', origin='lower', norm=LogNorm())
# plt.xlabel('x [pixels]')
# plt.ylabel('y [pixels]')
# plt.colorbar()
# plt.show()

sky_span =800

sky = np.zeros((sky_span,sky_span))
array_size = np.array([146,146]) # has to be numpy array # and equal for now!
# array_mask = create_bad_pix(array_size, pix_yield=0.8)

n_sources = 12
dist = 20
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

def make_check(length=250, spacing=5):
    # a = np.kron([[1, 0] * 20, [0, 1] * 20] * 20, np.ones((2, 2)))
    a = np.zeros((length,length))
    a[::spacing] = 1
    a[:,::spacing] = 1
    return a

def get_mona():
    import cv2
    image_file = cv2.imread("/mnt/data0/dodkins/monalisa.jpg", 0)
    image_file = image_file[:sky_span,:sky_span]
    image_file = image_file[::-1]
    # image_file = np.pad(image_file, ((0,0),(200,0)), mode = 'constant', constant_values=0)
    # print(type(image_file))
    # plt.imshow(image_file, origin='lower')
    # plt.show()
    return image_file

def make_psf_array():

    positions = 5
    sky_sample = np.zeros((positions**2, array_size[0], array_size[1]))
    # for ir, r in enumerate(np.linspace(0,90,positions)):
    #     sky_pos, _ = rot(sky, np.array([sky_span//2,sky_span//2]), r)
    #     sky_pos[sky_pos<=0] = 0
    ir = 0
    for ix in np.linspace(-30,30,positions):
        for iy in np.linspace(-30,30,positions):
            sky_pos = sky
            print(sky_pos.shape)

            plt.imshow(sky_pos, interpolation='none', origin='lower', norm=LogNorm())
            plt.show()

            sample_ind_x = (sky_pos.shape[0]-array_size[0])/2.
            sample_ind_y = (sky_pos.shape[1]-array_size[1])/2.

            print(sample_ind_x, sky_pos.shape[0], array_size[0])
            this_sky_sample = sky_pos[np.int(np.floor(sample_ind_x +ix)):np.int(np.floor(-sample_ind_x +ix)),
                                      np.int(np.floor(sample_ind_y +iy)):np.int(np.floor(-sample_ind_y +iy))]

            # sky_sample *= array_mask

            plt.imshow(this_sky_sample, interpolation='none', origin='lower', norm=LogNorm())
            plt.show()

            sky_sample[ir] = this_sky_sample
            ir += 1
    return sky_sample

sky_sample = get_mona()

def get_photonImage(sky_sample):
    '''In case you wanted to get some phootns and then covert them back into an image'''
    # plt.imshow(sky, interpolation='none', origin='lower', norm=LogNorm())
    # plt.show()

    photons = sample_cube(sky_sample, int(1e4))
    # print photons, photons.shape
    packets = np.transpose(photons)

    # np.save('fakedata.npy', packets)
    # packets = np.load('fakedata.npy')

    cube = arange_into_cube(packets, (array_size[0], array_size[1]))
    image = make_intensity_map(cube, (array_size[0], array_size[1]))
    return image

image = sky_sample
# plt.imshow(image)
# plt.show()

# # dithLogFilename = 'quickRotTest2.log'
# dithLogFilename = 'quickditherTest.log'
#
# ditherDict = drizzler.loadDitherLog(dithLogFilename)
# ditherDict = drizzler.getPixOff(ditherDict,con2pix=np.array([[1, 1], [1,1]]))
# # print ditherDict
#
# # Initialise empty image centered on Crab Pulsar
# virtualImage = rdi.RADecImage(nPixRA=250, nPixDec=250, vPlateScale=0.25,
#                               cenRA=0., cenDec=89*np.pi/180,
#                               ditherDict=ditherDict)
# # packets[:,0] *= 5375
# packets[:,0] *= ditherDict['relStartTimes'][1]
# # packets[:,0] += ditherDict['startTimes'][0]
# print(ditherDict['relStartTimes'])
# print(packets[:5])
# [virtualImage.nDPixRow, virtualImage.nDPixCol] = array_size
# virtualImage.photWeights = None
# virtualImage.detExpTimes = None
#
# for ix in range(ditherDict['nSteps']):
#
#     effExposure = packets[np.where((packets[:,0] > ditherDict['relStartTimes'][ix])
#                                 & (packets[:,0] < ditherDict['relEndTimes'][ix]))]
#     effExposure = effExposure.T
#     # print effExposure[:,5], effExposure.shape
#
#     thisImage, thisGridDec, thisGridRA = np.histogram2d(effExposure[1], effExposure[2], bins=146)
#     plt.imshow(thisImage)
#     plt.show()
#     virtualImage.stackExposure(effExposure, ditherInd=ix, doStack=True)

times = np.asarray([1545626973.913, 1545627075.141, 1545627177.153, 1545627278.285, 1545627379.466, 1545627481.493,
         1545627583.6, 1545627684.731, 1545627785.88, 1545627887.044, 1545627989.008, 1545628090.251, 1545628191.399,
         1545628292.356, 1545628393.472, 1545628495.468, 1545628596.664, 1545628697.956, 1545628799.12,
         1545628900.252, 1545629002.279, 1545629103.444, 1545629204.608, 1545629305.948, 1545629407.177])

def get_rot_rates(observatory='Subaru', multiplier=250):
    site = EarthLocation.of_site(observatory)
    unixtimes = time.Time(val=times, format='unix')
    coords = SkyCoord.from_name('* kap And')

    apo = Observer.at_site(observatory)
    altaz = apo.altaz(unixtimes, coords)

    # TODO each dither potion will have its own altaz rather than the stars. Implement that
    Earthrate = 2 * np.pi / u.sday.to(u.second)


    obs_const = Earthrate * np.cos(site.geodetic.lat.rad)
    rot_rate = obs_const * np.cos(altaz.az.radian) / np.cos(altaz.alt.radian)
    # rot_rate = [0]*len(times)#obs_const * np.cos(altaz.az.radian) / np.cos(altaz.alt.radian)

    if multiplier:
        rot_rate = rot_rate * multiplier
    return rot_rate

angles = np.cumsum(get_rot_rates())#*(times - times[0]))
angles -= angles[0]
# plt.plot(angles)
# plt.show()
# angles = get_rot_rates() #*range(len(get_rot_rates()))
print(angles)

def get_offsets(con2pix=250):
    xPos = np.array([-0.75, -0.75, -0.75, -0.75, -0.75, -0.375, -0.375, -0.375, -0.375, -0.375, 0.0, 0.0, 0.0, 0.0, 0.0, 0.375,
            0.375, 0.375, 0.375, 0.375, 0.75, 0.75, 0.75, 0.75, 0.75])
    yPos = np.array([-0.75, -0.375, 0.0, 0.375, 0.75, -0.75, -0.375, 0.0, 0.375, 0.75, -0.75, -0.375, 0.0, 0.375, 0.75, -0.75,
            -0.375, 0.0, 0.375, 0.75, -0.75, -0.375, 0.0, 0.375, 0.75])
    xPos *= con2pix
    yPos *= con2pix
    return np.column_stack((np.int_(np.round((xPos))),np.int_(np.round((yPos)))))

xys = get_offsets()

nVPix = 800
width =150
vPlateScale = 10e-3

from astropy import wcs
coords = SkyCoord.from_name('* kap And')

def get_header():
    #TODO implement something like this
    # w = mkidcore.buildwcs(self.nPixRA, self.nPixDec, self.vPlateScale, self.cenRA, self.cenDec)
    # TODO implement the PV distortion?
    # eg w.wcs.set_pv([(2, 1, 45.0)])

    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [nVPix/2., nVPix/2.]
    w.wcs.cdelt = np.array([vPlateScale, vPlateScale])
    w.wcs.crval = [coords.ra.deg, coords.dec.deg]
    w.wcs.ctype = ["RA-----", "DEC----"]
    w._naxis1 = nVPix
    w._naxis2 = nVPix

    return w

w = get_header()
print(w)
from drizzle import drizzle as stdrizzle
driz = stdrizzle.Drizzle(outwcs=w, pixfrac=1)

def makeImage(angle, xy, plot=False):
    x, y = xy
    # rot_sky, cent = rot(image, np.array([sky_span//2,sky_span//2]), np.rad2deg(angle))
    rot_sky = rot(image, np.array([700,400]), np.rad2deg(angle))
    # rot_sky = image
    if plot:
        plt.imshow(rot_sky, origin='lower')
        plt.show()
    left = rot_sky.shape[0]//2+x - width//2
    right = left+width
    bottom = rot_sky.shape[1]//2+y - width//2
    top = bottom +width
    print(rot_sky.shape,x,y, left, right, bottom, top)
    dither = rot_sky[bottom:top, left:right]
    if plot:
        plt.imshow(dither, origin='lower')
        plt.show()

    return dither

def get_dith_header(angle, xy):
    w = wcs.WCS(naxis=2)
    x, y = xy
    w.wcs.crpix = [width/2., width/2.]
    # w.wcs.cdelt = np.array([vPlateScale*(np.cos(angle) + np.sin(angle)),vPlateScale*(np.cos(angle) + np.sin(angle))])
    print(w.wcs.cdelt, x, y)
    rotmat = np.array([[np.cos(angle), np.sin(angle)],
                         [-np.sin(angle), np.cos(angle)]])
    print(rotmat)
    # print(np.array([coords.ra.deg + x * vPlateScale, coords.dec.deg + y * vPlateScale]) * rotmat)
    # w.wcs.crval = np.array([coords.ra.deg + x * vPlateScale, coords.dec.deg + y * vPlateScale])
    w.wcs.crval = np.dot(np.array([(x-300) * vPlateScale,
                                   y * vPlateScale]),
                         rotmat)    + np.array([coords.ra.deg + 300*vPlateScale,
                                                coords.dec.deg])
    w.wcs.ctype = ["RA-----", "DEC----"]
    w._naxis1 = width
    w._naxis2 = width
    # deg = np.rad2deg(angle)
    # w.wcs.cd = rotmat *vPlateScale*(np.cos(angle) + np.sin(angle))
    w.wcs.cd = rotmat.T*vPlateScale#*(np.cos(-angle) + np.sin(-angle))#np.asarray([[vPlateScale,0],[0,vPlateScale]])
    return w

for d, (angle, (x,y)) in enumerate(zip(angles[0:],xys[0:])):
    insci = makeImage(angle, (x,y))
    inwcs = get_dith_header(angle, (x,y))
    print(inwcs)
    # driz = stdrizzle.Drizzle(outwcs=w, pixfrac=1)
    driz.add_image(insci, inwcs)
    # if d % 5 == 0:
    if d == 24:
        plt.imshow(driz.outsci, origin='lower', alpha=0.9)#, vmax=1750)
        plt.imshow(sky_sample, origin='lower', alpha=0.2)
        plt.show()