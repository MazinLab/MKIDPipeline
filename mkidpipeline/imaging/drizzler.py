"""
TODO
Add astroplan, drizzle, to setup.py/yml. drizzle need to be pip installed. I found that astroplan needed to be pip
installed otherwise some astropy import fails

Move plotting functionality to another module

Update photon TOA positioning after derotation functionality

Sort out centering of frame vs center of rotation (star)

Usage
-----

python drizzle.py path/to/drizzler.yml

Author: Rupert Dodkins,                                 Date: Mar 2019

"""
import matplotlib
import os
import numpy as np
import time
import multiprocessing as mp
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as ticker
from scipy.ndimage import gaussian_filter
from astropy import wcs
from astropy.coordinates import EarthLocation, Angle, SkyCoord
import astropy.units as u
from astroplan import Observer
import astropy
from astropy.io import fits
from drizzle import drizzle as stdrizzle
from mkidcore import pixelflags
from mkidpipeline.hdf.photontable import ObsFile
import mkidcore.corelog as pipelinelog
import pickle
import mkidpipeline
from mkidcore.instruments import CONEX2PIXEL
import argparse


def ditherp_2_pixel(positions):
    """ A function to convert the connex offset to pixel displacement"""
    positions = np.asarray(positions)
    pix = np.asarray(CONEX2PIXEL(positions[:, 0], positions[:, 1])) - np.array(CONEX2PIXEL(0, 0)).reshape(2, 1)
    return pix


class DitherDescription(object):
    """
    Info on the dither

    rotate determines if the effective integrations are pupil stablised or not


    [1] Smart, W. M. 1962, Spherical Astronomy, (Cambridge: Cambridge University Press), p. 55

    """

    def __init__(self, dither, ConnexOrigin2COR=None, observatory='Subaru', target='* kap And',
                 determine_min_timestep=False, suggested_time_step=1):
        self.description = dither
        self.target = target
        try:
            self.coords = dither.obs[0].lookup_coodinates(queryname=target)
        except astropy.coordinates.name_resolve.NameResolveError:
            pipelinelog.getLogger(__name__).warning('Unable to resolve coordinates for target name {target}, using (0,0).')
            self.coords = self.SkyCoord('0 deg', '0 deg')
        self.starRA, self.starDec = self.coords.ra.deg, self.coords.dec.deg
        if ConnexOrigin2COR is None:
            ConnexOrigin2COR = (0, 0)
        assert suggested_time_step < dither.inttime, 'You must have at least a time sample per dither'

        self.ConnexOrigin2COR = np.array([list(ConnexOrigin2COR)]).T

        self.dith_pix_offset = ditherp_2_pixel(dither.pos)
        self.cenRes = self.dith_pix_offset - self.ConnexOrigin2COR
        self.xCenRes, self.yCenRes = self.cenRes
        inst_info = dither.obs[0].instrument_info
        self.xpix = inst_info.beammap.ncols
        self.ypix = inst_info.beammap.nrows
        self.platescale = inst_info.platescale.to(u.deg).value  # 10 mas
        self.apo = Observer.at_site(observatory)

        if determine_min_timestep:
            min_timestep = self.calc_min_timesamp()

            # sometimes the min timestep can be ~100s of seconds. We need it to be at least shorter
            # than the dith exposure time
            self.timestep = min(suggested_time_step, min_timestep)
        else:
            self.timestep = suggested_time_step

        pipelinelog.getLogger(__name__).debug(
            "Timestep to be used {}".format(self.timestep))

    def calc_min_timesamp(self):
        # get the field rotation rate at the start of each dither
        dith_start_times = np.array([o.start for o in dither.obs])

        site = EarthLocation.of_site(self.observatory)
        altaz = self.apo.altaz(astropy.time.Time(val=dith_start_times, format='unix'), self.coords)
        earthrate = 2 * np.pi / u.sday.to(u.second)

        lat = site.geodetic.lat.rad
        az = altaz.az.radian
        alt = altaz.alt.radian

        # Smart 1962
        dith_start_rot_rates = earthrate * np.cos(lat) * np.cos(az) / np.cos(alt)

        # get the minimum required timestep. One that would produce 1 pixel displacement at the
        # center of furthest dither
        dith_dists = np.sqrt(self.xCenRes**2 + self.yCenRes**2)
        pix_disp = 1.
        dith_angle = np.arctan(pix_disp/dith_dists)
        min_timestep = min(dith_angle/abs(dith_start_rot_rates))

        pipelinelog.getLogger(__name__).debug("Minimum required time step calculated to be {}".format(min_timestep))

        return min_timestep

    # def plot(self):
    #     rotationMatrix = np.array([[np.cos(self.parallactic_angles), -np.sin(self.parallactic_angles)],
    #                                [np.sin(self.parallactic_angles), np.cos(self.parallactic_angles)]]).T
    #
    #     centroidRotated = (np.dot(rotationMatrix, np.array([self.xCenRes, self.yCenRes])).diagonal(axis1=0, axis2=2) +
    #                        [self.ConnexOrigin2COR[0], self.ConnexOrigin2COR[1]])
    #
    #     plt.plot(-self.dith_pix_offset[0], -self.dith_pix_offset[1], '-o')
    #     plt.plot(-self.ConnexOrigin2COR[0], -self.ConnexOrigin2COR[1], marker='x')
    #     plt.plot(-centroidRotated[0], -centroidRotated[1], '-o')
    #     plt.show()

def get_wcs(x, y, ditherdesc, ditherind, nxpix=146, nypix=140, platescale=0.01 / 3600., derotate=True):
    dither = ditherdesc.description
    sample_times = np.arange(dither.obs[ditherind].start,
                             dither.obs[ditherind].stop,
                             ditherdesc.timestep)

    if derotate:
        reformatted_times = astropy.time.Time(val=sample_times, format='unix')
        parallactic_angles = ditherdesc.apo.parallactic_angle(reformatted_times, ditherdesc.coords).value  # radians

    else:
        reformatted_mid_time = astropy.time.Time(val=(dither.obs[0].start + dither.obs[-1].stop)/2, format='unix')
        parallactic_angles = np.ones_like(sample_times) * ditherdesc.apo.parallactic_angle(reformatted_mid_time,
                                                                                           ditherdesc.coords).value  # radians

    pipelinelog.getLogger(__name__).debug("Parallactic angles: %s", parallactic_angles)

    obs_wcs_seq = np.empty(sample_times)
    for t, pa in enumerate(parallactic_angles):
        rotation_matrix = np.array([[np.cos(pa), -np.sin(pa)],
                                   [np.sin(pa), np.cos(pa)]])

        offset_connex_frame = ditherdesc.dith_pix_offset[ditherind]
        offset_COR_frame = offset_connex_frame - ditherdesc.ConnexOrigin2COR
        offset_derotate_frame = np.dot(rotation_matrix, offset_COR_frame)

        w = wcs.WCS(naxis=2)
        w.wcs.crpix = [nxpix / 2., nypix / 2.]
        w.wcs.ctype = ["RA-----", "DEC----"]
        w._naxis1 = nxpix
        w._naxis2 = nypix

        w.wcs.crval = offset_derotate_frame * platescale + np.array([ditherdesc.coords.ra.deg,
                                                                     ditherdesc.coords.dec.deg])
        w.wcs.cd = rotation_matrix.T * platescale
        print(w)
        obs_wcs_seq[t] = w

    return obs_wcs_seq

#
# def get_wcs2(x, y, ditherdesc, ditherind, nxpix=146, nypix=140, toa_rotation=False, randoffset=False, nPhot=1,
#             platescale=0.01 / 3600.):
#     """
#     :param timestamps:
#     :param xPhotonPixels:
#     :param yPhotonPixels:
#     :param ditherind:
#     :param ditherdesc:
#     :param toa_rotation:
#     If False each dither position is a fixed orientation. If True the HA of each photon receives an additional
#     contribution based on the TOA allowing for rotation effects during each dither integration.
#
#     Add uniform random dither to each photon, distributed over a square
#     area of the same size and orientation as the originating pixel at
#     the time of observation (assume RA and dec are defined at center of pixel).
#
#     :return:
#     """
#
#     if toa_rotation:
#         raise NotImplementedError
#         # TODO update this rot_rate to altaz model
#         earthrate = 1. / 86164.1 * 2 * np.pi  # * 500
#         obs_const = earthrate * np.cos(np.deg2rad(19.7))
#         rot_rate = obs_const * np.cos(az) / np.cos(alt)
#         photHAs = time * 1e-6 * rot_rate
#
#         hourangles = ha_ref + photHAs
#
#         parallactic_angles = np.array([[np.cos(parallactic_angles), -np.sin(parallactic_angles)],
#                                        [np.sin(parallactic_angles), np.cos(parallactic_angles)]]).T
#
#         dith_pix_offset = np.array([-ditherdesc.xCenRes[ditherind] + x - nxpix / 2,
#                               -ditherdesc.yCenRes[ditherind] + y - nypix / 2])
#
#     else:
#
#         parallactic_angle = ditherdesc.parallactic_angles[ditherind]
#
#         rotationMatrix = np.array([[np.cos(parallactic_angle), -np.sin(parallactic_angle)],
#                                    [np.sin(parallactic_angle), np.cos(parallactic_angle)]])
#
#         # put each photon from the dither into its raster location on the virtual grid
#         vgrid_photons = np.array([ditherdesc.dith_pix_offset[0][ditherind] + x - nxpix / 2,
#                                   ditherdesc.dith_pix_offset[1][ditherind] + y - nypix / 2])
#
#         # offset these to the center of rotation
#         cor_photons = vgrid_photons - ditherdesc.ConnexOrigin2COR
#
#         # rotate that virtual grid so that that those photons now occupy the part of the sky that was sampled
#         rotated_vgrid = np.dot(rotationMatrix, cor_photons)
#
#         # undo the COR offset
#         skyframe_photons = rotated_vgrid + ditherdesc.ConnexOrigin2COR
#
#     # Add the photon arcsecond offset to the centroid offset.
#     photDecDeg = ditherdesc.starDec + platescale * skyframe_photons[1]
#     photRADeg = ditherdesc.starRA + platescale * skyframe_photons[0]
#
#     if randoffset:
#         np.random.seed(42)  # so random values always same
#         xRand = np.random.rand(nPhot) * plateScale - plateScale / 2.0
#         yRand = np.random.rand(nPhot) * plateScale - plateScale / 2.0  # Not the same array!
#         ditherRAs = xRand * np.cos(hourangles) - yRand * np.sin(hourangles)
#         ditherDecs = yRand * np.cos(hourangles) + xRand * np.sin(hourangles)
#     else:
#         ditherRAs = 0
#         ditherDecs = 0
#
#     photRADeg = photRADeg + ditherRAs
#     photDecDeg = photDecDeg + ditherDecs
#
#     return photRADeg, photDecDeg


class Drizzler(object):
    def __init__(self, photonlists, metadata):
        # TODO Implement
        # Assume obsfiles either have their metadata or needed metadata is passed, e.g. WCS information, target info,
        # etc

        # TODO determine appropirate value from area coverage of dataset and oversampling, even longerterm there
        # the oversampling should be selected to optimize total phase coverage to extract the most resolution at a
        # desired minimum S/N

        # self.randoffset = False # apply random spatial offset to each photon
        self.nPixRA = None
        self.nPixDec = None
        self.square_grid = True

        self.config = None
        self.files = photonlists

        self.xpix = metadata.xpix
        self.ypix = metadata.ypix
        self.starRA = metadata.coords.ra.deg
        self.starDec = metadata.coords.dec.deg
        self.vPlateScale = metadata.platescale
        self.ConnexOrigin2COR = metadata.ConnexOrigin2COR

        raMin, raMax, decMin, decMax = [], [], [], []
        for photonlist in photonlists:
            raMin.append(min(photonlist['photRARad']))
            raMax.append(max(photonlist['photRARad']))
            decMin.append(min(photonlist['photDecRad']))
            decMax.append(max(photonlist['photDecRad']))
        raMin = min(raMin)
        raMax = max(raMax)
        decMin = min(decMin)
        decMax = max(decMax)

        # Set size of virtual grid to accommodate.
        if self.nPixRA is None:
            self.nPixRA = (2 * np.max((raMax-self.starRA, self.starRA-raMin))//self.vPlateScale).astype(int)
        if self.nPixDec is None:
            self.nPixDec = (2 * np.max((decMax-self.starDec, self.starDec-decMin))//self.vPlateScale).astype(int)

        if self.square_grid:
            nPix = max((self.nPixRA, self.nPixDec))
            self.nPixRA, self.nPixDec = nPix, nPix

        self.generate_coordinate_grid()

        self.get_header()

    def generate_coordinate_grid(self):
        """
        Establish RA and dec coordinates for pixel boundaries in the virtual pixel grid,
        given the number of pixels in each direction (self.nPixRA and self.nPixDec), the
        location of the centre of the array (self.starRA, self.starDec), and the plate scale
        (self.vPlateScale).
        """
        # Note - +1's are because these are pixel *boundaries*, not pixel centers:
        self.gridRA = self.starRA + (self.vPlateScale * (np.arange(self.nPixRA + 1) - ((self.nPixRA + 1) // 2)))
        self.gridDec = self.starDec + (self.vPlateScale * (np.arange(self.nPixDec + 1) - ((self.nPixDec + 1) // 2)))

    def get_header(self, center_on_star=True):
        # TODO implement something like this
        # w = mkidcore.buildwcs(self.nPixRA, self.nPixDec, self.vPlateScale, self.starRA, self.starDec)
        # TODO implement the PV distortion?
        # eg w.wcs.set_pv([(2, 1, 45.0)])

        self.w = wcs.WCS(naxis=2)
        self.w.wcs.crpix = np.array([self.nPixRA / 2., self.nPixDec / 2.])
        if center_on_star:
            self.w.wcs.crpix -= np.array([self.ConnexOrigin2COR[0][0], self.ConnexOrigin2COR[1][0]])
        self.w.wcs.cdelt = np.array([self.vPlateScale, self.vPlateScale])
        self.w.wcs.crval = [self.starRA, self.starDec]
        self.w.wcs.ctype = ["RA-----", "DEC----"]
        self.w._naxis1 = self.nPixRA
        self.w._naxis2 = self.nPixDec
        pipelinelog.getLogger(__name__).debug(self.w)


class SpectralDrizzler(Drizzler):
    """ Generate a spatially dithered fits dataacube from a set dithered dataset """

    def __init__(self, photonlists, metadata, pixfrac=1.):
        self.nwvlbins = 3
        self.wvlbins = np.linspace(metadata.wvlMin, metadata.wvlMax, self.nwvlbins + 1)
        super().__init__(photonlists, metadata)
        self.drizcube = [stdrizzle.Drizzle(outwcs=self.w, pixfrac=pixfrac)] * self.nwvlbins

    def run(self, save_file=None):
        for ix, file in enumerate(self.files):

            pipelinelog.getLogger(__name__).debug('Processing %s', file)
            tic = time.clock()
            insci, inwcs = self.makeCube(file)
            pipelinelog.getLogger(__name__).debug('Image load done. Time taken (s): %s', time.clock() - tic)
            for iw in range(self.nwvlbins):
                self.drizcube[iw].add_image(insci[iw], inwcs, inwht=np.int_(np.logical_not(insci[iw] == 0)))

        self.cube = [d.outsci for d in self.drizcube]

        # TODO add the wavelength WCS

    def makeCube(self, file):
        sample = np.vstack((file['wavelengths'], file['photDecRad'], file['photRARad']))
        bins = np.array([self.wvlbins, self.ypix, self.xpix])

        datacube, (wavelengths, thisGridDec, thisGridRA) = np.histogramdd(sample.T, bins)

        # w = wcs.WCS(naxis=3)
        # w.wcs.crpix = [0., 0., 0.]
        # w.wcs.cdelt = np.array([wavelengths[1] - wavelengths[0],
        #                         thisGridRA[1]  - thisGridRA[0],
        #                         thisGridDec[1] - thisGridDec[0]])
        # w.wcs.crval = [wavelengths[0], thisGridRA[0], thisGridDec[0]]
        # w.wcs.ctype = ["WAVE", "RA---", "DEC-"]
        # w._naxis1 = 0
        # w._naxis2 = 250
        # w._naxis3 = 250

        w = wcs.WCS(naxis=2)
        w.wcs.crpix = [0., 0.]
        w.wcs.cdelt = np.array([thisGridRA[1] - thisGridRA[0], thisGridDec[1] - thisGridDec[0]])
        w.wcs.crval = [thisGridRA[0], thisGridDec[0]]
        w.wcs.ctype = ["RA---AIR", "DEC--AIR"]
        w._naxis1 = len(thisGridRA) - 1
        w._naxis2 = len(thisGridDec) - 1

        return datacube, w


class TemporalDrizzler(Drizzler):
    """
    Generate a spatially dithered fits 4D hypercube from a set dithered dataset. The cube size is
    ntimebins * ndithers X nwvlbins X nPixRA X nPixDec.
    """

    def __init__(self, photonlists, metadata, pixfrac=1., nwvlbins=2, timestep=0.1,
                 wvlMin=0, wvlMax=np.inf, startt=0, intt=10):

        super().__init__(photonlists, metadata)

        self.nwvlbins = nwvlbins
        self.timestep = timestep  # seconds

        self.ndithers = len(self.files)
        self.pixfrac = pixfrac
        self.wvlbins = np.linspace(wvlMin, wvlMax, self.nwvlbins + 1)
        self.ntimebins = int(intt / self.timestep)
        self.timebins = np.linspace(startt,
                                    startt + intt,
                                    self.ntimebins + 1) * 1e6  # timestamps are in microseconds
        self.totHypCube = None
        self.totWeightCube = None

    def run(self, save_file=None):
        tic = time.clock()

        self.totHypCube = np.zeros((self.ntimebins * self.ndithers, self.nwvlbins, self.nPixDec, self.nPixRA))
        self.totWeightCube = np.zeros((self.ntimebins, self.nwvlbins, self.nPixDec, self.nPixRA))
        for ix, file in enumerate(self.files):

            pipelinelog.getLogger(__name__).debug('Processing %s', file)

            insci, inwcs = self.makeHyper(file)

            thishyper = np.zeros((self.ntimebins, self.nwvlbins, self.nPixDec, self.nPixRA), dtype=np.float32)

            for it, iw in np.ndindex(self.ntimebins, self.nwvlbins):
                drizhyper = stdrizzle.Drizzle(outwcs=self.w, pixfrac=self.pixfrac)
                drizhyper.add_image(insci[it, iw], inwcs, inwht=np.int_(np.logical_not(insci[it, iw] == 0)))
                thishyper[it, iw] = drizhyper.outsci
                self.totWeightCube[it, iw] += thishyper[it, iw] != 0

            self.totHypCube[ix * self.ntimebins: (ix + 1) * self.ntimebins] = thishyper

        pipelinelog.getLogger(__name__).debug('Image load done. Time taken (s): %s', time.clock() - tic)
        # TODO add the wavelength WCS

    def makeHyper(self, file, applyweights=False, applymask=True, maxCountsCut=10):
        if applyweights:
            weights = file['weight']
        else:
            weights = None
        sample = np.vstack((file['timestamps'], file['wavelengths'], file['photDecRad'], file['photRARad']))
        bins = np.array([self.timebins, self.wvlbins, self.ypix, self.xpix])
        hypercube, bins = np.histogramdd(sample.T, bins, weights=weights, )

        if applymask:
            usablemask = file['usablemask'].T.astype(int)
            hypercube *= usablemask

        if maxCountsCut:
            hypercube *= np.int_(hypercube < maxCountsCut)

        times, wavelengths, thisGridDec, thisGridRA = bins

        w = wcs.WCS(naxis=2)
        w.wcs.crpix = [self.xpix / 2., self.ypix / 2.]
        w.wcs.cdelt = np.array([thisGridRA[1] - thisGridRA[0], thisGridDec[1] - thisGridDec[0]])
        w.wcs.crval = [thisGridRA[self.xpix // 2], thisGridDec[self.ypix // 2]]
        w.wcs.ctype = ["RA-----", "DEC----"]
        w._naxis1 = self.xpix
        w._naxis2 = self.ypix

        return hypercube, w


class SpatialDrizzler(Drizzler):
    """ Generate a spatially dithered fits image from a set dithered dataset """

    def __init__(self, photonlists, metadata, pixfrac=1.):
        Drizzler.__init__(self, photonlists, metadata)
        self.driz = stdrizzle.Drizzle(outwcs=self.w, pixfrac=pixfrac)
        self.timestep = ditherdesc.description.timestep

    def run(self, save_file=None, applymask=False):
        for ix, file in enumerate(self.files):
            pipelinelog.getLogger(__name__).debug('Processing %s', file)

            tic = time.clock()
            # insci, inwcs = self.makeImage(file, applymask=True)

            times = np.arange(0, len(file['obs_wcs_seq']) + 1, self.timestep)
            for t, inwcs in enumerate(file['obs_wcs_seq']):
                insci = self.makeImage(file, (times[t], times[t+1]),  applymask=True)

                if applymask:
                    insci *= ~self.hot_mask
                pipelinelog.getLogger(__name__).debug('Image load done. Time taken (s): %s', time.clock() - tic)
                inwht = (insci != 0).astype(int)
                self.driz.add_image(insci, inwcs, inwht=inwht)
            if save_file:
                self.driz.write(save_file)

        # TODO introduce total_exp_time variable and complete these steps

    def makeImage(self, file, (start, stop), applyweights=True, applymask=True, maxCountsCut=10000):

        weights = file['weight'] if applyweights else None

        # TODO mixing pixels and radians per variable names
        # thisImage, thisGridDec, thisGridRA = np.histogram2d(file['photDecRad'], file['photRARad'],
        #                                                     weights=weights,
        #                                                     bins=[self.ypix, self.xpix],
        #                                                     normed=False)
        #

        exposure =
        thisImage, _, _ = np.histogram2d(file['xPhotonPixels'], file['yPhotonPixels'],
                                                            weights=weights,
                                                            bins=[self.ypix, self.xpix],
                                                            normed=False)

        if applymask:
            pipelinelog.getLogger(__name__).debug("Applying bad pixel mask")
            # usablemask = np.rot90(file['usablemask']).astype(int)
            usablemask = file['usablemask'].T.astype(int)
            # thisImage *= ~usablemask
            thisImage *= usablemask

        if maxCountsCut:
            pipelinelog.getLogger(__name__).debug("Applying max pixel count cut")
            thisImage *= thisImage < maxCountsCut

        # w = wcs.WCS(naxis=2)
        # w.wcs.crpix = [self.xpix / 2., self.ypix / 2.]
        # w.wcs.cdelt = np.array([thisGridRA[1] - thisGridRA[0], thisGridDec[1] - thisGridDec[0]])
        # w.wcs.crval = [thisGridRA[self.xpix // 2], thisGridDec[self.ypix // 2]]
        # w.wcs.ctype = ["RA-----", "DEC----"]
        # w._naxis1 = self.xpix
        # w._naxis2 = self.ypix
        return thisImage

    def get_persistant_bad(self, metadata, dithfrac=0.1, min_count=500, plot=True):
        '''
        Compare the same pixels at different dithers to determine if they are bad

        :param metadata:
        :param dithfrac:
        :param min_count:
        :param plot:
        :return:

        '''
        ndithers = len(metadata.parallactic_angles)
        hot_cube = np.zeros((ndithers, metadata.ypix, metadata.xpix))
        dith_cube = np.zeros_like(hot_cube)
        for ix, file in enumerate(self.files):
            dith_cube[ix], _ = self.makeImage(file, applymask=False)
            # plt.imshow(dith_cube[ix], origin='lower')
            # plt.show()
        # hot_cube[dith_cube > min_count] = ma.masked
        hot_cube[dith_cube > min_count] = 1
        hot_amount_map = np.sum(hot_cube, axis=0)  # hot_cube.count(axis=0)
        self.hot_mask = hot_amount_map / ndithers > dithfrac
        if plot:
            plt.imshow(self.hot_mask, origin='lower')
            plt.show(block=True)


def annotate_axis(im, ax, width, platescale, cenCoords):
    rad = platescale * width / 2.

    xticks = np.linspace(-rad, rad, 5) + cenCoords[0]
    yticks = np.linspace(-rad, rad, 5) + cenCoords[1]
    xticklabels = ["{:0.4f}".format(i) for i in xticks]
    yticklabels = ["{:0.4f}".format(i) for i in yticks]
    ax.set_xticks(np.linspace(-0.5, width - 0.5, 5))
    ax.set_yticks(np.linspace(-0.5, width - 0.5, 5))
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)
    im.axes.tick_params(color='white', direction='in', which='both', right=True, top=True, width=1,
                        length=10)
    im.axes.tick_params(which='minor', length=5, width=0.5)
    ax.xaxis.set_minor_locator(ticker.FixedLocator(np.linspace(-0.5, width - 0.5, 33)))
    ax.yaxis.set_minor_locator(ticker.FixedLocator(np.linspace(-0.5, width - 0.5, 33)))

    ax.set_xlabel('RA ($^{\circ}$)')
    ax.set_ylabel('Dec ($^{\circ}$)')


def pretty_plot(image, platescale, cenCoords, log_scale=False, vmin=None, vmax=None, show=True):
    if log_scale:
        norm = LogNorm()
    else:
        norm = None
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.imshow(image, origin='lower', vmin=vmin, vmax=vmax, norm=norm)
    annotate_axis(cax, ax, image.shape[0], platescale, cenCoords)
    cb = plt.colorbar(cax)
    cb.ax.set_title('Counts')
    if show:
        plt.show(block=True)


# def write_fits(data, hdr, filename):
#     fits.writeto(filename, image, hdr, overwrite=True)


def load_data(ditherdesc, wvlMin, wvlMax, startt, intt, tempfile='drizzler_tmp_{}.pkl',
              tempdir='', usecache=True, clearcache=False):
    ndither = len(ditherdesc.description.obs)  # len(dither.obs)

    pkl_save = os.path.join(tempdir, tempfile.format(ditherdesc.target))
    if clearcache:  # TODO the cache must be autocleared if the query parameters would alter the contents
        os.remove(pkl_save)
    try:
        if not usecache:
            raise FileNotFoundError
        with open(pkl_save, 'rb') as f:
            data = pickle.load(f)
            pipelinelog.getLogger(__name__).info('loaded', pkl_save)
    except FileNotFoundError:
        begin = time.time()
        filenames = [o.h5 for o in ditherdesc.description.obs]
        if not filenames:
            pipelinelog.getLogger(__name__).info('No obsfiles found')

        def mp_worker(file, q, startt=startt, intt=intt, startw=wvlMin, stopw=wvlMax):
            obsfile = ObsFile(file)
            usableMask = np.array(obsfile.beamFlagImage) == pixelflags.GOODPIXEL

            photons = obsfile.query(startw=startw, stopw=stopw, startt=startt, intt=intt)
            weights = photons['SpecWeight'] * photons['NoiseWeight']
            pipelinelog.getLogger(__name__).info("Fetched {} photons from {}".format(len(photons), file))

            x, y = obsfile.xy(photons)
            del obsfile

            q.put({'file': file, 'timestamps': photons["Time"], 'xPhotonPixels': x, 'yPhotonPixels': y,
                   'wavelengths': photons["Wavelength"], 'weight': weights, 'usablemask': usableMask})

        pipelinelog.getLogger(__name__).info('stacking number of dithers: %i'.format(ndither))

        jobs = []
        data_q = mp.Queue()

        if ndither > 25:
            raise RuntimeError('Needs rewrite, will use too many cores')

        for f in filenames[:ndither]:
            p = mp.Process(target=mp_worker, args=(f, data_q))
            jobs.append(p)
            p.daemon = True
            p.start()

        data = []
        for t in range(ndither):
            data.append(data_q.get())

        # Wait for all of the processes to finish fetching their data, this should hang until all the data has been
        # fetched
        for j in jobs:
            j.join()

        data.sort(key=lambda k: filenames.index(k['file']))

        pipelinelog.getLogger(__name__).debug('Time spent: %f' % (time.time() - begin))

        with open(pkl_save, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Do the dither
    for i, d in enumerate(data):
        d['obs_wcs_seq'] = get_wcs(d['xPhotonPixels'], d['yPhotonPixels'], ditherdesc, i,
                                   nxpix=ditherdesc.xpix, nypix=ditherdesc.ypix,
                                   platescale=ditherdesc.platescale)
        # d['photRARad'], d['photDecRad'] = radec

    return data


def form(dither, dim='spatial', derotate=True, ConnexOrigin2COR=None, wvlMin=850, wvlMax=1100, startt=0, intt=60, pixfrac=.5,
         nwvlbins=1, timestep=1.):
    """

    :param dim: 2->image, 3->spectral cube, 4->sequence of spectral cubes. If drizzle==False then dim is ignored
    :param derotate: 0 or 1
    :param ConnexOrigin2COR: None or array/tuple
    :param wvlMin:
    :param wvlMax:
    :param startt:
    :param intt:
    :param pixfrac:
    :param drizzle: Bool for output format

    :return:
    """

    ditherdesc = DitherDescription(dither, derotate=derotate, target=dither.name, ConnexOrigin2COR=ConnexOrigin2COR)
    data = load_data(ditherdesc, wvlMin, wvlMax, startt, intt)

    if dim not in ['spatial', 'spectral', 'temporal']:
        raise ValueError('Not calling one of the available functions')

    elif dim == 'spatial':
        driz = SpatialDrizzler(data, ditherdesc, pixfrac=pixfrac)
        # driz.get_persistant_bad(ditherdesc)
        driz.run(applymask=False)
        outsci = driz.driz.outsci
        outwcs = driz.w

    elif dim == 'spectral':
        # TODO implement, is this even necessary. On hold till interface specification and dataproduct definition
        raise NotImplementedError

    elif dim == 'temporal':
        tdriz = TemporalDrizzler(data, ditherdesc, pixfrac=pixfrac, nwvlbins=nwvlbins, timestep=timestep,
                                 wvlMin=wvlMin, wvlMax=wvlMax, startt=startt, intt=intt)
        tdriz.run()
        outsci = tdriz.totHypCube
        outwcs = tdriz.w
        # weights = tdriz.totWeightCube.sum(axis=0)[0]
        # TODO: While we can still have a reference-point WCS solution this class needs a drizzled WCS helper as the
        # WCS solution changes with time, right?

    return outsci, outwcs


def get_star_offset(dither, wvlMin, wvlMax, startt, intt):
    '''
    Get the ConnexOrigin2COR offset parameter for DitherDescription

    :param dither:
    :param wvlMin:
    :param wvlMax:
    :param startt:
    :param intt:
    :return:
    '''

    fig, ax = plt.subplots()

    image, _ = form(dither=dither, dim='spatial', ConnexOrigin2COR=(0,0), derotate=False,
                    wvlMin=wvlMin, wvlMax=wvlMax, startt=startt, intt=intt, pixfrac=1)

    print("Click on the four satellite speckles and the star")
    cax = ax.imshow(image, origin='lower', norm=None)
    cb = plt.colorbar(cax)
    cb.ax.set_title('Counts')

    xlocs, ylocs = [], []
    def onclick(event):
        xlocs.append(event.xdata)
        ylocs.append(event.ydata)
        running_mean = [np.mean(xlocs), np.mean(ylocs)]
        print('xpix=%i, ypix=%i. Running mean=(%i,%i)'
              % (event.xdata, event.ydata, running_mean[0], running_mean[1]))

    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show(block=True)

    star_pix = np.array([np.mean(xlocs), np.mean(ylocs)]).astype(int)
    ConnexOrigin2COR = star_pix - np.array(image.shape)//2

    return ConnexOrigin2COR


if __name__ == '__main__':
    matplotlib.use('QT5Agg', force=True)
    matplotlib.rcParams['backend'] = 'Qt5Agg'
    import matplotlib.pylab as plt

    parser = argparse.ArgumentParser(description='Photon Drizzling Utility')
    parser.add_argument('cfg', type=str, help='The configuration file')
    parser.add_argument('-wl', type=float, dest='wvlMin', help='', default=850)
    parser.add_argument('-wh', type=float, dest='wvlMax', help='', default=1100)
    parser.add_argument('-t0', type=int, dest='startt', help='', default=0)
    parser.add_argument('-it', type=int, dest='intt', help='', default=60)
    args = parser.parse_args()

    # set up logging
    mkidpipeline.logtoconsole()

    pipelinelog.getLogger('mkidpipeline.hdf.photontable').setLevel('INFO')

    # load as a task configuration
    cfg = mkidpipeline.config.load_task_config(args.cfg)

    wvlMin = args.wvlMin
    wvlMax = args.wvlMax
    startt = args.startt
    intt = args.intt
    pixfrac = cfg.drizzler.pixfrac
    dither = cfg.dither

    # ConnexOrigin2COR = get_star_offset(dither, wvlMin, wvlMax, startt, intt)
    ConnexOrigin2COR = (20,20)

    image, drizwcs = form(dither, 'spatial', ConnexOrigin2COR=ConnexOrigin2COR, wvlMin=wvlMin,
                          wvlMax=wvlMax, startt=startt, intt=intt, pixfrac=pixfrac)

    pretty_plot(image, drizwcs.wcs.cdelt[0], drizwcs.wcs.crval, vmin=5, vmax=600)

    fits.writeto(cfg.dither.name + '_mean.fits', image, drizwcs.to_header(), overwrite=True)

    tess, drizwcs = form(dither, 'temporal', ConnexOrigin2COR=ConnexOrigin2COR, wvlMin=wvlMin,
                         wvlMax=wvlMax, startt=startt, intt=intt, pixfrac=pixfrac, nwvlbins=1)

    #Get median collpased image
    mask_cube = np.ma.masked_where(tess[:, 0] == 0, tess[:, 0])
    medDither = np.ma.median(mask_cube, axis=0).filled(0)

    pretty_plot(medDither, drizwcs.wcs.cdelt[0], drizwcs.wcs.crval, vmin=1, vmax=10)
