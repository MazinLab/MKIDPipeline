import mkidpipeline.config
import numpy as np
import matplotlib.pyplot as plt
from mkidpipeline.utils.smoothing import astropy_convolve
from mkidpipeline.utils.photometry import fit_sources
from astropy.coordinates import SkyCoord
import astropy.units as u
from mkidpipeline.photontable import Photontable
from mkidcore.corelog import getLogger
from mkidpipeline.config import MKIDDitherDescription, MKIDObservation
from mkidcore.instruments import CONEX2PIXEL
from astropy import wcs
from drizzle import drizzle as stdrizzle


class StepConfig(mkidpipeline.config.BaseStepConfig):
    yaml_tag = u'!wcscal_cfg'
    REQUIRED_KEYS = (('plot', 'none', 'none|all|summary'),
                     ('interpolate', 'True', 'whether to inerpolate the image before PSF fitting. Recommended if an '
                                             'MKIDObservation is used or data is noisy'),
                     ('sigma_psf', 2.0, 'standard deviation of the point spread functions to fit in the image '),
                     ('frame', 'icrs', 'same as SkyCoord frame kwarg'))


HEADER_KEYS = tuple()


PROBLEM_FLAGS = ('pixcal.hot', 'pixcal.cold', 'pixcal.dead', 'beammap.noDacTone', 'wavecal.bad',
                 'wavecal.failed_validation', 'wavecal.failed_convergence', 'wavecal.not_monotonic',
                 'wavecal.not_enough_histogram_fits', 'wavecal.no_histograms',
                 'wavecal.not_attempted')

class ClickCoords:
    def __init__(self, image, source_locs):
        self.coords = []
        self.image = image
        self.n_sources = len(source_locs)
        self.source_locs = source_locs
        self.counter = 0
        self.fig = plt.figure()
        self.cid = None

    def push_to_start(self):
        plt.imshow(self.image)
        plt.title('Click Anywhere to Start')
        if plt.waitforbuttonpress() is True:
            return True

    def get_coords(self):
        self.push_to_start()
        plt.clf()
        plt.imshow(self.image)
        plt.title(f'Select Location of Source at {self.source_locs[self.counter]}')
        plt.draw()
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.__onclick__)
        plt.show()
        return self.coords

    def __onclick__(self, click):
        point = (click.xdata, click.ydata)
        self.coords.append(point)
        self.counter += 1
        if self.counter == self.n_sources:
            self.fig.canvas.mpl_disconnect(self.cid)
            plt.close()
        plt.clf()
        plt.imshow(self.image)
        plt.scatter(*zip(*self.coords), c='red', marker='o', s=10)
        plt.title(f'Select Locations of Source at {self.source_locs[self.counter]}')
        plt.draw()
        return self.coords


def crop_image(image):
    """
    crops out buffer of pixels with 0 counts from an image
    :param image: 2D image array to be cropped
    :return: cropped 2D image array.
    """
    if np.all(image == 0):
        getLogger(__name__).warning('Entire image is empty. Check WCS data input.')
        exit()
    live_idxs = np.argwhere(image != 0)
    center_x, center_y = int(image.shape[0]/2), int(image.shape[1]/2)
    min_x, min_y = min(live_idxs[:,0]), min(live_idxs[0,:])
    max_x, max_y = max(live_idxs[:,0]), max(live_idxs[0,:])
    use_x = int(max(center_x - min_x, max_x - center_x))
    use_y = int(max(center_y - min_y, max_y - center_y))
    return image[center_x - use_x:center_x + use_x, center_y - use_y:center_y + use_y]


def get_pixel_space_wcs(conex_pos=None, pix_ref=None, conex_ref=None, shape=None):
    delta_y, delta_x = CONEX2PIXEL(conex_pos[0], conex_pos[1]) - CONEX2PIXEL(conex_ref[0], conex_ref[1])
    new_x = pix_ref[0] + delta_x
    new_y = pix_ref[1] + delta_y
    w = wcs.WCS(naxis=2)
    w.wcs.crpix = [new_x, new_y]
    w.wcs.crval = [int(shape[1]/2), int(shape[0]/2)]
    w.wcs.ctype = ['LINEAR', 'LINEAR']
    w.pixel_shape = shape
    w.wcs.pc = np.eye(2)
    w.wcs.cdelt = [1, 1]
    w.wcs.cunit = ['pixel', 'pixel']
    return w

def pixspace_drizzle(data, pixfrac=0.5, conex_ref=None, pix_ref=None):
    ref_wcs = get_pixel_space_wcs(conex_pos=conex_ref, pix_ref=pix_ref, conex_ref=conex_ref, shape=(140, 146))
    driz = stdrizzle.Drizzle(outwcs=ref_wcs, pixfrac=pixfrac, wt_scl='')
    for o in data.obs:
        hdul = o.photontable.get_fits()
        image = hdul[1].data
        wcs = get_pixel_space_wcs(conex_pos=(o.header['E_CONEXX'], o.header['E_CONEXY']), pix_ref=pix_ref,
                                  conex_ref=conex_ref, shape=image.shape)
        driz.add_image(image.T, wcs)
    return driz.outsci


def select_sources(image, source_locs):
    cc = ClickCoords(image, source_locs=source_locs)
    coords = cc.get_coords()
    return coords


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)


def difference(a, b):
    """
    finds the distance between two points a and b
    :param a: coordinate 1
    :param b: coordinate 2
    :return:
    """
    diff = np.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)
    return diff


def angle(a, b):
    """
    finds the angle between two coordinates a and b
    :param a: coordinate 1
    :param b: coordinate 2
    :return:
    """
    theta = np.arctan((a[0] - b[0])/(a[1] - b[1]))
    return theta


def dist_compare(coord_dict, frame='icrs'):
    """
    find the separations between an arbitrary number of objects in an image
    :param coord_dict: dictionary of coordinates with keys of (RA, DEC) (in units specified by 'frame') and values (x, y)
    :param frame: see astropy SkyCoord
    :return:
    """
    result = []
    for key1, val1 in coord_dict.items():
        for key2, val2 in coord_dict.items():
            if key1==key2:
                pass
            else:
                coord1 = SkyCoord(key1[0], key1[1], frame=frame)
                coord2 = SkyCoord(key2[0], key2[1], frame=frame)
                result.append(coord1.separation(coord2).to(u.mas).value/difference(val1, val2))
        return result


def theta_compare(coord_dict, frame='icrs'):
    """
    find the angle offset between an arbitrary number of objects in an image and their true position angle
    :param coord_dict: dictionary of coordinates with keys of (RA, DEC) (in units specified by 'frame') and values (x, y)
    :param frame: see astropy SkyCoord
    :return: list of rotation angle offsets
    """
    result = []
    for key1, val1 in coord_dict.items():
        for key2, val2 in coord_dict.items():
            if key1==key2:
                pass
            else:
                coord1 = SkyCoord(key1[0], key1[1], frame=frame)
                coord2 = SkyCoord(key2[0], key2[1], frame=frame)
                true_theta = coord1.position_angle(coord2).to(u.deg).value
                measured_theta = np.rad2deg(angle(val1, val2))
                result.append(true_theta - measured_theta)
        return result


def get_platescale(coord_dict, frame='icrs'):
    """
    calculates the platescale given an arbitrary number of points in an image
    :param coord_dict: dictionary of coordinates with keys of (RA, DEC) (in units specified by 'frame') and values (x, y)
    :param frame: see astropy SkyCoord
    :return: mean platescale value
    """
    pltscl = []
    dict_copy = coord_dict.copy()
    for key, value in coord_dict.items():
        pltscl += dist_compare(dict_copy, frame=frame)
        dict_copy.pop(key)
    return np.mean(pltscl) # TODO errors


def get_rotation_angle(coord_dict, frame='icrs'):
    """
    calculates the rotation angle given an arbitrary amount of points in an image
    :param coord_dict: dictionary of coordinates with keys of (RA, DEC) (in units specified by 'frame') and values (x, y)
    :param frame: see astropy SkyCoord
    :return: mean rotation angle value
    """
    angles = []
    dict_copy = coord_dict.copy()
    for key, value in coord_dict.items():
        angles += theta_compare(dict_copy, frame=frame)
        dict_copy.pop(key)
    return np.mean(angles) # TODO errors


def calculate_wcs_solution(image, source_locs=None, sigma_psf=2.0, interpolate=False, frame='icrs'):
    if interpolate:
        im = astropy_convolve(image)
    else:
        im = image
    cropped = crop_image(im)
    #returns clicked coordinates corresponding in order to the given ra/dec coords
    coord_dict = {k: None for k in source_locs}
    selected_coords = select_sources(cropped, source_locs=source_locs)
    sources, residuals = fit_sources(cropped, sigma_psf=sigma_psf)#, guesses=selected_coords)
    fit_coords = [(sources['x_0'][i], sources['y_0'][i]) for i in range(len(sources))]
    use_idxs = [closest_node(selected_coords[i], fit_coords) for i in range(len(selected_coords))]
    use_coord = [fit_coords[idx] for idx in use_idxs]
    for i, key in enumerate(coord_dict.keys()):
        coord_dict[key] = use_coord[i]
    pltscl = get_platescale(coord_dict, frame=frame)
    rot_angle = get_rotation_angle(coord_dict, frame=frame)
    return pltscl, rot_angle


def run_wcscal(data, source_locs, sigma_psf=None, wave_start=950*u.nm, wave_stop=1375*u.nm, ncpu=None, interpolate=True,
               frame='icrs', conex_ref=None, pix_ref=None):
    if isinstance(data, MKIDDitherDescription):
        getLogger(__name__).info('Using MKIDDitherDescription to find WCS Solution: Drizzling...')
        drizzled = pixspace_drizzle(data, conex_ref=conex_ref, pix_ref=pix_ref)
        image = drizzled
    else:
        getLogger(__name__).info('Using MKIDObservation to find WCS Solution: Generating fits file...')
        hdul = Photontable(data.h5).get_fits(wave_start=wave_start.to(u.nm).value,
                                             wave_stop=wave_stop.to(u.nm).value)
        image = hdul[1].data

    source_locs = [(s[0], s[1]) for s in source_locs]
    platescale, rotation_angle = calculate_wcs_solution(image, source_locs, sigma_psf=sigma_psf, interpolate=interpolate,
                                                        frame=frame)
    return platescale, rotation_angle


def apply(outputs, config=None, ncpu=None):
    solution_descriptors = outputs.wcscals
    try:
        solution_descriptors = solution_descriptors.wcscals
    except AttributeError:
        pass

    wcscfg = mkidpipeline.config.PipelineConfigFactory(step_defaults=dict(wcscal=StepConfig()), cfg=config, ncpu=ncpu,
                                                       copy=True)
    for sd in solution_descriptors:
        if isinstance(sd.data, MKIDObservation) or isinstance(sd.data, MKIDDitherDescription):
            pltscl, rot_ang = run_wcscal(sd.data, sd.source_locs, sigma_psf=wcscfg.wcscal.sigma_psf, wave_start=950*u.nm,
                                         wave_stop=1375*u.nm, ncpu=ncpu if ncpu is not None else 1,
                                         interpolate=wcscfg.wcscal.interpolate, frame=wcscfg.wcscal.frame,
                                         conex_ref=sd.conex_ref, pix_ref=sd.pixel_ref)
        else:
            pltscl = sd.data
            rot_ang = wcscfg.instrument.device_orientation_deg
        for o in set(outputs.to_wcscal):
            getLogger(__name__).info(f'Updating header for {o.name} with platescale {pltscl:.2f} mas/pixel and rotation '
                                     f'angle {rot_ang:.2f} degrees')
            pt = o.photontable
            pt.enablewrite()
            pt.update_header('E_PLTSCL', pltscl)
            pt.update_header('E_DEVANG', rot_ang)
            pt.disablewrite()


