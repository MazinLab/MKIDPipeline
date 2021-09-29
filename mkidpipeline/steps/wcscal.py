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
from scipy.optimize import root
from astropy.io import fits
import os


class StepConfig(mkidpipeline.config.BaseStepConfig):
    yaml_tag = u'!wcscal_cfg'
    REQUIRED_KEYS = (('plot', 'none', 'none|all|summary'),
                     ('interpolate', 'True', 'whether to inerpolate the image before PSF fitting. Recommended if an '
                                             'MKIDObservation is used or data is noisy'),
                     ('sigma_psf', 2.0, 'standard deviation of the point spread functions to fit in the image '),
                     ('frame', 'icrs', 'same as SkyCoord frame kwarg'),
                     ('param_guesses', '[1e-6, 1e-6, 50, 50, 45]', 'intitial guesses for hte wcs solution'
                                                                   ' [platescale in x, platescale in y, '
                                                                   'pixels per conex move in x, '
                                                                   'pixels per conex move in y, device angle]'))


HEADER_KEYS = tuple()


PROBLEM_FLAGS = ('pixcal.hot', 'pixcal.cold', 'pixcal.dead', 'beammap.noDacTone', 'wavecal.bad',
                 'wavecal.failed_validation', 'wavecal.failed_convergence', 'wavecal.not_monotonic',
                 'wavecal.not_enough_histogram_fits', 'wavecal.no_histograms',
                 'wavecal.not_attempted')

class ClickCoords:
    """
    Class for choosing approximate location in the image for each point source to use for the wcscal. Associates the
    (RA/DEC) coordinates given using the source_loc keyword in the data.yaml with an approximate (x, y) pixel value.
    """
    def __init__(self, image, source_locs, fig=None):
        self.coords = []
        self.image = image
        self.n_sources = len(source_locs)
        self.source_locs = source_locs
        self.counter = 0
        self.fig = plt.figure() if fig is None else fig
        self.cid = None

    def push_to_start(self):
        plt.imshow(self.image)
        plt.title('Click Anywhere to Start')
        if plt.waitforbuttonpress() is True:
            return True

    def get_coords(self):
        self.push_to_start()
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
            plt.close(self.fig)
        try:
            plt.imshow(self.image)
            plt.scatter(*zip(*self.coords), c='red', marker='o', s=10)
            plt.title(f'Select Locations of Source at {self.source_locs[self.counter]}')
            plt.draw()
        except IndexError:
            pass
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


def select_sources(image, source_locs):
    """
    runs interactive plotting function for selecting which points in an image correspond to which sources that have
    well constrained on sky coordinates
    :param image: Image in which to select points
    :param source_locs: (RA, DEC) coordinates of the sources in the image to select
    :return: list of (x, y) pixel coordinates for each source_loc
    """
    fig = plt.figure()
    cc = ClickCoords(image, source_locs=source_locs, fig=fig)
    coords = cc.get_coords()
    plt.close("all")
    return coords


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)


def generate_equations(x, *args):
    """
    generates the system of equations to solve to find the WCS solution
    :param x: variables to fit for
    :param args: known quantities
    :return: list of equations to fit
    """
    nu_x, nu_y, mu_x, mu_y, phi = x
    conex_pos, pix_coord, sky_coord, theta, s0, s1, frame = args
    equations = []
    for i, pos in enumerate(conex_pos):
        for j, pix in enumerate(pix_coord[i]):
            sky = SkyCoord(sky_coord[i][j][0], sky_coord[i][j][1], frame=frame)
            eq1 = nu_x*np.cos(theta)*(np.cos(phi)*pix[0] - np.sin(phi)*pix[1] + mu_x*pos[0]) - \
                  nu_y*np.sin(theta)*(np.sin(phi)*pix[0] + np.cos(phi)*pix[1] + mu_y*pos[1]) + s0 - \
                  sky.ra.to(u.deg).value
            eq2 = nu_x*np.sin(theta)*(np.cos(phi)*pix[0] - np.sin(phi)*pix[1] + mu_x*pos[0]) + \
                  nu_y*np.cos(theta)*(np.sin(phi)*pix[0] + np.cos(phi)*pix[1] + mu_y*pos[1]) + s1 - \
                  sky.dec.to(u.deg).value
            equations.append(eq1)
            equations.append(eq2)
    return equations


def solve_system_of_equations(coords, conex_positions, telescope_angle, ra, dec, frame='icrs',
                              guesses=np.array([1, 1, 1, 1, 1])):
    """
    Solves the system of equations to generate the wcs solution
    :param coords: dictionary where the keys are the on-sky coordinates of each object and the values are the MKID
    pixel array coordintes
    :param conex_positions: List of conex positions for each coordinate pair
    :param telescope_angle: List of telescope rotation angles
    :param ra: RA of the central object (telescope offset)
    :param dec: DEC of the central object (telescope offset)
    :param frame: see SkyCoord variable of the same name - reference coordinate system for the keys of the coords dict
    :return: platescale in x, platescale in y, x slope of the conex, y slope of the conex, device rotation angle
    """
    pix_coords=[]
    sky_coords=[]
    for i, c in enumerate(coords):
        pix_coords.append([i for i in c.values()])
        sky_coords.append([i for i in c.keys()])
    res = root(generate_equations, guesses,
               args=(conex_positions, pix_coords, sky_coords, telescope_angle, ra, dec, frame), method='lm')
    pltscl_x, pltscl_y, dp_dconx, dp_dcony, devang = res.x
    getLogger(__name__).info('\n Calculated WCS Solution: \n'
                             f'PLATESCALE: ({(pltscl_x*u.deg).to(u.mas).value:.2f}, '
                             f'{(pltscl_y*u.deg).to(u.mas).value:.2f}) mas/pix\n'
                             f'CONEX MOVE: ({dp_dconx:.2f}, {dp_dcony:.2f}) pixels/conex move\n'
                             f'TELESCOPE OFFSET: ({ra:.2f}, {dec:.2f}) (RA, DEC)\n'
                             f'DEVANG: {devang:.2f} degrees')
    return (pltscl_x*u.deg).to(u.mas), (pltscl_y*u.deg).to(u.mas), dp_dconx, dp_dcony, devang


def calculate_wcs_solution(images, source_locs=None, sigma_psf=2.0, interpolate=False, conex_positions=None,
                           telescope_angle = 0, ra=None, dec=None, guesses=None,frame='icrs'):
    """
    calculates the parameters needed to form a WCS solution
    :param images: list of the images or each different pointing, conex position, or rotation angle.
    :param source_locs: True locations (RA, DEC) of the sources in the image
    :param sigma_psf: if interpolate is True, what width Gaussian to use for the interpolation
    :param interpolate: if True will interpolate thei mages to improve the PSF fitting
    :param conex_positions: List of conex positions for each image
    :param telescope_angle: List of telescope angles for each image
    :param ra: RA of the central object (telescope offset)
    :param dec: DEC of the central object (telescope offset)
    :param frame: see SkyCoord variable of the same name - reference coordinate system for the keys of the coords dict
    :return: platescale in x, platescale in y, x slope of the conex, y slope of the conex, device rotation angle
    """
    coords = []
    for image in images:
        if interpolate:
            image[image==0] = np.nan
            im = astropy_convolve(image)
        else:
            im = image
        #returns clicked coordinates corresponding in order to the given ra/dec coords
        coord_dict = {k: None for k in source_locs}
        selected_coords = select_sources(im, source_locs=source_locs)
        sources, residuals = fit_sources(im, sigma_psf=sigma_psf)#, guesses=selected_coords)
        fit_coords = [(sources['x_fit'][i], sources['y_fit'][i]) for i in range(len(sources))]
        use_idxs = [closest_node(selected_coords[i], fit_coords) for i in range(len(selected_coords))]
        use_coord = [fit_coords[idx] for idx in use_idxs]
        for i, key in enumerate(coord_dict.keys()):
            coord_dict[key] = use_coord[i]
        coords.append(coord_dict)
    res = solve_system_of_equations(coords, conex_positions, telescope_angle, ra, dec, frame=frame, guesses=guesses)
    return res


def run_wcscal(data, source_locs, sigma_psf=None, wave_start=950*u.nm, wave_stop=1375*u.nm, interpolate=True,
               guesses=None, frame='icrs'):
    """
    main function for running the WCSCal
    :param data: MKIDDitherDescription or MKIDObservation
    :param source_locs: on-sky coordinates of objects in the image to be sued for the WCS cal. Need to be in a format
    compatible with frame
    :param sigma_psf: width of the Gaussian PSF to use for the PSF fitting
    :param wave_start: start wavelenth to use for generating the image (u.Quantity)
    :param wave_stop: stop wavelength to use for generating the image (u.Quantity)
    :param interpolate: If True will perform a gaussian interpolation of the image before doing the PSF fits
    :param frame: see astropy.SkyCoord - coordinate system of source_locs
    :param conex_ref: reference position of the conex
    :param pix_ref: reference pixel coordinate while conex is at conex_ref
    :return: platescale (in mas/pixel), rotation angle (in degrees), and x and y slopes of the conex mirror
    """
    if isinstance(data, MKIDDitherDescription):
        conex_positions = []
        images = []
        hdus = []
        getLogger(__name__).info('Using MKIDDitherDescription to find WCS Solution')
        for o in data.obs:
            hdul = Photontable(o.h5).get_fits(wave_start=wave_start.to(u.nm).value,
                                              wave_stop=wave_stop.to(u.nm).value)
            images.append(hdul[1].data)
            hdus.append(hdul[1].header)
            conex_positions.append((o.header['E_CONEXX'], o.header['E_CONEXY']))
            ra = o.metadata['RA'].values
            dec = o.metadata['DEC'].values
            if len(ra) == 0 or len(dec) == 0:
                try:
                    skycood = SkyCoord.from_name(o.header['OBJECT'])
                    ra, dec = skycood.ra.to(u.deg).value, skycood.dec.to(u.deg).value
                except AttributeError:
                    getLogger(__name__).warning('either RA and DEC or OBJECT must be specified in the metadata!')
    else:
        getLogger(__name__).info('Using MKIDObservation to find WCS Solution: Generating fits file...')
        hdul = Photontable(data.h5).get_fits(wave_start=wave_start.to(u.nm).value,
                                             wave_stop=wave_stop.to(u.nm).value)
        images = [hdul[1].data]
        conex_positions = [(data.header['E_CONEXX'], data.header['E_CONEXY'])]
        ra = data.metadata['RA'].to(u.deg).values
        dec = data.metadata['DEC'].to(u.deg).values
        if len(ra) == 0 or len(dec) == 0:
            try:
                skycood = SkyCoord.from_name(data.header['OBJECT'])
                ra, dec = skycood.ra.value, skycood.dec.value
            except AttributeError:
                getLogger(__name__).warning('either RA and DEC or OBJECT must be specified in the metadata!')
    source_locs = [(s[0], s[1]) for s in source_locs]
    pltscl_x, pltscl_y, dp_dconx, dp_dcony, devang = \
        calculate_wcs_solution(images, source_locs, sigma_psf=sigma_psf,interpolate=interpolate,
                               conex_positions=conex_positions, frame=frame, guesses=guesses, ra=ra, dec=dec)
    return pltscl_x, pltscl_y, dp_dconx, dp_dcony, devang, images, hdus


def fetch(solution_descriptors, config=None, ncpu=None):
    try:
        solution_descriptors = solution_descriptors.wcscals
    except AttributeError:
        pass

    wcscfg = mkidpipeline.config.PipelineConfigFactory(step_defaults=dict(wcscal=StepConfig()), cfg=config, ncpu=ncpu,
                                                       copy=True)
    for sd in solution_descriptors:
        if os.path.exists(sd.path[:-4] + '.fits'):
            continue
        if isinstance(sd.data, MKIDObservation) or isinstance(sd.data, MKIDDitherDescription):
            pltsclx, pltscly, dp_dconx, dp_dcony, devang, images , hdus = \
                run_wcscal(sd.data, sd.source_locs, sigma_psf=wcscfg.wcscal.sigma_psf, wave_start=950*u.nm,
                           wave_stop=1375*u.nm, interpolate=wcscfg.wcscal.interpolate, frame=wcscfg.wcscal.frame,
                           guesses=np.array(wcscfg.wcscal.param_guesses))
            if abs(pltscly-pltsclx) > 0.1*(max(pltsclx, pltscly)):
                getLogger(__name__).critical('Platescale in x and y directions differ by more than 10%! Check WCS '
                                             'dataset as it is likely some parameters are underconstrained')
                pltscl = np.mean((pltsclx.value, pltscly.value))
            else:
                pltscl = np.mean((pltsclx.value, pltscly.value))
                #TODO flesh out
            hdr = sd.data.obs[0].photontable.get_fits()[0].header
            hdul = fits.HDUList([fits.PrimaryHDU(header=hdr)])
            for i, im in enumerate(images):
                im[np.isnan(im)] = 0
                hdul.append(fits.ImageHDU(data=im, header=hdus[i], name='SCIENCE'))
                hdul[i+1].header['PLTSCL'] = (pltscl, 'platescale in mas/pixel')
                hdul[i+1].header['DPIXDCXX'] =  (dp_dconx, 'pixel move per conex move in x')
                hdul[i+1].header['DPIXDCXY'] = (dp_dcony, 'pixel move per conex move in y')
                hdul[i+1].header['DEVANG'] = (devang, 'device angle in degrees')
            hdul.writeto(sd.path[:-4] + '.fits')
        else:
            pltscl = sd.data
            devang = wcscfg.instrument.device_orientation_deg
            hdul = fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data=np.ones((140, 146)))])
            hdul[1].header['PLTSCL'] = (pltscl.value, 'platescale in mas/pixel')
            hdul[1].header['DEVANG'] = (devang.value, 'device angle in degrees')
            hdul.writeto(sd.path[:-4] + '.fits')

def apply(o):
    sol_path = o.wcscal.path[:-4] + '.fits'
    getLogger(__name__).info(f'Applying {sol_path} to {o.h5}')
    sol_hdul = fits.open(sol_path)
    pltscl = sol_hdul[1].header['PLTSCL']
    devang = sol_hdul[1].header['DEVANG']
    pt = o.photontable
    pt.enablewrite()
    pt.update_header('E_PLTSCL', pltscl)
    pt.update_header('E_DEVANG', devang)
    pt.disablewrite()
