import mkidpipeline.config
import numpy as np
import matplotlib.pyplot as plt
from mkidpipeline.utils.smoothing import astropy_convolve
from mkidpipeline.utils.photometry import fit_sources
from astropy.coordinates import SkyCoord
import astropy.units as u
from mkidpipeline.photontable import Photontable
from mkidcore.corelog import getLogger
from mkidpipeline.definitions import MKIDObservation, MKIDDither
from scipy.optimize import root
from astropy.io import fits
import os
from mkidpipeline.utils.smoothing import replace_nan


class StepConfig(mkidpipeline.config.BaseStepConfig):
    yaml_tag = u'!wcscal_cfg'
    REQUIRED_KEYS = (('plot', 'none', 'none|all|summary'),
                     ('interpolate', True, 'whether to inerpolate the image before PSF fitting. Recommended if an '
                                             'MKIDObservation is used or data is noisy'),
                     ('sigma_psf', 2.0, 'standard deviation of the point spread functions to fit in the image '),
                     ('param_guesses', [1e-6, 1e-6, 50, 50, 45], '(optional) intitial guesses for hte wcs solution'
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
        self.plot_coords = []
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
        # set origin to bottom left of array
        point = (click.xdata, 146 - click.ydata)
        plot_point = (click.xdata, click.ydata)
        self.coords.append(point)
        self.plot_coords.append(plot_point)
        self.counter += 1
        if self.counter == self.n_sources:
            self.fig.canvas.mpl_disconnect(self.cid)
            plt.close(self.fig)
        try:
            plt.imshow(self.image)
            plt.scatter(*zip(*self.plot_coords), c='red', marker='o', s=10)
            plt.title(f'Select Locations of Source at {self.source_locs[self.counter]}')
            plt.draw()
        except IndexError:
            pass
        return self.coords


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
    conex_pos, pix_coord, sky_coord, theta, t0, t1, frame, same_platescale, ref_pix, conex_ref = args
    equations = []
    if same_platescale:
        eta, mu_x, mu_y, phi = x
    else:
        eta_x, eta_y, mu_x, mu_y, phi = x
    for i, pos in enumerate(conex_pos):
        for j, pix in enumerate(pix_coord[i]):
            if same_platescale:
                sky = sky_coord[i][j]
                sx, sy = np.around(sky.ra.value, decimals=7), np.around(sky.dec.value, decimals=7)
                t0, t1 = np.around(t0, decimals=7), np.around(t1, decimals=7)
                p_refx, p_refy = ref_pix[0], ref_pix[1]
                c_refx, c_refy = conex_ref[0], conex_ref[1]
                x_cen, y_cen = 140 / 2, 146 / 2
                cx, cy = pos[0], pos[1]
                px, py = pix[0], pix[1]
                eq1 = np.cos(theta[i]) * (eta * (
                            px * np.cos(phi) - py * np.sin(phi) - x_cen * np.cos(phi) + y_cen * np.sin(
                        phi) + x_cen + mu_x * (cx - c_refx) - p_refx)) - \
                      np.sin(theta[i]) * (eta * (
                            px * np.sin(phi) + py * np.cos(phi) - x_cen * np.sin(phi) - y_cen * np.cos(
                        phi) + y_cen + mu_y * (cy - c_refy) - p_refy)) + \
                      t0 - sx
                eq2 = np.sin(theta[i]) * (eta * (
                            px * np.cos(phi) - py * np.sin(phi) - x_cen * np.cos(phi) + y_cen * np.sin(
                        phi) + x_cen + mu_x * (cx - c_refx) - p_refx)) + \
                      np.cos(theta[i]) * (eta * (
                            px * np.sin(phi) + py * np.cos(phi) - x_cen * np.sin(phi) - y_cen * np.cos(
                        phi) + y_cen + mu_y * (cy - c_refy) - p_refy)) + \
                      t1 - sy

                equations.append(eq1)
                equations.append(eq2)
            else:
                raise NotImplementedError
    return equations


def solve_system_of_equations(coords, conex_positions, telescope_angles, ra, dec, ref_pix, conex_ref, frame='icrs',
                              guesses=np.array([1, 1, 1, 1]), same_platescale=True):
    """
    Solves the system of equations to generate the wcs solution
    :param coords: dictionary where the keys are the on-sky coordinates of each object and the values are the MKID
    pixel array coordinates
    :param conex_positions: List of conex positions for each coordinate pair
    :param telescope_angles: List of telescope rotation angles
    :param ra: RA of the central object (telescope offset)
    :param dec: DEC of the central object (telescope offset)
    :param frame: see SkyCoord variable of the same name - reference coordinate system for the keys of the coords dict
    :return: platescale in x, platescale in y, x slope of the conex, y slope of the conex, device rotation angle
    """
    pix_coords=[]
    sky_coords=[]
    for i, c in enumerate(coords):
        pix_coords.append([i for i in c.values()])
        sky_coords.append([SkyCoord(i[0], i[1], unit='deg') for i in c.keys()])
    if same_platescale:
        # ra, dec, sky coords in degrees
        # args conex_pos, pix_coord, sky_coord, theta, t0, t1, frame, same_platescale, ref_pix, conex_ref
        res = root(generate_equations, guesses,
                   args=(conex_positions, pix_coords, sky_coords, telescope_angles, ra, dec, frame, same_platescale,
                         ref_pix, conex_ref),
                   method='lm',
                   options={'maxiter': 5000, 'xtol': 1e-10, 'ftol': 1e-10})
        pltscl, dp_dconx, dp_dcony, devang = res.x
        getLogger(__name__).info('\n Calculated WCS Solution: \n'
                                 f'PLATESCALE: {(pltscl * u.deg).to(u.arcsec).value} arcsec/pix\n'
                                 f'CONEX MOVE: ({dp_dconx:.2f}, {dp_dcony:.2f}) pixels/conex move\n'
                                 f'TELESCOPE OFFSET: ({ra:.5f}, {dec:.5f}) (RA, DEC)\n'
                                 f'DEVANG: {(devang * u.rad).to(u.deg).value:.2f} degrees')
        return (pltscl * u.deg).to(u.arcsec).value, dp_dconx, dp_dcony, (devang * u.rad).to(u.deg).value
    else:
        raise NotImplementedError
        # res = root(generate_equations, guesses,
        #            args=(conex_positions, pix_coords, sky_coords, telescope_angles.to(u.rad), ra, dec, frame,
        #                  same_platescale), method='lm')
        # pltscl_x, pltscl_y, dp_dconx, dp_dcony, devang = res.x
        # getLogger(__name__).info('\n Calculated WCS Solution: \n'
        #                          f'PLATESCALE: ({(pltscl_x*u.deg).to(u.arcsec).value:.2f}, '
        #                          f'{(pltscl_y*u.deg).to(u.arcsec).value:.2f}) mas/pix\n'
        #                          f'CONEX MOVE: ({dp_dconx:.2f}, {dp_dcony:.2f}) pixels/conex move\n'
        #                          f'TELESCOPE OFFSET: ({ra:.2f}, {dec:.2f}) (RA, DEC)\n'
        #                          f'DEVANG: {devang:.2f} degrees')
        # return (pltscl_x*u.deg).to(u.arcsec).value, (pltscl_y*u.deg).to(u.arcsec).value, dp_dconx, dp_dcony, devang


def calculate_wcs_solution(images, source_locs=None, sigma_psf=2.0, interpolate=False, conex_positions=None,
                           telescope_angles=None, ra=None, dec=None, ref_pix=None, conex_ref=None, guesses=None):
    """
    calculates the parameters needed to form a WCS solution
    :param images: list of the images or each different pointing, conex position, or rotation angle.
    :param source_locs: Array of SkyCoord objects - true locations of the sources in the image
    :param sigma_psf: if interpolate is True, what width Gaussian to use for the interpolation
    :param interpolate: if True will interpolate thei mages to improve the PSF fitting
    :param conex_positions: List of conex positions for each image
    :param telescope_angles: List of telescope angles for each image
    :param ra: RA of the central object (telescope offset)
    :param dec: DEC of the central object (telescope offset)
    :return: platescale in x, platescale in y, x slope of the conex, y slope of the conex, device rotation angle
    """
    coords = []
    for image in images:
        if interpolate:
            image[image==0] = np.nan
            im = replace_nan(image)
            im = astropy_convolve(im)
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
    res = solve_system_of_equations(coords, conex_positions, telescope_angles, ra, dec, ref_pix, conex_ref,
                                    guesses=guesses)
    return res


def run_wcscal(data, source_locs, sigma_psf=None, wave_start=950*u.nm, wave_stop=1375*u.nm, interpolate=True,
               ref_pix=None, conex_ref=None, guesses=None):
    """
    main function for running the WCSCal
    :param data: MKIDDither or MKIDObservation
    :param source_locs: on-sky coordinates of objects in the image to be sued for the WCS cal. Needs to be in icrs
    currently
    :param sigma_psf: width of the Gaussian PSF to use for the PSF fitting
    :param wave_start: start wavelenth to use for generating the image (u.Quantity)
    :param wave_stop: stop wavelength to use for generating the image (u.Quantity)
    :param interpolate: If True will perform a gaussian interpolation of the image before doing the PSF fits
    :param conex_ref: reference position of the conex
    :param pix_ref: reference pixel coordinate while conex is at conex_ref
    :return: platescale (in mas/pixel), rotation angle (in degrees), and x and y slopes of the conex mirror
    """
    sources = []
    for source in source_locs:
        loc = SkyCoord(source[0], source[1], unit=(u.hourangle, u.deg), frame='icrs')
        sources.append((loc.ra.value, loc.dec.value))

    if isinstance(data, MKIDDither):
        conex_positions = []
        images = []
        hdus = []
        telescope_ang = []
        getLogger(__name__).info('Using MKIDDither to find WCS Solution')
        for o in data.obs:
            hdul = Photontable(o.h5).get_fits(wave_start=wave_start.to(u.nm).value,
                                              wave_stop=wave_stop.to(u.nm).value,
                                              exclude_flags=('beammap.NoDacTone'))
            images.append(hdul[1].data)
            hdus.append(hdul[1].header)
            conex_positions.append((o.header['E_CONEXX'], o.header['E_CONEXY']))
            sky = SkyCoord(o.metadata['D_IMRRA'].values[0], o.metadata['D_IMRDEC'].values[0], unit=(u.hourangle, u.deg),
                           frame='icrs')
            ra = sky.ra.value
            dec = sky.dec.value
            telescope_ang.append((o.metadata['D_IMRPAD'].values[0] * u.deg).to(u.rad).value)
            if ra == 999 or dec == 999:
                try:
                    skycoord = SkyCoord.from_name(o.header['OBJECT'])
                    ra, dec = skycoord.ra.to(u.rad).value, skycoord.dec.to(u.rad).value
                except AttributeError:
                    getLogger(__name__).warning('either RA and DEC or OBJECT must be specified in the metadata!')
    else:
        getLogger(__name__).info('Using MKIDObservation to find WCS Solution: Generating fits file...')
        hdul = Photontable(data.h5).get_fits(wave_start=wave_start.to(u.nm).value,
                                             wave_stop=wave_stop.to(u.nm).value,
                                             exclude_flags={'beammap.NoDacTone'})
        images = [hdul[1].data]
        conex_positions = [(data.header['E_CONEXX'], data.header['E_CONEXY'])]
        ra = data.metadata['D_IMRRA'].values
        dec = data.metadata['D_IMRDEC'].values
        telescope_ang = [(data.metadata['D_IMRPAD'].values * u.deg).to(u.rad)]
        if ra == 999 or dec == 999:
            try:
                skycood = SkyCoord.from_name(data.header['OBJECT'])
                ra, dec = skycood.ra.value, skycood.dec.value
            except AttributeError:
                getLogger(__name__).warning('either RA and DEC or OBJECT must be specified in the metadata!')

    res = calculate_wcs_solution(images, sources, sigma_psf=sigma_psf, interpolate=interpolate,
                                 conex_positions=conex_positions, guesses=guesses, telescope_angles=telescope_ang,
                                 ra=ra, dec=dec, ref_pix=ref_pix, conex_ref=conex_ref)
    try:
        pltscl_x, pltscl_y, dp_dconx, dp_dcony, devang = res
    except ValueError:
        pltscl, dp_dconx, dp_dcony, devang = res
    return pltscl, dp_dconx, dp_dcony, devang


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
        if isinstance(sd.data, MKIDObservation) or isinstance(sd.data, MKIDDither):
            pltscl, dp_dconx, dp_dcony, devang = \
                run_wcscal(sd.data, sd.source_locs, sigma_psf=wcscfg.wcscal.sigma_psf, wave_start=950*u.nm,
                           wave_stop=1375 * u.nm, interpolate=wcscfg.wcscal.interpolate, ref_pix=sd.pixel_ref,
                           conex_ref=sd.conex_ref, guesses=np.array(wcscfg.wcscal.param_guesses))
            hdr = sd.data.obs[0].photontable.get_fits()[0].header
            hdul = fits.HDUList([fits.PrimaryHDU(header=hdr)])
            hdul[0].header['PLTSCL'] = (pltscl, 'platescale in arcsec/pixel')
            hdul[0].header['DPIXDCXX'] = (dp_dconx, 'pixel move per conex move in x')
            hdul[0].header['DPIXDCXY'] = (dp_dcony, 'pixel move per conex move in y')
            hdul[0].header['DEVANG'] = (devang, 'device angle in degrees')
            hdul.writeto(sd.path[:-4] + '.fits')
        else:
            pltscl = sd.data
            devang = wcscfg.instrument.device_orientation_deg
            hdul = fits.HDUList([fits.PrimaryHDU()])
            hdul[0].header['PLTSCL'] = (pltscl.to(u.arcsec).value, 'platescale in x arcsec/pixel')
            hdul[0].header['DEVANG'] = (devang.value, 'device angle in degrees')
            hdul.writeto(sd.path[:-4] + '.fits')


def apply(o):
    sol_path = o.wcscal.path[:-4] + '.fits'
    getLogger(__name__).info(f'Applying {sol_path} to {o.h5}')
    sol_hdul = fits.open(sol_path)
    pltscl = sol_hdul[0].header['PLTSCL']
    devang = sol_hdul[0].header['DEVANG']
    pt = o.photontable
    pt.enablewrite()
    pt.update_header('E_PLTSCL', pltscl)
    pt.update_header('E_DEVANG', devang)
    pt.disablewrite()
