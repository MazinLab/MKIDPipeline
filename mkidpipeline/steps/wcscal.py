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
from scipy.optimize import fsolve
from astropy.io import fits
import os
from mkidpipeline.utils.smoothing import replace_nan
from scipy.optimize import curve_fit
from astropy.utils.exceptions import AstropyUserWarning
import warnings
import tkinter as tk
from tkinter import messagebox

warnings.simplefilter('ignore', category=AstropyUserWarning)

class StepConfig(mkidpipeline.config.BaseStepConfig):
    yaml_tag = u'!wcscal_cfg'
    REQUIRED_KEYS = (('plot', 'none', 'none|all|summary'),
                     ('interpolate', True, 'whether to inerpolate the image before PSF fitting. Recommended if an '
                                             'MKIDObservation is used or data is noisy'),
                     ('sigma_psf', 2.0, 'standard deviation of the point spread functions to fit in the image '),
                     ('param_guesses', [-0.6], '(optional) initial guesse for device angle fitting '
                                               '(in radians)'))


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

    def __init__(self, image, source_locs, fig=None, conex_pos=None):
        self.coords = []
        self.image = image
        self.n_sources = len(source_locs)
        self.source_locs = source_locs
        self.counter = 0
        self.fig = plt.figure() if fig is None else fig
        self.cid1 = None
        self.cid2 = None
        self.conex_pos = conex_pos
        self.redo = False

    def push_to_start(self):
        plt.imshow(self.image)
        plt.title('Click Anywhere to Start')
        if plt.waitforbuttonpress() is True:
            return True

    def get_coords(self):
        self.push_to_start()
        plt.title(f'CONEX pos: {self.conex_pos}. \n Select Location of Source at '
                  f'({self.source_locs[self.counter][0]:.6f}, {self.source_locs[self.counter][1]:.6f})')
        plt.draw()
        self.cid1 = self.fig.canvas.mpl_connect('button_press_event', self.__onclick__)
        plt.show()
        return self.coords

    def __onclick__(self, click):
        try:
            point = (click.xdata, click.ydata)
        except TypeError:
            point = (None, None)
        self.coords.append(point)
        self.counter += 1
        if self.counter == self.n_sources:
            plt.scatter(*zip(*self.coords), c='red', marker='o', s=10)
            plt.draw()
            plt.title('Press "c" to Continue or "r" to Re-do')
            self.cid2 = self.fig.canvas.mpl_connect('key_press_event', self.__onkey__)
        try:
            plt.imshow(self.image)
            plt.scatter(*zip(*self.coords), c='red', marker='o', s=10)
            plt.title(f'CONEX pos: {self.conex_pos}. \n Select Locations of Source at '
                      f'({self.source_locs[self.counter][0]:.6f}, {self.source_locs[self.counter][1]:.6f})')
            plt.draw()
        except IndexError:
            pass
        return self.coords

    def __onkey__(self, event):
        cont_key = 'c'
        redo_key = 'r'
        cont = False
        while not cont:
            if event.key == cont_key:
                cont = True
            if event.key == redo_key:
                self.fig.canvas.mpl_disconnect(self.cid2)
                self.fig.canvas.mpl_disconnect(self.cid1)
                plt.close(self.fig)
                self.redo = True
                break
        self.fig.canvas.mpl_disconnect(self.cid2)
        self.fig.canvas.mpl_disconnect(self.cid1)
        plt.close(self.fig)
        return self.coords


def select_sources(image, source_locs, conex_pos):
    """
    runs interactive plotting function for selecting which points in an image correspond to which sources that have
    well constrained on sky coordinates
    :param image: Image in which to select points
    :param source_locs: (RA, DEC) coordinates of the sources in the image to select
    :return: list of (x, y) pixel coordinates for each source_loc
    """
    fig = plt.figure()
    cc = ClickCoords(image, source_locs=source_locs, fig=fig, conex_pos=conex_pos)
    coords = cc.get_coords()
    if cc.redo:
        coords = select_sources(image, source_locs, conex_pos)
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
    conex_pos, pix_coord, sky_coord, theta, t0, t1, frame, ref_pix, conex_ref, mux, muy, eta, idx = args
    phi = x
    theta = theta
    sx, sy = sky_coord.ra.value, sky_coord.dec.value
    p_refx, p_refy = ref_pix[0], ref_pix[1]
    c_refx, c_refy = conex_ref[0], conex_ref[1]
    x_cen, y_cen = 139 / 2, 145 / 2
    cx, cy = conex_pos[0], conex_pos[1]
    px, py = pix_coord[0], pix_coord[1]
    eq1 = np.cos(theta) * (eta * (
            px * np.sqrt(1 - phi ** 2) - py * phi - x_cen * np.sqrt(1 - phi ** 2) + y_cen * phi + x_cen + mux * (
                cx - c_refx) - p_refx)) - \
          np.sin(theta) * (eta * (
            px * phi + py * np.sqrt(1 - phi ** 2) - x_cen * phi - y_cen * np.sqrt(1 - phi ** 2) + y_cen + muy * (
                cy - c_refy) - p_refy)) + \
          t0 - sx
    eq2 = np.sin(theta) * (eta * (
            px * np.sqrt(1 - phi ** 2) - py * phi - x_cen * np.sqrt(1 - phi ** 2) + y_cen * phi + x_cen + mux * (
                cx - c_refx) - p_refx)) + \
          np.cos(theta) * (eta * (
            px * phi + py * np.sqrt(1 - phi ** 2) - x_cen * phi - y_cen * np.sqrt(1 - phi ** 2) + y_cen + muy * (
                cy - c_refy) - p_refy)) + \
          t1 - sy
    if idx == 0:
        return eq1
    if idx == 1:
        return eq2


def solve_system_of_equations(coords, conex_positions, telescope_angles, ra, dec, ref_pix, conex_ref, pltscl,
                              frame='icrs',
                              guesses=np.array([1, 1, 1, 1]), mux=None, muy=None):
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
    devangs = []
    for i, pos in enumerate(conex_positions):
        for j, pix in enumerate(pix_coords[i]):
            for k in range(2):
                sky = sky_coords[i][j]
                ang = telescope_angles[i]
                if np.isclose(ra - sky.ra.value, 0, rtol=1e-7) or np.isclose(dec - sky.dec.value, 0, rtol=1e-7):
                    pass
                else:
                    res = fsolve(generate_equations, guesses[-1],
                                 args=(pos, pix, sky, ang, ra, dec, frame, ref_pix, conex_ref, mux, muy,
                                       pltscl.to(u.deg).value, k), xtol=1e-12, maxfev=1000)
                    devangs.append((np.arcsin(res) * u.rad).to(u.deg).value)
    devang = np.mean(devangs)
    dp_dconx = mux
    dp_dcony = muy
    getLogger(__name__).info('\n Calculated WCS Solution: \n'
                             f'PLATESCALE: {(pltscl).to(u.arcsec).value} arcsec/pix\n'
                             f'CONEX MOVE: ({dp_dconx:.2f}, {dp_dcony:.2f}) pixels/conex move\n'
                             f'TELESCOPE OFFSET: ({ra:.5f}, {dec:.5f}) (RA, DEC)\n'
                             f'DEVANG: {(devang * u.deg).value:.2f} degrees, stddev: {np.std(devangs)}')
    return pltscl.to(u.deg), dp_dconx, dp_dcony, (devang * u.deg)


def calculate_wcs_solution(images, source_locs=None, sigma_psf=2.0, interpolate=False, conex_positions=None,
                           telescope_angles=None, ra=None, dec=None, ref_pix=None, conex_ref=None, guesses=None,
                           mux=None, muy=None):
    """
    calculates the parameters needed to form a WCS solution
    :param images: list of the images or each different pointing, conex position, or rotation angle.
    :param source_locs: Array of SkyCoord objects - true locations of the sources in the image
    :param sigma_psf: if interpolate is True, what width Gaussian to use for the interpolation
    :param interpolate: if True will interpolate the images to improve the PSF fitting
    :param conex_positions: List of conex positions for each image
    :param telescope_angles: List of telescope angles for each image
    :param ra: RA of the central object (telescope offset)
    :param dec: DEC of the central object (telescope offset)
    :return: platescale in x, platescale in y, x slope of the conex, y slope of the conex, device rotation angle
    """
    coords = []
    for i, image in enumerate(images):
        if interpolate:
            image[image==0] = np.nan
            im = replace_nan(image)
            im = astropy_convolve(im)
        else:
            im = image
        # returns clicked coordinates corresponding in order to the given ra/dec coords
        coord_dict = {k: None for k in source_locs}
        selected_coords = select_sources(im, source_locs=source_locs, conex_pos=conex_positions[i])
        sources, residuals = fit_sources(im, sigma_psf=sigma_psf)#, guesses=selected_coords)
        fit_coords = [(sources['x_fit'][i], sources['y_fit'][i]) for i in range(len(sources))]
        use_idxs = []
        for i, coord in enumerate(selected_coords):
            if coord[0] is None or coord[1] is None:
                use_idxs.append(np.nan)
            else:
                use_idxs.append(closest_node(selected_coords[i], fit_coords))

        use_coord = [fit_coords[idx] if not np.isnan(idx) else (0, 0) for idx in use_idxs]
        for i, key in enumerate(coord_dict.keys()):
            coord_dict[key] = use_coord[i]

        bad_keys = []
        for key, val in coord_dict.items():
            if val == (0, 0):
                bad_keys.append(key)
        for i, key in enumerate(bad_keys):
            del coord_dict[key]


        coords.append(coord_dict)
    if mux and muy:
        pass
    else:
        mux, muy = solve_conex(coords, conex_positions)

    pltscl = solve_platescale(coords, len(source_locs))
    res = solve_system_of_equations(coords, conex_positions, telescope_angles, ra, dec, ref_pix, conex_ref, pltscl,
                                    guesses=guesses, mux=mux, muy=muy)
    return res


def solve_conex(coords, conex_pos):
    pix_coords = []
    for i, c in enumerate(coords):
        pix_coords.append([i for i in c.values()])

    # get PSF centers and CONEX positions
    psf_centers = []
    conex_positions = []
    for i, pos in enumerate(conex_pos):
        psf_centers.append([pix_coords[i][0][0], pix_coords[i][0][1]])
        conex_positions.append([pos[0], pos[1]])
    # solve
    xPosFit = np.zeros(len(psf_centers))
    yPosFit = np.zeros(len(psf_centers))

    for i, pos in enumerate(psf_centers):
        xPosFit[i] = pos[0]
        yPosFit[i] = pos[1]

    xConFit = np.zeros(len(conex_positions))
    yConFit = np.zeros(len(conex_positions))

    for i, pos in enumerate(conex_positions):
        xConFit[i] = pos[0]
        yConFit[i] = pos[1]

    def func(x, slope, intercept):
        return x * slope + intercept

    xopt, xcov = curve_fit(func, xConFit, xPosFit)
    yopt, ycov = curve_fit(func, yConFit, yPosFit)
    mux = xopt[0]
    muy = yopt[0]
    getLogger(__name__).info(f'CONEX slopes calculated to be {mux} in x and {muy} in y. (0,0) CONEX position calculated'
                             f' to be ({xopt[1]}, {yopt[1]})')
    return mux, muy


def sep_between_two_points(p1, p2):
    dist = np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
    return dist


def solve_platescale(coords, n_sources):
    pix_coords = []
    sky_coords = []
    if n_sources > 2:
        getLogger(__name__).error('WCS Cal currently only supports an using a dataset with two unique sources')
        raise NotImplementedError
    for i, c in enumerate(coords):
        pix_coords.append([i for i in c.values()])
        sky_coords.append([SkyCoord(i[0], i[1], unit='deg') for i in c.keys()])
    unique_coords = []
    for i in range(n_sources):
        unique_coords.append(sky_coords[0][i])

    platescales = []
    for i, pix in enumerate(pix_coords):
        try:
            s1 = unique_coords[0]
            s2 = unique_coords[1]
            p1 = pix[0]
            p2 = pix[1]
            sky_sep = s1.separation(s2).to(u.mas).value
            pix_sep = sep_between_two_points(p1, p2)
            platescale = sky_sep / pix_sep
            platescales.append(platescale)
        except IndexError:
            pass
    return np.mean(platescales) * u.mas


def load_data(data, wave_start=950 * u.nm, wave_stop=1375 * u.nm):
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
    return images, conex_positions, telescope_ang, ra, dec


def run_wcscal(data, source_locs, sigma_psf=None, wave_start=950*u.nm, wave_stop=1375*u.nm, interpolate=True,
               ref_pix=None, conex_ref=None, guesses=None, mux=None, muy=None):
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

    images, conex_positions, telescope_ang, ra, dec = load_data(data, wave_start=wave_start, wave_stop=wave_stop)
    res = calculate_wcs_solution(images, sources, sigma_psf=sigma_psf, interpolate=interpolate,
                                 conex_positions=conex_positions, guesses=guesses, telescope_angles=telescope_ang,
                                 ra=ra, dec=dec, ref_pix=ref_pix, conex_ref=conex_ref, mux=mux, muy=muy)
    try:
        pltscl_x, pltscl_y, dp_dconx, dp_dcony, devang = res
    except ValueError:
        pltscl, dp_dconx, dp_dcony, devang = res
    return pltscl, dp_dconx, dp_dcony, devang


def display_message():
    messagebox.showinfo(title="WCS Calibration Instructions",
                        message="You specified an observation or dither and are about to run the WCS Calibration. \n"
                                "Some quick instructions before you get started: \n \n"
                                "- Depending on how many source locations were specified in your data you will need to "
                                "click the approximate location of each source. \n \n"
                                "- You will receive the RA/Dec location of each source you are supposed to click \n \n"
                                "- If you do not see a source in your image, click outside the image window \n \n"
                                "- After selecting all the sources you can press 'c' to move to the next image or 'r' "
                                "to reselect \n \n"
                                "Have Fun! ")

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
            ws = tk.Tk()
            ws.geometry('300x100')
            button = tk.Button(ws, text='Click to view WCSCal Directions', command=display_message)
            button.pack()
            ws.mainloop()
            tk.Tk().withdraw()

            pltscl, dp_dconx, dp_dcony, devang = \
                run_wcscal(sd.data, sd.source_locs, sigma_psf=wcscfg.wcscal.sigma_psf, wave_start=950*u.nm,
                           wave_stop=1375 * u.nm, interpolate=wcscfg.wcscal.interpolate, ref_pix=sd.pixel_ref,
                           conex_ref=sd.conex_ref, guesses=np.array(wcscfg.wcscal.param_guesses), mux=sd.dp_dcx,
                           muy=sd.dp_dcy)
            hdr = sd.data.obs[0].photontable.get_fits()[0].header
            hdul = fits.HDUList([fits.PrimaryHDU(header=hdr)])
            hdul[0].header['PLTSCL'] = (pltscl.to(u.deg).value, 'platescale in degree/pixel')
            hdul[0].header['E_DPDCX'] = (dp_dconx, 'pixel move per conex move in x')
            hdul[0].header['E_DPDCY'] = (dp_dcony, 'pixel move per conex move in y')
            hdul[0].header['DEVANG'] = (devang.to(u.deg).value, 'device angle in degrees')

            if sd.conex_ref is not None:
                hdul[0].header['E_CXREFX'] = (sd.conex_ref[0], 'Conex reference position in X')
                hdul[0].header['E_CXREFY'] = (sd.conex_ref[1], 'Conex reference position in Y')
            else:
                try:
                    if isinstance(sd.data, MKIDObservation):
                        time = sd.data.start
                        md = sd.data.metadata_at(time=time)
                    elif isinstance(sd.data, MKIDDither):
                        time = sd.data.obs[0].start
                        md = sd.data.obs[0].metadata_at(time=time)

                    hdul[0].header['E_CXREFX'] = (md['E_CXREFX'], 'Conex reference position in X')
                    hdul[0].header['E_CXREFY'] = (md['E_CXREFY'], 'Conex reference position in X')
                except KeyError:
                    raise RuntimeError('Conex reference either needs to be specified in the pipeline config or '
                                       'be present in the metadata. WCScal failes to apply.')

            if sd.pixel_ref is not None:
                hdul[0].header['E_PREFX'] = (sd.pixel_ref[0], 'Pixel reference position in X')
                hdul[0].header['E_PREFY'] = (sd.pixel_ref[1], 'Pixel reference position in Y')
            else:
                try:
                    if isinstance(sd.data, MKIDObservation):
                        time = sd.data.start
                        md = sd.data.metadata_at(time=time)
                    elif isinstance(sd.data, MKIDDither):
                        time = sd.data.obs[0].start
                        md = sd.data.obs[0].metadata_at(time=time)
                    hdul[0].header['E_PREFX'] = (md['E_PREFX'], 'Pixel reference position in X')
                    hdul[0].header['E_PREFY'] = (md['E_PREFY'], 'Pixel reference position in X')
                except KeyError:
                    raise RuntimeError('Pixel reference either needs to be specified in the pipeline config or '
                                       'be present in the metadata. WCScal failed to apply ')

            hdul.writeto(sd.path[:-4] + '.fits')
            getLogger(__name__).info(f'Saved WCS Solution to {sd.path[:-4]}.fits')
        else:
            pltscl = sd.data
            devang = wcscfg.instrument.device_orientation_deg
            hdul = fits.HDUList([fits.PrimaryHDU()])
            hdul[0].header['PLTSCL'] = (pltscl.to(u.deg).value, 'platescale in degree/pixel')
            hdul[0].header['E_DPDCX'] = (sd.dp_dcx, 'pixel move per conex move in x')
            hdul[0].header['E_DPDCY'] = (sd.dp_dcy, 'pixel move per conex move in y')
            hdul[0].header['DEVANG'] = (devang, 'device angle in degrees')
            hdul[0].header['E_CXREFX'] = (sd.conex_ref[0], 'Conex reference position in X')
            hdul[0].header['E_CXREFY'] = (sd.conex_ref[1], 'Conex reference position in Y')
            hdul[0].header['E_PREFX'] = (sd.pixel_ref[0], 'Pixel reference position in X')
            hdul[0].header['E_PREFY'] = (sd.pixel_ref[1], 'Pixel reference position in Y')
            hdul.writeto(sd.path[:-4] + '.fits')
            getLogger(__name__).info(f'Saved WCS Solution to {sd.path[:-4]}.fits')

def apply(o):
    sol_path = o.wcscal.path[:-4] + '.fits'
    getLogger(__name__).info(f'Applying {sol_path} to {o.h5}')
    sol_hdul = fits.open(sol_path)
    pltscl = sol_hdul[0].header['PLTSCL']
    devang = sol_hdul[0].header['DEVANG']
    dp_dcx = sol_hdul[0].header['E_DPDCX']
    dp_dcy = sol_hdul[0].header['E_DPDCY']
    cx_refx = sol_hdul[0].header['E_CXREFX']
    cx_refy = sol_hdul[0].header['E_CXREFY']
    p_refx = sol_hdul[0].header['E_PREFX']
    p_refy = sol_hdul[0].header['E_PREFY']
    pt = Photontable(o.h5)
    pt.enablewrite()
    pt.update_header('E_DPDCX', dp_dcx)
    pt.update_header('E_DPDCY', dp_dcy)
    pt.update_header('E_PLTSCL', pltscl)
    pt.update_header('E_DEVANG', devang)
    pt.update_header('E_CXREFX', cx_refx)
    pt.update_header('E_CXREFY', cx_refy)
    pt.update_header('E_PREFX', p_refx)
    pt.update_header('E_PREFY', p_refy)
    getLogger(__name__).info(f'Updated WCS info for {o.h5}')
    pt.disablewrite()
