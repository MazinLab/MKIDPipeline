import os
import scipy.ndimage as ndi
import numpy as np
from multiprocessing import Pool
from astropy.io import fits

from mkidcore.corelog import getLogger
from mkidcore.instruments import CONEX2PIXEL
from mkidpipeline.photontable import Photontable
from mkidpipeline.config import n_cpus_available
from mkidpipeline.steps import drizzler


def fetchimg(ob, kwargs):
    of = Photontable(ob.h5)
    im = of.get_fits(cube_type='wave' if kwargs['nwvl'] > 1 else None)['SCIENCE']
    del of

    #TODO move to photontable and fetch it from an import
    im.headerh['PIXELA'] = 10.0**2
    return im


def makeimage(data, mode, nwvl=1, wvlRange=(None,None), cfg=None, ncpu=None):
    """Make an image or cube from the data (Obs, list of obs, or dither) using sum, median, or average"""

    kw = dict(wvlStart=wvlRange[0], wvlStop=wvlRange[1], applyWeight=True, applyTPFWeight=True, wvlN=nwvl)

    dither = None
    try:
        obs = list(data)
        id = obs[0].id + '_{}obs'.format(len(obs))
    except TypeError:
        try:
            obs = data.obs
            dither = data
            id = dither.id
        except AttributeError:
            obs = [data]
            id = data.id

    pool = Pool(min(len(obs), ncpu if ncpu is not None else n_cpus_available()))
    hdus = pool.starmap(fetchimg, (ob, kw for ob in obs))
    pool.close()

    shifts = []
    angles = []
    if dither is not None:
        for h, p in zip(hdus, dither.pos):
            try:
                shifts.append(CONEX2PIXEL(h.header['CONEXX'], h.header['CONEXY']))
            except KeyError:
                getLogger(__name__).warning('If see this message and conex support has been added to photontable there is a '
                                            'bug.')
                shifts.append(CONEX2PIXEL(p))
            try:
                angles.append(h.header['PARAAOFF'])
            except KeyError:
                getLogger(__name__).warning('If see this message and parallactic angle offsets have been added to '
                                            'photontable there is a bug. Not rotating images')
                angles.append(0)
    else:
        angles = [0] * len(hdus)
        shifts = [(0, 0)] * len(hdus)

    #Combine hdus
    #TODO: MAJOR these shifts may be nonsense and need an offset or other zeroing
    shifts = np.array(shifts)
    padx_high, pady_high = shifts.max(0).clip(0,np.inf)
    padx_low, pady_low= np.abs(np.array(shifts).min(0).clip(-np.inf, 0))
    stack = []
    for h, angle, shift in zip(hdus, angles, shifts):
        padim = np.pad(h.data/h.header['PIXELA'], ((padx_low, padx_high), (pady_low, pady_high)), 'constant',
                       constant_values=0)
        im = ndi.shift(ndi.rotate(padim, angle, order=1, reshape=False), shift, order=1)
        stack.append(im*h.header['PIXELA'])

    image = getattr(np, mode)(stack, axis=0)

    hdu = fits.PrimaryHDU(data=image)
    for i, o in enumerate(obs):
        hdu.header['OBS{}'.format(i)] = o.id
    hdu.header['COMBMODE'] = mode
    hdul = fits.HDUList(hdus=[hdu])

    hdul.writeto('{id}_{mode}.fits'.format(id=id, mode=mode))


def drizzle(dither, config=None):
    cfg = config.PipelineConfigFactory(step_defaults=dict(drizzler=drizzler.StepConfig()), cfg=config, copy=True)

    out = drizzler.form(dither, mode=cfg.drizzler.mode, rotation_center=cfg.drizzler.rotation_origin,
                        wvlMin=dither.out.startw, wvlMax=dither.out.stopw,
                        pixfrac= cfg.drizzler.pixfrac, target_radec=cfg.drizzler.target_radec,
                        device_orientation=cfg.drizzler.device_orientation, derotate=dither.out.derotate)

    out.writefits(os.path.join(cfg.paths.out, '{name}_{kind}.fits'.format(dither.name, dither.out.kind)))