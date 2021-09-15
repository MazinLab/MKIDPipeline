import mkidpipeline.config as config
from astropy.io import fits
import logging
import pkg_resources as pkg
import csv
import astropy.units as u

pipe_cfg ='/work/steiger/new_pipeline_testing/pipe.yaml'
out_cfg='/work/steiger/new_pipeline_testing/out.yaml'
data_cfg='/work/steiger/new_pipeline_testing/data.yaml'

config.configure_pipeline(pipe_cfg)
outputs = config.MKIDOutputCollection(out_cfg, datafile=data_cfg)
logging.getLogger().setLevel('INFO')

def validate(outputs):
    for o in outputs:
        open_close(o)
        fits_keys(o)
        dimensions(o)
        calibration_steps(o)
        wcs(o)

    return None

def open_close(o):
    try:
        hdul = fits.open(o.filename)
        hdul.close()
    except OSError:
        logging.warning(f'Unable to open and close output {o.name}')
    logging.info(f'Successfully opened and closed {o.name}')


def fits_keys(o):
    """
    validate all required fits keys are present
    :param o:
    :return:
    """
    with open(pkg.resource_filename('mkidcore', 'mec_keys.csv')) as f:
        data = [row for row in csv.reader(f)]
    data = [{k.strip().lower().replace(' ', '_').replace('?', ''): v.strip() for k, v in zip(data[0], l)} for l in
            data[1:]]
    hdul = fits.open(o.filename)
    hdr = hdul[0].header
    mec_keys = [k['fits_card'] for k in data]
    res = all(key in mec_keys for key in hdr)
    if res:
        logging.info(f'output {o.name} contains all required fits keys')
    else:
        diff = set(hdr) ^ set(mec_keys)
        logging.warning(f'output {o.name} does not contain all required fits keys, missing {diff}')


def dimensions(o):
    """
    validate the dimensions of the output are correct
    :param o:
    :return:
    """
    hdul = fits.open(o.filename)
    if o.wants_image:
        if hdul[1].shape[-2:] == (140,146):
            logging.info(f'output {o.name} has appropriate dimensions for type {o.kind}')
        else:
            logging.warning(f'output {o.name} has WRONG dimensions for type {o.kind}')
    if o.wants_drizzled:
        if o.timestep > 0.0:
            n_steps = (o.duration/o.timestep) * len(o.data)
            if hdul[1].shape[0] == n_steps:
                logging.info(f'output {o.name} has appropriate dimensions for type {o.kind} and timestep {o.timestep}')
        elif o.wavestep.value > 0.0 and o.timestep == 0:
            n_steps = (o.max_wave - o.min_wave).to(u.nm)/o.wavestep.to(u.nm)
            if hdul[1].shape[0] == n_steps:
                logging.info(f'output {o.name} has appropriate dimensions for type {o.kind} and wavestep {o.wavestep}')
        elif o.wavestep.value > 0 and o.timestep !=0:
            n_wave = (o.max_wave - o.min_wave).to(u.nm)/o.wavestep.to(u.nm)
            n_time = (o.duration/o.timestep) * len(o.data)
            if hdul[1].shape[0] == n_time and hdul[1].shape[1] == n_wave:
                logging.info(f'output {o.name} has appropriate dimensions for type {o.kind} with wavestep {o.wavestep}'
                             f'and timestep {o.timestep}')
        if hdul[1].shape[-2:] == (500, 500):
            logging.info(f'output {o.name} has appropriate pixel dimensions for type {o.kind}')
        else:
            logging.warning(f'output {o.name} has WRONG dimensions for type {o.kind}')


def calibration_steps(o):
    """
    validate that the header signifies all the requested calibration steps have been run
    :param o:
    :return:
    """
    hdul = fits.open(o.filename)
    hdr = hdul[0].header
    if o.flatcal:
        if 'E_FLTCAL' not in hdr:
            logging.warning(f'E_FLTCAL not in fits header')
        elif hdr['E_FLTCAL'] == 'none':
            logging.warning(f'E_FLTCAL has no associated flatcal UUID and one is expected')
    if o.data.wavecal:
        if 'E_WAVCAL' not in hdr:
            logging.warning(f'E_WAVCAL not in fits header')
        elif hdr['E_WAVCAL'] == 'none':
            logging.warning(f'E_WAVCAL has no associated wavecal UUID and one is expected')
    if o.data.wavecal:
        if 'E_WCSCAL' not in hdr:
            logging.warning(f'E_WCSCAL not in fits header')
        elif hdr['E_WCSCAL'] == 'none':
            logging.warning(f'E_WCSCAL has no associated wcscal UUID and one is expected')
    if o.data.speccal:
        if 'E_SPECAL' not in hdr:
            logging.warning(f'E_SPECAL not in fits header')
        elif hdr['E_SPECAL'] == 'none':
            logging.warning(f'E_SPECAL has no associated speccal UUID and one is expected')


def wcs(o):
    """
    validate the wcs solution makes sense for the object
    :param o:
    :return:
    """
    return None

validate(outputs)
