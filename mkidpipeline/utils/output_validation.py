import mkidpipeline.definitions as definitions
import mkidpipeline.config as config
from astropy.io import fits
from mkidcore.corelog import getLogger
import pkg_resources as pkg
import csv


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
        getLogger(__name__).warning(f'Unable to open and close output {o.name}')
    getLogger(__name__).info(f'Successfully opened and closed {o.name}')


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
        getLogger(__name__).info(f'output {o.name} contains all required fits keys')
    else:
        diff = set(hdr) ^ set(mec_keys)
        getLogger(__name__).warning(f'output {o.name} does not contain all required fits keys, missing {diff}')


def dimensions(o):
    """
    validate the dimensions of the output are correct
    :param o:
    :return:
    """
    hdul = fits.open(o.filename)
    if o.wants_image:
        if hdul[1].shape == (140,146):
            getLogger(__name__).info(f'output {o.name} has appropriate dimensions for type {o.kind}')
        else:
            getLogger(__name__).warning(f'output {o.name} has inappropriate dimensions for type {o.kind}')
    if o.wants_drizzled:
        if hdul[1].shape == (1000, 1000):
            getLogger(__name__).info(f'output {o.name} has appropriate dimensions for type {o.kind}')
        else:
            getLogger(__name__).warning(f'output {o.name} has inappropriate dimensions for type {o.kind}')


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
            getLogger(__name__).warning(f'E_FLTCAL not in fits header')
        elif hdr['E_FLTCAL'] == 'none':
            getLogger(__name__).warning(f'E_FLTCAL has no associated flatcal UUID and one is expected')
    if o.data.wavecal:
        if 'E_WAVCAL' not in hdr:
            getLogger(__name__).warning(f'E_WAVCAL not in fits header')
        elif hdr['E_WAVCAL'] == 'none':
            getLogger(__name__).warning(f'E_WAVCAL has no associated wavecal UUID and one is expected')
    if o.data.wavecal:
        if 'E_WCSCAL' not in hdr:
            getLogger(__name__).warning(f'E_WCSCAL not in fits header')
        elif hdr['E_WCSCAL'] == 'none':
            getLogger(__name__).warning(f'E_WCSCAL has no associated wcscal UUID and one is expected')


def wcs(o):
    """
    validate the wcs solution makes sense for the object
    :param o:
    :return:
    """
    return None


if __name__== "__main__":
    pipe_cfg = '/path/to/pipe.yaml'
    out_cfg = '/path/to/out.yaml'
    data_cfg = '/path/to/data.yaml'

    config.configure_pipeline(pipe_cfg)
    outputs = definitions.MKIDOutputCollection(out_cfg, datafile=data_cfg)
    validate(outputs)
