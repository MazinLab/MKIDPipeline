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
    hdr = hdul[1].header
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
    return None


def calibration_steps(o):
    """
    validate that the header signifies all the requested calibration steps have been run
    :param o:
    :return:
    """
    return None


def wcs(o):
    """
    validate the wcs solution makes ense for the object
    :param o:
    :return:
    """
    return None


if __name__== "__main__":
    pipe_cfg = '/path/to/pipe.yaml'
    out_cfg = '/path/to/out.yaml'
    data_cfg = '/path/to/data.yaml'

    config.configure_pipeline(pipe_cfg)
    outputs = config.MKIDOutputCollection(out_cfg, datafile=data_cfg)
    validate(outputs)
