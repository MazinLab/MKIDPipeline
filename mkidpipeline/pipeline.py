from importlib import import_module
import pkgutil
import multiprocessing as mp
import time
import os

from mkidcore.config import getLogger
import mkidpipeline
import mkidpipeline.photontable as photontable
import mkidpipeline.config as config
import mkidpipeline.steps
from mkidpipeline.steps import wavecal
import mkidpipeline.bin2hdf
import mkidpipeline.imaging.movies

import mkidcore.config

log = getLogger('mkidpipeline')

PIPELINE_STEPS = {}
for info in pkgutil.iter_modules(mkidpipeline.steps.__path__):
    mod = import_module(f"mkidpipeline.steps.{info.name}")
    globals()[info.name] = mod
    PIPELINE_STEPS[info.name] = mod
    try:
        mkidcore.config.yaml.register_class(mod.StepConfig)
    except AttributeError:
        pass


class BaseConfig(mkidpipeline.config.BaseStepConfig):
    yaml_tag = u'!pipe_cfg'
    REQUIRED_KEYS = (('ncpu', 1, 'number of cpus'),
                     ('verbosity', 0, 'level of verbosity'),
                     ('flow', ('wavecal','metadata','flatcal','cosmiccal','photcal','lincal'), 'Calibration steps to apply'),
                     ('paths.dithers', '/darkdata/MEC/logs/','dither log location'),
                     ('paths.data', '/darkdata/ScienceData/Subaru/','bin file parent folder'),
                     ('paths.database', '/work/temp/database/', 'calibrations will be retrieved/stored here'),
                     ('paths.obslog', '/work/temp/database/obslog', 'obslog.json go here'),
                     ('paths.out', '/work/temp/out/', 'root of output'),
                     ('paths.tmp', '/work/temp/scratch/', 'use for data intensive temp files'))


mkidcore.config.yaml.register_class(BaseConfig)


def generate_default_config():
    cfg = BaseConfig()
    for name, step in PIPELINE_STEPS.items():
        try:
            cfg.register(name, step.StepConfig(), update=True)
        except AttributeError:
            getLogger(__name__).warning(f'Pipeline step mkidpipeline.steps.{name} does not '
                                        f'support automatic configuration discovery.')
    cfg.register('beammap', mkidcore.objects.Beammap(default='MEC'))
    cfg.register('instrument', mkidcore.instruments.InstrumentInfo('MEC'))
    return cfg


def metadata_apply(ob):
    o = photontable.Photontable(ob.h5, mode='w')
    mdl = config.select_metadata_for_h5(ob, config.load_observing_metadata())
    o.attach_observing_metadata(mdl)


def wavecal_apply(o):
    if o.wavecal is None:
        getLogger(__name__).info('No wavecal to apply for {}'.format(o.h5))
        return
    try:
        of = photontable.Photontable(o.h5, mode='a')
        of.apply_wavecal(wavecal.load_solution(o.wavecal.path))
        of.file.close()
    except Exception as e:
        getLogger(__name__).critical('Caught exception during run of {}'.format(o.h5), exc_info=True)


def flatcal_apply(o):
    if o.flatcal is None:
        getLogger(__name__).info('No flatcal to apply for {}'.format(o.h5))
        return
    try:
        of = photontable.Photontable(o.h5, mode='a')
        cfg = mkidpipeline.config.config
        of.apply_flatcal(o.flatcal.path, use_wavecal=cfg.flatcal.use_wavecal, startw=850, stopw=1375)
        of.file.close()
    except Exception as e:
        getLogger(__name__).critical('Caught exception during run of {}'.format(o.h5), exc_info=True)


def linearitycal_apply(o):
    try:
        of = photontable.Photontable(o, mode='a')
        cfg = mkidpipeline.config.config
        of.applyLinearitycal(dt=cfg.linearitycal.dt, tau=cfg.instrument.deadtime*1*10**6)
        of.file.close()
    except Exception as e:
        getLogger(__name__).critical('Caught exception during run of {}'.format(o), exc_info=True)


def badpix_apply(o):
    try:
        mkidpipeline.steps.badpix.mask_hot_pixels(o)
    except Exception as e:
        getLogger(__name__).critical('Caught exception during run of {}'.format(o), exc_info=True)


def batch_apply_metadata(dataset):
    """Function associates things not known at hdf build time (e.g. that aren't in the bin files)"""
    # Retrieve metadata database
    metadata = config.load_observing_metadata()
    # Associate metadata
    for ob in dataset.all_observations:
        o = photontable.Photontable(ob.h5, mode='w')
        mdl = config.select_metadata_for_h5(ob, metadata)
        o.attach_observing_metadata(mdl)
        del o


def batch_apply_wavecals(dset, ncpu=None):
    """ filter for unique h5 files, not responsible for mixed wavecal specs """
    wavecal.clear_solution_cache()
    pool = mp.Pool(ncpu if ncpu is not None else config.n_cpus_available(config.config.wavecal.ncpu))
    obs = {o.h5: o for o in dset.wavecalable if o.wavecal is not None}.values()
    pool.map(wavecal_apply, obs)
    pool.close()


def batch_apply_flatcals(dset, ncpu=None):
    """
    Will filter for unique h5 files, not responsible for mixed flatcal specs
    """
    pool = mp.Pool(ncpu if ncpu is not None else config.n_cpus_available())
    obs = {o.h5: o for o in dset.flatcalable if o.flatcal is not None}.values()
    pool.map(flatcal_apply, obs)
    pool.close()


def batch_apply_badpix(dset, ncpu=None):
    pool = mp.Pool(ncpu if ncpu is not None else config.n_cpus_available())
    pool.map(badpix_apply, set([o.h5 for o in dset.science_observations]))
    pool.close()


def batch_apply_linearitycal(dset, ncpu=None):
    pool = mp.Pool(ncpu if ncpu is not None else config.n_cpus_available())
    pool.map(linearitycal_apply, set([o.h5 for o in dset.science_observations]))
    pool.close()


def batch_build_hdf(timeranges):
    mkidpipeline.bin2hdf.buildtables(timeranges, ncpu=ncpu, remake=False)


def run_stage1(dataset):
    operations = (('Building H5s', mkidpipeline.bin2hdf.buildtables),
                  ('Attaching metadata', batch_apply_metadata),
                  ('Fetching wavecals', mkidpipeline.steps.wavecal.fetch),
                  ('Applying wavelength solutions', batch_apply_wavecals),
                  ('Applying wavelength solutions', batch_apply_badpix),
                  ('Applying linearity correction', batch_apply_linearitycal),
                  ('Fetching flatcals', mkidpipeline.steps.flatcal.fetch),
                  ('Applying flatcals', batch_apply_flatcals),
                  ('Fetching speccals', mkidpipeline.steps.spectralcal.fetch))

    toc = time.time()
    for task_name, task in operations:
        tic = time.time()
        getLogger(__name__).info(f'Stage 1: {task_name}')
        task(dataset)
        getLogger(__name__).info(f'Completed {task_name} in {time.time()-tic:.0f} s')

    getLogger(__name__).info(f'Stage 1 complete in {(time.time()-toc)/60:.0f} m')


def generate_outputs(outputs):
    mkidpipeline.steps.drizzler.fetch(outputs)

    for o in outputs:
        # TODO make into a batch process
        getLogger(__name__).info('Generating {}'.format(o.name))
        if o.wants_image:

            for obs in o.data.obs:
                h5 = mkidpipeline.photontable.Photontable(obs.h5)
                img = h5.get_fits(wvlStart=o.startw, wvlStop=o.stopw, applyWeight=o.enable_photom,
                                  applyTPFWeight=o.enable_noise, countRate=True)
                img.writeto(o.output_file)
                getLogger(__name__).info('Generated fits file for {}'.format(obs.h5))
        if o.wants_movie:
            getLogger('mkidpipeline.hdf.photontable').setLevel('DEBUG')
            mkidpipeline.imaging.movies.make_movie(o, inpainting=False)