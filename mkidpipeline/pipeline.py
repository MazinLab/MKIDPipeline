from importlib import import_module
import pkgutil

from mkidcore.config import getLogger
import mkidpipeline
import mkidpipeline.hdf.photontable
import mkidpipeline.config as config
import mkidpipeline.steps
from mkidpipeline.steps import *
from mkidpipeline.steps import wavecal
from mkidpipeline.hdf import bin2hdf
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


class BaseConfig(mkidcore.config.ConfigThing):
    yaml_tag = u'!pipe_cfg'
    REQUIRED_KEYS = (('ncpu', 1, 'number of cpus'),
                     ('verbosity', 0, 'level of verbosity'),
                     ('flow', ('wavecal','metadata','flatcal','cosmiccal','photcal','lincal'), 'Calibration steps to apply'),
                     ('paths.dithers', '/darkdata/MEC/logs/','dither log location'),
                     ('paths.data', '/darkdata/ScienceData/Subaru/','bin file parent folder'),
                     ('paths.database', '/work/temp/database/', 'calibrations will be retrieved/stored here'),
                     ('paths.out', '/work/temp/out/', 'root of output'),
                     ('paths.tmp', '/work/temp/scratch/', 'use for data intensive temp files'))

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v, c in self.REQUIRED_KEYS:
            self.register(k, v, comment=c, update=False)


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


def wavecal_apply(o):
    if o.wavecal is None:
        getLogger(__name__).info('No wavecal to apply for {}'.format(o.h5))
        return
    try:
        of = mkidpipeline.hdf.photontable.Photontable(o.h5, mode='a')
        of.applyWaveCal(wavecal.load_solution(o.wavecal.path))
        of.file.close()
    except Exception as e:
        getLogger(__name__).critical('Caught exception during run of {}'.format(o.h5), exc_info=True)


def flatcal_apply(o):
    if o.flatcal is None:
        getLogger(__name__).info('No flatcal to apply for {}'.format(o.h5))
        return
    try:
        of = mkidpipeline.hdf.photontable.Photontable(o.h5, mode='a')
        cfg = mkidpipeline.config.config
        of.applyFlatCal(o.flatcal.path, use_wavecal=cfg.flatcal.use_wavecal, startw=850, stopw=1375)
        of.file.close()
    except Exception as e:
        getLogger(__name__).critical('Caught exception during run of {}'.format(o.h5), exc_info=True)


def linearitycal_apply(o):
    try:
        of = mkidpipeline.hdf.photontable.Photontable(o, mode='a')
        cfg = mkidpipeline.config.config
        of.applyLinearitycal(dt=cfg.linearitycal.dt, tau=cfg.instrument.deadtime*1*10**6)
        of.file.close()
    except Exception as e:
        getLogger(__name__).critical('Caught exception during run of {}'.format(o), exc_info=True)


def batch_apply_metadata(dataset):
    """Function associates things not known at hdf build time (e.g. that aren't in the bin files)"""
    # Retrieve metadata database
    metadata = config.load_observing_metadata()
    # Associate metadata
    for ob in dataset.all_observations:
        o = mkidpipeline.hdf.photontable.Photontable(ob.h5, mode='w')
        mdl = config.select_metadata_for_h5(ob, metadata)
        o.attach_observing_metadata(mdl)
        del o


def batch_apply_wavecals(obs, ncpu=None):
    wavecal.clear_solution_cache()
    pool = mp.Pool(ncpu if ncpu is not None else config.n_cpus_available())
    obs = {o.h5: o for o in obs if o.wavecal is not None}.values()  # filter so unique h5 files, not responsible for a mixed wavecal specs
    pool.map(wavecal_apply, obs)
    pool.close()


def batch_apply_flatcals(obs, ncpu=None):
    pool = mp.Pool(ncpu if ncpu is not None else config.n_cpus_available())
    obs = {o.h5: o for o in obs if o.flatcal is not None}.values()  # filter so unique h5 files, not responsible for a mixed flatcal specs
    pool.map(flatcal_apply, obs)
    pool.close()


def batch_maskhot(obs, ncpu=None):
    pool = mp.Pool(ncpu if ncpu is not None else config.n_cpus_available())
    pool.map(badpix.mask_hot_pixels, set([o.h5 for o in obs]))
    pool.close()


def batch_apply_linearitycal(obs, ncpu=None):
    pool = mp.Pool(ncpu if ncpu is not None else config.n_cpus_available())
    pool.map(linearitycal_apply, set([o.h5 for o in obs]))
    pool.close()