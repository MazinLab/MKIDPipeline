from importlib import import_module
import pkgutil
import multiprocessing as mp
import time
from collections import defaultdict
import functools

from mkidcore.pixelflags import FlagSet, BEAMMAP_FLAGS
from mkidcore.config import getLogger
import mkidpipeline
import mkidpipeline.photontable as photontable
import mkidpipeline.config as config
import mkidpipeline.steps
from mkidpipeline.steps import wavecal
import mkidpipeline.steps.buildhdf
import mkidpipeline.imaging.movies

import mkidcore.instruments
import mkidcore.objects
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

_flags = {'beammap': BEAMMAP_FLAGS}
for name, step in PIPELINE_STEPS.items():
    try:
        _flags[name] = step.FLAGS
    except AttributeError:
        getLogger(__name__).debug(f"Step {name} does not export any pipeline flags.")
        pass

PIPELINE_FLAGS = FlagSet.define(*sorted([(f"{k}.{f.name.replace(' ', '_')}", i, f.description) for i, (k, f) in
                                         enumerate((k, f) for k, flagset in _flags.items() for f in flagset)]))
del _flags

PROBLEM_FLAGS = ('pixcal.hot', 'pixcal.cold', 'pixcal.unstable', 'beammap.noDacTone', 'wavecal.bad',
                 'wavecal.failed_validation', 'wavecal.failed_convergence', 'wavecal.not_monotonic',
                 'wavecal.not_enough_histogram_fits', 'wavecal.no_histograms',
                 'wavecal.not_attempted')


def generate_default_config():
    cfg = config.PipeConfig()
    for name, step in PIPELINE_STEPS.items():
        try:
            cfg.register(name, step.StepConfig(), update=True)
        except AttributeError:
            getLogger(__name__).warning(f'Pipeline step mkidpipeline.steps.{name} does not '
                                        f'support automatic configuration discovery.')
    cfg.register('beammap', mkidcore.objects.Beammap(specifier='MEC'))
    cfg.register('instrument', mkidcore.instruments.InstrumentInfo('MEC'))
    return cfg


def generate_sample_data():
    i = defaultdict(lambda: 0)

    def namer(name='Thing'):
        ret = f"{name}{i[name]}"
        i[name] = i[name] + 1
        return ret

    data = [config.MKIDTimerange(name=namer(), start=1602048870, duration=30,
                                 dark=config.MKIDTimerange(name=namer(), start=1602046500, stop=1602046510)),
            config.MKIDObservation(name=namer('star'), start=1602048875, duration=10, wavecal='wavecal0',
                                   dark=config.MKIDTimerange(name=namer(), start=1602046500, duration=10),
                                   flatcal='flactcal0', wcscal='wcscal0', speccal='speccal0'),
            # a wavecal
            config.MKIDWavecalDescription(name=namer('wavecal'), obs=(
                config.MKIDTimerange(name='850 nm', start=1602040820, duration=60,
                                     dark=config.MKIDTimerange(name=namer(), start=1602046500, duration=10),
                                     header=dict(laser='on', other='fits_key')),
                config.MKIDTimerange(name='950 nm', start=1602040895, duration=60,
                                     dark=config.MKIDTimerange(name=namer(), start=1602046500, duration=10)),
                config.MKIDTimerange(name='1.1 um', start=1602040970, duration=60,
                                     dark=config.MKIDTimerange(name=namer(), start=1602046500, duration=10)),
                config.MKIDTimerange(name='1.25 um', start=1602041040, duration=60,
                                     dark=config.MKIDTimerange(name=namer(), start=1602046500, duration=10)),
                config.MKIDTimerange(name='13750 A', start=1602041110, duration=60)
            )),
            # Flatcals
            config.MKIDFlatcalDescription(name=namer('flatcal'),
                                          data=config.MKIDObservation(name='950 nm', start=1602040900, duration=50,
                                                                      dark=config.MKIDTimerange(name=namer(),
                                                                                                start=1602046500,
                                                                                                duration=10),
                                                                      wavecal='wavecal0')),
            config.MKIDFlatcalDescription(name=namer('flatcal'), wavecal_duration=50.0, wavecal_offset=1,
                                          data='wavecal0'),
            # Speccal
            config.MKIDSpeccalDescription(name=namer('speccal'),
                                          data=config.MKIDObservation(name=namer('star'), start=340, duration=10,
                                                                      wavecal='wavecal0',
                                                                      spectrum='qualified/path/or/relative/'
                                                                               'todatabase/refspec.file'),
                                          aperture=('15h22m32.3', '30.32 deg', '200 mas')),

            # WCS cal
            config.MKIDWCSCalDescription(name=namer('wcscal'), dither_home=(107, 46), dither_ref=(-0.16, -0.4),
                                         data='10.40 mas'),
            config.MKIDWCSCalDescription(name=namer('wcscal'), comment='ob wcscals may be used to manually determine '
                                                                       'WCS parameters. They are not yet supported for '
                                                                       'automatic WCS parameter computation',
                                         data=config.MKIDObservation(name=namer('star'), start=360, duration=10,
                                                                     wavecal='wavecal0',
                                                                     dark=config.MKIDTimerange(name=namer(), start=350,
                                                                                               duration=10)),
                                         dither_home=(107, 46), dither_ref=(-0.16, -0.4)),
            # Dithers
            config.MKIDDitherDescription(name=namer('dither'), data=1602047815, wavecal='wavecal0',
                                         flatcal='flatcal0', speccal='speccal0', use='0,2,4-9', wcscal='wcscal0'),
            config.MKIDDitherDescription(name=namer('dither'), data='dither.logfile', wavecal='wavecal0',
                                         flatcal='flatcal0', speccal='speccal0', use=(1,), wcscal='wcscal0'),
            config.MKIDDitherDescription(name=namer('dither'), flatcal='', speccal='', wcscal='', wavecal='',
                                         data=(
                                         config.MKIDObservation(name=namer('HIP109427_'), start=1602047815, duration=10,
                                                                wavecal='wavecal0', dither_pos=(0.2, 0.3),
                                                                dark=config.MKIDTimerange(name=namer(), start=350,
                                                                                          duration=10)),
                                         config.MKIDObservation(name=namer('HIP109427_'), start=1602047825, duration=10,
                                                                wavecal='wavecal0', dither_pos=(0.1, 0.1),
                                                                wcscal='wcscal0'),
                                         config.MKIDObservation(name=namer('HIP109427_'), start=1602047835, duration=10,
                                                                wavecal='wavecal0', dither_pos=(-0.1, -0.1),
                                                                wcscal='wcscal0')
                                         )
                                         )
            ]
    return data


def generate_sample_output():
    i = defaultdict(lambda: 0)

    def namer(name='Thing'):
        ret = f"{name}{i[name]}"
        i[name] = i[name] + 1
        return ret

    data = [config.MKIDOutput(name=namer('out'), data='dither0', min_wave='850 nm', max_wave='1375 nm',
                              kind='spatial', noise=True, photom=True, ssd=True)]
    return data


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


def safe(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            getLogger(__name__).critical('Caught exception during run of {}'.format(o), exc_info=True)
            return None

    return wrapper_decorator


def batch_applier(func, obs, ncpu=None, unique_h5=True):
    if unique_h5:
        obs = {o.h5: o for o in obs}.values()
    ncpu = ncpu if ncpu is not None else config.n_cpus_available()
    ncpu = min(ncpu, len(obs))
    if ncpu == 1:
        for o in obs:
            safe(func)(o)
    else:
        pool = mp.Pool()
        pool.map(safe(func), obs)
        pool.close()


def batch_build_hdf(dset, ncpu=None):
    batch_applier(mkidpipeline.steps.buildhdf.buildtables, dset.timeranges, ncpu=ncpu, unique_h5=False)


def batch_apply_metadata(dset):
    """Function associates things not known at hdf build time (e.g. that aren't in the bin files)"""
    for ob in dset.all_observations:
        o = photontable.Photontable(ob.h5, mode='w')
        o.attach_observing_metadata(ob.metadata)
        del o


def batch_apply_wavecals(dset, ncpu=None):
    wavecal.clear_solution_cache()
    ncpu = ncpu if ncpu is not None else config.n_cpus_available(config.config.wavecal.ncpu)
    batch_applier(wavecal_apply, dset.wavecalable, ncpu=ncpu)


def batch_apply_flatcals(dset, ncpu=None):
    batch_applier(mkidpipeline.steps.flatcal.apply, dset.flatcalable, ncpu=ncpu)


def batch_apply_pixcals(dset, ncpu=None):
    batch_applier(mkidpipeline.steps.pixcal.apply, dset.pixcalable, ncpu=ncpu)


def batch_apply_lincals(dset, ncpu=None):
    batch_applier(mkidpipeline.steps.lincal.apply, dset.lincalable, ncpu=ncpu)


def batch_apply_speccals(dset, ncpu=None):
    batch_applier(mkidpipeline.steps.speccal.apply, dset.speccalable, ncpu=ncpu)


def batch_apply_cosmiccals(dset, ncpu=None):
    batch_applier(mkidpipeline.steps.cosmiccal.apply, dset.cosmiccalable, ncpu=ncpu)


def run_stage1(dataset):
    operations = (('Building H5s', mkidpipeline.steps.buildhdf.buildtables),
                  ('Attaching metadata', batch_apply_metadata),
                  ('Finding Cosmic-rays', batch_apply_cosmiccals),
                  ('Fetching wavecals', mkidpipeline.steps.wavecal.fetch),
                  ('Applying linearity correction', batch_apply_lincals),
                  ('Applying wavelength solutions', batch_apply_wavecals),
                  ('Applying wavelength solutions', batch_apply_pixcals),
                  ('Fetching flatcals', mkidpipeline.steps.flatcal.fetch),
                  ('Applying flatcals', batch_apply_flatcals),
                  ('Fetching speccals', mkidpipeline.steps.speccal.fetch),
                  ('Applying speccals', batch_apply_speccals))

    toc = time.time()
    for task_name, task in operations:
        tic = time.time()
        getLogger(__name__).info(f'Stage 1: {task_name}')
        task(dataset)
        getLogger(__name__).info(f'Completed {task_name} in {time.time() - tic:.0f} s')

    getLogger(__name__).info(f'Stage 1 complete in {(time.time() - toc) / 60:.0f} m')


def generate_outputs(outputs: config.MKIDOutputCollection):
    mkidpipeline.steps.drizzler.fetch(outputs)

    for o in outputs:
        # TODO make into a batch process
        getLogger(__name__).info('Generating {}'.format(o.name))
        if o.wants_image:

            for obs in o.data.obs:
                h5 = mkidpipeline.photontable.Photontable(obs.h5)
                img = h5.get_fits(wave_start=o.min_wave, wave_stop=o.max_wave, spec_weight=o.photom,
                                  noise_weight=o.noise, rate=True)
                img.writeto(o.output_file)
                getLogger(__name__).info('Generated fits file for {}'.format(obs.h5))
        if o.wants_movie:
            getLogger('mkidpipeline.hdf.photontable').setLevel('DEBUG')
            mkidpipeline.imaging.movies.make_movie(o, inpainting=False)
