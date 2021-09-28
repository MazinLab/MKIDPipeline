from importlib import import_module
import pkgutil
import multiprocessing as mp
import time
import functools
import numpy as np
import mkidcore.config
from mkidcore.pixelflags import FlagSet, BEAMMAP_FLAGS
from mkidcore.config import getLogger

import mkidpipeline
import mkidpipeline.config as config
import mkidpipeline.steps
from mkidpipeline.steps import wavecal
import mkidpipeline.steps.buildhdf
import mkidpipeline.steps.movies
import mkidpipeline.steps.drizzler


log = getLogger(__name__)

PIPELINE_STEPS = {}
for info in pkgutil.iter_modules(mkidpipeline.steps.__path__):
    if info.name == 'sample':
        continue
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

PROBLEM_FLAGS = ('pixcal.hot', 'pixcal.cold', 'pixcal.dead', 'beammap.noDacTone', 'wavecal.bad',
                 'wavecal.failed_validation', 'wavecal.failed_convergence', 'wavecal.not_monotonic',
                 'wavecal.not_enough_histogram_fits', 'wavecal.no_histograms',
                 'wavecal.not_attempted')


def generate_default_config(instrument='MEC'):
    cfg = config.PipeConfig(instrument=instrument)
    for name, step in PIPELINE_STEPS.items():
        try:
            if step.StepConfig is None:
                getLogger(__name__).debug(f'Pipeline step mkidpipeline.steps.{name} has no global settings.')
                continue
            cfg.register(name, step.StepConfig(), update=True)
        except AttributeError:
            getLogger(__name__).warning(f'Pipeline step mkidpipeline.steps.{name} does not '
                                        f'support automatic configuration discovery.')
    return cfg


def safe(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            getLogger(__name__).critical(f'Caught exception during run of {func.__module__}.{func.__name__}',
                                         exc_info=True)
            raise

    return wrapper_decorator


def batch_applier(func, obs, ncpu=None, unique_h5=True):
    if unique_h5:
        obs = {o.h5: o for o in obs}.values()

    if not len(obs):
        return

    ncpu = min(config.n_cpus_available(max=ncpu), len(obs))
    if ncpu == 1:
        for o in obs:
            safe(func)(o)
    else:
        pool = mp.Pool(processes=ncpu)
        pool.map(func, set(obs))
        pool.close()


def batch_build_hdf(dset, ncpu=None):
    batch_applier(mkidpipeline.steps.buildhdf.buildtables, dset.timeranges, ncpu=ncpu, unique_h5=False)


def batch_apply_metadata(dset):
    """Function associates things not known at hdf build time (e.g. that aren't in the bin files)"""
    timeranges = dset.input_timeranges
    data = {tr.h5: tr for tr in timeranges}
    if len(data) != len(timeranges):
        getLogger(__name__).warning(f'Timeranges are not all backed by unique h5 files, {len(timeranges)-len(data)} '
                                    "will be superseded by another timerange's metadata.")
    for tr in data.values():
        o = tr.photontable
        o.enablewrite()
        o.attach_observing_metadata(tr.metadata)
        o.disablewrite()


def batch_apply_wavecals(dset, ncpu=None):
    wavecal.clear_solution_cache()
    batch_applier(mkidpipeline.steps.wavecal.apply, dset.wavecalable, ncpu=ncpu)


def batch_apply_flatcals(dset, ncpu=None):
    batch_applier(mkidpipeline.steps.flatcal.apply, dset.flatcalable, ncpu=ncpu)


def batch_apply_pixcals(dset, ncpu=None):
    batch_applier(mkidpipeline.steps.pixcal.apply, dset.pixcalable, ncpu=ncpu)


def batch_apply_lincals(dset, ncpu=None):
    batch_applier(mkidpipeline.steps.lincal.apply, dset.lincalable, ncpu=ncpu)


def batch_apply_cosmiccals(dset, ncpu=None):
    batch_applier(mkidpipeline.steps.cosmiccal.apply, dset.cosmiccalable, ncpu=ncpu)


def run_stage1(dataset):
    operations = (('Building H5s', mkidpipeline.steps.buildhdf.buildtables),
                  ('Attaching metadata', batch_apply_metadata),
                  ('Fetching wavecals', mkidpipeline.steps.wavecal.fetch),
                  ('Finding Cosmic-rays', batch_apply_cosmiccals),
                  ('Applying linearity correction', batch_apply_lincals),
                  ('Applying wavelength solutions', batch_apply_wavecals),
                  ('Applying pixel masks', batch_apply_pixcals),
                  ('Fetching flatcals', mkidpipeline.steps.flatcal.fetch),
                  ('Applying flatcals', batch_apply_flatcals),
                  ('Fetching speccals', mkidpipeline.steps.speccal.fetch)
                  )

    toc = time.time()
    for task_name, task in operations:
        tic = time.time()
        getLogger(__name__).info(f'Stage 1: {task_name}')
        task(dataset)
        getLogger(__name__).info(f'Completed {task_name} in {time.time() - tic:.0f} s')

    getLogger(__name__).info(f'Stage 1 complete in {(time.time() - toc) / 60:.0f} m')
