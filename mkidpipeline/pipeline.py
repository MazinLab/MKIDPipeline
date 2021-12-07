from importlib import import_module
import pkgutil
import multiprocessing as mp
import functools
import mkidcore.config
from mkidcore.pixelflags import FlagSet, BEAMMAP_FLAGS
from mkidcore.config import getLogger

import mkidpipeline.config as config
import mkidpipeline.steps


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


def _safe(func):
    @functools.wraps(func)
    def wrapper_decorator(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception:
            getLogger(__name__).critical(f'Caught exception during run of {func.__module__}.{func.__name__}',
                                         exc_info=True)
            raise

    return wrapper_decorator


def _batch_apply_metadata(dset):
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


def batch_applier(step, obs, ncpu=None, unique_h5=True):
    if step == 'attachmeta':
        _batch_apply_metadata(obs)
        return
    if step == 'buildhdf':
        PIPELINE_STEPS['buildhdf'].buildtables(obs.input_timeranges, ncpu=ncpu)
        return

    func = PIPELINE_STEPS[step].apply

    if unique_h5:
        obs = {o.h5: o for o in obs}.values()
    else:
        obs = list(obs)

    if not len(obs):
        return

    ncpu = min(config.n_cpus_available(max=ncpu), len(obs))
    if ncpu == 1:
        for o in obs:
            _safe(func)(o)
    else:
        pool = mp.Pool(processes=ncpu)
        pool.map(func, set(obs))
        pool.close()
