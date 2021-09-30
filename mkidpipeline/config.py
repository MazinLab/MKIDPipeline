import os
import multiprocessing as mp
import pathlib

import mkidcore.config
from mkidcore.corelog import getLogger
from mkidcore.objects import Beammap
from mkidcore.instruments import InstrumentInfo
from mkidcore.metadata import DEFAULT_CARDSET

# Ensure that the beammap gets registered with yaml, the import does this
# but without this note an IDE or human might remove the import
Beammap()

config = None
_dataset = None
_metadata = {}

yaml = mkidcore.config.yaml


def dump_dataconfig(data, file):
    """
    writes data to the yaml file
    :param data: tuple containing all of the data. Can be of type MKIDObservation, MKIDWavecalDescription,
    MKIDFlatcalDescription, MKIDSpeccalDescription, MKIDWCSCalDescription, or MKIDDitherDescription
    :param file: yaml file to which to write the data
    :return:
    """
    with open(file, 'w') as f:
        mkidcore.config.yaml.dump(data, f)
    # patch bug in yaml export
    with open(file, 'r') as f:
        lines = f.readlines()
    for l in (l for l in lines if ' - !' in l and ':' in l):
        x = list(l.partition(l.partition(':')[0].split()[-1] + ':'))
        x.insert(1, '\n' + ' ' * l.index('!'))
        lines[lines.index(l)] = ''.join(x)
    with open(file, 'w') as f:
        f.writelines(lines)


# Note that in contrast to the Keys or DataBase these don't work quite the same way
# required keys specify items that the resulting object is required to have, not that use
# user is required to pass, they are
class BaseStepConfig(mkidcore.config.ConfigThing):
    """
    registers and verifies the fields of the pipeline config related to a apecific calibration step
    """
    REQUIRED_KEYS = tuple()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v, c in self.REQUIRED_KEYS:
            self.register(k, v, comment=c, update=False)

    @classmethod
    def from_yaml(cls, loader, node):
        """
        wrapper for mkidcore.config.ConfigThing.from_yaml. Verifies attributes and errors in the mkidpipeline
        config
        """
        ret = super().from_yaml(loader, node)
        errors = ret._verify_attribues() + ret._vet_errors()

        if errors:
            raise ValueError(f'{ret.yaml_tag} collected errors: \n' + '\n\t'.join(errors))
        return ret

    def _verify_attribues(self):
        """Returns a list missing keys from the pipeline config"""
        missing = [key for key, default, comment in self.REQUIRED_KEYS if key not in self]
        return ['Missing required keys: ' + ', '.join(missing)] if missing else []

    def _vet_errors(self):
        """Returns a list of errors found in the pipeline config"""
        return []


_pathroot = os.path.join('/work', os.environ.get('USER', ''))


class PipeConfig(BaseStepConfig):
    """Populates the fields of the pipeline config not related to a specific pipeline step"""
    yaml_tag = u'!pipe_cfg'
    REQUIRED_KEYS = (('ncpu', 1, 'number of cpus'),
                     ('verbosity', 0, 'level of verbosity'),
                     ('flow', ('metadata', 'wavecal', 'pixcal', 'flatcal', 'cosmiccal', 'speccal'),
                      'Calibration steps to apply'),
                     ('paths.data', '/darkdata/ScienceData/Subaru/',
                      'bin file parent folder, must contain YYYYMMDD/*.bin and YYYYMMDD/logs/'),
                     ('paths.database', os.path.join(_pathroot, 'database'),
                      'calibrations will be retrieved/stored here'),
                     ('paths.out', os.path.join(_pathroot, 'out'), 'root of output'),
                     ('paths.tmp', os.path.join(_pathroot, 'scratch'), 'use for data intensive temp files'),
                     ('beammap', None, 'A Beammap to use'),
                     ('instrument', None, 'An mkidcore.instruments.InstrumentInfo instance')
                     )

    def __init__(self, *args, defaults: dict = None, instrument='MEC', **kwargs):
        super().__init__(*args, **kwargs)
        if self.beammap is None:
            self.register('beammap', Beammap(specifier=instrument), update=True)
        self.register('instrument', InstrumentInfo(instrument), update=True)
        if defaults is not None:
            for k, v in defaults.items():
                self.register(k, v, update=True)


mkidcore.config.yaml.register_class(PipeConfig)


def PipelineConfigFactory(step_defaults: dict = None, cfg=None, ncpu=None, copy=True):
    """
    Return a pipeline config with the specified step.
    cfg will take precedence over an existing pipeline config
    ncpu will take precedence (at the root level only so if a step has defaults those will control for the step!)
    the step defaults will only be used if the step is not configured
    if copy is set is returned such that it is safe to edit, if not set any defaults will be updated
    into cfg (if passed) or the global config (if extant)
    """
    global config
    if cfg is None:
        cfg = PipeConfig(instrument='MEC') if config is None else config
    if copy:
        cfg = cfg.copy()
    if step_defaults:
        for name, defaults in step_defaults.items():
            # NB this will add in NCPU for steps that set it in their defaults, overriding
            # the inherited default
            cfg.register(name, defaults, update=False)
    if ncpu is not None:
        config.update('ncpu', ncpu)
    return cfg


def configure_pipeline(pipeline_config):
    """Load a pipeline config, configuring the pipeline. Any existing configuration will be replaced"""
    global config
    config = mkidcore.config.load(pipeline_config, namespace=None)
    return config


def update_paths(d):
    """updates the path fields in the pipeline config"""
    global config
    for k, v in d.items():
        config.update(f'paths.{k}', v)


def get_paths(config=None, output_collection=None):
    """Returns a set of all the required paths from the pipeline config"""
    if config is None:
        config = globals()['config']
    output_dirs = [] if output_collection is None else [os.path.dirname(o.filename) for o in output_collection]
    return set([config.paths.out, config.paths.database, config.paths.tmp] + list(output_dirs))


def verify_paths(config=None, output_collection=None, return_missing=False):
    """
    If return_missing=True, returns a list of all the missing paths from the pipeline config. If return_missing=False
    then returns True if there are paths missing, and False if all paths are present.
    """
    paths = get_paths(config=config, output_collection=output_collection)
    missing = list(filter(lambda p: p and not os.path.exists(p), paths))
    return missing if return_missing else not bool(missing)


def make_paths(config=None, output_collection=None):
    """Creates all paths returned from get_paths that do not already exist."""
    paths = get_paths(config=config, output_collection=output_collection)

    for p in filter(os.path.exists, paths):
        getLogger(__name__).info(f'"{p}" exists, and will be used.')

    for p in filter(lambda p: p and not os.path.exists(p), paths):
        getLogger(__name__).info(f'Creating "{p}"')
        pathlib.Path(p).mkdir(parents=True, exist_ok=True)


class H5Subset:
    """
    Defines a set of new h5 files that are of the same or shorter duration than a set of existing h5 files.

    For example, one can define a set of h5 files that all begin one second after currently defined h5 files, allowing
    for different calibration steps to be applied to (essentially) the same data.

    This is notably used for laser flats where an H5Subset of the wavecal laser h5s is desired.
    """
    def __init__(self, timerange, duration=None, start=None, relative=False):
        """ if relative the start is taken as an offset relative to the timerange """
        self.timerange = timerange
        self.h5start = int(timerange.start)
        if relative and start is not None:
            start = float(start) + float(self.h5start)
        self.start = float(self.h5start) if start is None else float(start)
        self.duration = timerange.duration if duration is None else float(duration)

    @property
    def photontable(self):
        """
        Convenience method for a photontable, file must exist, creates a new photon table on every call,

        Not for use in multithreaded/pool situation where enablewrite may be called
        """
        from mkidpipeline.photontable import Photontable
        return Photontable(self.timerange.h5)

    @property
    def first_second(self):
        """Convenience method for finding the first second of the new h5 files"""
        return self.start - self.h5start

    def __str__(self):
        return f'{os.path.basename(self.timerange.h5)} @ {self.start} for {self.duration}s'


def inspect_database(detailed=False):
    """Warning detailed=True will load each thing in the database for detailed inspection"""
    from glob import glob

    for f in glob(config.config.paths.database + '*'):
        print(f'{f}')


def n_cpus_available(max=1):
    """Returns n threads -4 modulo pipelinesettings"""
    global config
    if max is None:
        try:
            max = int(config.ncpu)
        except Exception:
            max = 1
    mcpu = min(mp.cpu_count() * 2 - 4, max)
    # try:
    #     mcpu = int(min(config.ncpu, mcpu))
    # except Exception:
    #     pass
    return mcpu
