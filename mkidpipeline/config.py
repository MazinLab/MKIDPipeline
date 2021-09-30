import os
import multiprocessing as mp
import astropy.units as u
import ruamel.yaml.comments
import pathlib

from mkidpipeline.definitions import MKIDTimerange, MKIDObservation, MKIDWavecalDescription, MKIDFlatcalDescription, \
    MKIDSpeccalDescription, MKIDWCSCalDescription, MKIDDitherDescription, MKIDOutput
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


class UnassociatedError(RuntimeError):
    pass


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


class Key:
    """Class that defines a Key which consists of a name, default value, comment, and data type"""
    def __init__(self, name='', default=None, comment='', dtype=None):
        self.name = str(name)
        self.default = default
        self.comment = str(comment)
        self.dtype = dtype


class DataBase:
    """Superclass to handle all MKID data. Verifies and sets all required keys"""
    KEYS = tuple()
    REQUIRED = tuple()  # May set individual elements to tuples of keys if they are alternates e.g. stop/duration
    EXPLICIT_ALLOW = tuple()  # Set to names that are allowed keys and are also used as properties

    def __init__(self, *args, **kwargs):
        from collections import defaultdict
        self._key_errors = defaultdict(list)
        self._keys = {k.name: k for k in self.KEYS}
        self.extra_keys = []

        # Check disallowed
        for k in kwargs:
            if getattr(self, k, None) is not None and k not in self.EXPLICIT_ALLOW or k.startswith('_'):
                self._key_errors[k] += ['Not an allowed key']

        self.name = kwargs.get('name', f'Unnamed !{self.yaml_tag}')  # yaml_tag defined by subclass
        self.extra_keys = [k for k in kwargs if k not in self.key_names]

        # Check for the existence of all required keys (or key sets)
        for key_set in self.REQUIRED:
            if isinstance(key_set, str):
                key_set = (key_set,)
            found = 0
            for k in key_set:
                found += int(k in kwargs)
            if len(key_set) == 1:
                key_set = key_set[0]
            if not found:
                self._key_errors[key_set] += ['missing']
            elif found > 1:
                if not found:
                    self._key_errors[key_set] += ['multiple specified']

        # Process keys
        for k, v in kwargs.items():
            if k in self._keys:
                required_type = self._keys[k].dtype
                try:
                    required_type[0]
                except TypeError:
                    required_type = (required_type,)

                if tuple in required_type and isinstance(v, list):
                    v = tuple(v)
                if float in required_type and v is not None:  # and isinstance(v, str) and v.endswith('inf'):
                    try:
                        v = float(v)
                    except (ValueError, TypeError):
                        pass
                if required_type[0] is not None and not isinstance(v, required_type):
                    self._key_errors[k] += [f' {v} not an instance of {tuple(map(lambda x: x.__name__, required_type))}']

            if isinstance(v, str):
                try:
                    v = u.Quantity(v)
                except (TypeError, ValueError):
                    if v.startswith('_'):
                        raise ValueError(f'Keys may not start with an underscore: "{v}". Check {self.name}')
            try:
                setattr(self, k, v)
            except AttributeError:
                try:
                    setattr(self, '_' + k, v)
                    getLogger(__name__).debug(f'Storing {k} as _{k} for use by subclass')
                except AttributeError:
                    pass

        # Set defaults
        for key in (key for key in self.KEYS if key.name not in kwargs and key.name not in self.EXPLICIT_ALLOW):
            try:
                if key.default is None and key.dtype is not None:
                    default = key.dtype[0]() if isinstance(key.dtype, tuple) else key.dtype()
                else:
                    default = key.default
            except Exception:
                default = None
                getLogger(__name__).debug(f'Unable to create default instance of {key.dtype} for '
                                          f'{key.name}, using None')
            try:
                setattr(self, key.name, default)
            except Exception:
                getLogger(__name__).debug(f'Key {key.name} is shadowed by property, prepending _')
                setattr(self, '_' + key.name, default)

        # # Check types
        # for k:
        #     if key.dtype is not None:
        #         try:
        #             if not isinstance(getattr(self, key.name), key.dtype):
        #                 self._key_errors[key.name] += [f'not an instance of {key.dtype}']
        #         except AttributeError:
        #             pass

    def _vet(self):
        """Returns a copy of all of the key errors"""
        return self._key_errors.copy()

    def extra(self):
        """Returns a dictionary of the extra keys (keys not included in KEYS)"""
        return {k: getattr(self, k) for k in self.extra_keys}

    @classmethod
    def from_yaml(cls, loader, node):
        return cls(**dict(loader.construct_pairs(node, deep=True)))

    @classmethod
    def to_yaml(cls, representer, node, use_underscore=tuple()):
        d = node.__dict__.copy()
        for k in use_underscore:
            d[k] = d.pop(f"_{k}", d[k])
        # We want to write out all the keys needed to recreate the definition
        #  keys that are explicitly allowed are used in __init__ to support dual definition (e.g. stop/duration)
        #  we exclude th to prevent redundancy
        #  we want to include any user defined keys
        keys = [k for k in node._keys if k not in cls.EXPLICIT_ALLOW] + d.pop('extra_keys')
        store = {}
        for k in keys:
            if type(d[k]) not in representer.yaml_representers:
                if not isinstance(d[k], u.Quantity):
                    getLogger(__name__).debug(f'{node.name} ({cls.__name__}.{k}) is a {type(d[k])} and '
                                              f'will be cast to string ({str(d[k])}) for yaml representation ')
                store[k] = str(d[k])
            else:
                # getLogger(__name__).debug(f'{node.name} ({cls.__name__}.{k}) is a {type(d[k])} and '
                #                           f'will be stored as ({d[k]}) for yaml representation ')
                store[k] = d[k]
        if 'header' in store:
            cm = ruamel.yaml.comments.CommentedMap(store['header'])
            for k in store['header']:
                try:
                    descr = mkidcore.metadata.MEC_KEY_INFO[k].description
                except KeyError:
                    descr = '!UNKNOWN MEC HEADER KEY!'
                cm.yaml_add_eol_comment(descr, key=k)
            store['header'] = cm
        cm = ruamel.yaml.comments.CommentedMap(store)
        for k in store:
            cm.yaml_add_eol_comment(node._keys[k].comment if k in node._keys else 'User added key', key=k)
        return representer.represent_mapping(cls.yaml_tag, cm)

    @property
    def key_names(self):
        """Convenience method for returning all of the names fo the KEYS"""
        return tuple([k.name for k in self.KEYS])


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


yaml.register_class(MKIDTimerange)
yaml.register_class(MKIDObservation)
yaml.register_class(MKIDWavecalDescription)
yaml.register_class(MKIDFlatcalDescription)
yaml.register_class(MKIDSpeccalDescription)
yaml.register_class(MKIDWCSCalDescription)
yaml.register_class(MKIDDitherDescription)
yaml.register_class(MKIDOutput)
