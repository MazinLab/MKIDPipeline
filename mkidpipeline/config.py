import astropy.units.core
import numpy as np
import os
from glob import glob
import hashlib
from datetime import datetime
import multiprocessing as mp
import pkg_resources as pkg
import json
import astropy.units as u
from astropy.coordinates import SkyCoord
import ruamel.yaml.comments

from mkidcore.utils import parse_ditherlog
from mkidcore.legacy import parse_dither_log
import mkidcore.config
from mkidcore.corelog import getLogger, create_log, MakeFileHandler
from mkidcore.utils import getnm, derangify
from mkidcore.objects import Beammap


# Ensure that the beammap gets registered with yaml, the import does this
# but without this note an IDE or human might remove the import
Beammap()

config = None
_dataset = None
_parsed_dither_logs = {}

yaml = mkidcore.config.yaml


STANDARD_KEYS = (
'ra', 'dec', 'airmass', 'az', 'el', 'ha', 'equinox', 'parallactic', 'target', 'utctcs', 'laser', 'flipper',
'filter', 'observatory', 'utc', 'comment', 'device_orientation', 'instrument', 'dither_ref', 'dither_home',
'dither_pos', 'platescale')

REQUIRED_KEYS = ('ra', 'dec', 'target', 'observatory', 'instrument', 'dither_ref', 'dither_home', 'platescale',
                 'device_orientation')


def load_task_config(file, use_global_config=True):
    """
    Load a task specific yml configuration

    If the pipeline is not configured then do all needed to get it online,
    loading defaults and overwriting them with the task config. If pipeline has been
    configured by user then there is a choice of which settings take precedence (pipeline or task
    via use_global_config), thought the config will be updated with any additional pipeline
    settings. Will never edit an existing pipeline config.

    :param file: Config file (or config object) to load
    :param use_global_config: config/pipe precedence
    :return:
    """
    global config

    # Allow pass-through of a config
    cfg = mkidcore.config.load(file) if isinstance(file, str) else file
    pipeline_settings = ('beammap', 'paths', 'instrument', 'ncpu')
    if config is None:
        configure_pipeline(pkg.resource_filename('mkidpipeline', 'pipe.yml'))
        for k in pipeline_settings:
            try:
                config.update(k, cfg.get(k))
            except KeyError:
                pass

    for k in pipeline_settings:
        cfg.register(k, config.get(k), update=use_global_config)

    return cfg


def configure_pipeline(pipeline_config):
    """ Load a pipeline config, configuring the pipeline. Any existing configuration will be replaced"""
    global config
    config = mkidcore.config.load(pipeline_config, namespace=None)
    return config


def update_paths(d):
    global config
    for k, v in d.items():
        config.update(f'paths.{k}', v)


def get_ditherinfo(time):
    global _parsed_dither_logs
    if not _parsed_dither_logs:
        for f in glob(os.path.join(config.paths.dithers, 'dither_*.log')):
            parsed_log = parse_ditherlog(f)
            _parsed_dither_logs.update(parsed_log)

    if isinstance(time, datetime):
        time = time.timestamp()

    for (t0, t1), v in _parsed_dither_logs.items():
        if t0 - (t1 - t0) <= time <= t1:
            return v
    raise ValueError('No dither found for time {}'.format(time))


class BaseStepConfig(mkidcore.config.ConfigThing):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for k, v, c in self.REQUIRED_KEYS:
            self.register(k, v, comment=c, update=False)

    @classmethod
    def from_yaml(cls, loader, node):
        ret = super().from_yaml(loader, node)
        errors = ret._verify_attribues() + ret._vet_errors()

        if errors:
            raise ValueError(f'{ret.yaml_tag} collected errors: \n' + '\n\t'.join(errors))
        return ret

    def _verify_attribues(self):
        missing = [key for key, default, comment in self.REQUIRED_KEYS if key not in self]
        return ['Missing required keys: ' + ', '.join(missing)] if missing else []

    def _vet_errors(self):
        return []


class PipeConfig(BaseStepConfig):
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


mkidcore.config.yaml.register_class(PipeConfig)


def make_paths(config=None, output_dirs=tuple()):
    if config is None:
        config = globals()['config']

    paths = set([config.paths.out, config.paths.database, config.paths.tmp] + list(output_dirs))

    for p in filter(os.path.exists, paths):
        getLogger(__name__).info(f'"{p}" exists, and will be used.')

    for p in filter(lambda p: not os.path.exists(p), paths):
        if not p:
            continue
        getLogger(__name__).info(f'Creating "{p}"')
        os.makedirs(p, exist_ok=True)


class H5Subset:
    def __init__(self, timerange, duration=None, start=None, relative=False):
        """if relative the start is taken as an offset relative to the timerange"""
        self.timerange = timerange
        self.h5start = int(timerange.start)
        if relative and start is not None:
            start = float(start) + float(self.h5start)
        self.start = float(self.h5start) if start is None else float(start)
        self.duration = timerange.duration if duration is None else float(duration)

    @property
    def photontable(self):
        from photontable import Photontable
        return Photontable(self.timerange.h5)

    @property
    def first_second(self):
        return self.start - self.h5start

    def __str__(self):
        return f'{os.path.basename(self.timerange.h5)} @ {self.start} for {self.duration}s'


class Key:
    def __init__(self, name='', default=None, comment='', dtype=None):
        self.name = str(name)
        self.default = default
        self.comment = str(comment)
        self.dtype = dtype


class DataBase:
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
                key_set=key_set[0]
            if not found:
                self._key_errors[key_set] += ['missing']
            elif found > 1:
                if not found:
                    self._key_errors[key_set] += ['multiple specified']

        # Process keys
        for k, v in kwargs.items():
            if k in self._keys:
                required_type = self._keys[k].dtype
                if required_type == tuple and isinstance(v, list):
                    v = tuple(v)
                if required_type == float and isinstance(v, str) and v.endswith('inf'):
                    try:
                        v = float(v)
                    except ValueError:
                        pass
                if required_type is not None and not isinstance(v, required_type):
                    self._key_errors[k] += [f'not an instance of {required_type}']

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
                    setattr(self, '_'+k, v)
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
                setattr(self, '_'+key.name, default)

        # # Check types
        # for k:
        #     if key.dtype is not None:
        #         try:
        #             if not isinstance(getattr(self, key.name), key.dtype):
        #                 self._key_errors[key.name] += [f'not an instance of {key.dtype}']
        #         except AttributeError:
        #             pass

    @property
    def key_names(self):
        return tuple([k.name for k in self.KEYS])

    def _vet(self):
        def joiner(x):
            return ', '.join(x)
        errors = [f'{k}:{joiner(v)}' for k, v in self._key_errors.items()]
        return f"{type(self).__name__}: {errors}" if errors else ''

    def extra(self):
        return {k: getattr(self, k) for k in self.extra_keys}

    @classmethod
    def from_yaml(cls, loader, node):
        return cls(**dict(loader.construct_pairs(node, deep=True)))

    @classmethod
    def to_yaml(cls, representer, node):
        d = node.__dict__.copy()
        d.pop('extra_keys')
        for k in list(d.keys()):
            if k.startswith('_'):
                d.pop(k)
            elif type(d[k]) not in representer.yaml_representers:
                getLogger(__name__).debug(f'{node.name} ({cls.__name__}.{k}) is a {type(d[k])} and '
                                          f'will be cast to string ({str(d[k])}) for yaml representation ')
                d[k] = str(d[k])

        cm = ruamel.yaml.comments.CommentedMap(d)
        for k in d:
            cm.yaml_add_eol_comment(node._keys[k].comment if k in node._keys else 'User added key', key=k)
        return representer.represent_mapping(cls.yaml_tag, cm)


class MKIDTimerange(DataBase):
    yaml_tag = u'!MKIDTimerange'
    KEYS = (
        Key(name='name', default=None, comment='A name', dtype=str),
        Key('start', None, 'The start unix time, float ok, rounded down for H5 creation.', (float, int)),
        Key('duration', None, 'A duration in seconds, float ok. If not specified stop must be', (float,int)),
        Key('stop', None, 'A stop unit time, float ok. If not specified duration must be', (float,int)),
        Key('dark', None, 'An MKIDTimerange to use for a dark reference.', None)
    )
    REQUIRED = ('name', 'start', ('duration', 'stop'))
    EXPLICIT_ALLOW = ('duration',)  # if a key is allows AND is a property or method name it must be listed here

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self,'_duration'):
            self.stop = self.start + self._duration

    def __str__(self):
        tn = type(self).split('.')[-1]
        return f'{self.name} ({tn}): {self.duration}s @ {self.start}'

    def _vet(self):
        if self.duration > 43200:
            getLogger(__name__).warning(f'Duration of {self.name} longer than 12h!')
        if self.stop < self.start:
            self._key_errors['stop'] += [f'Stop ({self.stop}) must come after start ({self.start})']
        return super()._vet()

    @property
    def date(self):
        return datetime.utcfromtimestamp(self.start)

    # @property
    # def platescale(self):
    #     return config.instrument.nominal_platescale_mas

    @property
    def beammap(self):
        return config.beammap

    @property
    def duration(self):
        return self.stop - self.start

    @property
    def timerange(self):
        return self.start, self.stop

    @property
    def timeranges(self):
        yield self.timerange
        if self.dark is not None:
            yield self.dark.timerange

    @property
    def h5(self):
        return os.path.join(config.paths.out, '{}.h5'.format(int(self.start)))

    @property
    def photontable(self):
        """Convenience method for a photontable, file must exist, creates a new photon table on every call"""
        from photontable import Photontable
        return Photontable(self.h5)


class MKIDObservation(MKIDTimerange):
    """requires keys name, wavecal, flatcal, wcscal, and all the things from ob"""
    yaml_tag = u'!MKIDObservation'
    KEYS = MKIDTimerange.KEYS + (
        Key('wavecal', None, 'A MKIDWavedata or name of the same', None),
        Key('flatcal', None, 'A MKIDFlatdata or name of the same', None),
        Key('wcscal', None, 'A MKIDWCSCal or name of the same', None),
        Key('speccal', None, 'A MKIDSpecdata or name of the same', None),
    )
    REQUIRED = MKIDTimerange.REQUIRED+('wavecal', 'flatcal', 'wcscal', 'speccal')
    EXPLICIT_ALLOW = MKIDTimerange.EXPLICIT_ALLOW
    OPTIONAL = ('standard', 'conex_pos')

    @property
    def metadata(self):
        exclude = ('wavecal', 'flatcal', 'wcscal', 'speccal', 'start', 'stop')
        d = {k: v for k, v in self.__dict__.items() if k not in exclude}
        try:
            wc = self.wavecal.id
        except AttributeError:
            wc = 'None'
        try:
            fc = self.flatcal.id
        except AttributeError:
            fc = 'None'
        try:
            sc = self.speccal.id
        except AttributeError:
            sc = 'None'
        # TODO make this play nice with fits headers and the like
        d2 = dict(wavecal=wc, flatcal=fc, speccal=sc, platescale=self.wcscal.platescale,
                  dither_ref=self.wcscal.dither_ref, dither_home=self.wcscal.dither_home,
                  device_orientation=self.wcscal.device_orientation)
        d.update(d2)
        return d

    @property
    def obs(self):
        yield self


class CalDefinitionMixin:
    @property
    def path(self):
        return os.path.join(config.paths.database, self.id + '.npz')

    def id(self, cfg=None):
        """
        Compute a wavecal id string from a wavedata id string and either the active or a specified wavecal config
        """
        id = str(self) + '_' + hashlib.md5(str(self).encode()).hexdigest()[-8:]
        if cfg is None:
            global config
            cfg = config.get(self.STEPNAME)
        config_hash = hashlib.md5(str(cfg).encode()).hexdigest()
        return f'{self.STEPNAME}_{id}_{config_hash[-8:]}'

    @property
    def timeranges(self):
        for o in self.obs:
            for tr in o.timeranges:
                yield tr


class MKIDWavecalDescription(DataBase, CalDefinitionMixin):
    """requires keys name and data"""
    yaml_tag = u'!MKIDWavecalDescription'
    KEYS = (
        Key(name='name', default='', comment='A name', dtype=str),
        Key('obs', None, 'List of MKIDTimerange named like 950 nm', tuple),
    )
    REQUIRED = ('name', 'obs')
    STEPNAME = 'wavecal'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for d in self.obs:
            if not isinstance(d, MKIDTimerange):
                self._key_errors['obs'] += [f'Element {d} of obs is not an MKIDTimerange']
        if not self.obs:
            self._key_errors['obs'] += ['obs must be a list of MKIDTimerange']

    def __str__(self):
        start = min(x.start for x in self.data)
        stop = min(x.stop for x in self.data)
        date = datetime.utcfromtimestamp(start).strftime('%Y-%m-%d-%H%M_')
        return f'{self.name} (MKIDWavedata): {start}-{stop}\n' + '\n '.join(str(x) for x in self.obs)

    @property
    def wavelengths(self):
        return tuple([getnm(x.name) for x in self.obs])

    @property
    def darks(self):
        return {w: ob.dark for w, ob in zip(self.wavelengths, self.obs)}


class MKIDFlatcalDescription(DataBase, CalDefinitionMixin):
    yaml_tag = u'!MKIDFlatcalDescription'
    KEYS = (
        Key(name='name', default=None, comment='A name', dtype=str),
        Key('ob', None, 'An MKIDTimerange to use for a whitelight flat. If not '
                        'specified wavecal will be used for a laser flat', None),
        Key('wavecal_duration', None, 'Number of seconds of the wavecal to use, float ok. Required if not ob', float),
        Key('wavecal_offset', None, 'An offset in seconds (>=1) from the start of the wavecal '
                                    'timerange. Required if not ob', int),
        Key('wavecal', None, 'A MKIDWavedata or name of the same', None),
    )
    REQUIRED = ('name', 'wavecal',)
    STEPNAME = 'flatcal'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.ob is None:
            try:
                if self.wavecal_offset < 1:
                    self._key_errors['wavecal_offset'] += ['must be >= 1s']
            except AttributeError:
                self._key_errors['wavecal_offset'] += ['required for a wavecal flat (i.e. no ob specified)']
            except TypeError:
                pass  # covered by super init
            try:
                if self.wavecal_duration < 1:
                    self._key_errors['wavecal_duration'] += ['must be >= 1s']
            except AttributeError:
                self._key_errors['wavecal_duration'] += ['required for a wavecal flat (i.e. no ob specified)']
            except TypeError:
                pass  # covered by super init
        else:
            if not isinstance(self.ob, MKIDTimerange):
                self._key_errors['ob'] += ['must be an MKIDTimerange']
            # if hasattr(self, 'wavecal_offset'):
            #     self._key_errors['wavecal_offset'] += ['not allowed for a whitelight flat (i.e. no specified ob)']
            # if hasattr(self, 'wavecal_duration'):
            #     self._key_errors['wavecal_duration'] += ['not allowed for a whitelight flat (i.e. no specified ob)']

        if not isinstance(getattr(self,'wavecal', None), (MKIDWavecalDescription, str)):
            self._key_errors['wavecal'] += ['must be an MKIDWavecalDescription or name of the same']

    def __str__(self):
        return '{}: {}'.format(self.name, self.ob.start if self.ob is not None else self.wavecal.id)

    @property
    def method(self):
        return 'laser' if self.ob is None else 'white'

    @property
    def h5s(self):
        """Returns MKIDObservations for the wavelengths of the wavecal, will raise errors for whitelight flats"""
        if self.ob is not None:
            raise NotImplementedError('h5s only available for laser flats')
        return {w: ob for w, ob in zip(self.wavecal.wavelengths, self.obs)}

    @property
    def obs(self):
        if self.ob is not None:
            yield self.ob
        else:
            for ob in self.wavecal.obs:
                o = MKIDObservation(f'{self.name}_{ob.name}', ob.start + self.wavecal_offset,
                                    duration=min(self.wavecal_duration, ob.duration - self.wavecal_offset),
                                    dark=ob.dark, wavecal=self.wavecal, **ob.extra())
                yield o


class MKIDSpeccalDescription(DataBase, CalDefinitionMixin):
    yaml_tag = u'!MKIDSpeccalDescription'
    KEYS = (
        Key(name='name', default=None, comment='A name', dtype=str),
        Key('obs', None, 'MKIDObservation or MKIDDither', None),
        Key('aperture', 'satellite', 'A 3-tuple (x/RA, y/Dec, r) or "satellite"', None),
    )
    REQUIRED = ('name', 'obs', 'aperture')
    STEPNAME = 'speccal'

    def __init__(self, *args, **kwargs):
        self.aperture_info=None
        super().__init__(*args, **kwargs)
        if isinstance(self.aperture, str):
            if self.aperture != 'satellite':
                self._key_errors['aperture'] += ['satellite is the only acceptable string']
        else:
            try:
                if len(self.aperture) != 3:
                    raise IndexError
                try:
                    self._aperture_info = tuple(map(float, self.aperture))
                except ValueError:
                    self._aperture_info = (SkyCoord(self.aperture[0], self.aperture[1]), u.Quantity(self.aperture[2]))
            except (TypeError, ValueError, IndexError) as e:
                getLogger(__name__).debug(f'Conversion of {self.aperture} failed: {e}')
                self._key_errors['aperture'] += ['3-tuple must in the form of (x/RA, y/Dec, radius) and '
                                                 'be parsable by float or SkyCoord+Quantity']


class MKIDWCSCalDescription(DataBase, CalDefinitionMixin):
    """
    The MKIDWCSCalDescription defines the coordinate relation between

    Keys are
    name - required

    Either:
    ob - The name of nn MKIDObservation from whitch to extract platescale dirter_ref, and dither_home. Presently unsupported
    Or:
    platescale - float (the platescale in mas, though note that TODO is the authoratative def. on units)
    dither_ref - 2 tuple (dither controller position for dither_hope)
    dither_home - 2 tuple (pixel position of optical axis at dither_ref)
    """
    yaml_tag = '!MKIDWCSCalDescription'
    KEYS = (
        Key(name='name', default=None, comment='A name', dtype=str),
        Key('ob', None, 'MKIDObservation or MKIDDither', None),
        Key('platescale', None, 'How may mas/pixel ', str),
        Key('dither_home', None, 'The pixel position of the target centroid when on '
                                'axis and the conex is at dither_home', tuple),
        Key('dither_ref', None, 'The conex (x,y) position, [0, 1.0], when the target is at dither_ref ', tuple),
    )
    REQUIRED = ('name', ('ob', 'platescale'),)
    STEPNAME = 'wcscal'

    def __init__(self, *args, **kwargs):
        super(MKIDWCSCalDescription, self).__init__(*args, **kwargs)
        if self.ob is None and self.dither_home is None:
            self._key_errors['dither_ref'] += ['must be an (x,y) position for the central source at dither_home']
        if self.ob is None and self.dither_ref is None:
            self._key_errors['dither_home'] += ['must be a conex (x,y) position when the target is at dither_ref']

        if self.dither_ref is not None:
            try:
                assert (len(self.dither_ref) == 2 and
                        0 <= self.dither_ref[0] < 1.0 and
                        0 <= self.dither_ref[1] < 1.0)
            except Exception:
                self._key_errors['dither_ref'] += ['must be a valid conex position (x,y), x & y in [0,1.0]']

        if self.dither_home is not None:
            try:
                assert len(self.dither_home) == 2
                if config is None or config.beammap is None:
                    getLogger(__name__).debug(f'Beammap not configured not checking dither_home validity')
                else:
                     assert (0 <= self.dither_home[0] < config.beammap.ncols and
                             0 <= self.dither_home[1] < config.beammap.nrows)
            except (TypeError, AssertionError):
                getLogger(__name__).debug(f'Dither home {self.dither_home} not in beammap '
                                          f'domain {config.beammap.ncols},{config.beammap.nrows}')
                self._key_errors['dither_home'] += ['must be a valid pixel (x,y) position']

        if self.platescale is not None:
            try:
                self.platescale.to('arcsec')
            except Exception:
                self._key_errors['platescale'] += ['must be a valid angular unit e.g. "10 mas"']

    @property
    def obs(self):
        yield self.ob


class MKIDDitherDescription(DataBase):
    yaml_tag = '!MKIDDitherDescription'
    KEYS = (
        Key(name='name', default=None, comment='A name', dtype=str),
        Key('data', tuple(), 'A list of !sob composing the dither, a unix time that falls within the range of a '
                          'dither in a dither log in paths.dithers, or a legacy (starttimes, endtimes, xpos,ypos) '
                          'dither file name (relative to paths.dithers or fully qualified)', None),
        Key('wavecal', None, 'A MKIDWavedata or name of the same', str),
        Key('flatcal', None, 'A MKIDFlatdata or name of the same', str),
        Key('wcscal', None, 'A MKIDWCSCal or name of the same', str),
        Key('speccal', None, 'A MKIDSpecdata or name of the same', str),
        Key('use', None, 'Specify which dither obs to use, list or range specification string e.g. #,#-#,#,#', None),
    )
    REQUIRED = ('name', 'data', 'wavecal', 'flatcal', 'wcscal')
    STEPNAME = 'dither'

    def __init__(self, *args, **kwargs):
        """
        Obs, byLegacy, or byTimestamp must be specified. byTimestamp is normal.

        Obs must be a list of MKIDObservations
        byLegacyFile must be a legacy dither log file (starttimes, endtimes, xpos,ypos)
        byTimestamp mut be a timestamp or a datetime that falls in the range of a dither in a ditherlog on the path
        obs>byTimestamp>byLegacyFile
        """
        super().__init__(*args, **kwargs)

        def check_use(maxn):
            if self.use is None:
                self.use = list(range(len(maxn)))
            else:
                try:
                    rspec = self.use
                    self.use = [self.use] if isinstance(self.use, int) else derangify(self.use)
                except Exception:
                    self.use = list(range(len(maxn)))
                    self._key_error['use'] += [f'Failed to degangify {rspec}, using all positions']
            if min(self.use)<0 or max(self.use)>=maxn:
                self._key_error['use'] += [f'Values must be be between 0 and {maxn}']

        try:
            if isinstance(self.data, str):  # by old file
                file = self.data
                if not os.path.isfile(file):
                    getLogger(__name__).info(f'Treating {file} as relative dither path.')
                    file = os.path.join(config.paths.dithers, file)

                try:
                    startt, endt, pos, inttime = parse_dither_log(file)
                except Exception as e:
                    self._key_error['data'] += [f'Unable to load legacy dither {file}: {e}']

            elif isinstance(self.data, (int, float)):  # by timestamp
                getLogger(__name__).info(f'Searching for dither containing time {self.data}')
                try:
                    startt, endt, pos = get_ditherinfo(self.data)
                except ValueError:
                    self._key_error['data'] += [f'Unable to find a dither at time {self.data}']
                    getLogger(__name__).warning(f'No dither found for {self.name} @ {self.data} in {config.paths.dithers}')
                    endt, startt, pos = [], [], []
            else:
                if self.use is not None:
                    check_use(len(self.data))
                self.obs = [self.data[i] for i in self.use]

                for o in self.obs:
                    try:
                        assert len(o.dither_pos)==2 and 0<=o.dither_pos[0]<=1 and 0<=o.dither_pos[0]<=1
                    except Exception:
                        self._key_error['data'] += [f'{o} does not specify a dither_pos for the conex (x,y) [0,1]']
                return

            check_use(len(startt))

            startt = [startt[i] for i in self.use]
            endt = [endt[i] for i in self.use]
            pos = [pos[i] for i in self.use]

            self.obs = [MKIDObservation(f'{self.name}_{i}/{len(self.obs)}', b, stop=e, dither_pos=p,
                                        wavecal=self.wavecal, flatcal=self.flatcal, wcscal=self.wcscal,
                                        speccal=self.speccal, **self.extra())
                        for i, b, e, p in zip(self.use, startt, endt, pos)]
        except:
            pass

    @property
    def inttime(self):
        return [o.duration for o in self.obs]

    @property
    def pos(self):
        return [o.dither_pos for o in self.obs]

    @property
    def timeranges(self):
        for o in self.obs:
            for tr in o.timeranges:
                yield tr


class MKIDObservingDataset:
    def __init__(self, yml):
        self.yml = yml
        self.meta = mkidcore.config.load(yml)
        names = [d.name for d in self.meta]
        if len(names) != len(set(names)):
            msg = 'Duplicate names not allowed in {}.'.format(yml)
            getLogger(__name__).critical(msg)
            raise ValueError(msg)

        wcdict = {w.name: w for w in self.wavecals}
        fcdict = {f.name: f for f in self.flatcals}
        wcsdict = {w.name: w for w in self.wcscals}
        scdict = {s.name: s for s in self.speccals}

        for o in self.all_observations:
            o.wavecal = wcdict.get(o.wavecal, o.wavecal)
            o.speccal = scdict.get(o.speccal, o.speccal)
            o.flatcal = fcdict.get(o.flatcal, o.flatcal)
            o.wcscal = wcsdict.get(o.wcscal, o.wcscal)

        for o in self.science_observations:
            o.flatcal = fcdict.get(o.flatcal, o.flatcal)
            o.wcscal = wcsdict.get(o.wcscal, o.wcscal)
            o.speccal = scdict.get(o.speccal, o.speccal)

        for fc in self.flatcals:
            try:
                fc.wavecal = wcdict.get(fc.wavecal, fc.wavecal)
            except AttributeError:
                pass

        for sc in self.speccals:
            for d in sc.data:
                try:
                    d.wavecal = wcdict.get(d.wavecal, d.wavecal)
                except AttributeError:
                    pass
                try:
                    d.flatcal = fcdict.get(d.flatcal, d.flatcal)
                except AttributeError:
                    pass
                try:
                    d.wcscal = wcsdict.get(d.wcscal, d.wcscal)
                except AttributeError:
                    pass

        for d in self.dithers:
            try:
                d.wavecal = wcdict.get(d.wavecal, d.wavecal)
            except AttributeError:
                pass
            try:
                d.flatcal = fcdict.get(d.flatcal, d.flatcal)
            except AttributeError:
                pass
            try:
                d.wcscal = wcsdict.get(d.wcscal, d.wcscal)
            except AttributeError:
                pass

    @property
    def timeranges(self):
        for x in self.meta:
            for tr in x.timeranges:
                yield tr

    @property
    def wavecals(self):
        return [r for r in self.meta if isinstance(r, MKIDWavecalDescription)]

    @property
    def flatcals(self):
        return [r for r in self.meta if isinstance(r, MKIDFlatcalDescription)]

    @property
    def wcscals(self):
        return [r for r in self.meta if isinstance(r, MKIDWCSCalDescription)]

    @property
    def dithers(self):
        return [r for r in self.meta if isinstance(r, MKIDDitherDescription)]

    @property
    def speccals(self):
        return [r for r in self.meta if isinstance(r, MKIDSpeccalDescription)]

    @property
    def sobs(self):
        return [r for r in self.meta if isinstance(r, MKIDObservation)]

    @property
    def all_observations(self):
        speccal_obs = [o for d in self.meta if isinstance(d, MKIDSpeccalDescription) for o in d.obs]
        return ([o for o in self.meta if isinstance(o, MKIDObservation)] +
                [o for d in self.meta if isinstance(d, MKIDDitherDescription) for o in d.obs] +
                [o for d in self.meta if isinstance(d, MKIDFlatcalDescription) for o in d.obs] +
                [d.ob for d in self.meta if isinstance(d, MKIDWCSCalDescription) and d.ob is not None] +
                speccal_obs)

    @property
    def science_observations(self):
        return ([o for o in self.meta if isinstance(o, MKIDObservation)] +
                [o for d in self.meta if isinstance(d, MKIDDitherDescription) for o in d.obs])

    @property
    def wavecalable(self):
        return self.all_observations

    @property
    def flatcalable(self):
        return self.science_observations

    def by_name(self, name):
        d = [d for d in self.meta if d.name == name]
        try:
            if len(d) > 1:
                getLogger(__name__).warning(f'There are {len(d)} things named {name}, returning the first')
            return d[0]
        except IndexError:
            raise ValueError(f'Item "{name}" not found in {self.yml}')

    @property
    def description(self):
        """Return a string describing the data"""
        s = ("Wavecals:\n{wc}\n"
             "Flatcals:\n{fc}\n"
             "Dithers:\n{dithers}\n"
             "Single Obs:\n{obs}".format(wc=('\t-' + '\n\t-'.join([str(w).replace('\n', '\n\t')
                                                                   for w in self.wavecals])) if self.wavecals else '\tNone',
                                         fc=('\t-' + '\n\t-'.join(
                                             [str(f) for f in self.flatcals])) if self.flatcals else
                                         '\tNone',
                                         dithers='Not implemented',
                                         obs='Not implemented'))
        return s


class MKIDOutput(DataBase):
    yaml_tag = '!MKIDOutput'
    KEYS = (
        Key(name='name', default='', comment='A name', dtype=str),
        Key('data', '', 'An data name', str),
        Key('kind', 'image', "('stack', 'spatial', 'temporal', 'list', 'image', 'movie')", str),
        Key('min_wave', float('-inf'), 'Wavelength start for wavelength sensitive outputs', float),
        Key('max_wave', float('inf'), 'Wavelength stop for wavelength sensitive outputs, ', float),
        Key('filename', '', 'relative or fully qualified path, defaults to name+output type,'
                            'so set if making multiple outputs with different settings', str),
        Key('ssd', True, '', bool),
        Key('noise', True, '', bool),
        Key('photom', True, '', bool)
    )
    REQUIRED = ('name', 'data', 'kind')
    EXPLICIT_ALLOW = ('filename',)

    def __init__(self, *args, **kwargs):
        """
        :param name: a name
        :param dataname: a name of a data association
        :param kind: stack|spatial|temporal|list|image|movie
        :param startw: wavelength start
        :param stopw: wavelength stop
        :param filename: an optional relative or fully qualified path, defaults to name+output type,
            so set if making multiple outputs with different settings

        Kind 'movie' requires _extra keys timestep and either frameduration or movieduration with frameduration
        taking precedence. startt and stopt may be included as well and are RELATIVE to the start of the file.

        image - uses photontable.get_fits to the a simple image of the data, applies to a single h5
        stack - uses drizzler.SpatialDrizzler
        spatial - uses drizzler.SpatialDrizzler
        temporal - uses drizzler.TemporalDrizzler
        list - drizzler.ListDrizzler to assign photons an RA and Dec
        movie -

        """
        super().__init__(*args, **kwargs)
        self.name = name
        self.startw = getnm(startw) if startw is not None else None
        self.stopw = getnm(stopw) if stopw is not None else None
        self.kind = kind.lower()
        opt = ('stack', 'spatial', 'temporal', 'list', 'image', 'movie')
        # self.exp_timestep=1  # 'duration of time bins in the output cube, required by temporal only, nbins=frametime/exp_timestep '
        if kind.lower() not in opt:
            raise ValueError('Output {} kind "{}" is not one of "{}" '.format(name, kind, ', '.join(opt)))
        self.enable_noise = True
        self.enable_photom = True
        self.enable_ssd = True
        self.filename = filename
        self.data = dataname
        if _extra is not None:
            for k in _extra:
                if k not in self.__dict__:
                    self.__dict__[k] = _extra[k]

    @property
    def wants_image(self):
        return self.kind == 'image'

    @property
    def wants_drizzled(self):
        return self.kind in ('stack', 'spatial', 'temporal', 'list')

    @property
    def wants_movie(self):
        return self.kind == 'movie'

    @classmethod
    def from_yaml(cls, loader, node):
        d = dict(loader.construct_pairs(node))
        return cls(d.pop('name'), d.pop('data'), d.pop('kind'),
                   startw=d.pop('startw', None), stopw=d.pop('stopw', None),
                   filename=d.pop('filename', ''), _extra=d)

    @property
    def input_timeranges(self):
        return list(self.data.timeranges) + list(self.data.wavecal.timeranges) + list(self.data.flatcal.timeranges)

    @property
    def output_file(self):
        global config
        if self.filename:
            file = self.filename
        else:
            if self.kind in ('stack', 'spatial', 'temporal', 'image'):
                ext = 'fits'
            elif self.kind is 'movie':
                ext = 'gif'
            else:
                ext = 'h5'
            file = f'{self.name}_{self.kind}.{ext}'

        if os.pathsep in file:
            return file
        else:
            return os.path.join(config.paths.out,
                                self.data if isinstance(self.data, str) else self.data.name,
                                file)


class MKIDOutputCollection:
    def __init__(self, file, datafile=''):
        self.file = file
        self.meta = mkidcore.config.load(file)

        if datafile:
            data = load_data_description(datafile, no_global=True)
        else:
            global _dataset
            data = _dataset

        self.dataset = data

        for o in self.meta:
            try:
                o.data = data.by_name(o.data)
            except ValueError as e:
                getLogger(__name__).critical(f'Unable to find data description for "{o.data}"')

    def __iter__(self):
        for o in self.meta:
            yield o

    def __str__(self):
        return f'MKIDOutputCollection: {self.file}'

    @property
    def input_timeranges(self):
        return set([r for o in self for r in o.input_timeranges])


def validate_metadata(md, warn=True, error=False):
    fail = False
    for k in REQUIRED_KEYS:
        if k not in md:
            if error:
                raise KeyError(msg)
            fail = True
            msg = '{} missing from {}'.format(k, md)
            if warn:
                getLogger(__name__).warning(msg)
    return fail


def select_metadata_for_h5(mkidobs, metadata_source):
    """
    Metadata that goes into an H5 consists of records within the duration

    requires metadata_source be an indexable iterable with an attribute utc pointing to a datetime
    """
    # Select the nearest metadata to the midpoint
    start = datetime.fromtimestamp(mkidobs.start)
    time_since_start = np.array([(md.utc - start).total_seconds() for md in metadata_source])
    ok = (time_since_start < mkidobs.duration) & (time_since_start >= 0)
    mdl = [metadata_source[i] for i in np.where(ok)[0]]
    if not mdl:
        mdl = [mkidcore.config.ConfigThing()]
    bad = False
    for md in mdl:
        md.registerfromkvlist(mkidobs.metadata.items(), namespace='')
        bad |= validate_metadata(md, warn=True, error=False)
    if bad:
        raise RuntimeError("Did not specify all the necessary metadata")
    return mdl


def parse_obslog(file):
    """Return a list of configthings for each record in the observing log filterable on the .utc attribute"""
    with open(file, 'r') as f:
        lines = f.readlines()
    ret = []
    for l in lines:
        ct = mkidcore.config.ConfigThing(json.loads(l).items())
        ct.register('utc', datetime.strptime(ct.utc, "%Y%m%d%H%M%S"), update=True)
        ret.append(ct)
    return ret


def load_observing_metadata(files=tuple(), include_database=True):
    """Return a list of mkidcore.config.ConfigThings with the contents of the metadata from observing"""
    global config
    files = list(files)
    if config is not None and include_database:
        files += glob(os.path.join(config.paths.obslog, 'obslog*.json'))
    elif include_database:
        getLogger(__name__).warning('No pipleline database configured.')
    metadata = []
    for f in files:
        metadata += parse_obslog(f)
    return metadata


def load_data_description(file, no_global=False):
    dataset = MKIDObservingDataset(file)
    wcdict = {w.name: w for w in dataset.wavecals}
    for o in dataset.all_observations:
        o.wavecal = wcdict.get(o.wavecal, o.wavecal)
    for d in dataset.dithers:
        try:
            d.wavecal = wcdict.get(d.wavecal, d.wavecal)
        except AttributeError:
            pass

    # TODO what is going on with this code, looks redundant with that in MKIDObservingDataset.__init__
    for fc in dataset.flatcals:
        try:
            fc.wavecal = wcdict.get(fc.wavecal, fc.wavecal)
        except AttributeError:
            pass
    for s in dataset.speccals:
        try:
            s.wavecal = wcdict.get(s.wavecal, s.wavecal)
        except AttributeError:
            pass

    fcdict = {f.name: f for f in dataset.flatcals}
    for o in dataset.science_observations:
        o.flatcal = fcdict.get(o.flatcal, o.flatcal)
    for d in dataset.dithers:
        try:
            d.flatcal = fcdict.get(d.flatcal, d.flatcal)
        except AttributeError:
            pass
    for s in dataset.speccals:
        try:
            s.flatcal = fcdict.get(s.flatcal, s.flatcal)
        except AttributeError:
            pass

    scdict = {s.name: s for s in dataset.speccals}
    for o in dataset.science_observations:
        o.speccal = scdict.get(o.speccal, o.speccal)
    for d in dataset.dithers:
        try:
            d.speccal = scdict.get(d.speccal, d.speccal)
        except AttributeError:
            pass

    if not no_global:
        global _dataset
        _dataset = dataset

    return dataset


def n_cpus_available(max=np.inf):
    """Returns n threads -4 modulo pipelinesettings"""
    global config
    mcpu = min(mp.cpu_count() * 2 - 4, max)
    try:
        mcpu = int(min(config.ncpu, mcpu))
    except Exception:
        pass
    return mcpu


def logtoconsole(file='', **kwargs):
    logs = (create_log('mkidcore', **kwargs), create_log('mkidreadout', **kwargs), create_log('mkidpipeline', **kwargs),
            create_log('__main__', **kwargs))
    if file:
        import logging
        handler = MakeFileHandler(file)
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s (pid=%(process)d)'))
        for l in logs:
            l.addHandler(handler)


yaml.register_class(MKIDTimerange)
yaml.register_class(MKIDObservation)
yaml.register_class(MKIDWavecalDescription)
yaml.register_class(MKIDFlatcalDescription)
yaml.register_class(MKIDSpeccalDescription)
yaml.register_class(MKIDWCSCalDescription)
yaml.register_class(MKIDDitherDescription)
yaml.register_class(MKIDOutput)
