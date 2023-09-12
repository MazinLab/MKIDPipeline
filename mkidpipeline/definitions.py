import hashlib
import os
from collections import defaultdict
from datetime import datetime, timezone
from typing import Set
import ruamel.yaml
import astropy.units
import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord

import mkidcore.config
import mkidcore.utils
import mkidpipeline.config as mkpc
from mkidcore.corelog import getLogger
from mkidcore.legacy import parse_dither as parse_legacy_dither
from mkidcore.utils import derangify


class UnassociatedError(RuntimeError):
    pass


class Key:
    """Class that defines a Key which consists of a name, default value, comment, and data type"""
    def __init__(self, name='', default=None, comment='', dtype=None):
        self.name = str(name)
        self.default = default
        self.comment = str(comment)
        self.dtype = dtype


class _Base:
    """Superclass to handle all MKID data. Verifies and sets all required keys"""
    KEYS = tuple()
    REQUIRED = tuple()  # May set individual elements to tuples of keys if they are alternates e.g. stop/duration
    UNIXTIMESTAMPKEYS = tuple()

    def __init__(self, *args, **kwargs):
        from collections import defaultdict
        self._key_errors = defaultdict(list)
        self._keys = {k.name: k for k in self.KEYS}
        self.extra_keys = []

        # Check disallowed
        for k in kwargs:
            if hasattr(self, f'_{k}'):
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
                self._key_errors[key_set] += ['multiple specified']

        # Process keys
        for k, v in kwargs.items():

            if k in self.UNIXTIMESTAMPKEYS:
                if isinstance(v, datetime):
                    v = v.replace(tzinfo=timezone.utc).timestamp()
                elif isinstance(v, str):
                    try:
                        v = datetime.strptime(v, '%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
                    except ValueError:
                        pass

            # Deal with types
            if k in self._keys:
                required_type = self._keys[k].dtype
                try:
                    required_type[0]
                except TypeError:
                    required_type = (required_type,)

                if tuple in required_type and isinstance(v, (list, ruamel.yaml.comments.CommentedSeq)):
                    v = tuple(v)
                if float in required_type and v is not None:  # and isinstance(v, str) and v.endswith('inf'):
                    try:
                        v = float(v)
                    except (ValueError, TypeError):
                        pass
                if required_type[0] is not None and not isinstance(v, required_type):
                    self._key_errors[k] += [f' {v} not an instance of {tuple(map(lambda x: x.__name__, required_type))}']

            # Try converting strings to Astropy quantities
            if isinstance(v, str):
                try:
                    v = u.Quantity(v)
                except (TypeError, ValueError):
                    if v.startswith('_'):
                        raise ValueError(f'Keys may not start with an underscore: "{v}". Check {self.name}')

            # Set the attribute, prepend _ if is a property/method
            try:
                setattr(self, k, v)
            except AttributeError:
                try:
                    setattr(self, '_' + k, v)
                    getLogger(__name__).debug(f'Storing {k} as _{k} for use by subclass')
                except AttributeError:
                    getLogger(__name__).warning(f'Could not store {k} as _{k} for use by subclass')

        # Set defaults keys that don't exist
        extant = dir(self)+list(kwargs.keys())
        for key in (key for key in self.KEYS if key.name not in extant):
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

        #TODO consider adding support for outputing optional specifications
        # add or f'_{k}' in dir(node) to the line below and then handle cases where e.g. stop and duration are now both
        # in the keys list
        keys = [k for k in node._keys if k not in dir(cls)] + d.pop('extra_keys')
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
        key_info = mkidcore.metadata.INSTRUMENT_KEY_MAP[mkpc.instrument()]['keys']
        if 'header' in store:
            cm = ruamel.yaml.comments.CommentedMap(store['header'])
            for k in store['header']:
                try:
                    descr = key_info[k].description
                except KeyError:
                    descr = '!UNKNOWN HEADER KEY FOR INSTRUMENT!'
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


class MKIDTimerange(_Base):
    """
    Basic MKID data type. By definition, an MKIDTimerange can specify no calibrations to be applied and only consists
    of a start, either a stop or a duration, and optional dark and header fields. A dark, if specified, is itself an
    MKIDTimerange
    """
    yaml_tag = u'!MKIDTimerange'
    KEYS = (
        Key(name='name', default=None, comment='A name', dtype=str),
        Key('start', None, 'The start unix time, float ok, rounded down for H5 creation. yyyy-mm-dd HH:MM:SS ok', (float, int)),
        Key('duration', None, 'A duration in seconds, float ok. If not specified stop must be', (float, int)),
        Key('stop', None, 'A stop unit time, float ok, yyyy-mm-dd HH:MM:SS ok. If not specified duration must be', (float, int)),
        Key('dark', None, 'An MKIDTimerange to use for a dark reference.', None),
        Key('header', None, 'A dictionary of fits header key overrides.', dict)
    )
    REQUIRED = ('name', 'start', ('duration', 'stop'))
    UNIXTIMESTAMPKEYS = ('start', 'stop')

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if hasattr(self, '_duration'):
            self.stop = self.start + self._duration

    def __str__(self):
        return f'{self.name} ({type(self).__name__}): {self.duration}s @ {self.start}'

    def __hash__(self):
        """
        Note that this means that a set of MKIDTimeranges where one MKIDTimerange has a dark and another MKIDTimerange
        has the same start but no dark, makes no promises about which you will get.

        DO NOT add to this to include self.dark without factoring in that sets of MKIDTimeranges would now contain
        members with identical (start, stop) (i.e. much of the pipeline execution path)
        """
        return hash((self.start, self.stop))

    def _vet(self):
        """Returns key errors corresponding to nonsensical start, stop, and duration values"""
        if self.duration > 43200:
            getLogger(__name__).warning(f'Duration of {self.name} longer than 12h!')
        if self.stop < self.start:
            self._key_errors['stop'] += [f'Stop ({self.stop}) must come after start ({self.start})']
        return super()._vet()

    @property
    def _metadata(self):
        """Return a dict of the metadata unique to self"""
        key_info = mkidcore.metadata.INSTRUMENT_KEY_MAP[mkpc.instrument()]['keys']
        d = dict(UNIXSTR=self.start, UNIXEND=self.stop,
                 E_DARK=f'{self.dark.duration}@{self.dark.start}' if self.dark else 'None')
        for k in self.header:
            if k in key_info:
                d[k] = self.header[k]
            else:  #TODO remove once updated
                x = mkidcore.metadata._LEGACY_OBSLOG_MAP.get(k, '')
                if x in key_info:
                    getLogger(__name__).warning(f'{k} is a legacy key name and will be replaced by {x}. '
                                                f'Update your data definition for {self.name}')
                    d[x] = self.header[k]
                else:
                    getLogger(__name__).warning(f'Header key {k} in definition of {self.name} is not valid. Ignoring')
        return d

    @property
    def date(self):
        """Returns the start time in UTC"""
        return datetime.utcfromtimestamp(self.start)

    @property
    def beammap(self):
        """Returns the beammmap"""
        return mkpc.config.beammap

    @property
    def duration(self):
        """Returns the duration in seconds"""
        return self.stop - self.start

    @property
    def timerange(self):
        return self

    @property
    def input_timeranges(self):
        """Yields all of the MKIDTimerange objects including self, and any nested dark MKIDTimeranges"""
        yield self.timerange
        if self.dark is not None:
            yield self.dark.timerange

    @property
    def h5(self):
        """Returns the full h5 file path associated with the MKIDTimerange"""
        if hasattr(self, 'h5_file'):
            return os.path.join(mkpc.config.paths.out, self.h5_file)
        return os.path.join(mkpc.config.paths.out, '{}.h5'.format(int(self.start)))

    @property
    def photontable(self):
        """
        Convenience method for a photontable, file must exist, creates a new photon table on every call,

        Not for use in multithreaded/pool situation where enablewrite may be called
        """
        from mkidpipeline.photontable import Photontable
        try:
            return Photontable(self.h5)
        except ValueError:
            getLogger(__name__).warning(f'H5 already opened for writing, returning table in write mode')
            return Photontable(self.h5, mode='w')

    @property
    def metadata(self):
        """
        Returns a dict of KEY:mkidcore.metadata.MetadataSeries|value pairs, likely a subset of all keys

        Does not return default key values, that would be the responsibility of e.g. get_metadata for a photontable
        """
        obslog_files = mkidcore.utils.get_obslogs(mkpc.config.paths.data, start=self.start)
        data = mkidcore.metadata.load_observing_metadata(files=obslog_files, use_cache=True,
                                                         instrument=mkpc.config.instrument.name)

        #TODO one way to override defaults with settings from the pipeline is to develop a "defaults" dict to pass
        # as a new feature to observing_metadata_for_timerange which would supplant those defined in the csv file
        # defaults = {'E_DEVANG':mkpc.config.instrument.device_orientation_deg,
        #             E_PLTSCL':nominal_platescale_mas'
        #             }

        metadata = mkidcore.metadata.observing_metadata_for_timerange(int(self.start), np.round(self.duration), data,
                                                                      instrument=mkpc.config.instrument.name)
        metadata['INSTRUME'] = mkpc.instrument()
        for k, v in self._metadata.items():
            metadata[k.upper()] = v  # override

        mkidcore.metadata.validate_metadata_dict(metadata, warn='required', error=False,
                                                 allow_missing=mkidcore.metadata.PIPELINE_KEYS,
                                                 instrument=mkpc.config.instrument.name)

        return metadata

    def metadata_at(self, time='start'):
        """Returns the metadata values at 'time'"""
        if time == 'start':
            time = self.start
        md = self.metadata
        ret = {}
        for k, v in md.items():
            try:
                ret[k] = v.get(time) if isinstance(v, mkidcore.metadata.MetadataSeries) else v
            except ValueError:
                pass
        return ret


class MKIDObservation(MKIDTimerange):
    """requires keys name, wavecal, flatcal, wcscal, and all the things from ob"""
    yaml_tag = u'!MKIDObservation'
    KEYS = MKIDTimerange.KEYS + (
        Key('wavecal', '', 'A MKIDWavedata name', str),
        Key('flatcal', '', 'A MKIDFlatdata name', str),
        Key('wcscal', '', 'A MKIDWCSCal name', str),
        Key('speccal', '', 'A MKIDSpecdata nam', str),
    )
    REQUIRED = MKIDTimerange.REQUIRED + ('wavecal', 'flatcal', 'wcscal', 'speccal')

    @classmethod
    def to_yaml(cls, representer, node):
        return super().to_yaml(representer, node, use_underscore=('wavecal', 'flatcal', 'wcscal', 'speccal'))  # save as
        # named references!

    @property
    def _metadata(self):
        """Updates the metadata to include necessary keywords for the WCScal"""
        d = super()._metadata
        try:
            d.update(dict(E_CXREFX=self.wcscal.conex_ref[0], E_CXREFY=self.wcscal.conex_ref[1],
                          E_PREFX=self.wcscal.pixel_ref[0], E_PREFY=self.wcscal.pixel_ref[1],
                          E_PLTSCL=self.wcscal.platescale if self.wcscal.platescale else self.photontable.query_header(
                              'E_PLTSCL'),
                          E_DPDCX=self.wcscal.dp_dcx if self.wcscal.dp_dcx else self.photontable.query_header(
                              'E_DPDCX'),
                          E_DPDCY=self.wcscal.dp_dcy if self.wcscal.dp_dcy else self.photontable.query_header(
                              'E_DPDCY')))
        except (AttributeError, KeyError) as error:
            pass
        return d

    @property
    def obs(self):
        return [self]

    @property
    def dither_pos(self):
        """Returns the (x,y) position of the CONEX mirror when the data was taken"""
        return np.array([self._metadata['E_CONEXX'], self._metadata['E_CONEXY']])

    @property
    def skycoord(self):
        """Returns an astropy SkyCoord object by querying the metadata"""
        try:
            return mkidcore.metadata.skycoord_from_metadata(self.header)
        except KeyError as e:
            getLogger(__name__).debug(f'No coordinates available for {self.name}: {e}')
        return None

    @property
    def input_timeranges(self):
        """Return all of the MKIDTimeranges (NB this, by definition includes subclasses) that go in to making the obs"""
        for tr in super().input_timeranges:
            yield tr
        if self.wavecal:
            for tr in self.wavecal.input_timeranges:
                yield tr
        if self.flatcal:
            for tr in self.flatcal.input_timeranges:
                yield tr
        if self.wcscal:
            for tr in self.wcscal.input_timeranges:
                yield tr
        if self.speccal:
            for tr in self.speccal.input_timeranges:
                yield tr

    def associate(self, **kwargs):
        """Call with dicts for wavecal, flatcal, speccal, and wcscal"""
        for k in ('wavecal', 'flatcal', 'speccal', 'wcscal'):
            if k not in kwargs:
                continue
            item = getattr(self, k)
            if isinstance(item, str) and item in kwargs[k]:
                setattr(self, f'_{k}', item)  # move the item to _name
                setattr(self, k, kwargs[k][item])  # pull item from kwargs[k]


class CalibMixin:
    """Supplementary class to _Base for the calibration data descriptions"""

    @property
    def path(self):
        """Convenience function for returning the full file path of the calibration data product"""
        return os.path.join(mkpc.config.paths.database, self.id + '.npz')

    @property
    def timeranges(self):
        """
        Returns the MKIDTimeranges associated with each MKIDObservation. Note this does not include nested
        MKIDTimeranges, use input_timeranges if nested MKIDTimeranges are desired.
        """
        for o in self.obs:
            yield o.timerange

    @property
    def input_timeranges(self):
        """Returns all of the MKIDTimeranges involved in the calibration. Includes nested MKIDTimeranges."""
        try:
            x = self.obs
        except AttributeError:
            x = self.data
        for o in x:
            for tr in o.input_timeranges:
                yield tr

    def id_for_config(self, cfg):
        """
        Compute a cal definition id string for a specified config (must have STEPNAME namespace)
        """
        id = str(self.name) + '_' + hashlib.md5(str(self).encode()).hexdigest()[-8:]
        cfg = mkpc.config.get(self.STEPNAME)
        config_hash = hashlib.md5(str(cfg).encode()).hexdigest()
        return f'{id}_{config_hash[-8:]}.{self.STEPNAME}'

    @property
    def id(self):
        """ Compute a cal definition id string for the active config (must have STEPNAME namespace) """
        return self.id_for_config(mkpc.config)


class MKIDWavecal(_Base, CalibMixin):
    """Data description for the wavelength calibration data - requires keys 'name' and 'data'"""
    yaml_tag = u'!MKIDWavecal'
    KEYS = (
        Key(name='name', default='', comment='A name', dtype=str),
        Key('data', None, 'List of MKIDTimerange named like 950 nm', tuple),
    )
    REQUIRED = ('name', 'data')
    STEPNAME = 'wavecal'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for d in self.data:
            if not isinstance(d, MKIDTimerange):
                self._key_errors['data'] += [f'Element {d} of data is not an MKIDTimerange']
        if not self.data:
            self._key_errors['data'] += ['obs must be a list of MKIDTimerange']

    def __str__(self):
        start = min(x.start for x in self.data)
        stop = min(x.stop for x in self.data)
        return f'{self.name} (MKIDWavecal): {start}-{stop}\n' + '\n '.join(str(x) for x in self.data)

    @property
    def wavelengths(self):
        """Returns the laser wavelengths used as astropy.units.Quantity objects in units of 'nm'"""
        # TODO update this to use metadata when avaialable
        return tuple([astropy.units.Quantity(x.name).to('nm') for x in self.data])

    @property
    def darks(self):
        """
        Returns a dictionary with keys for each wavelength laser used and values corresponding to either the
        MKIDTimerange for the dark to be applied, or None.
        """
        return {w: ob.dark for w, ob in zip(self.wavelengths, self.data)}

    def associate(self, *args, **kwargs):
        """Provided for interface compatibility"""
        pass


class MKIDFlatcal(_Base, CalibMixin):
    """Data description for the flat calibration data - requires keys 'name' and 'data'"""
    yaml_tag = u'!MKIDFlatcal'
    KEYS = (
        Key(name='name', default=None, comment='A name', dtype=str),
        Key('data', None, 'An MKIDObservation (for a whitelight flat) or an MKIDWavedata '
                          '(or name) for a lasercal flat', None),
        Key('wavecal_duration', None, 'Number of seconds of the wavecal to use, float ok. '
                                      'Required if using wavecal', None),
        Key('wavecal_offset', None, 'An offset in seconds (>=1) from the start of the wavecal '
                                    'timerange. Required if not ob', None),
        Key('lincal', False, 'Apply lincal to h5s ', bool),
        Key('pixcal', True, 'Apply pixcal to data ', bool),
        Key('cosmiccal', False, 'Apply cosmiccal to data ', bool)
    )
    REQUIRED = ('name', 'data',)
    STEPNAME = 'flatcal'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(self.data, (MKIDWavecal, str)):
            try:
                if self.wavecal_offset < 1:
                    self._key_errors['wavecal_offset'] += ['must be >= 1s']
            except (AttributeError, TypeError):
                self._key_errors['wavecal_offset'] += ['required for a wavecal flat (i.e. no ob specified)']
            try:
                if self.wavecal_duration < 1:
                    self._key_errors['wavecal_duration'] += ['must be >= 1s']
            except (AttributeError, TypeError):
                self._key_errors['wavecal_duration'] += ['required for a wavecal flat (i.e. no ob specified)']

        else:
            if not isinstance(self.data, MKIDObservation):
                self._key_errors['data'] += [
                    'must be an MKIDObservation, MKIDWavecal, or name of the latter']
            if not isinstance(getattr(self.data, 'wavecal', None), (MKIDWavecal, str)):
                self._key_errors['data'] += ['data must specify a wavecal when an MKIDObservation']
            self.wavecal_offset = None
            self.wavecal_duration = None

    def __str__(self):
        return '{}: {}'.format(self.name, self.data.start if self.method != 'laser' else self.data.id)

    @property
    def method(self):
        """Returns the method specified for the determination of the flatcal solution- either 'laser' or 'white'"""
        return 'white' if isinstance(self.data, MKIDObservation) else 'laser'

    @property
    def h5s(self):
        """Returns MKIDObservation objects for the wavelengths of the wavecal, will raise errors for white light flats"""
        if self.method != 'laser':
            raise NotImplementedError('h5s only available for laser flats')
        return {w: ob for w, ob in zip(self.data.wavelengths, self.obs)}

    @property
    def obs(self):
        """Returns all of the MKIDObservation objects associated with the flatcal"""
        if isinstance(self.data, MKIDObservation):
            yield self.data
        else:
            if isinstance(self.data, str):
                raise UnassociatedError(f'Must associate wavecal {self.data} prior to calling')
            for tr in self.data.data:
                stop = tr.start + self.wavecal_offset + min(self.wavecal_duration, tr.duration - self.wavecal_offset)
                o = MKIDObservation(name=f'{self.name}_{tr.name}', start=tr.start + self.wavecal_offset,
                                    stop=stop, dark=tr.dark, wavecal=self.data, **tr.extra())
                yield o

    def associate(self, **kwargs):
        """Pulls identifiers for associated datasets needed for MKIDFlatcal"""
        if isinstance(self.data, str):
            self.data = kwargs['wavecal'].get(self.data, self.data)
        else:
            self.data.associate(**kwargs)


class MKIDSpeccal(_Base, CalibMixin):
    """Data description for the spectrophotometric calibration data - requires keys name, data, and aperture"""
    yaml_tag = u'!MKIDSpeccal'
    KEYS = (
        Key(name='name', default='', comment='A name', dtype=str),
        Key('data', None, 'MKIDObservation or MKIDDither', None),
        Key('aperture', 'satellite', 'A 3-tuple (x/RA, y/Dec, r) or "satellite"', None),
    )
    REQUIRED = ('name', 'data', 'aperture')
    STEPNAME = 'speccal'

    def __init__(self, *args, **kwargs):
        self.aperture_info = None
        super().__init__(*args, **kwargs)
        if not isinstance(self.data, (MKIDObservation, MKIDDither, str)):
            self._key_errors['data'] += ['Much be an MKIDObservation, an MKIDDither, or name of the latter']
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

    @property
    def obs(self):
        """Returns all of the MKIDObservation objects associated with the speccal"""
        if isinstance(self.data, MKIDObservation):
            yield self.data
        else:
            for o in self.data.obs:
                yield o

    def associate(self, **kwargs):
        """Pulls identifiers for associated datasets needed for MKIDSpeccal"""
        if isinstance(self.data, str):
            self.data = kwargs['dither'].get(self.data, self.data)
        else:
            self.data.associate(**kwargs)

    def __str__(self):
        try:
            s = f'{self.name} (MKIDSpeccal):' + '\n '.join(str(x) for x in self.data)
        except TypeError:
            s = f'{self.name} (MKIDSpeccal):' + '\n '.join(str(x) for x in self.data.obs)
        return s


class MKIDWCSCal(_Base, CalibMixin):
    """
    The MKIDWCSCal defines the coordinate relation between detector pixels and on-sky RA/DEC

    Keys are
    name - required

    Either:
    data - The name of nn MKIDObservation from which to extract platescale conex_ref, and pixel_ref.
            (presently unsupported) OR  the platescale in mas
    conex_ref - 2 tuple (dither controller position for pixel_ref)
    pixel_ref - 2 tuple (pixel position of optical axis at conex_ref)
    """
    yaml_tag = '!MKIDWCSCal'
    KEYS = (
        Key(name='name', default=None, comment='A name', dtype=str),
        Key('data', None, 'MKIDObservation, MKIDDither (or name), or platescale (e.g. 10 mas)', None),
        Key('pixel_ref', None, 'The pixel position of the target centroid when on '
                               'axis and the conex is at conex_ref', tuple),
        Key('conex_ref', None, 'The conex (x,y) position, [0, 1.0], when the target is at pixel_ref ', tuple),
        Key('source_locs', None, 'The RA/DEC coordinates of the sources in the image ', list),
        Key('dp_dcx', None, 'Change in pixels/change in CONEX position in x ', float),
        Key('dp_dcy', None, 'Change in pixels/change in CONEX position in y ', float),
    )
    REQUIRED = ('name',)
    STEPNAME = 'wcscal'

    def __init__(self, *args, **kwargs):
        super(MKIDWCSCal, self).__init__(*args, **kwargs)
        if isinstance(self.data, u.Quantity):
            try:
                self.data.to('arcsec')
            except Exception:
                self._key_errors['platescale'] += ['must be a valid angular unit e.g. "10 mas"']
        elif isinstance(self.data, (MKIDObservation, MKIDDither, str)):
            if self.pixel_ref is None:
                self._key_errors['pixel_ref'] += ['must be an (x,y) position for the central source at conex_ref']
            if self.conex_ref is None:
                self._key_errors['conex_ref'] += ['must be a conex (x,y) position when the target is at pixel_ref']
            if self.source_locs is None:
                self._key_errors['source_locs'] += ['must specify source_locations in the image']
        else:
            self._key_errors['data'] += [
                'MKIDObservation, MKIDDither, the name of either or a platescale (e.g. 10 mas)']

        if self.conex_ref is not None:
            try:
                assert (len(self.conex_ref) == 2 and
                        -1.0 <= self.conex_ref[0] < 1.0 and
                        -1.0 <= self.conex_ref[1] < 1.0)
            except Exception:
                self._key_errors['conex_ref'] += ['must be a valid conex position (x,y), x & y in [0,1.0]']

        if self.pixel_ref is not None:
            try:
                assert len(self.pixel_ref) == 2
                if mkpc.config is None or mkpc.config.beammap is None:
                    getLogger(__name__).debug(f'Beammap not configured not checking pixel_ref validity')
                else:
                    assert (0 <= self.pixel_ref[0] < mkpc.config.beammap.ncols and
                            0 <= self.pixel_ref[1] < mkpc.config.beammap.nrows)
            except (TypeError, AssertionError):
                getLogger(__name__).debug(f'Dither home {self.pixel_ref} not in beammap '
                                          f'domain {mkpc.config.beammap.ncols},{mkpc.config.beammap.nrows}')
                self._key_errors['pixel_ref'] += ['must be a valid pixel (x,y) position']

    @property
    def platescale(self):
        """Returns the platescale as an astropy.unit.Quantity object in units of arcseconds"""
        if not isinstance(self.data, u.Quantity):
            getLogger(__name__).info('WCSCal not defined by a platescale but by an observation. Platescale set to None '
                                     'until WCSCal is run.')
            return None
        return self.data.to('arcsec')

    @property
    def conex_slopes(self):
        if self.dp_dcx is None or self.dp_dcy is None:
            return tuple((None, None))
        else:
            return tuple((self.dp_dcx, self.dp_dcy))

    @property
    def obs(self):
        """Returns all of the MKIDObservation objects associated with the wcscal"""
        if isinstance(self.data, str):
            raise UnassociatedError(f'Must associate wavecal {self.data} prior to calling')
        elif isinstance(self.data, u.Quantity):
            return
            yield
        else:
            for o in self.data.obs:
                yield o

    def associate(self, **kwargs):
        """Pulls identifiers for associated datasets needed for MKIDWCSCal"""
        if isinstance(self.data, str):
            self.data = kwargs['dither'].get(self.data, self.data)
        elif isinstance(self.data, (MKIDObservation, MKIDDither)):
            self.data.associate(**kwargs)

    def __str__(self):
        try:
            s = f'{self.name} (MKIDWCSCal):' + '\n '.join(str(x) for x in self.data)
        except TypeError:
            try:
                s = f'{self.name} (MKIDWCSCal):' + '\n '.join(str(x) for x in self.data.obs)
            except AttributeError:
                s = f'{self.name} (MKIDWCSCal):' + '\n '.join(str(self.data))
        return s

class MKIDDither(_Base):
    """Data description for dithered data - requires keys name, data, wavecal, flatcal, and wcscal"""
    yaml_tag = '!MKIDDither'
    KEYS = (
        Key(name='name', default=None, comment='A name', dtype=str),
        Key('data', None, 'A list of !sob composing the dither, a unix time that falls within the range of a '
                          'dither log in paths.data, or a fully qualified legacy (starttimes, endtimes, xpos,ypos) '
                          'dither file.', (tuple, float, str)),
        Key('wavecal', '', 'A MKIDWavedata or name of the same', str),
        Key('flatcal', '', 'A MKIDFlatdata or name of the same', str),
        Key('wcscal', '', 'A MKIDWCSCal or name of the same', str),
        Key('speccal', '', 'A MKIDSpecdata or name of the same', str),
        Key('use', None, 'Specify which dither obs to use, list or range specification string e.g. #,#-#,#,#', None),
        Key('header', None, 'A dictionary of fits header key overrides, to be applied to the observations', dict),
    )
    REQUIRED = ('name', 'data', 'wavecal', 'flatcal', 'wcscal')
    STEPNAME = 'dither'

    def __add__(self, other):
        wavecal = self.wavecal if self.wavecal==other.wavecal else ''
        flatcal = self.flatcal if self.flatcal == other.flatcal else ''
        speccal = self.speccal if self.speccal == other.speccal else ''
        wcscal = self.wcscal if self.wcscal == other.wcscal else ''
        header = dict(self.header)
        for k,v in other.header.items():
            if header.get(k,v)!=v:
                raise ValueError(f'Cannot combine dithers due to incompatible headers ({k}:{header.get(k)} v. {k}:{v})')
        header.update(other.header)
        # use = self.use
        # base=len(self.data)
        # use += tuple([a + base for a in other.use])
        d = MKIDDither(name=f'{self.name}+{other.name}', flatcal=flatcal, speccal=speccal,
                       wcscal=wcscal, wavecal=wavecal, header=header,
                       #use=use,
                       data=tuple(self.obs+other.obs))
        return d


    def __init__(self, *args, **kwargs):
        """
        Obs, byLegacy, or byTimestamp must be specified. byTimestamp is normal.

        Obs must be a list of MKIDObservations
        byLegacyFile must be a legacy dither log file (starttimes, endtimes, xpos,ypos)
        byTimestamp mut be a timestamp or a datetime that falls in the range of a dither in a ditherlog on the path
        obs>byTimestamp>byLegacyFile
        """
        self.obs = None
        super().__init__(*args, **kwargs)

        try:
            dither_path = mkpc.config.paths.data
        except AttributeError:
            dither_path = ''
            getLogger(__name__).debug('Pipeline mkpc.config.paths.data not configured, '
                                      'dither discovery will use working directory')

        def check_duration_consistency(startt, endt, name):
            delta = np.array([int(y - x) for x, y in zip(startt, endt)])
            w = np.where(delta == np.median(delta))[0]
            nn = len(delta) - len(w)
            if nn != 0:
                getLogger(__name__).warning(f'Data named {name} has inconsitent dwell times. {nn} outliers removed.')
                self.use = tuple([self.use[i] for i in w])

        def check_use(maxn):
            """Determines whether the 'use' keyword contains sensical values and is appropriately formatted"""
            if self.use is None or not self.use:
                self.use = tuple(range(maxn))
            else:
                if isinstance(self.use, str):
                    try:
                        rspec = self.use
                        self.use = derangify(self.use)
                    except Exception:
                        self.use = tuple(range(maxn))
                        self._key_errors['use'] += [f'Failed to derangify {rspec}, using all positions']
                elif isinstance(self.use, int):
                    self.use = (self.use,)
                else:
                    try:
                        self.use = tuple(map(int, self.use))
                    except Exception:
                        self._key_errors['use'] += [f'Failed to parse use, not int, list of ints, range spec or none']
            try:
                if self.use and (min(self.use) < 0 or max(self.use) >= maxn):
                    self._key_errors['use'] += [f'Values must be in [0, {maxn}]']
                    getLogger(__name__).info('Clearing use due to illegal/out-of-range values.')
                    self.use = list(range(maxn))
            except TypeError:
                self._key_errors['use'] += [f'Failed to parse use, check that if type string there are no bounding '
                                            f'brackets (i.e. 1-5 not [1-5])']
        try:
            if isinstance(self.data, (list, tuple)):
                check_use(len(self.data))
                self.obs = [self.data[i] for i in self.use]

                for i, o in enumerate(self.obs):
                    try:
                        assert isinstance(o, MKIDObservation)
                    except AssertionError:
                        self._key_errors['data'] += [f'data[{i}] ({o}) is not an MKIDObservation']
                        continue

                    try:
                        assert len(o.dither_pos) == 2 and -1 <= o.dither_pos[0] <= 1 and -1 <= o.dither_pos[0] <= 1
                    except Exception:
                        self._key_errors['data'] += [f'data[{i}]={o} does not specify a dither_pos '
                                                     f'for the conex (x,y) in [-1,1]']
                return

            if isinstance(self.data, str):  # by old file
                file = self.data
                if not os.path.isfile(file):
                    getLogger(__name__).info(f'Treating {file} as relative dither path.')
                    file = os.path.join(dither_path, file)

                try:
                    startt, endt, pos, inttime = parse_legacy_dither(file)
                except Exception as e:
                    self._key_errors['data'] += [f'Unable to load legacy dither {file}: {e}']
                    endt, startt, pos = [], [], []

            elif isinstance(self.data, (int, float)):  # by timestamp
                getLogger(__name__).info(f'Searching for dither containing time {self.data}...')
                try:
                    startt, endt, pos = mkidcore.utils.get_ditherdata_for_time(dither_path, self.data)
                    getLogger(__name__).info(f'... found. Dither associated.')
                except ValueError:
                    self._key_errors['data'] += [f'Unable to find a dither at time {self.data}']
                    getLogger(__name__).warning(f'No dither found for {self.name} @ {self.data} in {dither_path}')
                    endt, startt, pos = [], [], []

            else:
                self._key_errors['data'] += [f'data is neither a timestamp, MKIDObservation list, nor dither logfile']
                self.obs = []
                return

            n = len(startt)
            check_use(n)
            if 'dither' in kwargs['name'] and n > 1:
                check_duration_consistency(startt, endt, kwargs['name'])

            startt = [startt[i] for i in self.use]
            endt = [endt[i] for i in self.use]
            pos = [pos[i] for i in self.use]
            n = len(startt)
            self.obs = [MKIDObservation(name=f'{self.name}_{i+1}/{n}', start=b, stop=e, wavecal=self.wavecal,
                                        flatcal=self.flatcal, wcscal=self.wcscal, speccal=self.speccal,
                                        header=dict(E_CONEXX=p[0], E_CONEXY=p[1]), **self.extra())
                        for i, b, e, p in zip(self.use, startt, endt, pos)]
            for o in self.obs:
                o.header.update(self.header)
        except:
            getLogger(__name__).critical('During creation of dither definition: ', exc_info=True)
            pass

    def associate(self, **kwargs):
        """Pulls identifiers for associated datasets needed for MKIDDither"""
        for k in ('wavecal', 'flatcal', 'speccal', 'wcscal'):
            if k not in kwargs:
                continue
            if isinstance(getattr(self, k), str):
                setattr(self, f'_{k}', getattr(self, k))  # store the name
                setattr(self, k, kwargs.get(k, getattr(self, k)))  # pull the object from the kwargs if preset
        for o in self.obs:
            o.associate(**kwargs)

    def obs_for_time(self, timestamp):
        """Returns the MKIDObservation object within an MKIDDither containing time 'timestamp'"""
        for o in self.obs:
            if o.start <= timestamp <= o.stop:
                return o
        raise ValueError(f'Dither {self.name} does not cover time {timestamp}')

    @property
    def inttime(self):
        """Returns a list of the durations of the MKIDObservation objects that compose the MKIDDither"""
        return [o.duration for o in self.obs]

    @property
    def duration(self):
        """Returns the minimum duration of the MKIDObservations that make the dither to be used as the default"""
        return min(self.inttime)

    @property
    def pos(self):
        """Returns a list of the CONEX positions of the MKIDObservation objects that compose the MKIDDither"""
        return [o.dither_pos for o in self.obs]

    @property
    def timeranges(self):
        """
        Returns the MKIDTimerange objects directly associated with the MKIDDither. This includes darks, but
        not the nested MKIDtTimeranges for the associated calibration steps
        """
        for o in self.obs:
            for tr in o.timeranges:
                yield tr

    @property
    def input_timeranges(self):
        """
        Returns all of the MKIDTimerange objects associated with the MKIDDither including the nested
        MKIDTimerange objects in the associated MKIDWavecal, MKIDFlatcal, MKIDWCSCal,
        and MKIDSpeccal (as applicable).
        """
        for o in self.obs:
            for tr in o.input_timeranges:
                yield tr

class MKIDMCMCWCSol(_Base, CalibMixin):
    """

    """
    yaml_tag = '!MKIDMCMCWCSol'

    KEYS = (
        Key(name='name', default='', comment='A name', dtype=str),
        # Key('data', '', 'A data name', str),
        # Key('wcscal', '', 'The wcscal name to update with he new solution in the data.yaml', str),
        Key('cor_spot_ref', [], 'X,Y guess for the chronograph', list),
        Key('conex_ref', [], 'X,Y conex for the chronograph guess', list),
        Key('spot_ref1', [], 'X,Y guess for the first satellite spot', list),
        Key('spot_ref2', [], 'X,Y guess for the second satellite spot', list),
        Key('spot_ref3', [], 'X,Y guess for the third satellite spot', list),
        Key('spot_ref4', [], 'X,Y guess for the fourth satellite spot', list),
        Key('slopes', [], 'Chronograph position to conex slope guess', list),
        # Key('start_offset', 0, 'An offset in seconds (>=0) from the start of the data timerange', int),
        # Key('sol', {}, 'The dpdc pixel solution at conex 0 with errors', dict),
    )
    REQUIRED = ('name','cor_spot_ref','conex_ref') #,'data', 'wcscal'

    def __init__(self, *args, **kwargs):
        super(MKIDMCMCWCSol, self).__init__(*args, **kwargs)

    @property
    def obs(self):
        return self.data

class MKIDObservingDataset:
    """Class that manages all of the data specified in the data configuration file"""
    def __init__(self, yml):
        self.yml = yml
        self.meta = mkidcore.config.load(yml)
        names = [d.name for d in self.meta]
        if len(names) != len(set(names)):
            msg = 'Duplicate names not allowed in {}.'.format(yml)
            getLogger(__name__).critical(msg)
            raise ValueError(msg)

        self.datadict = {x.name: x for x in self.meta}

        wcdict = {w.name: w for w in self.wavecals}
        fcdict = {f.name: f for f in self.flatcals}
        wcsdict = {w.name: w for w in self.wcscals}
        scdict = {s.name: s for s in self.speccals}
        dithdict = {d.name: d for d in self.dithers}

        missing = defaultdict(lambda: set())

        for f in self.flatcals:
            f.associate(wavecal=wcdict, flatcal=fcdict, wcscal=wcsdict, speccal=scdict, dither=dithdict)

        for f in self.wcscals:
            f.associate(wavecal=wcdict, flatcal=fcdict, wcscal=wcsdict, speccal=scdict, dither=dithdict)

        for f in self.speccals:
            f.associate(wavecal=wcdict, flatcal=fcdict, wcscal=wcsdict, speccal=scdict, dither=dithdict)

        for f in self.dithers:
            f.associate(wavecal=wcdict, flatcal=fcdict, wcscal=wcsdict, speccal=scdict, dither=dithdict)

        for o in self.all_observations:
            o.associate(wavecal=wcdict, flatcal=fcdict, wcscal=wcsdict, speccal=scdict, dither=dithdict)

        try:
            for o in self.wavecalable:
                if isinstance(o.wavecal, str) and o.wavecal:
                    missing['wavecal'].add(o.wavecal)

            for o in self.flatcalable:
                if isinstance(o.flatcal, str) and o.flatcal:
                    missing['flatcal'].add(o.flatcal)

            for o in self.wcscalable:
                if isinstance(o.wcscal, str) and o.wcscal:
                    missing['wcscal'].add(o.wcscal)

            for o in self.speccalable:
                if isinstance(o.speccal, str) and o.speccal:
                    missing['speccal'].add(o.speccal)
        except:
            getLogger(__name__).error('Failure during name/data association', exc_info=True)
        self.missing_cal_defs = missing

    def __getitem__(self, item):
        return self.datadict[item]

    def __iter__(self):
        getLogger(__name__).warning('Iterating on a dataset excludes nested definitions')
        for o in self.meta:
            yield o

    def _find_nested(self, attr, kind, look_in):
        """
        Internal function for finding nested data descriptions associated with another data description (i.e. find an
        MKIDWavecal associated with an MKIDObservation). Yields an object of type 'kind'.
        """
        for r in self.meta:
            if isinstance(r, kind):
                yield r
            # This is necessary as we allow the user to define directly where they are used
            if isinstance(r, look_in):
                try:
                    for o in r.obs:
                        x = getattr(o, attr, None)
                        if isinstance(x, kind):
                            yield x
                except UnassociatedError:
                    getLogger(__name__).debug(f'Skipping nested search of unassociated "{r.data}" for {attr}')

    def unique_data(self):
        """
        returns True if there are no distinct datasets defined by the same timeranges. All h5 files must be unique to a
        given data definition.
        Does not search through nested definitions since this is allowed.
        """
        start_times = []
        data = [o for o in self.meta]
        names = [o.name for o in self.meta]
        for d in data:
            try:
                if d.data.name in names:
                    continue
            except AttributeError:
                pass
            if isinstance(d, MKIDWavecal):
                start_times.extend([t.start for t in d.input_timeranges])
            elif  isinstance(d, MKIDMCMCWCSol): continue
            else:
                try:
                    for o in d.obs:
                        start_times.append(o.start)
                except AttributeError:
                    start_times.append(d.start)
        return len(set(start_times)) == len(start_times)

    def validate(self, return_errors=False, error=False):
        """
        Ensures that there is no missing or ill-defined data in the data configuration. Returns True if everything is
        good and all is associated. If error=True raise an exception instead of returning False
        """
        errors = {}
        if not self.unique_data():
            errors['Non unique data'] = 'all observations must be defined by unique data'
        for x in self:
            issues = x._vet()
            if issues:
                name = f'{x.name} ({repr(x)})' if x.name in errors else x.name
                errors[name] = issues

        if self.missing_cal_defs:
            errors['missing calibrations'] = [f'{k}(s): {v}' for k, v in self.missing_cal_defs.items()]

        if return_errors:
            return errors
        if error and errors:
            raise Exception('Validation failed')
        return len(errors) == 0

    @property
    def all_timeranges(self) -> Set[MKIDTimerange]:
        tr = set([tr for x in self.meta for tr in x.input_timeranges])
        return tr

    @property
    def wavecals(self):
        """
        Returns a set of all of the MKIDWavecalDescriptions that are associated with each MKIDObservation,
        MKIDWCSCal, MKIDDither, or MKIDFlatcal.
        """
        look_in = (MKIDObservation, MKIDWCSCal, MKIDDither, MKIDFlatcal,
                   MKIDSpeccal)
        return set(self._find_nested('wavecal', MKIDWavecal, look_in))

    @property
    def flatcals(self):
        """
        Returns a set of all of the MKIDFlatcalDescriptions that are associated with each MKIDObservation,
        MKIDWCSCal, MKIDDither, or MKIDSpeccal.
        """
        look_in = (MKIDObservation, MKIDWCSCal, MKIDDither, MKIDSpeccal)
        return set(self._find_nested('flatcal', MKIDFlatcal, look_in))

    @property
    def wcscals(self):
        """
        Returns a set of all of the MKIDWCSCalDescriptions that are associated with each MKIDObservation,
        MKIDDither, or MKIDSpeccal.
        """
        look_in = (MKIDObservation, MKIDDither, MKIDSpeccal)
        return set(self._find_nested('wcscal', MKIDWCSCal, look_in))

    @property
    def dithers(self):
        """
        Returns a set of all of the MKIDDitherDescriptions that are associated with each MKIDDither,
        or MKIDSpeccal.
        """
        for r in self.meta:
            if isinstance(r, MKIDDither):
                yield r
            if isinstance(r, MKIDSpeccal) and isinstance(r.data, MKIDDither):
                yield r.data

    @property
    def speccals(self):
        """
        Returns a set of all of the MKIDSpeccalDescriptions that are associated with each MKIDDither,
        or MKIDObservation.
        """
        look_in = (MKIDObservation, MKIDDither)
        return set(self._find_nested('speccal', MKIDSpeccal, look_in))

    @property
    def all_observations(self):
        """
        Returns a set of all of the MKIDObservations that are associated with each MKIDFlatcal,
        MKIDWCSCal, MKIDSpeccal, or MKIDDither.

        Note that nested definitions are not included
        """
        for o in self.meta:
            if isinstance(o, MKIDObservation):
                yield o
        for d in self.meta:
            if isinstance(d, MKIDFlatcal):
                for o in d.obs:
                    yield o
        for d in self.meta:
            if isinstance(d, MKIDWCSCal):
                for o in d.obs:
                    if o:
                        yield o
        for d in self.meta:
            if isinstance(d, MKIDSpeccal):
                for o in d.obs:
                    yield o
        for d in self.meta:
            if isinstance(d, MKIDDither):
                for o in d.obs:
                    yield o

    @property
    def wavecalable(self):
        """
        Returns all items that can have wavecal applied. Must return EVERY item in the dataset that might have .wavecal
        """
        return self.all_observations

    @property
    def pixcalable(self):
        """Returns all items that can have pixcal applied"""
        return self.all_observations

    @property
    def cosmiccalable(self):
        """Returns all items that can have cosmiccal applied"""
        return self.all_observations

    @property
    def lincalable(self):
        """Returns all items that can have lincal applied"""
        return self.all_observations

    @property
    def flatcalable(self):
        """
        Returns all items that can have flatcal applied. Must return EVERY item in the dataset that might have .flatcal
        """
        return ([o for o in self.meta if isinstance(o, MKIDObservation)] +
                [o for d in self.meta if isinstance(d, MKIDDither) for o in d.obs] +
                [o for d in self.meta if isinstance(d, MKIDSpeccal) for o in d.obs] +
                [o for d in self.meta if isinstance(d, MKIDWCSCal) for o in d.obs if o])

    @property
    def wcscalable(self):
        """
        Returns all items that can have wcscal applied. Must return EVERY item in the dataset that might have .wcscal
        """
        return ([o for d in self.meta if isinstance(d, MKIDSpeccal) for o in d.obs] +
                [o for o in self.meta if isinstance(o, MKIDObservation)] +
                [o for d in self.meta if isinstance(d, MKIDDither) for o in d.obs])

    @property
    def speccalable(self):
        """
        Returns all items that can have speccal applied. Must return EVERY item in the dataset that might have .speccal
        """
        return ([o for o in self.meta if isinstance(o, MKIDObservation)] +
                [o for d in self.meta if isinstance(d, MKIDDither) for o in d.obs])

    @property
    def description(self):
        """Return a string describing the data"""
        s = ("Wavecals:\n{wc}\n"
             "Flatcals:\n{fc}\n"
             "Dithers:\n{dithers}\n"
             "Single Obs:\n{obs}".format(wc=('\t-' + '\n\t-'.join([str(w).replace('\n', '\n\t')
                                                                   for w in
                                                                   self.wavecals])) if self.wavecals else '\tNone',
                                         fc=('\t-' + '\n\t-'.join(
                                             [str(f) for f in self.flatcals])) if self.flatcals else
                                         '\tNone',
                                         dithers='Not implemented',
                                         obs='Not implemented'))
        return s

# class MKIDMcmc(_Base):
#     """
#
#     """
#     yaml_tag = '!MKIDMcmc'
#
#     KEYS = (
#         Key(name='name', default='', comment='A name', dtype=str),
#         Key('data', '', 'A data name', str),
#         Key('wcscal', '', 'The wcscal name to update with he new solution in the data.yaml', str),
#         Key('length', [80, 75, 85], 'Length of the satellite spot arm: i.e. [value, min, max]', list),
#         Key('amplitude', [500, 1, 1000], 'Amplitude of the satellite spot: i.e. [value, min, max]', list),
#         Key('angle1', [45, 44, 46], 'Angle of the first pair of satellite spots vs to X: i.e. [value, min, max]', list),
#         Key('angle2', [90, 89, 91], 'Angle of the second pair of satellite spots vs to the first: i.e. [value, min, max]', list),
#         Key('fwhm_x', [7, 3, 20], 'X FWHM of the satellite spots: i.e. [value, min, max]', list),
#         Key('fwhm_y', [7, 3, 20], 'Y FWHM of the satellite spots: i.e. [value, min, max]', list),
#         Key('cor_spot_ref', [], 'X,Y guess for the chronograph', list),
#         Key('conex_ref', [], 'X,Y conex for the chronograph guess', list),
#         Key('spot_ref1', [], 'X,Y guess for the first satellite spot', list),
#         Key('spot_ref2', [], 'X,Y guess for the second satellite spot', list),
#         Key('spot_ref3', [], 'X,Y guess for the third satellite spot', list),
#         Key('spot_ref4', [], 'X,Y guess for the fourth satellite spot', list),
#         Key('slopes', [], 'Chronograph position to conex slope guess', list),
#         Key('start_offset', 0, 'An offset in seconds (>=0) from the start of the data timerange', int),
#     )
#     REQUIRED = ('name', 'data', 'kind','wcscal','lenght','amplitude','angle1','angle2','fwhm_x','fwhm_y','cor_spot_ref','conex_ref')
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)


class MKIDOutput(_Base):
    """

    Kind 'movie' requires exp_timestep and either frameduration or movieduration with frameduration
    taking precedence. startt and stopt may be included as well and are RELATIVE to the start of the file.

    image - uses photontable.get_fits to the a simple image of the data, applies to a single h5
    stack - uses drizzler.Drizzler
    drizzle- uses drizzler.Drizzler
    movie - uses movie.make_movie to make an animation

    """
    yaml_tag = '!MKIDOutput'
    KEYS = (
        Key(name='name', default='', comment='A name', dtype=str),
        Key('data', '', 'An data name', str),
        Key('kind', 'image', "drizzle - Use drizzler"
                             "image - Generate a simple image (use photontable.get_fits)"
                             "movie - Use movies to generate a gif animation"
                             "tcube - use photontable.get_fits"
                             "scube - use photontable.get_fits", str),

        Key('exclude_flags', None, 'A list of pixel flag names to exclude', None),
        Key('min_wave', float('-inf'), 'Wavelength start for wavelength sensitive outputs', str),
        Key('max_wave', float('inf'), 'Wavelength stop for wavelength sensitive outputs, ', str),
        Key('start_offset', 0, 'start offset (s) into data (per frame if dithered)', float),
        Key('duration', None, 'number of seconds od data to use (per frame if dithered), None=all', (float, int)),
        Key('filename', '', 'relative or fully qualified path, defaults to name+output type,'
                            'so set if making multiple outputs with different settings', str),
        Key('units', 'photons', 'photons|photons/s. Drizzled data will always be in photons/s', str),
        Key('use_weights', True, 'Use photon weights', bool),
        Key('adi_mode', False, 'Output compatible with ADI pipeline - do not apply any rotation to outputs '
                               '(If True will overwrite derotate)', bool),
        Key('lincal', False, 'Apply and use lincal, slow', bool),
        Key('pixcal', True, 'Apply pixcal', bool),
        Key('cosmical', False, 'Determine cosmic ray hits, slow', bool),
        Key('flatcal', True, 'Apply flatcal', bool),
        Key('movie_runtime', None, 'Runtime of movie, defaults to realtime', float),
        # NB wavecal is applied and used if the underlying data specifies them, min/max wave allow ignoring it
        # there is no speccal key as it isn't something that is applied to the data
        # speccals are just fetched and determined for
        Key('timestep', None, 'Duration of time bins in output cubes with a temporal axis (None means no temporal axis)',
            float),
        Key('wavestep', None,
            'Width of wavelength bins in output cubes with a wavelength axis (None means no wavelength'
            ' axis)', (None, str))
    )
    REQUIRED = ('name', 'data', 'kind')

    # OPTIONAL = tuple

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kind = self.kind.lower()
        opt = ('drizzle', 'image', 'movie', 'tcube', 'scube')
        if self.kind not in opt:
            self._key_errors['kind'] += [f"Must be one of: {opt}"]
        if self.kind == 'movie':
            if not self.movie_runtime:
                self._key_errors['movie_runtime'] += [f"Runtime required"]
        # TODO add exclude flag checking
        # TODO improve extra keys settings
        self._data = ''

    def associate(self, data):
        if not isinstance(self.data, str):
            raise RuntimeError('Association already complete')
        if self.wants_drizzled and not isinstance(data, MKIDDither):
            raise ValueError(f'Output {self.kind} requires a dither')
        if self.data != data.name:
            getLogger(__name__).warning(f'Data named {data.name} used of dataset {self.data}')
        self._data = self.data
        self.data = data

    @property
    def output_settings_dict(self):
        """Returns a dict of kwargs from the various output settings"""
        step, cube_type = None, None
        wave_step = self.wavestep if isinstance(self.wavestep, u.Quantity) else 0.0 * u.nm
        time_step = self.timestep
        if self.kind in ('tcube', 'movie'):
            cube_type = 'time'
            bin_type = 'time'
        elif self.kind == 'scube':
            cube_type = 'wave'
            bin_type = 'wave' if self.wavestep is None or self.wavestep.unit != 'eV' else 'energy'
        else:
            bin_type = 'energy'
        kwargs = dict(start=self.start_offset, duration=self.duration, weight=self.use_weights,
                      wave_start=self.min_wave, wave_stop=self.max_wave, rate=self.units == 'photons/s',
                      bin_type=bin_type, cube_type=cube_type, wvl_bin_width=wave_step, time_bin_width=time_step,
                      ADI_mode = self.adi_mode, exclude_flags=mkidcore.pixelflags.PROBLEM_FLAGS)
        return mkidcore.config.ConfigThing().registerfromkvlist(kwargs.items(), namespace='')

    @property
    def wants_image(self):
        """Returns True if the output type specified is an image ('image', 'tcube', 'scube'), otherwise returns False"""
        return self.kind in ('image', 'tcube', 'scube')

    @property
    def wants_drizzled(self):
        """
        Returns True if the output type specified is a drizzled output ('drizzle'.), otherwise returns False
        """
        return self.kind in ('drizzle',)

    @property
    def wants_movie(self):
        """Returns True if the output type specified is 'movie', otherwise returns False"""
        return self.kind == 'movie'

    @property
    def input_timeranges(self) -> Set[MKIDTimerange]:
        """Returns a set of all input timeranges"""
        return set(self.data.input_timeranges)

    @property
    def duration(self):
        """Returns the duration of the data associated with the output"""
        try:
            return self._duration
        except AttributeError:
            if isinstance(self.data, str):
                raise RuntimeError('Must associate dataset to get default duration')
            return self.data.duration



    @property
    def filename(self):
        """
        Returns the name of the full file path to which the output will be written. If kind is ('stack', 'drizzle',
        'image', 'scube', 'scube') this will be a fits file. if kind is 'movie' it will be a gif and otherwise it will
        remain an h5 file."""
        if hasattr(self, '_filename'):
            file = self._filename
        else:
            if self.kind in ('stack', 'drizzle', 'image', 'scube', 'tcube'):
                ext = 'fits'
            else:
                ext = self.movie_format if self.kind == 'movie' else 'h5'
            file = f'{self.name}_{self.kind}.{ext}'

        if os.pathsep in file:
            return file
        else:
            full_path = os.path.join(mkpc.config.paths.out, self.data if isinstance(self.data, str) else self.data.name)
            return os.path.join(full_path, file)


class MKIDOutputCollection:
    """Class that manages all of the outputs and relevant dependencies specified in the out configuration"""
    def __init__(self, file, datafile=''):
        self.file = file
        self.meta = mkidcore.config.load(file)
        self.dataset = MKIDObservingDataset(datafile) if datafile else None

        if self.dataset is not None:
            for o in self.meta:
                try:
                    if '+' in o.data:
                        a = self.dataset.datadict[o.data.split('+')[0]]
                        for i in o.data.split('+')[1:]: a += self.dataset.datadict[i]
                        o.associate(a)
                    else:
                        o.associate(self.dataset.datadict[o.data])
                except KeyError:
                    getLogger(__name__).error(f'Unable to find data description for "{o.data}"')
                except ValueError:
                    getLogger(__name__).error(f'Data description for {o.data} is not of the type required for '
                                              f'output {o.name}')

    def __iter__(self) -> MKIDOutput:
        for o in self.meta:
            yield o

    def __str__(self):
        return f'MKIDOutputCollection: {self.file}'

    def validation_summary(self, null_success=False):
        """Nicely formats the errors returned by self.validate"""
        errors = self.validate(return_errors=True)
        if not errors:
            return '' if null_success else 'Validation Successful, no issues identified'

        def format_errors(errors: dict):
            if isinstance(errors, str):
                return errors
            elif isinstance(errors, dict):
                return '\n'.join([f"{k}:\n\t"+format_errors(v).replace('\n\t', '\n\t\t') for k, v in errors.items()])
            else:
                return '\n\t'.join(errors)

        return (f'Validation failed, please these issues fix before continuing.\n'
                f'=============================================================\n'
                f'{format_errors(errors)}\n'
                f'=============================================================\n')

    def validate(self, error=False, return_errors=False):
        """
        Ensures that there is no missing or ill-defined data in the data configuration. Returns True if everything is
        good and all is associated. If error=True raise an exception instead of returning False
        """
        errors = {}
        for x in self:
            issues = x._vet()
            if issues:
                name = f'{x.name} ({repr(x)})' if x.name in errors else x.name
                errors[name] = issues

        name = os.path.basename(self.file)
        if self.dataset is not None:
            e = self.dataset.validate(return_errors=True)
            if e:
                errors[f'{name}.dataset ({os.path.basename(self.dataset.yml)})'] = e

        e = [f'{o.wavecal} missing for {o.name} ' for o in set(self.to_wavecal) if isinstance(o.wavecal, str)]
        if e:
            errors[f'{name} wavecal'] = e
        e = [f'{o.flatcal} missing for {o.name} ' for o in set(self.to_flatcal) if isinstance(o.flatcal, str)]
        if e:
            errors[f'{name} flatcal'] = e
        e = [f'{o.speccal} missing for {o.name} ' for o in set(self.to_speccal) if isinstance(o.speccal, str)]
        if e:
            errors[f'{name} speccal'] = e

        if return_errors:
            return errors
        if error and errors:
            raise RuntimeError('Validation failed')
        return len(errors) == 0

    @property
    def input_timeranges(self) -> Set[MKIDTimerange]:
        """Returns a set of all input timeranges"""
        return set([r for o in self for r in o.input_timeranges])

    @property
    def wavecals(self):
        """
        Returns a set of all of the MKIDObservations affiliated with an output that have an associated
        MKIDWavecal
        """
        return set([o.wavecal for o in self.to_wavecal if o.wavecal])

    @property
    def flatcals(self):
        """
        Returns a set of all of the MKIDObservations affiliated with an output that have an associated
        MKIDFlatcal
        """
        return set([o.flatcal for o in self.to_flatcal if o.flatcal])

    @property
    def speccals(self):
        """
        Returns a set of all of the MKIDObservations affiliated with an output that have an associated
        MKIDSpeccal
        """
        return set([o.speccal for o in self.to_speccal if o.speccal])

    @property
    def mcmcwcssols(self):
        """
        Returns a set of all of the MKIDObservations affiliated with an output that have an associated
        MKIDWCSCal. Does not search for nested MKIDWCSCalDescriptions except with the speccal
        """
        return set([o for o in self.to_mcmcwcssol if o.mcmc])
        # return set([o.wcscal for o in self.to_wcscal if o.wcscal])

    @property
    def wcscals(self):
        """
        Returns a set of all of the MKIDObservations affiliated with an output that have an associated
        MKIDWCSCal. Does not search for nested MKIDWCSCalDescriptions except with the speccal
        """
        return set([o.wcscal for o in self.to_wcscal if o.wcscal])

    @property
    def to_lincal(self):
        def input_observations(obs):
            for o in obs:
                if o.flatcal and o.flatcal.lincal:
                    for x in o.flatcal.obs:
                        yield x
                if o.speccal:
                    for x in o.speccal.obs:
                        if x.flatcal and x.flatcal.lincal:
                            for y in x.flatcal.obs:
                                yield y
                if o.wcscal:
                    for x in o.wcscal.obs:
                        if x.flatcal and x.flatcal.lincal:
                            for y in x.flatcal.obs:
                                yield y

        for out in self:
            if out.lincal:
                for o in out.data.obs:
                    yield o
            for o in input_observations(out.data.obs):
                yield o

    @property
    def to_buildhdf(self):
        return self

    @property
    def to_attachmeta(self):
        return self

    @property
    def to_wavecal(self):
        def input_observations(obs):
            try:
                for o in obs:
                    if o.flatcal:
                        for x in o.flatcal.obs:
                            yield x
                    if o.speccal:
                        for x in o.speccal.obs:
                            yield x
                            if x.flatcal:
                                for y in x.flatcal.obs:
                                    yield y
                            if x.wcscal:
                                for y in x.wcscal.obs:
                                    yield y
                                    if y.flatcal:
                                        for z in y.flatcal.obs:
                                            yield z
                    if o.wcscal:
                        for x in o.wcscal.obs:
                            yield x
                            if x.flatcal:
                                for y in x.flatcal.obs:
                                    yield y
            except AttributeError as e:
                if "'str' object has not attribute obs" in str(e):
                    raise KeyError(f'An input {o} to {self.name} has was not associated properly')
        for out in self:
            for o in out.data.obs:
                yield o
            for o in input_observations(out.data.obs):
                yield o

    @property
    def to_pixcal(self):
        def input_observations(obs):
            for o in obs:
                if o.flatcal and o.flatcal.pixcal:
                    for x in o.flatcal.obs:
                        yield x
                if o.speccal:
                    for x in o.speccal.obs:
                        yield x
                        if x.flatcal and x.flatcal.pixcal:
                            for y in x.flatcal.obs:
                                yield y
                        if x.wcscal:
                            for y in x.wcscal.obs:
                                yield y
                                if y.flatcal and y.flatcal.pixcal:
                                    for z in y.flatcal.obs:
                                        yield z
                if o.wcscal:
                    for x in o.wcscal.obs:
                        yield x
                        if x.flatcal and x.flatcal.pixcal:
                            for y in x.flatcal.obs:
                                yield y

        for out in self:
            if out.pixcal:
                for o in out.data.obs:
                    yield o
            for o in input_observations(out.data.obs):
                yield o

    @property
    def to_cosmiccal(self):
        def input_observations(obs):
            for o in obs:
                if o.flatcal and o.flatcal.cosmiccal:
                    for x in o.flatcal.obs:
                        yield x
                if o.speccal:
                    for x in o.speccal.obs:
                        # yield x
                        if x.flatcal and x.flatcal.cosmiccal:
                            for y in x.flatcal.obs:
                                yield y
                        if x.wcscal:
                            for y in x.wcscal.obs:
                                # yield y
                                if y.flatcal and y.flatcal.cosmiccal:
                                    for z in y.flatcal.obs:
                                        yield z
                if o.wcscal:
                    for x in o.wcscal.obs:
                        # yield x
                        if x.flatcal and x.flatcal.cosmiccal:
                            for y in x.flatcal.obs:
                                yield y

        for out in self:
            if out.cosmical:
                for o in out.data.obs:
                    yield o
            for o in input_observations(out.data.obs):
                yield o

    @property
    def to_flatcal(self):
        def input_observations(obs):
            for o in obs:
                if o.speccal:
                    for x in o.speccal.obs:
                        if x.flatcal:
                            yield x
                        if x.wcscal:
                            for y in x.wcscal.obs:
                                if y.flatcal:
                                    yield y
                if o.wcscal:
                    for x in o.wcscal.obs:
                        if x.flatcal:
                            yield x

        for out in self:
            if out.flatcal:
                for o in out.data.obs:
                    if o.flatcal:
                        yield o
            for o in input_observations(out.data.obs):
                yield o

    @property
    def to_speccal(self):
        for out in self:
            # speccals aren't applied to the data, so just return them all for generation so
            # that the user has them for later
            # if out.speccal:
            for o in out.data.obs:
                if o.speccal:
                    yield o

    @property
    def to_mcmcwcssol(self):
        # def input_observations(obs):
        #     for o in obs:
        #         if o.speccal:
        #             for x in o.speccal.obs:
        #                 if x.wcscal:
        #                     yield x
        # for out in self:
        #     for o in out.data.obs:
        #         if o.wcscal:
        #             yield o
        #
        #     for o in input_observations(out.data.obs):
        #         yield o
        for out in self:
            for o in out.data.obs:
                yield o


    @property
    def to_wcscal(self):
        def input_observations(obs):
            for o in obs:
                if o.speccal:
                    for x in o.speccal.obs:
                        if x.wcscal:
                            yield x
        for out in self:
            for o in out.data.obs:
                if o.wcscal:
                    yield o
            for o in input_observations(out.data.obs):
                yield o

    @property
    def to_drizzle(self):
        for out in self:
            if isinstance(out.data, MKIDDither):
                yield out.data
            if out.data.speccal and isinstance(out.data.speccal.data, MKIDDither):
                yield out.data.speccal.data
            if out.data.wcscal and isinstance(out.data.wcscal.data, MKIDDither):
                yield out.data.wcscal.data


# mkidcore.config.yaml.register_class(MKIDMcmc)
mkidcore.config.yaml.register_class(MKIDMCMCWCSol)
mkidcore.config.yaml.register_class(MKIDTimerange)
mkidcore.config.yaml.register_class(MKIDObservation)
mkidcore.config.yaml.register_class(MKIDWavecal)
mkidcore.config.yaml.register_class(MKIDFlatcal)
mkidcore.config.yaml.register_class(MKIDSpeccal)
mkidcore.config.yaml.register_class(MKIDWCSCal)
mkidcore.config.yaml.register_class(MKIDDither)
mkidcore.config.yaml.register_class(MKIDOutput)
