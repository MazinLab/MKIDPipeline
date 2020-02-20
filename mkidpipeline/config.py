
import numpy as np
import os
from glob import glob
import hashlib
from datetime import datetime
import multiprocessing as mp
import pkg_resources as pkg
import astropy.units as units
import json
from astropy.coordinates import SkyCoord
from collections import namedtuple
import ast

import mkidcore.config
from mkidcore.corelog import getLogger, create_log, MakeFileHandler
from mkidcore.utils import getnm, derangify
from mkidcore.objects import Beammap

#TODO this is a placeholder to help integrating metadata
InstrumentInfo = namedtuple('InstrumentInfo', ('beammap', 'platescale'))

#Ensure that the beammap gets registered with yaml, technically the import does this
#but without this note an IDE or human might remove the import
Beammap()

config = None
_dataset = None
_parsedDitherLogs = {}

yaml = mkidcore.config.yaml

pipline_settings = ('beammap', 'paths', 'templar', 'instrument', 'ncpu')


STANDARD_KEYS = ('ra','dec', 'airmass','az','el','ha','equinox','parallactic','target','utctcs','laser','flipper',
                 'filter','observatory','utc','comment','device_orientation','instrument','dither_ref','dither_home',
                 'dither_pos','platescale')

REQUIRED_KEYS = ('ra','dec','target','observatory','instrument','dither_ref','dither_home','platescale',
                 'device_orientation', 'dither_ref')


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

    if config is None:
        configure_pipeline(pkg.resource_filename('mkidpipeline', 'pipe.yml'))
        for k in pipline_settings:
            try:
                config.update(k, cfg.get(k))
            except KeyError:
                pass

    for k in pipline_settings:
        cfg.register(k, config.get(k), update=use_global_config)

    return cfg


def configure_pipeline(*args, **kwargs):
    global config
    config = mkidcore.config.load(*args, **kwargs)
    return config


def h5_for_MKIDodd(observing_data_desc):
    return os.path.join(config.paths.out, '{}.h5'.format(observing_data_desc.start))


def wavecal_id(wavedata_id, wavecal_cfg=None):
    """
    Compute a wavecal id string from a wavedata id string and either the active or a specified wavecal config
    """
    if wavecal_cfg is None:
        global config
        wavecal_cfg = config.wavecal
    config_hash = hashlib.md5(str(wavecal_cfg).encode()).hexdigest()
    return 'wavcal_{}_{}'.format(wavedata_id, config_hash[-8:])


class MKIDTimerange(object):
    yaml_tag = u'!ob'

    def __init__(self, name, start, duration=None, stop=None, _common=None, wavecal=None, flatcal=None, background=None):

        if _common is not None:
            self.__dict__.update(_common)

        if duration is None and stop is None:
            raise ValueError('Must specify stop or duration')
        if duration is not None and stop is not None:
            raise ValueError('Must only specify stop or duration')
        if duration is not None and duration > 43200:
            raise ValueError('Specified duration is longer than 12 hours!')
        self.start = int(start)

        if duration is not None:
            self.stop = self.start + int(np.ceil(duration))

        if stop is not None:
            self.stop = int(np.ceil(stop))

        if self.stop < self.start:
            raise ValueError('Stop ({}) must come after start ({})'.format(self.stop,self.start))


        self.name = str(name)
        self.background = str(background)
    def __str__(self):
        return '{} t={}:{}s'.format(self.name, self.start, self.duration)

    @property
    def date(self):
        return datetime.utcfromtimestamp(self.start)

    @property
    def instrument_info(self):
        #TODO remove or sync this with the metadata in the H5 files
        return InstrumentInfo(beammap=self.beammap, platescale=config.instrument.platescale * units.mas)

    @property
    def beammap(self):
        #TODO need to move beammap from pipe.yml to data.yml
        return config.beammap

    @property
    def duration(self):
        return self.stop-self.start

    @classmethod
    def from_yaml(cls, loader, node):
        d = dict(loader.construct_pairs(node))  #WTH this one line took half a day to get right
        name = d.pop('name')
        start = d.pop('start', None)
        stop = d.pop('stop', None)
        duration = d.pop('duration', None)
        background = d.pop('background', None)
        return cls(name, start, duration=duration, stop=stop, wavecal=d.pop('wavecal', None),
                   flatcal=d.pop('flatcal', None),background=background, _common=d)

    @property
    def timerange(self):
        return self.start, self.stop

    @property
    def timeranges(self):
        return self.timerange,

    @property
    def h5(self):
        return h5_for_MKIDodd(self)


class MKIDObservation(object):
    """requires keys name, wavecal, flatcal, wcscal, and all the things from ob"""
    yaml_tag = u'!sob'
    #TODO make subclass of MKIDTimerange, keep in sync for now
    def __init__(self, name, start, duration=None, stop=None, wavecal=None, flatcal=None, wcscal=None, _common=None):

        if _common is not None:
            self.__dict__.update(_common)

        if duration is None and stop is None:
            raise ValueError('Must specify stop or duration')
        if duration is not None and stop is not None:
            raise ValueError('Must only specify stop or duration')
        self.start = int(start)

        if duration is not None:
            self.stop = self.start + int(np.ceil(duration))

        if stop is not None:
            self.stop = int(np.ceil(stop))

        if self.stop < self.start:
            raise ValueError('Stop ({}) must come after start ({})'.format(self.stop,self.start))

        self.wavecal = wavecal
        self.flatcal = flatcal
        self.wcscal = wcscal

        self.name = str(name)

    def __str__(self):
        return '{} t={}:{}s'.format(self.name, self.start, self.duration)

    @property
    def date(self):
        return datetime.utcfromtimestamp(self.start)

    @property
    def instrument_info(self):
        #TODO remove or sync this with the metadata in the H5 files
        return InstrumentInfo(beammap=self.beammap, platescale=config.instrument.platescale * units.mas)

    @property
    def beammap(self):
        #TODO need to move beammap from pipe.yml to data.yml
        return config.beammap

    @property
    def duration(self):
        return self.stop-self.start

    @classmethod
    def from_yaml(cls, loader, node):
        d = dict(loader.construct_pairs(node))  #WTH this one line took half a day to get right
        name = d.pop('name')
        start = d.pop('start', None)
        stop = d.pop('stop', None)
        duration = d.pop('duration', None)
        return cls(name, start, duration=duration, stop=stop, wavecal=d.pop('wavecal', None),
                   flatcal=d.pop('flatcal', None), wcscal=d.pop('wcscal', None), _common=d)

    @property
    def timerange(self):
        return self.start, self.stop

    @property
    def timeranges(self):
        return self.timerange,

    @property
    def h5(self):
        return h5_for_MKIDodd(self)

    @property
    def metadata(self):
        exclude = ('wavecal', 'flatcal', 'wcscal', 'start', 'stop')
        d = {k: v for k, v in self.__dict__.items() if k not in exclude}
        try:
            wc = wavecal_id(self.wavecal.id)
        except AttributeError:
            wc = 'None'
        try:
            fc = self.flatcal.id
        except AttributeError:
            fc = 'None'
        d2 = dict(wavecal=wc, flatcal=fc, platescale=self.wcscal.platescale,
                  dither_ref=self.wcscal.dither_ref, dither_home=self.wcscal.dither_home,
                  device_orientation=self.wcscal.device_orientation)
        d.update(d2)
        return d


class MKIDWavedataDescription(object):
    """requires keys name and data"""
    yaml_tag = u'!wc'

    def __init__(self, name, data, backgrounds):

        self.name = name
        self.data = data
        self.backgrounds = backgrounds
        self.data.sort(key=lambda x: x.start)
        self.wavelengths = []
        self.backgrounds_list = []

    @property
    def timeranges(self):
        for o in self.data:
            yield o.timerange
        if self.backgrounds:
            for o in self.backgrounds:
                yield o.timerange

    @property
    def wavelengths(self):
        return [getnm(x.name) for x in self.data]

    @property
    def backgrounds_list(self):
        backgrounds_list = np.zeros(len(self.wavelengths), dtype=[('wavelength', 'float'), ('background', '<U11'),
                                                                  ('start_time', 'int')])
        backgrounds_list['wavelength'] = self.wavelengths
        if len(self.backgrounds) == 0:
            return backgrounds_list
        backgrounds_list['background'] = [x.background if x.background else '' for x in self.data]
        backgrounds_list['start_time'] = [x.start if x.name == y.background else 0 for y in self.data
                                          for x in self.backgrounds]
        return backgrounds_list

    def __str__(self):
        return '\n '.join("{} ({}-{})".format(x.name, x.start, x.stop) for x in self.data)

    @property
    def id(self):
        meanstart = int(np.mean([x[0] for x in self.timeranges]))
        hash = hashlib.md5(str(self).encode()).hexdigest()
        return datetime.utcfromtimestamp(meanstart).strftime('%Y-%m-%d-%H%M_') + hash[-8:]

    @property
    def path(self):
        return os.path.join(config.paths.database, wavecal_id(self.id)+'.npz')


class MKIDFlatdataDescription(object):
    """attributes name and either ob or wavecal"""
    yaml_tag = u'!fc'

    def __init__(self, name, ob=None, wavecal=None, _common=None):
        if _common is not None:
            self.__dict__.update(_common)

        self.name = name
        self.ob = ob
        self.wavecal = wavecal

    @property
    def id(self):
        try:
            return 'flatcal_{}.h5'.format(self.ob.start)
        except AttributeError:
            return 'flatcal_{}.h5'.format(wavecal_id(self.wavecal.id))

    @property
    def path(self):
        #TODO flat data doesn't really have a path, tts the flatcal that has a phath
        return os.path.join(config.paths.database, self.id)

    @property
    def timerange(self):
        return self.ob.timerange if self.ob is not None else None

    @property
    def timeranges(self):
        return (self.timerange, ) if self.timerange else tuple()

    def __str__(self):
        return '{}: {}'.format(self.name, self.ob if self.ob is not None else self.wavecal)

    @classmethod
    def from_yaml(cls, loader, node):
        d = dict(loader.construct_pairs(node))  #WTH this one line took half a day to get right
        name = d.pop('name')
        ob = d.pop('ob', None)
        wavecal = d.pop('wavecal', None)
        return cls(name, ob=ob, wavecal=wavecal, _common=d)


class MKIDWCSCalDescription(object):
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
    yaml_tag = '!wcscal'

    def __init__(self, name, ob=None, platescale=None, dither_ref=None, _common=None,
                 dither_home=None):
        self.name = name
        self.ob = ob
        self.platescale = platescale
        self.dither_ref = dither_ref
        self.dither_home = dither_home

        if (platescale is None or dither_ref is None or dither_home is None) and ob is None:
            raise ValueError('ob must be specified if platescale, dither_ref, dither_home are not')

        if _common is not None:
            self.__dict__.update(_common)

    @classmethod
    def from_yaml(cls, loader, node):
        d = dict(loader.construct_pairs(node))
        name = d.pop('name')
        ob = d.pop('ob', None)
        platescale = d.pop('platescale', None)
        dither_ref = d.pop('dither_ref', None)
        dither_home = d.pop('dither_home', None)
        return cls(name, ob=ob, platescale=platescale, dither_ref=dither_ref, _common=d, dither_home=dither_home)


def parseLegacyDitherLog(file):
    with open(file) as f:
        lines = f.readlines()

    tofloat = lambda x: list(map(float, x.replace('[', '').replace(']', '').split(',')))
    proc = lambda x: str.lower(str.strip(x))
    d = dict([list(map(proc, l.partition('=')[::2])) for l in lines])

    # Support legacy legacy names
    if 'endtimes' not in d:
        d['endtimes'] = d['stoptimes']

    inttime = int(d['inttime'])

    startt = tofloat(d['starttimes'])
    endt = tofloat(d['endtimes'])
    xpos = tofloat(d['xpos'])
    ypos = tofloat(d['ypos'])

    return startt, endt, list(zip(xpos, ypos)), inttime


class MKIDDitheredObservation(object):
    yaml_tag = '!dither'

    def __init__(self, name, wavecal, flatcal, wcscal, obs=None, byLegacyFile=None, byTimestamp=None,
                 use=None, _common=None):
        """
        Obs, byLegacy, or byTimestamp must be specified. byTimestamp is normal.

        Obs must be a list of MKIDObservations
        byLegacyFile must be a legacy dither log file (starttimes, endtimes, xpos,ypos)
        byTimestamp mut be a timestamp or a datetime that falls in the range of a dither in a ditherlog on the path
        obs>byTimestamp>byLegacyFile
        """
        if _common is not None:
            self.__dict__.update(_common)

        self.name = name
        self.file = byLegacyFile
        self.wavecal = wavecal
        self.flatcal = flatcal
        self.wcscal = wcscal

        if obs is not None:
            self.obs=obs
            self.pos = None
            self.inttime = None
            return
        elif byTimestamp is not None:
            startt, endt, pos = getDitherInfoByTime(byTimestamp)
            self.inttime = (np.array(endt) - np.array(startt))[0]

        else:
            startt, endt, pos, inttime= parseLegacyDitherLog(byLegacyFile)
            self.inttime = inttime

        if use is None:
            self.use = list(range(len(startt)))
        else:
            self.use = [use] if isinstance(use, int) else derangify(use)

        startt = [startt[i] for i in self.use]
        endt = [endt[i] for i in self.use]

        self.pos = [pos[i] for i in self.use]

        self.obs = []
        for i, b, e, p in zip(self.use, startt, endt, self.pos):
            name = '{}_({})_{}'.format(self.name, '', i) #TODO: removed self.file - fix w/ something sensible
            _common.pop('dither_pos', None)
            _common['dither_pos'] = p
            self.obs.append(MKIDObservation(name, b, stop=e, wavecal=wavecal, flatcal=flatcal, wcscal=wcscal,
                                            _common=_common))

    @classmethod
    def from_yaml(cls, loader, node):
        d = dict(loader.construct_pairs(node))
        if 'approximate_time' in d:
            d.pop('file',None)
            return cls(d.pop('name'), d.pop('wavecal', None), d.pop('flatcal', None), d.pop('wcscal'),
                       byTimestamp=d.pop('approximate_time'), use=d.pop('use', None), _common=d)

        if not os.path.isfile(d['file']):
            getLogger(__name__).info('Treating {} as relative dither path.'.format(d['file']))
            d['file'] = os.path.join(config.paths.dithers, d['file'])
        return cls(d.pop('name'), d.pop('wavecal', None), d.pop('flatcal', None), d.pop('wcscal'), byLegacyFile=d.pop('file'),
                   use=d.pop('use', None), _common=d)

    @property
    def timeranges(self):
        for o in self.obs:
            yield o.timerange


class MKIDObservingDataset(object):
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

        for o in self.all_observations:
            o.wavecal = wcdict.get(o.wavecal, o.wavecal)

        for o in self.science_observations:
            o.flatcal = fcdict.get(o.flatcal, o.flatcal)
            o.wcscal = wcsdict.get(o.wcscal, o.wcscal)

        for fc in self.flatcals:
            try:
                fc.wavecal = wcdict.get(fc.wavecal, fc.wavecal)
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
            try:
                for tr in x.timeranges:
                    yield tr
            except AttributeError:
                try:
                    yield x.timerange
                except AttributeError:
                    pass
            except StopIteration:
                pass

    @property
    def wavecals(self):
        return [r for r in self.meta if isinstance(r, MKIDWavedataDescription)]

    @property
    def flatcals(self):
        return [r for r in self.meta if isinstance(r, MKIDFlatdataDescription)]

    @property
    def wcscals(self):
        return [r for r in self.meta if isinstance(r, MKIDWCSCalDescription)]

    @property
    def dithers(self):
        return [r for r in self.meta if isinstance(r, MKIDDitheredObservation)]

    @property
    def sobs(self):
        return [r for r in self.meta if isinstance(r, MKIDObservation)]

    @property
    def all_observations(self):
        return ([o for o in self.meta if isinstance(o, MKIDObservation)] +
                [o for d in self.meta if isinstance(d, MKIDDitheredObservation) for o in d.obs] +
                [d.ob for d in self.meta if isinstance(d, MKIDFlatdataDescription) and d.ob is not None] +
                [d.ob for d in self.meta if isinstance(d, MKIDWCSCalDescription) and d.ob is not None])

    @property
    def science_observations(self):
        return ([o for o in self.meta if isinstance(o, MKIDObservation)] +
                [o for d in self.meta if isinstance(d, MKIDDitheredObservation) for o in d.obs])

    @property
    def wavecalable(self):
        return self.all_observations

    def by_name(self, name):
        d = [d for d in self.meta if d.name == name]
        try:
            return d[0]
        except IndexError:
            raise ValueError('Item "{}" not found in data {}'.format(name, self.yml))

    @property
    def description(self):
        """Return a string describing the data"""
        s = ("Wavecals:\n{wc}\n"
             "Flatcals:\n{fc}\n"
             "Dithers:\n{dithers}\n"
             "Single Obs:\n{obs}".format(wc=('\t-'+'\n\t-'.join([str(w).replace('\n','\n\t')
                                                               for w in self.wavecals])) if  self.wavecals else
                                         '\tNone',
                                      fc=('\t-'+'\n\t-'.join([str(f) for f in self.flatcals])) if self.flatcals else
                                         '\tNone',
                                      dithers='Not implemented',
                                      obs='Not implemented'))
        return s


class MKIDOutput(object):
    yaml_tag = '!out'

    def __init__(self, name, dataname, kind, startw=None, stopw=None, filename='',_extra=None):
        """
        :param name: a name
        :param dataname: a name of a data association
        :param kind: stack|spatial|spectral|temporal|list|image|movie
        :param startw: wavelength start
        :param stopw: wavelength stop
        :param filename: an optional relative or fully qualified path

        Kind 'movie' requires _extra keys timestep and either frameduration or movieduration with frameduration
        taking precedence. startt and stopt may be included as well and are RELATIVE to the start of the file.
        """
        self.name = name
        self.startw = getnm(startw) if startw is not None else None
        self.stopw = getnm(stopw) if stopw is not None else None
        self.kind = kind.lower()
        opt = ('stack', 'spatial', 'spectral', 'temporal', 'list', 'image','movie')
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
        return self.kind in ('stack','spatial','spectral','temporal','list')

    @property
    def wants_movie(self):
        return self.kind == 'movie'

    @classmethod
    def from_yaml(cls, loader, node):
        d = dict(loader.construct_pairs(node))
        return cls(d.pop('name'), d.pop('data'), d.pop('kind'),
                   startw=d.pop('startw', None), stopw=d.pop('stopw', None),
                   filename=d.pop('filename', ''), _extra=d)
        # #TODO I don't know why I used extract_from_node here and dict(loader.construct_pairs(node)) elsewhere
        # d = mkidcore.config.extract_from_node(loader, ('name', 'data', 'kind', 'stopw', 'startw', 'filename'), node)
        # return cls(d['name'], d['data'], d['kind'], d.get('startw', None), d.get('stopw', None),
        #            d.get('filename', ''), )

    @property
    def input_timeranges(self):
        return list(self.data.timeranges)+list(self.data.wavecal.timeranges)+list(self.data.flatcal.timeranges)

    @property
    def output_file(self):
        global config
        if not self.filename:
            # TODO generate the filename programatically if one isn't specified
            raise ValueError('No output filename for output, it may be time to add code for a default')
        if os.pathsep in self.filename:
            return self.filename
        else:
            return os.path.join(config.paths.out, self.filename)


class MKIDOutputCollection:
    def __init__(self, file, datafile=''):
        self.yml = file
        self.meta = mkidcore.config.load(file)

        if datafile:
            data = load_data_description(datafile)
        else:
            global _dataset
            data = _dataset

        self.dataset = data

        for o in self.meta:
            try:
                o.data = data.by_name(o.data)
            except ValueError as e:
                getLogger(__name__).critical(e)

    @property
    def outputs(self):
        return self.meta

    @property
    def input_timeranges(self):
        return set([r for o in self.outputs for r in o.input_timeranges])

    def __str__(self):
        return 'Output "{}"'.format(self.name)


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


def parse_ditherlog(file):
    global _parsedDitherLogs
    with open(file, 'r') as f:
        lines = f.readlines()
    for i, l in enumerate(lines):
        if not l.strip().startswith('starts'):
            continue
        try:
            assert lines[i+1].strip().startswith('ends') and lines[i+2].strip().startswith('path')
            starts = ast.literal_eval(l.partition('=')[2])
            ends = ast.literal_eval(lines[i + 1].partition('=')[2])
            pos = ast.literal_eval(lines[i + 2].partition('=')[2])
        except (AssertionError, IndexError, ValueError, SyntaxError):
            # Bad dither
            getLogger(__name__).error('Dither l{}:{} corrupt'.format(i-1, lines[i-1]))
            continue
        _parsedDitherLogs[(min(starts), max(ends))] = (starts, ends, pos)


def getDitherInfoByTime(time):
    global _parsedDitherLogs
    if not _parsedDitherLogs:
        for f in glob(os.path.join(config.paths.dithers, 'dither_*.log')):
            parse_ditherlog(f)

    if isinstance(time, datetime):
        time = time.timestamp()

    for (t0, t1), v in _parsedDitherLogs.items():
        if t0 - (t1 - t0) <= time <= t1:
            return v

    raise ValueError('No dither found for time {}'.format(time))


def load_observing_metadata(files=tuple(), include_database=True):
    """Return a list of mkidcore.config.ConfigThings with the contents of the metadata from observing"""
    global config
    files = list(files)
    if config is not None and include_database:
        files+=glob(os.path.join(config.paths.database, 'obslog*.json'))
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
    for fc in dataset.flatcals:
        try:
            fc.wavecal = wcdict.get(fc.wavecal, fc.wavecal)
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

    if not no_global:
        global _dataset
        _dataset = dataset

    return dataset

load_output_description = MKIDOutputCollection


def n_cpus_available(max=np.inf):
    """Returns n threads -4 modulo pipelinesettings"""
    global config
    mcpu = min(mp.cpu_count()*2 - 4, max)
    try:
        mcpu = int(min(config.ncpu, mcpu))
    except Exception:
        pass
    return mcpu


def logtoconsole(file=''):
    logs = (create_log('mkidcore'), create_log('mkidreadout'), create_log('mkidpipeline'), create_log('__main__'))
    if file:
        import logging
        handler = MakeFileHandler(file)
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s (pid=%(process)d)'))
        for l in logs:
            l.addHandler(handler)


yaml.register_class(MKIDTimerange)
yaml.register_class(MKIDObservation)
yaml.register_class(MKIDWavedataDescription)
yaml.register_class(MKIDFlatdataDescription)
yaml.register_class(MKIDWCSCalDescription)
yaml.register_class(MKIDDitheredObservation)
yaml.register_class(MKIDOutput)

