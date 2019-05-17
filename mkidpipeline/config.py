import mkidcore.config
import numpy as np
import os
from glob import glob
import hashlib
from datetime import datetime
import multiprocessing as mp
from mkidcore.corelog import getLogger, create_log
from mkidcore.utils import getnm
import pkg_resources as pkg
from mkidcore.objects import Beammap, DashboardState
from astropy.coordinates import SkyCoord
from collections import namedtuple
import astropy.units as units
from mkidpipeline.hdf.photontable import ObsFile

#TODO this is a placeholder to help integrating metadata
InstrumentInfo = namedtuple('InstrumentInfo', ('beammap', 'platescale'))

#Ensure that the beammap gets registered with yaml, technically the import does this
#but without this note an IDE or human might remove the import
Beammap()

config = None

yaml = mkidcore.config.yaml

load_data_description = mkidcore.config.load

pipline_settings = ('beammap', 'paths', 'templar', 'instrument', 'ncpu')

_COMMON_KEYS = ('comments', 'meta', 'header', 'out')


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


def _build_common(yaml_loader, yaml_node):
    # TODO flesh out as needed
    pairs = yaml_loader.construct_pairs(yaml_node)
    return {k: v for k, v in pairs if k in _COMMON_KEYS}


def h5_for_MKIDodd(observing_data_desc):
    return os.path.join(config.paths.out, '{}.h5'.format(observing_data_desc.start))


class MKIDTimerange(object):
    yaml_tag = u'!ob'

    def __init__(self, name, start, duration=None, stop=None, _common=None, wavecal=None, flatcal=None):

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
                   flatcal=d.pop('flatcal', None), _common=d)

    @property
    def timerange(self):
        return self.start, self.stop

    @property
    def timeranges(self):
        return self.timerange,

    @property
    def h5(self):
        return h5_for_MKIDodd(self)

    def lookup_coordinates(self, queryname=''):
        return SkyCoord.from_name(queryname if queryname else self.name)


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
        d = {k: v for k,v in self.__dict__.items() if k not in exclude}
        d2 = dict(wavecal=self.wavecal.id, flatcal=self.flatcal.id, platescale=self.wcscal.platescale,
                  dither_ref=self.wcscal.dither_ref, dither_home=self.wcscal.dither_home)
        d.update(d2)
        return d


class MKIDWavedataDescription(object):
    """requires keys name and data"""
    yaml_tag = u'!wc'

    def __init__(self, name, data):

        self.name = name
        self.data = data
        self.data.sort(key=lambda x: x.start)
        self.wavelengths

    @property
    def timeranges(self):
        for o in self.data:
            yield o.timerange

    @property
    def wavelengths(self):
        return [getnm(x.name) for x in self.data]

    def __str__(self):
        return '\n '.join("{} ({}-{})".format(x.name, x.start, x.stop) for x in self.data)

    @property
    def id(self):
        meanstart = int(np.mean([x[0] for x in self.timeranges]))
        hash = hashlib.md5(str(self).encode()).hexdigest()
        return datetime.utcfromtimestamp(meanstart).strftime('%Y-%m-%d %H%M') + hash + '.npz'

    @property
    def path(self):
        return os.path.join(config.paths.database, self.id)


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
            return 'flatcal_{}.h5'.format(str(self.wavecal).replace(os.path.sep, '_'))

    @property
    def path(self):
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


class MKIDDitheredObservation(object):
    yaml_tag = '!dither'

    def __init__(self, name, file, wavecal, flatcal, wcscal, _common=None):

        if _common is not None:
            self.__dict__.update(_common)

        self.name = name
        self.file = file
        self.wavecal = wavecal
        self.flatcal = flatcal
        self.wcscal = wcscal

        with open(file) as f:
            lines = f.readlines()

        tofloat = lambda x: map(float, x.replace('[','').replace(']','').split(','))
        proc = lambda x: str.lower(str.strip(x))
        d = dict([list(map(proc, l.partition('=')[::2])) for l in lines])

        # Support legacy names
        if 'npos' not in d:
            d['npos'] = d['nsteps']
        if 'endtimes' not in d:
            d['endtimes'] = d['stoptimes']

        self.inttime = int(d['inttime'])
        self.nsteps = int(d['npos'])
        self.pos = list(zip(tofloat(d['xpos']), tofloat(d['ypos'])))

        self.obs = []
        for i, (b, e) in enumerate(zip(tofloat(d['starttimes']), tofloat(d['endtimes']))):
            name = '{}_({})_{}'.format(self.name, os.path.basename(self.file), i)
            self.obs.append(MKIDObservation(name, b, stop=e, wavecal=wavecal, flatcal=flatcal,
                                            wcscal=wcscal, _common=_common))

    @classmethod
    def from_yaml(cls, loader, node):
        #TODO I don't know why I used extract_from_node here and dict(loader.construct_pairs(node)) elsewhere
        d = mkidcore.config.extract_from_node(loader, ('file', 'name', 'wavecal', 'flatcal', 'wcscal'), node)
        if not os.path.isfile(d['file']):
            file = os.path.join(config.paths.dithers, d['file'])
            getLogger(__name__).info('Treating {} as relative dither path.'.format(d['file']))
        else:
            file = d['file']
        return cls(d['name'], file, d['wavecal'], d['flatcal'], d['wcscal'], _common=_build_common(loader, node))

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

    def __init__(self, name, dataname, kind, startw=None, stopw=None, filename=''):
        """
        :param name: a name
        :param dataname: a name of a data association
        :param kind: stack|spatial|spectral|temporal|list|image
        :param startw: wavelength start
        :param stopw: wavelength stop
        :param filename: an optional relative or fully qualified path
        """
        self.name = name
        self.startw = getnm(startw) if startw is not None else None
        self.stopw = getnm(stopw) if stopw is not None else None
        self.kind = kind.lower()
        opt = ('stack', 'spatial', 'spectral', 'temporal', 'list', 'image')
        if kind.lower() not in opt:
            raise ValueError('Output kind "{}" is not one of "{}"'.format(name, ', '.join(opt)))
        self.enable_noise = True
        self.enable_photom = True
        self.enable_ssd = True
        self.filename = filename
        self.data = dataname

    @property
    def wants_image(self):
        return self.kind == 'image'

    @property
    def wants_drizzled(self):
        return self.kind != 'image'

    @classmethod
    def from_yaml(cls, loader, node):
        d = mkidcore.config.extract_from_node(loader, ('name', 'data', 'kind', 'stopw', 'startw', 'filename'), node)
        return cls(d['name'], d['data'], d['kind'], d.get('startw', None), d.get('stopw', None),
                   d.get('filename', ''))

    @property
    def input_timeranges(self):
        return list(self.data.timeranges)+list(self.data.wavecal.timeranges)+list(self.data.flatcal.timeranges)

    @property
    def output_file(self):
        global config
        #TODO generate the filename programatically if one isn't specified
        return os.path.join(config.paths.out, self.filename)


class MKIDOutputCollection:
    def __init__(self, file, datafile=''):
        self.yml = file
        self.meta = mkidcore.config.load(file)

        if datafile:
            data = load_data_description(datafile)
        else:
            global global_dataset
            data = global_dataset

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


def select_metadata_for_h5(starttime, duration, metadata_source):
    """Metadata that goes into an H5 consists of records within the duration"""
    # Select the nearest metadata to the midpoint
    start = datetime.fromtimestamp(starttime)
    time_since_start = np.array([(md.utc - start).total_seconds() for md in metadata_source])
    ok = (time_since_start < duration) & (time_since_start >= 0)
    return [metadata_source[i] for i in np.where(ok)[0]]


def associate_metadata(dataset):
    """Function associates things not known at hdf build time (e.g. that aren't in the bin files)"""

    # Retrieve metadata database
    metadata = load_observing_metadata()

    # Associate metadata
    for ob in dataset.all_observations:
        o = ObsFile(ob.h5, mode='w')
        ob_md = ob.metadata
        md = select_metadata_for_h5(o.startTime, o.duration, metadata)
        for m in md:
            m.update(ob_md)
        o.attach_metadata(md)
        del o


def load_observing_metadata(files=tuple()):
    global config
    files = list(files) + glob(os.path.join(config.paths.database), 'obslog*.json')
    data = []
    for f in files:
        with open(f, 'r') as of:
            data += of.readlines()
    metadata = [DashboardState(d) for d in data]

    return metadata


def load_data_description(file):
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

    return dataset


def n_cpus_available(max=np.inf):
    """Returns n threads -4 modulo pipelinesettings"""
    global config
    mcpu = min(mp.cpu_count()*2 - 4, max)
    try:
        mcpu = int(min(config.ncpu, mcpu))
    except Exception:
        pass
    return mcpu


def logtoconsole():
    create_log('mkidcore')
    create_log('mkidreadout')
    create_log('mkidpipeline')
    create_log('__main__')


yaml.register_class(MKIDTimerange)
yaml.register_class(MKIDObservation)
yaml.register_class(MKIDWavedataDescription)
yaml.register_class(MKIDFlatdataDescription)
yaml.register_class(MKIDWCSCalDescription)
yaml.register_class(MKIDDitheredObservation)
yaml.register_class(MKIDOutput)

