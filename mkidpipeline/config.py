import mkidcore.config
import numpy as np
import os
import hashlib
from datetime import datetime
import multiprocessing as mp
from mkidcore.corelog import getLogger, create_log
from mkidcore.utils import getnm
import pkg_resources as pkg
from mkidcore.objects import Beammap
from astropy.coordinates import SkyCoord
from collections import namedtuple
import astropy.units as units

InstrumentInfo = namedtuple('InstrumentInfo',('beammap','platescale'))

#Ensure that the beammap gets registered with yaml, technically the import does this
#but without this note an IDE or human might remove the import
Beammap()

config = None

yaml = mkidcore.config.yaml

pipline_settings = ('beammap', 'paths', 'templar', 'instrument', 'ncpu')


def load_task_config(file, use_global_config=True):
    #if pipeline is not configured then do all needed to get it online, loading defaults and overwriting them with
    # the task cfg
    #if pipeline has been configured by user then choice of pip or task, but update with all pipeline stuff
    #Never edit an existing pipeline config

    global config

    cfg = mkidcore.config.load(file) if isinstance(file, str) else file  #assume someone passing through a config

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


load_data_description = mkidcore.config.load


_COMMON_KEYS = ('comments', 'meta', 'header', 'out')


def _build_common(yaml_loader, yaml_node):
    # TODO flesh out as needed
    pairs = yaml_loader.construct_pairs(yaml_node)
    return {k: v for k, v in pairs if k in _COMMON_KEYS}


def h5_for_MKIDodd(observing_data_desc):
    return os.path.join(config.paths.out, '{}.h5'.format(observing_data_desc.start))


class MKIDObservingDataDescription(object):
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

        self.wavecal = wavecal
        self.flatcal = flatcal

        self.name = str(name)

    def __str__(self):
        return '{} t={}:{}s'.format(self.name, self.start, self.duration)

    @property
    def date(self):
        return datetime.utcfromtimestamp(self.start)

    @property
    def instrument_info(self):
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
        return cls(name, start, duration=duration, stop=stop, _common=d,
                   wavecal=d.pop('wavecal', None), flatcal=d.pop('flatcal', None))

    @property
    def timerange(self):
        return self.start, self.stop

    @property
    def h5(self):
        return h5_for_MKIDodd(self)

    def lookup_coordinates(self, queryname=''):
        return SkyCoord.from_name(queryname if queryname else self.name)


class MKIDWavedataDescription(object):
    yaml_tag = u'!wc'

    def __init__(self, data):
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


class MKIDFlatdataDescription(object):
    yaml_tag = u'!fc'

    @property
    def id(self):
        try:
            return 'flatcal_{}.h5'.format(self.ob.start)
        except AttributeError:
            return 'flatcal_{}.h5'.format(str(self.wavecal).replace(os.path.sep, '_'))

    @property
    def timerange(self):
        return self.ob.timerange

    def __str__(self):
        return '{}: {}'.format(self.name, self.ob if hasattr(self,'ob') else self.wavecal)


class MKIDObservingDither(object):
    yaml_tag = '!dither'

    def __init__(self, name, file, wavecal, flatcal, _common=None):

        if _common is not None:
            self.__dict__.update(_common)

        self.name = name
        self.file = file
        self.wavecal = wavecal
        self.flatcal = flatcal
        with open(file) as f:
            lines = f.readlines()

        tofloat = lambda x: map(float, x.replace('[','').replace(']','').split(','))
        proc = lambda x: str.lower(str.strip(x))
        d = dict([list(map(proc, l.partition('=')[::2])) for l in lines])

        #support legacy names
        if 'npos' not in d:
            d['npos'] = d['nsteps']
        if 'endtimes' not in d:
            d['endtimes'] = d['stoptimes']

        self.inttime = int(d['inttime'])
        self.nsteps = int(d['npos'])
        self.pos = list(zip(tofloat(d['xpos']), tofloat(d['ypos'])))
        self.obs = [MKIDObservingDataDescription('{}_({})_{}'.format(self.name, os.path.basename(self.file), i),
                                                 b, stop=e, wavecal=wavecal, flatcal=flatcal)
                    for i, (b, e) in enumerate(zip(tofloat(d['starttimes']), tofloat(d['endtimes'])))]


    @classmethod
    def from_yaml(cls, loader, node):
        d = mkidcore.config.extract_from_node(loader, ('file', 'name', 'wavecal', 'flatcal'), node)
        if not os.path.isfile(d['file']):
            file = os.path.join(config.paths.dithers, d['file'])
            getLogger(__name__).info('Treating {} as relative dither path.'.format(d['file']))
        else:
            file = d['file']
        return cls(d['name'], file, d['wavecal'], d['flatcal'], _common=_build_common(loader, node))

    @property
    def timeranges(self):
        for o in self.obs:
            yield o.timerange


class MKIDObservingDataset(object):
    def __init__(self, yml):
        self.yml = yml
        self.meta = mkidcore.config.load(yml)

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
    def dithers(self):
        return [d for d in self.meta if isinstance(d, MKIDObservingDither)]

    @property
    def all_observations(self):
        return ([o for o in self.meta if isinstance(o, MKIDObservingDataDescription)] +
                [o for d in self.meta if isinstance(d, MKIDObservingDither) for o in d.obs] +
                [d.ob for d in self.meta if isinstance(d, MKIDFlatdataDescription) and hasattr(d,'ob')])

    @property
    def science_observations(self):
        return ([o for o in self.meta if isinstance(o, MKIDObservingDataDescription)] +
                [o for d in self.meta if isinstance(d, MKIDObservingDither) for o in d.obs])

    @property
    def wavecalable(self):
        return self.all_observations

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

    def __init__(self, name, kind, startw=None, stopw=None, filename=''):
        """
        :param name: a name
        :param kind: stack|spatial|spectral|temporal|list
        :param startw: wavelength start
        :param stopw: wavelength stop
        :param filename: an optional relative or fully qualified path
        """
        self.name = name
        self.startw = getnm(startw) if startw is not None else None
        self.stopw = getnm(stopw) if stopw is not None else None
        self.kind = kind
        self.enable_noise = True
        self.enable_photom = True
        self.enable_ssd = True
        self.filename = filename

    @classmethod
    def from_yaml(cls, loader, node):
        d = mkidcore.config.extract_from_node(loader, ('name', 'kind', 'stopw', 'startw', 'filename'), node)
        return cls(d['name'], d['kind'], d.get('startw', None), d.get('stopw', None), d.get('filename', ''))

    @property
    def input_timeranges(self):
        try:
            tr = self.ob.timeranges
        except AttributeError:
            tr = [self.ob.timerange]
        return tr + self.ob.wavecal.timeranges + [self.flatcal.timerange]

    @property
    def output_file(self):
        global config
        #TODO generate the filename programatically if one isn't specified
        return os.path.join(config.paths.out, self.filename)


def load_data_description(file):
    dataset = MKIDObservingDataset(file)
    wcdict = {w.name: os.path.join(config.paths.database, w.id) for w in dataset.wavecals}
    for o in dataset.all_observations:
        o.wavecal = wcdict.get(o.wavecal, o.wavecal)

    for fc in dataset.flatcals:
        try:
            fc.wavecal = wcdict.get(fc.wavecal, fc.wavecal)
        except AttributeError:
            pass

    fcdict = {f.name: os.path.join(config.paths.database, f.id) for f in dataset.flatcals}
    for o in dataset.science_observations:
        o.flatcal = fcdict.get(o.flatcal, o.flatcal)

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


yaml.register_class(MKIDObservingDataDescription)
yaml.register_class(MKIDWavedataDescription)
yaml.register_class(MKIDFlatdataDescription)
yaml.register_class(MKIDObservingDither)
yaml.register_class(MKIDOutput)