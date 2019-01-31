import mkidcore.config
import numpy as np
import os
import hashlib
from datetime import datetime
import astropy.units
import multiprocessing as mp
from mkidcore.corelog import getLogger, create_log
import pkg_resources as pkg
from mkidreadout.configuration.beammap.beammap import Beammap

#Ensure that the beammap gets registered with yaml
Beammap()

config = None

yaml = mkidcore.config.yaml


def load_task_config(file):
    global config
    if not isinstance(file, str):
        return file
    cfg = mkidcore.config.load(file)
    if config is not None:
        cfg.register('beammap', config.beammap, update=True)
        cfg.register('paths', config.paths, update=True)
        cfg.register('templar', config.templar, update=True)
        cfg.register('instrument', config.instrument, update=True)
    else:
        getLogger(__name__).warning('Loading task configuration when pipeline not fully configured.')
    return cfg


def configure_pipeline(*args, **kwargs):
    global config
    config = mkidcore.config.load(*args, **kwargs)
    return config


configure_pipeline(pkg.resource_filename('mkidpipeline','pipe.yml'))

load_data_description = mkidcore.config.load


_COMMON_KEYS = ('comments', 'meta', 'header', 'out')


def _build_common(yaml_loader, yaml_node):
    # TODO flesh out as needed
    pairs = yaml_loader.construct_pairs(yaml_node)
    return {k: v for k, v in pairs if k in _COMMON_KEYS}


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


class MKIDWavedataDescription(object):
    yaml_tag = u'!wc'

    def __init__(self, data):
        self.data = data
        self.data.sort(key=lambda x: x.start)

    @property
    def timeranges(self):
        for o in self.data:
            yield o.start, o.stop

    @property
    def wavelengths(self):
        def getnm(x):
            try:
                return astropy.units.Unit(x).to('nm')
            except astropy.units.UnitConversionError:
                return float(x)
        return [getnm(x.name) for x in self.data]

    def __str__(self):
        return '\n '.join("{} ({}-{})".format(x.name, x.start, x.stop) for x in self.data)

    @property
    def id(self):
        meanstart = int(np.mean([x[0] for x in self.timeranges]))
        hash = hashlib.md5(str(self).encode()).hexdigest()
        return datetime.utcfromtimestamp(meanstart).strftime('%Y-%m-%d %H%M') + hash


class MKIDFlatdataDescription(object):
    yaml_tag = u'!fc'

    @property
    def id(self):
        return 'calsol_{}.h5'.format(self.data.start)

    # def __init__(self, data):
    #     self.data = data

    # @classmethod
    # def from_yaml(cls, loader, node):
    #     return MKIDObservingDataDescription.from_yaml(cls, loader, node)
    #     d = dict(loader.construct_pairs(node))  #WTH this one line took half a day to get right
    #     name = d.pop('name')
    #     start = d.pop('start', None)
    #     stop = d.pop('stop', None)
    #     duration = d.pop('duration', None)
    #     return cls(name, start, duration=duration, stop=stop, _common=d)


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
        self.obs = [MKIDObservingDataDescription('{}_({})_{}'.format(self.name,os.path.basename(self.file), i),
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
            yield o.start, o.stop


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
                    yield x.start, x.stop
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
    def observations(self):
        return ([o for o in self.meta if isinstance(o, MKIDObservingDataDescription)] +
                [o for d in self.meta if isinstance(d, MKIDObservingDither) for o in d.obs])

    @property
    def wavecalable(self):
        return [fc.ob for fc in self.flatcals]+self.observations


def load_data_description(file):
    return MKIDObservingDataset(file)


def get_h5_path(obs_data_descr):
    global config
    return os.path.join(config.paths.out, '{}.h5'.format(obs_data_descr.start))


def n_cpus_available():
    global config
    mcpu = mp.cpu_count()*2 - 4
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


def assiciate_wavecals(dataset):
    wcdict = {w.name: os.path.join(config.paths.database, w.id+'.npz') for w in dataset.wavecals}
    return [(o, wcdict.get(o.wavecal, o.wavecal)) for o in dataset.wavecalable if o.wavecal is not None]

yaml.register_class(MKIDObservingDataDescription)
yaml.register_class(MKIDWavedataDescription)
yaml.register_class(MKIDFlatdataDescription)
yaml.register_class(MKIDObservingDither)
