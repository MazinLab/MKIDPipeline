import mkidcore.config
import numpy as np
import os

config = None

yaml = mkidcore.config.yaml


def load(*args, **kwargs):
    global config
    c = mkidcore.config.load(*args, **kwargs)
    config = c
    return c


configure_pipeline = load

load_data_description = mkidcore.config.load


_COMMON_KEYS = ('comments', 'meta', 'header', 'out')


def _build_common(yaml_node):
    # TODO flesh out as needed
    return {x[0].value: x[1].value for x in yaml_node.value if x[0].value.lower() in _COMMON_KEYS}


class MKIDObservingDataDescription(object):
    yaml_tag = u'!ob'

    def __init__(self, name, start, duration=None, stop=None, _common=None):

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

        if stop < start:
            RuntimeWarning('Stop must come after start')

        self.name = str(name)

    @property
    def duration(self):
        return self.stop-self.start

    # @classmethod
    # def to_yaml(cls, representer, node):
    #     raise NotImplementedError
    #     return representer.represent_mapping(cls.yaml_tag, dict(node))


    @classmethod
    def from_yaml(cls, loader, node):
        d = dict(loader.construct_pairs(node))  #WTH this one line took half a day to get right
        name = d.pop('name')
        start = d.pop('start', None)
        stop = d.pop('stop', None)
        duration = d.pop('duration', None)
        return MKIDObservingDataDescription(name, start, duration=duration, stop=stop, _common=d)


yaml.register_class(MKIDObservingDataDescription)



class MKIDWavedataDescription(object):
    yaml_tag = u'!wc'

    def __init__(self, data):
        self.data = data

    @property
    def timeranges(self):
        for o in self.data:
            yield o.start, o.stop


yaml.register_class(MKIDWavedataDescription)


class MKIDFlatdataDescription(MKIDObservingDataDescription):
    yaml_tag = u'!fc'

    def __init__(self, data):
        self.data = data

yaml.register_class(MKIDFlatdataDescription)


class MKIDObservingDither(object):
    yaml_tag = '!dither'

    def __init__(self, name, file, _common=None):

        if _common is not None:
            self.__dict__.update(_common)

        self.name = name
        self.file = file
        with open(file) as f:
            lines = f.readlines()

        tofloat = lambda x: map(float, x.replace('[','').replace(']','').split(','))
        proc = lambda x: str.lower(str.strip(x))
        d = dict([list(map(proc, l.partition('=')[::2])) for l in lines])
        self.inttime = int(d['inttime'])
        self.nsteps = int(d['nsteps'])
        self.pos = list(zip(tofloat(d['xpos']), tofloat(d['ypos'])))
        self.obs = [MKIDObservingDataDescription('{}_({})_{}'.format(self.name,os.path.basename(self.file),i),
                                                 b, stop=e)
                    for i, (b, e) in enumerate(zip(tofloat(d['starttimes']), tofloat(d['endtimes'])))]

    @classmethod
    def from_yaml(cls, loader, node):
        d = mkidcore.config.extract_from_node(('file', 'name'), node)
        return cls(d['name'], d['file'], _common=_build_common(node))

    @property
    def timeranges(self):
        for o in self.obs:
            yield o.start, o.stop


yaml.register_class(MKIDObservingDither)


class MKIDObservingDataset(object):
    def __init__(self, yml):
        self.yml = yml
        self.meta = load(yml)

    @property
    def timeranges(self):
        for x in self.yml:
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
