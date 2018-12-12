import mkidcore.config

config = None

yaml = mkidcore.config.yaml


def load(*args, **kwargs):
    global config
    c = mkidcore.config.load(*args, **kwargs)
    config = c
    return c


