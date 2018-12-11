from mkidcore.config import load as _load

config = None


def load(*args, **kwargs):
    global config
    c = _load(*args, **kwargs)
    config = c
    return c