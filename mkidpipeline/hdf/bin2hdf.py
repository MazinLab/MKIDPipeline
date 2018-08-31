import psutil
import tempfile
import subprocess
import os
from mkidcore.corelog import getLogger
from mkidcore.config import yaml, yaml_object


BIN2HDFCONFIGTEMPLATE = ('{x} {y}\n'
                         '{datadir}\n'
                         '{starttime}\n'
                         '{inttime}\n'
                         '{beamdir}\n'
                         '1\n'
                         '{outdir}')


def makehdf(cfgORcfgs, maxprocs=2, polltime=.1, executable_path=''):
    """
    Run b2n2hdf on the config(s). Takes a config or iterable of configs.

    maxprocs(2) keyword may be used to specify the maximum number of processes
    polltime(.1) sets how often processes are checked for output and output logged
    """
    if isinstance(cfgORcfgs, (tuple, list, set)):
        cfgs = tuple(cfgORcfgs)
    else:
        cfgs = (cfgORcfgs,)

    keepconfigs=False

    nproc = min(len(cfgs), maxprocs)
    polltime = max(.01, polltime)

    tfiles=[]
    for cfg in cfgs:
        with tempfile.NamedTemporaryFile('w',suffix='.cfg', delete=False) as tfile:
            tfile.write(BIN2HDFCONFIGTEMPLATE.format(datadir=cfg.datadir, starttime=cfg.starttime,
                                                     inttime=cfg.inttime, beamdir=cfg.beamdir,
                                                     outdir=cfg.outdir, x=cfg.x, y=cfg.y))
            tfiles.append(tfile)

    things = list(zip(tfiles, cfgs))
    procs = []
    while things:
        tfile, cfg = things.pop()
        procs.append(psutil.Popen((os.path.join(executable_path,'bin2hdf'),tfile.name),
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  shell=False, cwd=None, env=None, creationflags=0))
        while len(procs) >= nproc:
            #TODO consider repalcing with https://gist.github.com/bgreenlee/1402841
            for i, proc in enumerate(procs):
                out, err = proc.communicate(timeout=polltime)
                getLogger(__name__ + '.bin2hdf_{}'.format(i)).info(out)
                getLogger(__name__ + '.bin2hdf_{}'.format(i)).error(err)
            procs = list(filter(lambda p: p.poll() is None, procs))

    #Clean up temp files
    while tfiles and not keepconfigs:
        tfile = tfiles.pop()
        try:
            os.remove(tfile.name)
        except IOError:
            getLogger(__name__).debug('Failed to delete temp file {}'.format(tfile.name))


@yaml_object(yaml)
class Bin2HdfConfig(object):
    def __init__(self, datadir='./', beamdir='./', starttime=None, inttime=None,
                 outdir='./', x=140, y=145, writeto=None):
        self.datadir = datadir
        self.starttime = starttime
        self.inttime = inttime
        self.beamdir = beamdir
        self.outdir = outdir
        self.x = x
        self.y = y
        if writeto is not None:
            self.write(writeto)

    def write(self, file):
        with open(file, 'w') as wavefile:
            wavefile.write(BIN2HDFCONFIGTEMPLATE.format(datadir=self.datadir, starttime=self.starttime,
                                                         inttime=self.inttime, beamdir=self.beamdir,
                                                         outdir=self.outdir, x=self.x, y=self.y))

    def load(self):
        raise NotImplementedError
