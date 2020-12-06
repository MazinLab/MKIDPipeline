
from mkidpipeline.hdf.bin2hdf import gen_configs
import os.path
def make_aws_dataset(dataset):
    cfg = gen_configs(dataset.timeranges)

    files = [os.path.join(c.datadir, '{}.bin'.format(t))
             for c in cfg for t in range(c.starttime-1, c.starttime+c.inttime+1)]

    for f in files:
        #link