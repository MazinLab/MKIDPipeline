#!/bin/env python3
import mkidpipeline.definitions as definitions
import mkidpipeline.config as config
import mkidpipeline.pipeline as pipe
from importlib import resources as rs
import mkidcore.corelog

mkidcore.corelog.getLogger('mkidcore', setup=True,
                           configfile=rs.files('mkidpipeline').joinpath('./utils/logging.yaml'))
pipe.getLogger('mkidpipeline.steps.wavecal').setLevel('INFO')

config.configure_pipeline(rs.files('mkidpipeline').joinpath('pipe.yml'))
paths = dict(data='/darkdata/ScienceData/Subaru/', out='/work/tmp/tests/out',
             database='/work/tmp/tests/database', tmp='')
config.update_paths(paths)
config.make_paths()
config.config.update('ncpu', 10)
dataset = definitions.MKIDObservingDataset(rs.files('mkidpipeline').joinpath('tests/test_data.yml'))

bin2hdf.buildtables(dataset.timeranges, remake=False)
pipe.wavecal.fetch(dataset.wavecals, verbose=False)
