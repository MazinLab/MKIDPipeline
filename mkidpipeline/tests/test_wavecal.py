#!/bin/env python3
import mkidpipeline.config as config
import mkidpipeline.pipeline as pipe
import pkg_resources as pkg

config.logtoconsole()
pipe.getLogger('mkidpipeline.steps.wavecal').setLevel('INFO')

config.configure_pipeline(pkg.resource_filename('mkidpipeline', 'pipe.yml'))
paths = dict(data='/darkdata/ScienceData/Subaru/', out='/work/tmp/tests/out',
             database='/work/tmp/tests/database', tmp='')
config.update_paths(paths)
config.make_paths()
config.config.update('ncpu', 10)
dataset = config.load_data_description(pkg.resource_filename('mkidpipeline', 'tests/test_data.yml'))

bin2hdf.buildtables(dataset.timeranges, remake=False)
pipe.wavecal.fetch(dataset.wavecals, verbose=False)
