import os
os.environ['NUMEXPR_MAX_THREADS'] = '32'
os.environ['NUMEXPR_NUM_THREADS'] = '16'
os.environ["TMPDIR"] = '/mnt/data0/tmp/'
import tables.parameters
tables.parameters.MAX_BLOSC_THREADS = 4

import mkidpipeline as pipe

datafile = '/scratch/baileyji/mec/data.yml'
cfgfile = '/scratch/baileyji/mec/pipe.yml'

pipe.logtoconsole()

pcfg = pipe.configure_pipeline(cfgfile)
dataset = pipe.load_data_description(datafile)


pipe.getLogger('mkidpipeline.calibration.wavecal').setLevel('INFO')
pipe.getLogger('mkidpipeline.hdf.photontable').setLevel('INFO')

ncpu=20


pipe.bin2hdf.buildtables(dataset.timeranges, ncpu=ncpu, remake=False, chunkshape=250)

pipe.wavecal.fetch(dataset.wavecals, verbose=False, ncpu=ncpu)

pipe.batch_apply_wavecals(dataset.wavecalable, ncpu=ncpu)

pipe.flatcal.fetch(dataset.flatcals, ncpu=ncpu)

pipe.batch_apply_flatcals(dataset.science_observations, ncpu=ncpu)

pipe.getLogger('mkidpipeline.hdf.photontable').setLevel('DEBUG')

pipe.batch_maskhot(dataset.science_observations, ncpu=ncpu)
