#!/bin/env python3
import os
import time
import numpy as np
os.environ['NUMEXPR_MAX_THREADS'] = '32'
os.environ['NUMEXPR_NUM_THREADS'] = '16'
os.environ["TMPDIR"] = '/mnt/data0/tmp/'
import tables.parameters
tables.parameters.MAX_BLOSC_THREADS = 4

import mkidpipeline as pipe

datafile = 'data_HD1160.yml'
cfgfile = 'pipe_HD1160.yml'

pipe.logtoconsole()

pcfg = pipe.configure_pipeline(cfgfile)

pipe.getLogger('mkidpipeline.steps.wavecal').setLevel('INFO')
pipe.getLogger('mkidpipeline.steps.badpix').setLevel('INFO')
pipe.getLogger('mkidpipeline.hdf.photontable').setLevel('INFO')

ncpu = 7

def run_stage1(dataset):
    times = []
    times.append(time.time())
    pipe.bin2hdf.buildtables(dataset.timeranges, ncpu=ncpu, remake=False, chunkshape=250)
    times.append(time.time())
    pipe.batch_apply_metadata(dataset)
    times.append(time.time())
    pipe.wavecal.fetch(dataset.wavecals, verbose=False, ncpu=ncpu)
    times.append(time.time())
    pipe.batch_apply_wavecals(dataset.wavecalable, ncpu=ncpu)
    times.append(time.time())
    pipe.flatcal.fetch(dataset.flatcals, ncpu=ncpu)
    times.append(time.time())
    for o in dataset.science_observations:
        pipe.flatcal_apply(o)
    #pipe.batch_apply_flatcals(dataset.science_observations, ncpu=ncpu)
    times.append(time.time())
    pipe.getLogger('mkidpipeline.hdf.photontable').setLevel('DEBUG')
    times.append(time.time())
    pipe.batch_maskhot(dataset.science_observations, ncpu=ncpu)
    times.append(time.time())

    print(np.diff(times).astype(int))
    print(int(times[-1] - times[0])/60)


def generate_outputs(outputs):
    from mkidpipeline.config import config
    import mkidpipeline.steps.drizzler as drizzler
    import mkidpipeline as pipe
    for o in outputs:
        pipe.getLogger(__name__).info('Generating {}'.format(o))
        if o.wants_image:
            import mkidpipeline.photontable as photontable
            for obs in o.data.obs:
                h5 = photontable.Photontable(obs.h5)
                img = h5.get_fits(wave_start=o.startw, wave_stop=o.stopw, spec_weight=o.enable_photom,
                                  noise_weight=o.enable_noise, rate=True)
                img.writeto(o.output_file)
                pipe.getLogger(__name__).info('Generated fits file for {}'.format(obs.h5))
        if o.wants_drizzled:
            import mkidpipeline.imaging
            if not isinstance(o.data, mkidpipeline.config.MKIDDitherDescription):
                raise TypeError('a dither is not specified in the out.yml')
            drizzled = drizzler.form(o.data, mode=o.kind, wvlMin=o.startw, wvlMax=o.stopw,
                                     pixfrac=config.drizzler.pixfrac, usecache=False)
            drizzled.writefits(o.output_file)

dataset = pipe.load_data_description(datafile) #NB using this may result in processing more than is strictly required
out_collection = pipe.load_output_description('out_HD1160.yml', datafile='data_HD1160.yml') #make sure you give this a datafile
outputs = out_collection.outputs

# dataset = out_collection.dataset

# First we need to process data
run_stage1(dataset)
# Generate desired outputs
generate_outputs(outputs)

