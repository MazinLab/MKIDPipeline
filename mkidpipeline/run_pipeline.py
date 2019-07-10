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

datafile = '/scratch/baileyji/mec/data.yml'
cfgfile = '/scratch/baileyji/mec/pipe.yml'

pipe.logtoconsole()

pcfg = pipe.configure_pipeline(cfgfile)

pipe.getLogger('mkidpipeline.calibration.wavecal').setLevel('INFO')
pipe.getLogger('mkidpipeline.badpix').setLevel('INFO')
pipe.getLogger('mkidpipeline.hdf.photontable').setLevel('INFO')

ncpu=7


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
    pipe.batch_apply_flatcals(dataset.science_observations, ncpu=ncpu)
    times.append(time.time())
    pipe.getLogger('mkidpipeline.hdf.photontable').setLevel('DEBUG')
    times.append(time.time())
    pipe.batch_maskhot(dataset.science_observations, ncpu=ncpu)
    times.append(time.time())

    print(np.diff(times).astype(int))
    print(int(times[-1] - times[0])/60)


def generate_outputs(outputs):
    from mkidpipeline.config import config
    import mkidpipeline.imaging.drizzler as drizzler
    import mkidpipeline as pipe
    for o in outputs:
        pipe.getLogger(__name__).info('Generating {}'.format(o))
        if o.wants_image:
            import mkidpipeline.hdf.photontable
            for obs in o.data.obs:
                h5 = mkidpipeline.hdf.photontable.ObsFile(obs.h5)
                img = h5.getFits(wvlStart=o.startw, wvlStop=o.stopw, applyWeight=o.enable_photom,
                                applyTPFWeight=o.enable_noise, countRate=True)
                img.writeto(o.output_file + h5.fileName.split('.')[0] + ".fits")
                pipe.getLogger(__name__).info('Generated fits file for {}'.format(obs.h5))
        if o.wants_drizzled:
            import mkidpipeline.imaging
            if not isinstance(o.data, mkidpipeline.config.MKIDDitheredObservation):
                raise TypeError('a dither is not specified in the out.yml')
            drizzled = drizzler.form(o.data, mode=o.kind, wvlMin=o.startw, wvlMax=o.stopw,
                                     pixfrac=config.drizzler.pixfrac, usecache=False, ncpu=config.ncpu)
            drizzled.writefits(o.output_file)


out_collection = pipe.load_output_description('')
outputs = out_collection.outputs
dataset = pipe.load_data_description(datafile) # NB using this may result in processing more than is strictly required
# dataset = out_collection.dataset

# First we need to process data
run_stage1(dataset)
# generate_outputs(outputs)

