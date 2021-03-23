#!/bin/env python3
import os
import time
import numpy as np
os.environ['NUMEXPR_MAX_THREADS'] = '32'
os.environ['NUMEXPR_NUM_THREADS'] = '16'
os.environ["TMPDIR"] = '/scratch/tmp/'
import tables.parameters
tables.parameters.MAX_BLOSC_THREADS = 4
import mkidcore.pixelflags as pixelflags
import mkidpipeline.pipeline as pipe

datafile = '/scratch/baileyji/mec/data.yml'
cfgfile = '/scratch/baileyji/mec/pipe.yml'
outfile = '/scratch/baileyji/mec/out.yml'

pipe.logtoconsole()

pipe.configure_pipeline(cfgfile)
config = pipe.config.config

pipe.getLogger('mkidpipeline.steps.wavecal').setLevel('INFO')
pipe.getLogger('mkidpipeline.steps.badpix').setLevel('INFO')
pipe.getLogger('mkidpipeline.hdf.photontable').setLevel('INFO')

ncpu=7


def run_stage1(dataset):
    times = []
    times.append(time.time())
    bin2hdf.buildtables(dataset.timeranges, ncpu=ncpu, remake=False, chunkshape=250)
    times.append(time.time())
    pipe.batch_apply_metadata(dataset)
    times.append(time.time())
    pipe.wavecal.fetch(dataset.wavecals, verbose=False, ncpu=ncpu)
    times.append(time.time())
    pipe.batch_apply_wavecals(dataset.wavecalable, ncpu=ncpu)
    times.append(time.time())
    pipe.batch_maskhot(dataset.science_observations, ncpu=ncpu)
    times.append(time.time())
    pipe.batch_apply_linearitycal(dataset.science_observations, ncpu=ncpu)
    times.append(time.time())
    pipe.flatcal.fetch(dataset, ncpu=ncpu)
    times.append(time.time())
    pipe.batch_apply_flatcals(dataset.science_observations, ncpu=ncpu)
    times.append(time.time())
    pipe.spectralcal.fetch(dataset, ncpu=ncpu)
    times.append(time.time())
    pipe.getLogger('mkidpipeline.hdf.photontable').setLevel('INFO')
    times.append(time.time())

    steps = ('H5 Creation', 'Metadata Application', 'Wavecal Fetch', 'Wavecal Application',
             'Flatcal Fetch', 'Flatcal Application', 'Hotpixel Masking')
    intervals = np.diff(times).astype(int)
    pipe.log.info('Stage one took {:.1f} m'.format(int(times[-1] - times[0])/60))
    for s, dt in zip(steps[:len(intervals)], intervals):
        pipe.log.info('    {} took {:.0f} s'.format(s, dt))


def generate_outputs(outputs):
    for o in outputs:
        pipe.getLogger(__name__).info('Generating {}'.format(o.name))
        if o.wants_image:
            import mkidpipeline.photontable as photontable
            for obs in o.data.obs:
                h5 = photontable.Photontable(obs.h5)
                img = h5.getFits(wvlStart=o.startw, wvlStop=o.stopw, applyWeight=o.enable_photom,
                                 applyTPFWeight=o.enable_noise, countRate=True)
                img.writeto(o.output_file + h5.fileName.split('.')[0] + ".fits")
                pipe.getLogger(__name__).info('Generated fits file for {}'.format(obs.h5))
        if o.wants_drizzled:
            import mkidpipeline.imaging
            if not isinstance(o.data, mkidpipeline.config.MKIDDitheredObservation):
                raise TypeError('a dither is not specified in the out.yml')
            drizzled = pipe.drizzler.form(o.data, mode=o.kind, wvlMin=o.startw, wvlMax=o.stopw,
                                          pixfrac=config.drizzler.pixfrac, wcs_timestep=config.drizzler.wcs_timestep,
                                          exp_timestep=config.drizzler.exp_timestep, exclude_flags=pixelflags.PROBLEM_FLAGS,
                                          usecache=config.drizzler.usecache, ncpu=config.ncpu,
                                          derotate=config.drizzler.derotate, align_start_pa=config.drizzler.align_start_pa,
                                          whitelight=config.drizzler.whitelight, save_file=config.drizzler.save_file)
            drizzled.write(o.output_file)
        if o.wants_movie:
            pipe.getLogger('mkidpipeline.hdf.photontable').setLevel('DEBUG')
            pipe.movies.make_movie(o, inpainting=False)


full_dataset = pipe.load_data_description(datafile)  # NB using this may result in processing more than is strictly required
out_collection = pipe.load_output_description(outfile)
outputs = out_collection.outputs
dataset = out_collection.dataset

# dataset = out_collection.dataset

# First we need to process data
run_stage1(dataset)
# Generate desired outputs
generate_outputs(outputs)