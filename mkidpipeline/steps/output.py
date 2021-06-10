from importlib import import_module
import pkgutil
import multiprocessing as mp
import time
from collections import defaultdict
import functools
import os
import mkidcore.pixelflags
from mkidcore.config import getLogger
import mkidpipeline.config as config
from mkidpipeline.steps import movies, drizzler
import mkidpipeline.steps.movies


def generate(outputs: config.MKIDOutputCollection):

    for o in outputs:
        # TODO make into a batch process
        getLogger(__name__).info('Generating {}'.format(o.name))
        if o.wants_image:
            # if we are putting out more than one image we need to give them unique file names
            if len(o.data.obs) > 1:
                f, ext = os.path.splitext(o.filename)
                filename = f'{f}.{{}}of{len(o.data.obs)+1}{ext}'
            else:
                filename = o.filename

            for i, obs in enumerate(o.data.obs):
                obs.photontable.get_fits(**o.output_settings_dict).writeto(filename.format(i+1))
                getLogger(__name__).info(f'Generated fits file for {obs}')

        if o.wants_movie:
            mkidpipeline.steps.movies.make_movie(o, **o.output_settings_dict)
        if o.wants_drizzled:
            config = mkidpipeline.config.PipelineConfigFactory(step_defaults=dict(drizzler=mkidpipeline.steps.drizzler.StepConfig()),
                                                               copy=True)

            drizzler.form(o.data, mode=o.kind, wvlMin=o.min_wave, wvlMax=o.max_wave,
                                nwvlbins=config.drizzler.n_wave, pixfrac=config.drizzler.pixfrac,
                                wcs_timestep=config.drizzler.wcs_timestep, exp_timestep=o.exp_timestep,
                                exclude_flags=mkidcore.pixelflags.PROBLEM_FLAGS,
                                usecache=config.drizzler.usecache, ncpu=config.get('drizzler.ncpu'),
                                derotate=config.drizzler.derotate, align_start_pa=config.drizzler.align_start_pa,
                                whitelight=config.drizzler.whitelight, save_steps=config.drizzler.save_steps,
                                output_file=o.filename)
