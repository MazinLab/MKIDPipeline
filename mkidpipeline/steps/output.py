from importlib import import_module
import pkgutil
import multiprocessing as mp
import time
from collections import defaultdict
import functools
import os

import mkidpipeline.definitions as definitions
import mkidcore.pixelflags
from mkidcore.config import getLogger
import mkidpipeline.config as config
from mkidpipeline.steps import movies, drizzler
import mkidpipeline.steps.movies
import astropy.units as u

StepConfig = None


def generate(outputs: definitions.MKIDOutputCollection):
    for o in outputs:
        if os.path.exists(o.filename):
            getLogger(__name__).info(f'Output {o.filename} for {o.name} already exists. Skipping')
            continue
        getLogger(__name__).info('Generating {}'.format(o.name))
        if o.wants_image:
            # if we are putting out more than one image we need to give them unique file names
            if len(o.data.obs) > 1:
                f, ext = os.path.splitext(o.filename)
                filename = f'{f}.{{}}of{len(o.data.obs)}{ext}'
            else:
                filename = o.filename

            for i, obs in enumerate(o.data.obs):
                file = filename.format(i+1)
                if os.path.exists(file):
                    getLogger(__name__).info(f'Output {file} for {o.name} already exists. Skipping')
                    continue
                kwargs = o.output_settings_dict
                kwargs['wave_start'] = kwargs['wave_start'].to(u.nm).value
                kwargs['wave_stop'] = kwargs['wave_stop'].to(u.nm).value
                for k in ('wvl_bin_width', 'time_bin_width'):
                    try:
                        val = kwargs[k].to(u.nm).value
                    except AttributeError:
                        val = kwargs[k]
                    if val > 0:
                        kwargs['bin_width'] = val
                    kwargs.pop(k)
                obs.photontable.get_fits(**kwargs).writeto(file)
                getLogger(__name__).info(f'Output {file} for {o.name} generated')

        if o.wants_movie:
            if o.output_settings_dict['time_bin_width'] == 0.0:
                raise ValueError(f'need to specify a timestep for {o.name} of output type {o.kind}')
            mkidpipeline.steps.movies.fetch(o, **o.output_settings_dict)

        if o.wants_drizzled:
            config = mkidpipeline.config.PipelineConfigFactory(step_defaults=dict(drizzler=mkidpipeline.steps.drizzler.StepConfig()),
                                                               copy=True)
            kwargs = o.output_settings_dict
            kwargs['mode'] = o.kind
            kwargs['output_file'] = o.filename
            for k in ('cube_type', 'rate', 'bin_type'):
                kwargs.pop(k)
            drizzler.form(o.data, pixfrac=config.drizzler.pixfrac,
                          wcs_timestep=config.drizzler.wcs_timestep, usecache=config.drizzler.usecache,
                          ncpu=config.get('drizzler.ncpu'), derotate=config.drizzler.derotate,
                          align_start_pa=config.drizzler.align_start_pa, whitelight=config.drizzler.whitelight,
                          debug_dither_plot=config.drizzler.plots == 'all', **kwargs)
