from importlib import import_module
import pkgutil
import multiprocessing as mp
import time


from mkidcore.pixelflags import FlagSet, BEAMMAP_FLAGS
from mkidcore.config import getLogger
import mkidpipeline
import mkidpipeline.photontable as photontable
import mkidpipeline.config as config
import mkidpipeline.steps
from mkidpipeline.steps import wavecal
import mkidpipeline.steps.buildhdf
import mkidpipeline.imaging.movies

import mkidcore.instruments
import mkidcore.objects
import mkidcore.config

log = getLogger('mkidpipeline')

PIPELINE_STEPS = {}
for info in pkgutil.iter_modules(mkidpipeline.steps.__path__):
    mod = import_module(f"mkidpipeline.steps.{info.name}")
    globals()[info.name] = mod
    PIPELINE_STEPS[info.name] = mod
    try:
        mkidcore.config.yaml.register_class(mod.StepConfig)
    except AttributeError:
        pass


_flags = {'beammap': BEAMMAP_FLAGS}
for name, step in PIPELINE_STEPS.items():
    try:
        _flags[name] = step.FLAGS
    except AttributeError:
        getLogger(__name__).debug(f"Step {name} does not export any pipeline flags.")
        pass

PIPELINE_FLAGS = FlagSet.define(*sorted([(f"{k}.{f.name.replace(' ','_')}", i, f.description) for i, (k, f) in
                                         enumerate((k, f) for k, flagset in _flags.items() for f in flagset)]))
del _flags

PROBLEM_FLAGS = ('pixcal.hot', 'pixcal.cold', 'pixcal.unstable', 'beammap.noDacTone', 'wavecal.bad',
                 'wavecal.failed_validation', 'wavecal.failed_convergence', 'wavecal.not_monotonic',
                 'wavecal.not_enough_histogram_fits', 'wavecal.no_histograms',
                 'wavecal.not_attempted')


def generate_default_config():
    cfg = config.PipeConfig()
    for name, step in PIPELINE_STEPS.items():
        try:
            cfg.register(name, step.StepConfig(), update=True)
        except AttributeError:
            getLogger(__name__).warning(f'Pipeline step mkidpipeline.steps.{name} does not '
                                        f'support automatic configuration discovery.')
    cfg.register('beammap', mkidcore.objects.Beammap(specifier='MEC'))
    cfg.register('instrument', mkidcore.instruments.InstrumentInfo('MEC'))
    return cfg


def generate_sample_data():
    from collections import defaultdict
    i = defaultdict(lambda: 0)

    def namer(name='Thing'):
        ret = f"{name}{i[name]}"
        i[name] = i[name]+1
        return ret

    data = [config.MKIDTimerange(name=namer(), start=1602048870, duration=30,
                                 dark=config.MKIDTimerange(name=namer(), start=1602046500, stop=1602046510)),
            config.MKIDObservation(name=namer('star'), start=1602048875, duration=10, wavecal='wavecal0',
                                   dark=config.MKIDTimerange(name=namer(), start=1602046500, duration=10),
                                   flatcal='flactcal0', wcscal='wcscal0', speccal='speccal0'),
            # a wavecal
            config.MKIDWavecalDescription(name=namer('wavecal'), obs=(
                config.MKIDTimerange(name='850 nm', start=1602040820, duration=60,
                                     dark=config.MKIDTimerange(name=namer(), start=1602046500, duration=10),
                                     header=dict(laser='on', other='fits_key')),
                config.MKIDTimerange(name='950 nm', start=1602040895, duration=60,
                                     dark=config.MKIDTimerange(name=namer(), start=1602046500, duration=10)),
                config.MKIDTimerange(name='1.1 um', start=1602040970, duration=60,
                                     dark=config.MKIDTimerange(name=namer(), start=1602046500, duration=10)),
                config.MKIDTimerange(name='1.25 um', start=1602041040, duration=60,
                                     dark=config.MKIDTimerange(name=namer(), start=1602046500, duration=10)),
                config.MKIDTimerange(name='13750 A', start=1602041110, duration=60)
            )),
            # Flatcals
            config.MKIDFlatcalDescription(name=namer('flatcal'),
                                          data=config.MKIDObservation(name='950 nm', start=1602040900, duration=50,
                                                                      dark=config.MKIDTimerange(name=namer(),
                                                                                                start=1602046500,
                                                                                                duration=10),
                                                                      wavecal='wavecal0')),
            config.MKIDFlatcalDescription(name=namer('flatcal'), wavecal_duration=50.0, wavecal_offset=1,
                                          data='wavecal0'),
            # Speccal
            config.MKIDSpeccalDescription(name=namer('speccal'),
                                          data=config.MKIDObservation(name=namer('star'), start=340, duration=10,
                                                                      wavecal='wavecal0',
                                                                      spectrum='qualified/path/or/relative/'
                                                                              'todatabase/refspec.file'),
                                          aperture=('15h22m32.3', '30.32 deg', '200 mas')),

            # WCS cal
            config.MKIDWCSCalDescription(name=namer('wcscal'), comment='ob wcscals may be used to manually determine '
                                                                       'WCS parameters. They are not yet supported for '
                                                                       'automatic WCS parameter computation',
                                         ob=config.MKIDObservation(name=namer('star'), start=360, duration=10,
                                                                   wavecal='wavecal0',
                                                                   dark=config.MKIDTimerange(name=namer(), start=350,
                                                                                             duration=10)),
                                         dither_home=(107,46), dither_ref=(-0.16, -0.4), platescale='10.40 mas'),
            # Dithers
            config.MKIDDitherDescription(name=namer('dither0'), data=1602047815, wavecal='wavecal0',
                                         flatcal='flatcal0', speccal='speccal0', use='0,2,4-9', wcscal='wcscal0'),
            config.MKIDDitherDescription(name=namer('dither1'), data='dither.logfile', wavecal='wavecal0',
                                         flatcal='flatcal0', speccal='speccal0', use=(1,), wcscal='wcscal0'),
            config.MKIDDitherDescription(name=namer('dither'), flatcal='', speccal='', wcscal='',wavecal='',
                                         data=(config.MKIDObservation(name=namer('HIP 109427'), start=1602047815, duration=10,
                                                                      wavecal='wavecal0', dither_pos=(0.2,0.3),
                                                                      dark=config.MKIDTimerange(name=namer(), start=350,
                                                                                                duration=10)),
                                               config.MKIDObservation(name=namer('HIP 109427'), start=1602047825, duration=10,
                                                                      wavecal='wavecal0', dither_pos=(0.1, 0.1),
                                                                      wcscal='wcscal0'),
                                               config.MKIDObservation(name=namer('HIP 109427'), start=1602047835, duration=10,
                                                                      wavecal='wavecal0', dither_pos=(-0.1, -0.1),
                                                                      wcscal='wcscal0')
                                               )
                                         )
            ]
    return data


def metadata_apply(ob):
    o = photontable.Photontable(ob.h5, mode='w')
    mdl = config.select_metadata_for_h5(ob, config.load_observing_metadata())
    o.attach_observing_metadata(mdl)


def wavecal_apply(o):
    if o.wavecal is None:
        getLogger(__name__).info('No wavecal to apply for {}'.format(o.h5))
        return
    try:
        of = photontable.Photontable(o.h5, mode='a')
        of.apply_wavecal(wavecal.load_solution(o.wavecal.path))
        of.file.close()
    except Exception as e:
        getLogger(__name__).critical('Caught exception during run of {}'.format(o.h5), exc_info=True)


def flatcal_apply(o):
    if o.flatcal is None:
        getLogger(__name__).info('No flatcal to apply for {}'.format(o.h5))
        return
    try:
        of = photontable.Photontable(o.h5, mode='a')
        cfg = mkidpipeline.config.config
        of.apply_flatcal(o.flatcal.path, use_wavecal=cfg.flatcal.use_wavecal, startw=850, stopw=1375)
        of.file.close()
    except Exception as e:
        getLogger(__name__).critical('Caught exception during run of {}'.format(o.h5), exc_info=True)


def lincal_apply(o):
    try:
        of = photontable.Photontable(o, mode='a')
        cfg = mkidpipeline.config.config
        of.apply_lincal(dt=cfg.lincal.dt, tau=cfg.instrument.deadtime*1*10**6)
        of.file.close()
    except Exception as e:
        getLogger(__name__).critical('Caught exception during run of {}'.format(o), exc_info=True)


def badpix_apply(o):
    try:
        mkidpipeline.steps.pixcal.mask_hot_pixels(o)
    except Exception as e:
        getLogger(__name__).critical('Caught exception during run of {}'.format(o), exc_info=True)


def batch_apply_metadata(dataset):
    """Function associates things not known at hdf build time (e.g. that aren't in the bin files)"""
    # Retrieve metadata database
    metadata = config.load_observing_metadata()
    # Associate metadata
    for ob in dataset.all_observations:
        o = photontable.Photontable(ob.h5, mode='w')
        mdl = config.select_metadata_for_h5(ob, metadata)
        o.attach_observing_metadata(mdl)
        del o


def batch_apply_wavecals(dset, ncpu=None):
    """ filter for unique h5 files, not responsible for mixed wavecal specs """
    wavecal.clear_solution_cache()
    pool = mp.Pool(ncpu if ncpu is not None else config.n_cpus_available(config.config.wavecal.ncpu))
    obs = {o.h5: o for o in dset.wavecalable if o.wavecal is not None}.values()
    pool.map(wavecal_apply, obs)
    pool.close()


def batch_apply_flatcals(dset, ncpu=None):
    """
    Will filter for unique h5 files, not responsible for mixed flatcal specs
    """
    pool = mp.Pool(ncpu if ncpu is not None else config.n_cpus_available())
    obs = {o.h5: o for o in dset.flatcalable if o.flatcal is not None}.values()
    pool.map(flatcal_apply, obs)
    pool.close()


def batch_apply_badpix(dset, ncpu=None):
    pool = mp.Pool(ncpu if ncpu is not None else config.n_cpus_available())
    pool.map(badpix_apply, set([o.h5 for o in dset.pixcalable]))
    pool.close()


def batch_apply_lincal(dset, ncpu=None):
    pool = mp.Pool(ncpu if ncpu is not None else config.n_cpus_available())
    pool.map(lincal_apply, set([o.h5 for o in dset.all_observations]))
    pool.close()


def batch_build_hdf(timeranges, ncpu=None):
    """will also accept an opject with a .timeranges (e.g. a dataset)"""
    ncpu = ncpu if ncpu is not None else config.n_cpus_available()
    mkidpipeline.steps.buildhdf.buildtables(timeranges, ncpu=ncpu, remake=False)


def run_stage1(dataset):
    operations = (('Building H5s', mkidpipeline.steps.buildhdf.buildtables),
                  ('Attaching metadata', batch_apply_metadata),
                  ('Fetching wavecals', mkidpipeline.steps.wavecal.fetch),
                  ('Applying wavelength solutions', batch_apply_wavecals),
                  ('Applying wavelength solutions', batch_apply_badpix),
                  ('Applying linearity correction', batch_apply_lincal),
                  ('Fetching flatcals', mkidpipeline.steps.flatcal.fetch),
                  ('Applying flatcals', batch_apply_flatcals),
                  ('Fetching speccals', mkidpipeline.steps.speccal.fetch))

    toc = time.time()
    for task_name, task in operations:
        tic = time.time()
        getLogger(__name__).info(f'Stage 1: {task_name}')
        task(dataset)
        getLogger(__name__).info(f'Completed {task_name} in {time.time()-tic:.0f} s')

    getLogger(__name__).info(f'Stage 1 complete in {(time.time()-toc)/60:.0f} m')


def generate_outputs(outputs):
    mkidpipeline.steps.drizzler.fetch(outputs)

    for o in outputs:
        # TODO make into a batch process
        getLogger(__name__).info('Generating {}'.format(o.name))
        if o.wants_image:

            for obs in o.data.obs:
                h5 = mkidpipeline.photontable.Photontable(obs.h5)
                img = h5.get_fits(wave_start=o.startw, wave_stop=o.stopw, spec_weight=o.enable_photom,
                                  noise_weight=o.enable_noise, rate=True)
                img.writeto(o.output_file)
                getLogger(__name__).info('Generated fits file for {}'.format(obs.h5))
        if o.wants_movie:
            getLogger('mkidpipeline.hdf.photontable').setLevel('DEBUG')
            mkidpipeline.imaging.movies.make_movie(o, inpainting=False)