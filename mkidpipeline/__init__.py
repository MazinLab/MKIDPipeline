import sys
import multiprocessing as mp

if sys.version_info.major == 3:
    import mkidpipeline.hdf.bin2hdf as bin2hdf
    import mkidpipeline.calibration.wavecal as wavecal
    import mkidpipeline.calibration.flatcal as flatcal
    import mkidpipeline.imaging.drizzler as drizzler
    import mkidpipeline.imaging.movies as movies
    import mkidpipeline.badpix as badpix
    import mkidpipeline.config as config
    import mkidpipeline.hdf.photontable
    from mkidpipeline.hdf.photontable import ObsFile

from mkidpipeline.config import configure_pipeline, load_data_description, load_task_config, load_output_description, \
    logtoconsole
from mkidcore.config import getLogger

log = getLogger('mkidpipeline')


def wavecal_apply(o):
    if o.wavecal is None:
        getLogger(__name__).info('No wavecal to apply for {}'.format(o.h5))
        return
    try:
        of = mkidpipeline.hdf.photontable.ObsFile(o.h5, mode='a')
        of.applyWaveCal(wavecal.load_solution(o.wavecal.path))
        of.file.close()
    except Exception as e:
        getLogger(__name__).critical('Caught exception during run of {}'.format(o.h5), exc_info=True)


def flatcal_apply(o):
    if o.flatcal is None:
        getLogger(__name__).info('No flatcal to apply for {}'.format(o.h5))
        return
    try:
        of = mkidpipeline.hdf.photontable.ObsFile(o.h5, mode='a')
        cfg = mkidpipeline.config.config
        of.applyFlatCal(o.flatcal.path, use_wavecal=cfg.flatcal.use_wavecal)
        of.file.close()
    except Exception as e:
        getLogger(__name__).critical('Caught exception during run of {}'.format(o.h5), exc_info=True)


def batch_apply_metadata(dataset):
    """Function associates things not known at hdf build time (e.g. that aren't in the bin files)"""
    # Retrieve metadata database
    metadata = config.load_observing_metadata()
    # Associate metadata
    for ob in dataset.all_observations:
        o = mkidpipeline.hdf.photontable.ObsFile(ob.h5, mode='w')
        mdl = config.select_metadata_for_h5(ob, metadata)
        o.attach_observing_metadata(mdl)
        del o


def batch_apply_wavecals(obs, ncpu=None):
    wavecal.clear_solution_cache()
    pool = mp.Pool(ncpu if ncpu is not None else config.n_cpus_available())
    obs = {o.h5: o for o in obs if o.wavecal is not None}.values()  # filter so unique h5 files, not responsible for a mixed wavecal specs
    pool.map(wavecal_apply, obs)
    pool.close()


def batch_apply_flatcals(obs, ncpu=None):
    pool = mp.Pool(ncpu if ncpu is not None else config.n_cpus_available())
    obs = {o.h5: o for o in obs if o.flatcal is not None}.values()  # filter so unique h5 files, not responsible for a mixed flatcal specs
    pool.map(flatcal_apply, obs)
    pool.close()


def batch_maskhot(obs, ncpu=None):
    pool = mp.Pool(ncpu if ncpu is not None else config.n_cpus_available())
    pool.map(badpix.mask_hot_pixels, set([o.h5 for o in obs]))
    pool.close()

