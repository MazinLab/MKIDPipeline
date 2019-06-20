import sys
import multiprocessing as mp

if sys.version_info.major == 3:
    import mkidpipeline.hdf.bin2hdf as bin2hdf
    import mkidpipeline.calibration.wavecal as wavecal
    import mkidpipeline.calibration.flatcal as flatcal
    import mkidpipeline.badpix as badpix
    import mkidpipeline.config as config
    import mkidpipeline.hdf.photontable

from mkidpipeline.config import configure_pipeline, load_data_description, load_task_config, load_output_description, \
    logtoconsole
from mkidcore.config import getLogger


def wavecal_apply(o):
    of = mkidpipeline.hdf.photontable.ObsFile(o.h5, mode='a')
    of.applyWaveCal(wavecal.load_solution(o.wavecal.path))
    of.file.close()


def flatcal_apply(o):
    of = mkidpipeline.hdf.photontable.ObsFile(o.h5, mode='a')
    of.applyFlatCal(o.flatcal.path)
    of.file.close()


def batch_apply_metadata(dataset):
    """Function associates things not known at hdf build time (e.g. that aren't in the bin files)"""
    # Retrieve metadata database
    metadata = config.load_observing_metadata()
    # Associate metadata
    for ob in dataset.all_observations:
        o = mkidpipeline.hdf.photontable.ObsFile(ob.h5, mode='w')
        mdl = config.select_metadata_for_h5(o.startTime, o.duration, metadata)
        for md in mdl:
            md.registerfromkvlist(ob.metadata.items())
        o.attach_observing_metadata(mdl)
        del o


def batch_apply_wavecals(obs, ncpu=None):
    pool = mp.Pool(ncpu if ncpu is not None else config.n_cpus_available())
    obs = {o.h5: o for o in obs}.values()  # filter so unique h5 files, not responsible for a mixed wavecal specs
    pool.map(wavecal_apply, obs)
    pool.close()


def batch_apply_flatcals(obs, ncpu=None):
    pool = mp.Pool(ncpu if ncpu is not None else config.n_cpus_available())
    obs = {o.h5: o for o in obs}.values()  # filter so unique h5 files, not responsible for a mixed flatcal specs
    pool.map(flatcal_apply, obs)
    pool.close()


def batch_maskhot(obs, ncpu=None):
    pool = mp.Pool(ncpu if ncpu is not None else config.n_cpus_available())
    pool.map(badpix.mask_hot_pixels, set([o.h5 for o in obs]))
    pool.close()

