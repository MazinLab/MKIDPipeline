import sys
import multiprocessing as mp

if sys.version_info.major==3:
    import mkidpipeline.hdf.bin2hdf as bin2hdf
    import mkidpipeline.calibration.wavecal as wavecal
    import mkidpipeline.calibration.flatcal as flatcal
    import mkidpipeline.badpix as badpix
    import mkidpipeline.config as config
    import mkidpipeline.hdf.photontable

from mkidpipeline.config import configure_pipeline, load_data_description, load_task_config, logtoconsole
from mkidcore.config import getLogger


def wavecal_apply(o):
    of = mkidpipeline.hdf.photontable.ObsFile(o.h5, mode='a')
    of.applyWaveCal(wavecal.load_solution(o.wavecal))
    of.file.close()


def flatcal_apply(o):
    of = mkidpipeline.hdf.photontable.ObsFile(o.h5, mode='a')
    of.applyFlatCal(o.flatcal)
    of.file.close()


def batch_apply_wavecals(obs, ncpu=None):
    pool = mp.Pool(ncpu if ncpu is not None else config.n_cpus_available())
    #TODO filter so that any files don't get opened concurrently
    pool.map(wavecal_apply, obs)
    pool.close()


def batch_apply_flatcals(obs, ncpu=None):
    pool = mp.Pool(ncpu if ncpu is not None else config.n_cpus_available())
    # TODO filter so that any files don't get opened concurrently
    pool.map(flatcal_apply, obs)
    pool.close()


def batch_maskhot(obs, ncpu=None):
    pool = mp.Pool(ncpu if ncpu is not None else config.n_cpus_available())
    pool.map(badpix.mask_hot_pixels, set([o.h5 for o in obs]))
    pool.close()

