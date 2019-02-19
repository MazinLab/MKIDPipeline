import os
os.environ['NUMEXPR_MAX_THREADS'] = '64'
os.environ['NUMEXPR_NUM_THREADS'] = '32'
os.environ["TMPDIR"] = '/mnt/data0/tmp/'

import mkidpipeline.hdf.bin2hdf as bin2hdf
import mkidpipeline.calibration.wavecal as wavecal
import mkidpipeline.calibration.flatcal as flatcal
import mkidpipeline.badpix as badpix
import mkidpipeline.config
import mkidpipeline.hdf.photontable
from mkidcore.config import getLogger
import multiprocessing as mp


def wavecal_apply(o):
    of = mkidpipeline.hdf.photontable.ObsFile(mkidpipeline.config.get_h5_path(o), mode='a')
    of.applyWaveCal(wavecal.load_solution(o.wavecal))
    of.file.close()


def flatcal_apply(o):
    of = mkidpipeline.hdf.photontable.ObsFile(mkidpipeline.config.get_h5_path(o), mode='a')
    of.applyFlatCal(wavecal.load_solution(o.flatcal))
    of.file.close()


def batch_apply_wavecals(wavecal_pairs, ncpu=None):
    pool = mp.Pool(ncpu if ncpu is not None else mkidpipeline.config.n_cpus_available())
    pool.map(wavecal_apply, wavecal_pairs)
    pool.close()


def batch_apply_flatcals(flatcal_pairs, ncpu=None):
    pool = mp.Pool(ncpu if ncpu is not None else mkidpipeline.config.n_cpus_available())
    pool.map(wavecal_apply, flatcal_pairs)
    pool.close()


def batch_maskhot(obs, ncpu=None):
    pool = mp.Pool(ncpu if ncpu is not None else mkidpipeline.config.n_cpus_available())
    pool.map(badpix.mask_hot_pixels, set([o.h5 for o in obs]))
    pool.close()


datafile = '/scratch/baileyji/mec/data.yml'
cfgfile = '/scratch/baileyji/mec/pipe.yml'

mkidpipeline.config.logtoconsole()

pcfg = mkidpipeline.config.configure_pipeline(cfgfile)
dataset = mkidpipeline.config.load_data_description(datafile)


getLogger('mkidpipeline.calibration.wavecal').setLevel('INFO')
getLogger('mkidpipeline.hdf.photontable').setLevel('DEBUG')


bin2hdf.buildtables(dataset.timeranges, ncpu=15, remake=False, timesort=False)

wavecal.fetch(dataset.wavecals, verbose=False)

batch_apply_wavecals(dataset.wavecalable, 10)
#
# flatcal.fetch(dataset.flatcals)
#
# batch_apply_flatcals(dataset.science_observations, 10)

# batch_maskhot(dataset.science_observations)

# for h5 in set([o.h5 for o in dataset.science_observations]): badpix.mask_hot_pixels(h5)

