import os
os.environ['NUMEXPR_MAX_THREADS'] = '32'
os.environ['NUMEXPR_NUM_THREADS'] = '16'
os.environ["TMPDIR"] = '/mnt/data0/tmp/'

import tables.parameters
tables.parameters.MAX_BLOSC_THREADS = 4

import mkidpipeline.hdf.bin2hdf as bin2hdf
import mkidpipeline.calibration.wavecal as wavecal
import mkidpipeline.calibration.flatcal as flatcal
import mkidpipeline.badpix as badpix
import mkidpipeline.config
import mkidpipeline.hdf.photontable
from mkidcore.config import getLogger
import multiprocessing as mp


def wavecal_apply(o):
    of = mkidpipeline.hdf.photontable.ObsFile(o.h5, mode='a')
    of.applyWaveCal(wavecal.load_solution(o.wavecal))
    of.file.close()


def flatcal_apply(o):
    of = mkidpipeline.hdf.photontable.ObsFile(o.h5, mode='a')
    of.applyFlatCal(o.flatcal)
    of.file.close()


def batch_apply_wavecals(obs, ncpu=None):
    pool = mp.Pool(ncpu if ncpu is not None else mkidpipeline.config.n_cpus_available())
    #TODO filter so that any files don't get opened concurrently
    pool.map(wavecal_apply, obs)
    pool.close()


def batch_apply_flatcals(obs, ncpu=None):
    pool = mp.Pool(ncpu if ncpu is not None else mkidpipeline.config.n_cpus_available())
    # TODO filter so that any files don't get opened concurrently
    pool.map(flatcal_apply, obs)
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
getLogger('mkidpipeline.hdf.photontable').setLevel('INFO')

ncpu=20

bin2hdf.buildtables(dataset.timeranges, ncpu=ncpu, remake=False, chunkshape=250)

wavecal.fetch(dataset.wavecals, verbose=False, ncpu=ncpu)

batch_apply_wavecals(dataset.wavecalable, ncpu=ncpu)

flatcal.fetch(dataset.flatcals, ncpu=ncpu)

batch_apply_flatcals(dataset.science_observations, ncpu=ncpu)

getLogger('mkidpipeline.hdf.photontable').setLevel('DEBUG')

batch_maskhot(dataset.science_observations, ncpu=ncpu)
