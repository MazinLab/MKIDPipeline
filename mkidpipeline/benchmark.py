import os, time
os.environ['NUMEXPR_MAX_THREADS'] = '32'
os.environ['NUMEXPR_NUM_THREADS'] = '16'
os.environ["TMPDIR"] = '/scratch/tmp/'

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


datafile = '/scratch/baileyji/mec/data.yml'
cfgfile = '/scratch/baileyji/mec/pipe.yml'

mkidpipeline.config.logtoconsole()

pcfg = mkidpipeline.config.configure_pipeline(cfgfile)
dataset = mkidpipeline.config.load_data_description(datafile)
print(dataset.description)

bin2hdf.buildtables(dataset.timeranges, ncpu=7, remake=False, timesort=False)

of = mkidpipeline.hdf.photontable.ObsFile(pcfg.paths.out+'/1545542463.h5')
q1=of.query(startt=1, stopt=5)
tic=time.time()
of.photonTable.read()
print(time.time()-tic)
ofraid = mkidpipeline.hdf.photontable.ObsFile('/mnt/data0/baileyji/mec/out/1545542463.h5')
q1=ofraid.query(startt=1, stopt=5)
tic=time.time()
ofraid.photonTable.read()
print(time.time()-tic)
wavecal.fetch(dataset.wavecals, verbose=False)
batch_apply_wavecals(dataset.wavecalable, 10)

# flatcal.fetch(dataset.flatcals)
#
# batch_apply_flatcals(dataset.science_observations, 10)
