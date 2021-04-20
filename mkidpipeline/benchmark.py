import os, time
os.environ['NUMEXPR_MAX_THREADS'] = '32'
os.environ['NUMEXPR_NUM_THREADS'] = '16'
os.environ["TMPDIR"] = '/scratch/tmp/'

import tables.index
tables.index.profile = True

import mkidpipeline.hdf.bin2hdf as bin2hdf
import mkidpipeline.calibration.wavecal as wavecal
import mkidpipeline.calibration.flatcal as flatcal
import mkidpipeline.badpix as badpix
import mkidpipeline.config
import mkidpipeline.hdf.photontable
from mkidcore.config import getLogger
import multiprocessing as mp
from mkidpipeline.hdf.photontable import Photontable


def wavecal_apply(o):
    of = mkidpipeline.hdf.photontable.Photontable(o.h5, mode='a')
    of.applyWaveCal(wavecal.load_solution(o.wavecal))
    of.file.close()


def flatcal_apply(o):
    of = mkidpipeline.hdf.photontable.Photontable(o.h5, mode='a')
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


def pipe_time():
    datafile = '/scratch/baileyji/mec/data.yml'
    cfgfile = '/scratch/baileyji/mec/pipe.yml'

    mkidpipeline.config.logtoconsole()

    pcfg = mkidpipeline.config.configure_pipeline(cfgfile)
    dataset = mkidpipeline.config.load_data_description(datafile)
    print(dataset.description)

    bin2hdf.buildtables(dataset.timeranges, ncpu=7, remake=False, timesort=False)

    of = mkidpipeline.hdf.photontable.Photontable(pcfg.paths.out + '/1545542463.h5')
    q1=of.query(startt=1, stopt=5)
    tic=time.time()
    of.photonTable.read()
    print(time.time()-tic)
    ofraid = mkidpipeline.hdf.photontable.Photontable('/mnt/data0/baileyji/mec/out/1545542463.h5')
    q1=ofraid.query(startt=1, stopt=5)
    tic=time.time()
    ofraid.photonTable.read()
    print(time.time()-tic)
    wavecal.fetch(dataset.wavecals, verbose=False)
    batch_apply_wavecals(dataset.wavecalable, 10)

    flatcal.fetch(dataset.flatcals)
    batch_apply_flatcals(dataset.science_observations, 10)


from line_profiler import LineProfiler


import tables
#@profile
def tsectest(f):
    of = Photontable(f)
    # p=LineProfiler(of.photonTable._where)
    # p.enable()
    of.query(startt=0, intt=30)

    #All the time is in the tables.tableextension.Row iterator returned by table._where,
    # p.disable()
    # p.dump_stats('benchmark.py.lpstats')
    # p.print_stats()


import mkidpipeline.config
mkidpipeline.config.logtoconsole()
# tsectest('/scratch/baileyji/mec/out/1545545212_slowtest.h5')
tsectest('/scratch/baileyji/mec/out/1545544477_slowtest.h5')


# from mkidpipeline.hdf.photontable import Photontable
# import mkidpipeline.config
# mkidpipeline.config.logtoconsole()
# f='/mnt/data0/isabel/highcontrastimaging/Jan2019Run/20190112/Trapezium/trap_ditherwavecalib/1547374834.h5'
# obsfile = Photontable(f)
# obsfile.getPixelCountImage(firstSec=0, integrationTime=30, applyWeight=True, applyTPFWeight=True, scaleByEffInt=False)
# #2019-02-15 16:54:41,351 DEBUG Feteched 27438555/27438555 rows in 111.714s using indices ('Time',) for query (Time < stopt)
#
#
# f='/scratch/baileyji/mec/out/1545545212_slowtest.h5'
# obsfile = Photontable(f,mode='w')
# obsfile.getPixelCountImage(firstSec=0, integrationTime=30, applyWeight=True, applyTPFWeight=True, scaleByEffInt=False)
# #2019-02-15 17:13:18,137 DEBUG Feteched 34003232/72271775 rows in 863.709s using indices ('Time',) for query (Time < stopt)
#
# obsfile.photonTable.autoindex=0
# r0=obsfile.photonTable[:1]
# obsfile.photonTable.modify_rows(0,1,rows=r0)
# obsfile.photonTable.cols._g_col('Time').index
# #2019-02-15 17:47:49,351 DEBUG Feteched 34003232/72271775 rows in 824.571s using indices () for query (Time < stopt)
#
#
# f='/scratch/baileyji/mec/out/1545544477_slowtest.h5'
# obsfile = Photontable(f,mode='w')
# obsfile.getPixelCountImage(firstSec=0, integrationTime=30, applyWeight=True, applyTPFWeight=True, scaleByEffInt=False)
# #2019-02-15 16:58:02,389 DEBUG Feteched 33126732/70860428 rows in 15.331s using indices () for query (Time < stopt)
# #obsfile.photonTable.cols._g_col('Time').index shows dirty
# obsfile.photonTable.reindex_dirty()
# #2019-02-15 17:50:12,871 DEBUG Feteched 33126732/70860428 rows in 12.114s using indices ('Time',) for query (Time < stopt)
#
