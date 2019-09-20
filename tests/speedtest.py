import os
#Dark has 16 cores, 2 threads/core
os.environ['NUMEXPR_MAX_THREADS'] = '64'
os.environ['NUMEXPR_NUM_THREADS'] = '32'
import numexpr
import numpy as np
import tables
import shutil
import time
import timeit
import os
import copy

import mkidpipeline
from mkidpipeline.hdf.photontable import ObsFile
import mkidpipeline.hdf.bin2hdf as bin2hdf




# Options to play with
#  Chunk shape None=Auto, 250 was best with pytables 3.4.4
# Filter options: compression level 0-9 (bug no real impact on speed), shuffle, bitshuffle
# Index Options optlevel(0-9), kind, ndx_shuffle, ndx_bitshuffle
# kind – full specifies complete track of the row position (64-bit). medium, light or ultralight only specify in
#        which chunk the row is (using 32-bit, 16-bit and 8-bit respectively).

# parameters.NODE_CACHE_SLOTS

# Index shuffling should affect size on disk and index read/write time, the latter of which should not dominate,
# biggest impact for full (as largest index)
# Index kind and optimization level will impact disk size and creation time
# Index kind and possibly optimization level will impact query times
# Shuffling and bit shuffling should impact compression times and thereby query times. They may be impacted by
# chunkshape depending on how the blocks are defined but should be able to be investigated using a lage CS and a
# small CS


firstSec = 2
shortT = 2
longT = 30
obsT = 1200
wvlStart = -90
wvlStop = -80
resid = 91416
resids = [91416,101119,90116,80881,81572,101471,90267,10603,80867,80180]


TEST_QUERIES = [dict(resid=resid), dict(resid=resids),
                dict(startt=firstSec, intt=shortT),
                dict(startt=firstSec, intt=longT),
                dict(startw=wvlStart, stopw=wvlStop),
                dict(resid=resids, startt=firstSec, intt=intTime, startw=wvlStart, stopw=wvlStop),
                dict(startt=firstSec, intt=obsT)]


def setting_id(sdict):
    return hex(hash(str(sdict)))


def test_settings(cfg, settings, queries, cleanup=True):
    results = {}
    builders = []
    h5s = [cfg.h5file[:-3] + setting_id(s) + '.h5' for s in settings]

    for s, h5 in zip(settings, h5s):
        builder = bin2hdf.HDFBuilder(copy.copy(cfg))
        builder.cfg.h5file = h5
        builder.kwargs = s
        builders.append(builder)

    for b in builders:
        tic = time.time()
        b.run()
        toc = time.time()
        results[str(b.kwargs)] = dict(creation=toc - tic,
                                      size=os.stat(b.cfg.h5file).st_size/1024**2)

    for q in queries:
        for f in h5s:
            of = ObsFile(f)
            tic = time.time()
            result = of.query(**q)
            toc = time.time()
            result[str(s)]['nphotons'] = len(of.photonTable)
            result[str(s)]['file_nfo'] = str(of)
            del of
            result[str(s)][str(q)] = toc-tic
            result[str(s)]['nphotons'] = nphotons

    if cleanup:
        for f in h5s:
            os.remove(f)

    return results


def test_query_format(fn, xCoord, yCoord, firstSec, intTime, wvlStart, wvlStop, nIters=1):
    beamim = tables.open_file(fn, mode='r').get_node('/BeamMap/Map').read()
    resID = beamim[xCoord][yCoord]
    tickspsec = int(1.0 / 1e-6)
    startTime = int(firstSec * tickspsec)  # convert to us
    endTime = startTime + int(intTime * tickspsec)
    rq='&'.join(['(ResID=={})'.format(r) for r in resID])
    rq = '('+rq+')' if '&' in rq else rq
    endTime = '(Time<{})'.format(endTime)
    startTime = '(Time>={})'.format(startTime)
    wvlStop='(Wavelength<{})'.format(wvlStop)
    wvlStart='(Wavelength>={})'.format(wvlStart)
    queryforms = ['{}&({}&({}&({}&{})))',

                  '{}&({}&{}&{}&{})',
                  '{}&({}&{}&({}&{}))',
                  '{}&(({}&{})&{}&{})',
                  '{}&(({}&{})&({}&{}))',

                  '{}&{}&{}&{}&{}',
                  '({}&{})&{}&{}&{}',
                  '{}&{}&{}&({}&{})',
                  '{}&{}&({}&{})&{}']
    queries = []
    for q in queryforms:
        queries.extend([q.format(rq, startTime, endTime, wvlStart, wvlStop)]+
                       [q.format(rq, wvlStart, wvlStop, startTime, endTime)]+
                       [q.format(rq, wvlStart, startTime, wvlStop, endTime)]+
                       [q.format(wvlStart, startTime, wvlStop, endTime, rq)]+
                       [q.format(startTime, endTime, wvlStart, wvlStop, rq)])
    querienames = []
    for q in queryforms:
        querienames.extend([q.format('res', 'time-', 'time+', 'wave-', 'wave+')]+
                           [q.format('res', 'wave-', 'wave+', 'time-', 'time+')]+
                           [q.format('res', 'wave-', 'time-', 'wave+', 'time+')]+
                           [q.format('wave-', 'time-', 'wave+', 'time+', 'res')]+
                           [q.format('time-', 'time+', 'wave-', 'wave+', 'res')])

    qtimes = [timeit.timeit('doquery(h5, "{}")'.format(query),
                            setup='from __main__ import tables, doquery; h5=tables.open_file("{}", mode="r")'.format(fn),
                            number=nIters,  globals=locals()) for query in queries]

    return zip(querienames, qtimes)


def test_query_format_multirID(fn, xCoord, yCoord, firstSec, intTime, wvlStart, wvlStop, nIters=1):
    beamim = tables.open_file(fn, mode='r').get_node('/BeamMap/Map').read()
    resID = beamim[xCoord, yCoord]
    tickspsec = int(1.0 / 1e-6)
    startTime = int(firstSec * tickspsec)  # convert to us
    endTime = startTime + int(intTime * tickspsec)
    rq='|'.join(['(ResID=={})'.format(r) for r in resID])
    rq = '('+rq+')' if '&' in rq else rq
    endTime = '(Time<{})'.format(endTime)
    startTime = '(Time>={})'.format(startTime)
    wvlStop='(Wavelength<{})'.format(wvlStop)
    wvlStart='(Wavelength>={})'.format(wvlStart)
    q = '({})&(({}&{})&({}&{}))'.format(rq, startTime, endTime, wvlStart, wvlStop)
    queriename = q.format('res', 'time-', 'time+', 'wave-', 'wave+')  # Best from test_query_format
    qtime=0
    qtime = timeit.timeit('doquery(h5, "{}")'.format(q),
                          setup='from __main__ import tables, doquery; h5=tables.open_file("{}", mode="r")'.format(fn),
                          number=nIters,  globals=locals())

    return queriename, qtime


def doquery(h5, query):
    return h5.get_node('/Photons/PhotonTable').read_where(query)


def cachetest():
    q='Time<3e5'
    binfile = '/mnt/data0/baileyji/mec/h5/1545542180.h5'
    csifile = '/mnt/data0/baileyji/mec/h5/1545542180_pytables_csi.h5'
    ulifile = '/mnt/data0/baileyji/mec/h5/1545542180_pytables_ul6.h5'
    # tables.parameters.CHUNK_CACHE_NELMTS = 1e6
    # tables.parameters.CHUNK_CACHE_SIZE = 20 * 1024*1024*1024
    # tables.parameters.TABLE_MAX_SIZE = 20 * 1024*1024*1024  #default 1MB
    # tables.parameters.SORTEDLR_MAX_SIZE = 1 *1024*1024*1024 # default 8MB
    # tables.parameters.SORTED_MAX_SIZE = 1 *1024*1024*1024 # default 1MB
    # tables.parameters.LIMBOUNDS_MAX_SIZE = 1 *1024*1024*1024
    tic = time.time()
    f=tables.open_file(binfile, mode="r")
    print(f.get_node('/Photons/PhotonTable').will_query_use_indexing(q))
    bin = f.get_node('/Photons/PhotonTable').read_where(q)
    f.close()
    toc1 = time.time()
    f=tables.open_file(csifile, mode="r")
    print(f.get_node('/Photons/PhotonTable').will_query_use_indexing(q))
    csi = f.get_node('/Photons/PhotonTable').read_where(q)
    f.close()
    toc2 = time.time()
    f = tables.open_file(ulifile, mode="r")
    print(f.get_node('/Photons/PhotonTable').will_query_use_indexing(q))
    uli =f.get_node('/Photons/PhotonTable').read_where(q)
    f.close()
    toc3 = time.time()
    print(('Bin: {} rows {:.2f} s\n'
          'csi: {} rows {:.2f} s\n'
          'uli: {} rows {:.2f} s').format(len(bin),toc1-tic,len(csi),toc2-toc1, len(uli), toc3-toc2))



import mkidpipeline as pipe
if __name__ == '__main__':
    pipe.logtoconsole(file='/scratch/baileyji/mec/speedtest/lastrun-pold.log')
    pipe.configure_pipeline('/home/baileyji/src/mkidpipeline/tests/h5speed_pipe.yml')
    d = pipe.load_data_description('/home/baileyji/src/mkidpipeline/tests/h5speed_data.yml')

    # Basic checks
    print(numexpr.get_vml_version())

    b2h_configs = pipe.bin2hdf.gen_configs(d.timeranges)

    # Index shuffling
    settings = [dict(index=('full', 9), chunkshape=None, timesort=False, shuffle=True,
                     bitshuffle=False, ndx_bitshuffle=ndx_bshuffle, ndx_shuffle=ndx_shuffle)
                for ndx_shuffle in (True, False) for ndx_bshuffle in (True, False)]

    results = test_settings(b2h_configs[:1], settings[:1], TEST_QUERIES, cleanup=True)

    # ndx_bshuffle = False
    # ndx_shuffle = True
    #
    # # Chunk shape
    # settings = [dict(index=('ultralight', 3), chunkshape=cshape, timesort=tsort, shuffle=True,
    #                  bitshuffle=False, ndx_bitshuffle=ndx_bshuffle, ndx_shuffle=ndx_shuffle)
    #             for tsort in (True, False)
    #             for cshape in (None, 20, 60, 100, 200, 260, 400, 500, 1000, 2000)]
    # tsort = False
    # cshape = 250
    #
    # # Index type. Use ndx_shuffle and ndx_bshuffle from test 1, include that run in the results
    # settings = [dict(index=ndx, chunkshape=cshape, timesort=False, shuffle=True,
    #                  bitshuffle=False, ndx_bitshuffle=ndx_bshuffle, ndx_shuffle=ndx_shuffle)
    #             for ndx in [(k, l) for k in ('medium', 'ultralight') for l in (3, 6, 9)]]
    # ndx = ('full', 9)
    #
    # # Shuffling. Should just be a gain or loss at the ideal cs (but also test an extremal value
    # settings = [dict(index=ndx, chunkshape=cshape_i, timesort=False, shuffle=shuffle,
    #                  bitshuffle=bshuffle, ndx_bitshuffle=False, ndx_shuffle=True)
    #             for shuffle in (True, False) for bshuffle in (True, False)
    #             for cshape_i in (20, None)]
    #
    # shuffle = True
    # bshuffle = False
    #
    # # Apparent Ideal
    # ideal = dict(index=ndx, chunkshape=cshape, timesort=False, shuffle=shuffle,
    #              bitshuffle=bshuffle, ndx_bitshuffle=False, ndx_shuffle=True)

    results = test_settings(b2h_configs, settings, TEST_QUERIES, cleanup=True)

#Legacy test results
# dfile = '/mnt/data0/baileyji/mec/h5/1507093430.h5'  #darkness flat 60s
# csifile = '/mnt/data0/baileyji/mec/h5/1545542180_csi.h5'
# csitimefile = '/mnt/data0/baileyji/mec/h5/1545542180_csi_time.h5'
# binfile = '/mnt/data0/baileyji/mec/h5/1545542180_bin2hdf.h5'
# ulitimefile = '/mnt/data0/baileyji/mec/h5/1545542180_time.h5'
# uli9timefile = '/mnt/data0/baileyji/mec/h5/1545542180_time_ul9.h5'
# ulitimenoindexfile = '/mnt/data0/baileyji/mec/h5/1545542180_time_notindex.h5'
# ulinoindexfile = '/mnt/data0/baileyji/mec/h5/1545542180_noridindex.h5'
# ulifile = '/mnt/data0/baileyji/mec/h5/1545542180_ul6.h5'
#
#
#
#
# for f in [ObsFile(uli9timefile)]:
#     f.query(resid=resid, startt=firstSec, intt=intTime, startw=wvlStart, stopw=wvlStop)
#     f.query(resid=resid)
#     f.query(resid=resids)
#     f.query(startt=0, intt=10)
#
#
# files = [binfile, csifile, ulitimefile, ulitimenoindexfile, ulinoindexfile, ulifile, csitimefile]
# of = [ObsFile(f) for f in files]
# for f in of:
#     # f=ObsFile(fi)
#     f.query(resid=resid, startt=firstSec, intt=intTime, startw=wvlStart, stopw=wvlStop)
#     # f.file.close()
# # 2019-01-23 09:33:44,465 DEBUG Retrieved 148 rows in 1.731s using indices ('Wavelength', 'Time', 'ResID') for query (ResID==91416)&(((Time >= startt) & (Time < stopt))&((Wavelength >= startw) & (Wavelength < stopw))) (pid=29222)
# # 2019-01-23 09:33:45,770 DEBUG Retrieved 148 rows in 1.287s using indices ('Wavelength', 'Time', 'ResID') for query (ResID==91416)&(((Time >= startt) & (Time < stopt))&((Wavelength >= startw) & (Wavelength < stopw))) (pid=29222)
# # 2019-01-23 09:33:46,027 DEBUG Retrieved 148 rows in 0.239s using indices ('Wavelength', 'Time', 'ResID') for query (ResID==91416)&(((Time >= startt) & (Time < stopt))&((Wavelength >= startw) & (Wavelength < stopw))) (pid=29222)
# # 2019-01-23 09:33:50,189 DEBUG Retrieved 148 rows in 4.146s using indices ('Wavelength', 'ResID') for query (ResID==91416)&(((Time >= startt) & (Time < stopt))&((Wavelength >= startw) & (Wavelength < stopw))) (pid=29222)
# # 2019-01-23 09:33:53,467 DEBUG Retrieved 148 rows in 3.263s using indices ('Wavelength', 'Time') for query (ResID==91416)&(((Time >= startt) & (Time < stopt))&((Wavelength >= startw) & (Wavelength < stopw))) (pid=29222)
# # 2019-01-23 09:33:53,876 DEBUG Retrieved 148 rows in 0.391s using indices ('Wavelength', 'Time', 'ResID') for query (ResID==91416)&(((Time >= startt) & (Time < stopt))&((Wavelength >= startw) & (Wavelength < stopw))) (pid=29222)
# # 2019-01-23 10:44:45,547 DEBUG Retrieved 148 rows in 1.350s using indices ('Time', 'ResID', 'Wavelength') for query (ResID==91416)&(((Time >= startt) & (Time < stopt))&((Wavelength >= startw) & (Wavelength < stopw))) (pid=31119)
#
# for f in of:
#     # f = ObsFile(fi)
#     f.query(resid=resid)
#     # f.file.close()
# # 2019-01-23 09:33:54,056 DEBUG Retrieved 47412 rows in 0.163s using indices ('ResID',) for query (ResID==91416) (pid=29222)
# # 2019-01-23 09:33:54,097 DEBUG Retrieved 47783 rows in 0.024s using indices ('ResID',) for query (ResID==91416) (pid=29222)
# # 2019-01-23 09:33:58,423 DEBUG Retrieved 47783 rows in 4.310s using indices ('ResID',) for query (ResID==91416) (pid=29222)
# # 2019-01-23 09:34:03,180 DEBUG Retrieved 47783 rows in 4.742s using indices ('ResID',) for query (ResID==91416) (pid=29222)
# # 2019-01-23 09:34:05,442 DEBUG Retrieved 47783 rows in 2.249s using indices () for query (ResID==91416) (pid=29222)
# # 2019-01-23 09:34:05,507 DEBUG Retrieved 47783 rows in 0.050s using indices ('ResID',) for query (ResID==91416) (pid=29222)
# # 2019-01-23 10:44:51,968 DEBUG Retrieved 47783 rows in 6.420s using indices ('ResID',) for query (ResID==91416) (pid=31119)
#
# for fi in files:
#     f = ObsFile(fi)
#     f.query(resid=resids)
#     f.file.close()
# # 2019-01-23 09:34:06,995 DEBUG Retrieved 432866 rows in 1.473s using indices ('ResID',) for query ((ResID==91416)|(ResID==101119)|(ResID==90116)|(ResID==80881)|(ResID==81572)|(ResID==101471)|(ResID==90267)|(ResID==10603)|(ResID==80867)|(ResID==80180)) (pid=29222)
# # 2019-01-23 09:34:24,873 DEBUG Retrieved 436192 rows in 17.854s using indices ('ResID',) for query ((ResID==91416)|(ResID==101119)|(ResID==90116)|(ResID==80881)|(ResID==81572)|(ResID==101471)|(ResID==90267)|(ResID==10603)|(ResID==80867)|(ResID==80180)) (pid=29222)
# # 2019-01-23 09:34:30,365 DEBUG Retrieved 436192 rows in 5.477s using indices ('ResID',) for query ((ResID==91416)|(ResID==101119)|(ResID==90116)|(ResID==80881)|(ResID==81572)|(ResID==101471)|(ResID==90267)|(ResID==10603)|(ResID==80867)|(ResID==80180)) (pid=29222)
# # 2019-01-23 09:34:36,367 DEBUG Retrieved 436192 rows in 5.985s using indices ('ResID',) for query ((ResID==91416)|(ResID==101119)|(ResID==90116)|(ResID==80881)|(ResID==81572)|(ResID==101471)|(ResID==90267)|(ResID==10603)|(ResID==80867)|(ResID==80180)) (pid=29222)
# # 2019-01-23 09:34:57,121 DEBUG Retrieved 436192 rows in 20.737s using indices () for query ((ResID==91416)|(ResID==101119)|(ResID==90116)|(ResID==80881)|(ResID==81572)|(ResID==101471)|(ResID==90267)|(ResID==10603)|(ResID==80867)|(ResID==80180)) (pid=29222)
# # 2019-01-23 09:35:14,994 DEBUG Retrieved 436192 rows in 17.857s using indices ('ResID',) for query ((ResID==91416)|(ResID==101119)|(ResID==90116)|(ResID==80881)|(ResID==81572)|(ResID==101471)|(ResID==90267)|(ResID==10603)|(ResID==80867)|(ResID==80180)) (pid=29222)
# # 2019-01-23 10:44:59,295 DEBUG Retrieved 436192 rows in 7.327s using indices ('ResID',) for query ((ResID==91416)|(ResID==101119)|(ResID==90116)|(ResID==80881)|(ResID==81572)|(ResID==101471)|(ResID==90267)|(ResID==10603)|(ResID==80867)|(ResID==80180)) (pid=31119)
#
# for fi in files:
#     f = ObsFile(fi)
#     f.query(startt=0, intt=10)
#     f.file.close()
# # 2019-01-23 09:36:05,951 DEBUG Retrieved 20543781 rows in 50.942s using indices ('Time',) for query ((Time >= startt) & (Time < stopt)) (pid=29222)
# # 2019-01-23 09:38:50,549 DEBUG Retrieved 20543781 rows in 162.294s using indices ('Time',) for query ((Time >= startt) & (Time < stopt)) (pid=29222)
# # 2019-01-23 09:38:58,417 DEBUG Retrieved 20543781 rows in 7.797s using indices ('Time',) for query ((Time >= startt) & (Time < stopt)) (pid=29222)
# # 2019-01-23 09:39:08,505 DEBUG Retrieved 20543781 rows in 10.073s using indices () for query ((Time >= startt) & (Time < stopt)) (pid=29222)
# # 2019-01-23 09:41:50,704 DEBUG Retrieved 20543781 rows in 162.184s using indices ('Time',) for query ((Time >= startt) & (Time < stopt)) (pid=29222)
# # 2019-01-23 09:44:33,428 DEBUG Retrieved 20543781 rows in 162.653s using indices ('Time',) for query ((Time >= startt) & (Time < stopt)) (pid=29222)
# # 2019-01-23 10:45:07,916 DEBUG Retrieved 20543781 rows in 8.621s using indices ('Time',) for query ((Time >= startt) & (Time < stopt)) (pid=31119)
#
#
#
#
# qtime = timeit.timeit('cachetest()', setup='from __main__ import cachetest', number=1,  globals=locals())
#
#
# binres = list(testqueries(binfile, xCoord, yCoord, firstSec, intTime, wvlStart, wvlStop))
# csires = list(testqueries(csifile, xCoord, yCoord, firstSec, intTime, wvlStart, wvlStop))
# ulires = list(testqueries(ulifile, xCoord, yCoord, firstSec, intTime, wvlStart, wvlStop))
#
# ulires2 = testqueriesmultires(ulifile, xCoord, yCoord, firstSec, intTime, wvlStart, wvlStop)
#
# msg='Query form {}: {:.2f} {:.2f} {:.2f}'
# for b,c,u in zip(binres,csires,ulires):
#     q,bt = b
#     _,ct = c
#     _,ut = u
#     print(msg.format(q.replace('0',''),bt,ct,ut))
#
# msg='Query form {}: {:.2f} s'
# ulires2.sort(key=lambda x: x[1])
# for q, t in ulires:
#     print(msg.format(q,t))
#
#
# # tables.parameters.CHUNK_CACHE_NELMTS = 1e6
# # tables.parameters.CHUNK_CACHE_SIZE = 20 * 1024*1024*1024
# # tables.parameters.TABLE_MAX_SIZE = 20 * 1024*1024*1024  #default 1MB
# # tables.parameters.SORTEDLR_MAX_SIZE = 1 *1024*1024*1024 # default 8MB
# # tables.parameters.SORTED_MAX_SIZE = 1 *1024*1024*1024 # default 1MB
# # tables.parameters.LIMBOUNDS_MAX_SIZE = 1 *1024*1024*1024
#
#
# """
# Query form res&(wave-&wave+&(time-&time+)): 0.44 s
# Query form res&((wave-&wave+)&(time-&time+)): 0.44 s
# Query form res&(time-&time+&(wave-&wave+)): 0.44 s
# Query form res&((time-&time+)&(wave-&wave+)): 0.44 s
# Query form time-&time+&(wave-&wave+)&res: 0.45 s
#
# Query form time-&time+&wave-&wave+&res: 4.38 s
# Query form res&((time-&time+)&wave-&wave+): 4.38 s
# Query form res&(time-&time+&wave-&wave+): 4.38 s
# Query form (time-&time+)&wave-&wave+&res: 4.39 s
# Query form time-&time+&wave-&(wave+&res): 4.39 s
# Query form res&((wave-&wave+)&time-&time+): 4.42 s
# Query form res&(wave-&wave+&time-&time+): 4.42 s
# Query form res&time-&time+&(wave-&wave+): 4.42 s
# Query form res&wave-&wave+&(time-&time+): 4.57 s
# Query form res&(wave-&(wave+&(time-&time+))): 4.93 s
#
# Query form res&(time-&(time+&(wave-&wave+))): 7.30 s
#
# Query form res&wave-&(wave+&time-)&time+: 8.33 s
# Query form wave-&((time-&wave+)&(time+&res)): 8.34 s
# Query form (wave-&time-)&wave+&time+&res: 8.34 s
# Query form time-&((time+&wave-)&(wave+&res)): 8.35 s
# Query form time-&((time+&wave-)&wave+&res): 8.35 s
# Query form time-&(time+&wave-&(wave+&res)): 8.35 s
# Query form res&(wave-&time-&wave+&time+): 8.35 s
# Query form time-&(time+&(wave-&(wave+&res))): 8.35 s
# Query form wave-&((time-&wave+)&time+&res): 8.35 s
# Query form wave-&(time-&wave+&time+&res): 8.35 s
# Query form (res&wave-)&wave+&time-&time+: 8.35 s
# Query form res&time-&time+&wave-&wave+: 8.35 s
# Query form wave-&(time-&(wave+&(time+&res))): 8.35 s
# Query form wave-&time-&(wave+&time+)&res: 8.36 s
# Query form wave-&time-&wave+&time+&res: 8.36 s
# Query form res&wave-&wave+&time-&time+: 8.36 s
# Query form res&wave-&time-&wave+&time+: 8.36 s
# Query form wave-&(time-&wave+&(time+&res)): 8.36 s
# Query form (res&wave-)&time-&wave+&time+: 8.37 s
# Query form time-&(time+&wave-&wave+&res): 8.37 s
# Query form res&wave-&(time-&wave+)&time+: 8.37 s
# Query form wave-&time-&wave+&(time+&res): 8.37 s
# Query form res&time-&(time+&wave-)&wave+: 8.38 s
# Query form res&wave-&time-&(wave+&time+): 8.38 s
# Query form (res&time-)&time+&wave-&wave+: 8.48 s
# Query form res&(wave-&(time-&(wave+&time+))): 8.50 s
# Query form res&((wave-&time-)&(wave+&time+)): 8.53 s
# Query form res&((wave-&time-)&wave+&time+): 8.54 s
# Query form res&(wave-&time-&(wave+&time+)): 8.55 s
# """