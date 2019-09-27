import os
#Dark has 16 cores, 2 threads/core
os.environ['NUMEXPR_MAX_THREADS'] = '64'
os.environ['NUMEXPR_NUM_THREADS'] = '16'
import numexpr
import numpy as np
import tables
import time, timeit, copy, pickle, hashlib, re, collections, ast
from glob import glob
from pkg_resources import resource_filename
import matplotlib.pyplot as plt

import mkidpipeline as pipe
from mkidpipeline.hdf.photontable import ObsFile
import mkidpipeline.hdf.bin2hdf as bin2hdf
from mkidcore.corelog import getLogger



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
obsT = 900
wvlStart = -90
wvlStop = -80
resid = 91416
resids = [91416,101119,90116,80881,81572,101471,90267,10603,80867,80180]

DATASET_NAMES = ['Vega', 'HD36546', 'LkCa15']
QUERY_NAMES = ['1Res', '10Res', '2s', '30s', '90s', 'Wave', 'Combined']
DURATIONS = {1567922621: 900, 1547280150: 900, 1545552775: 1545553007 - 1545552775}

TEST_QUERIES = [dict(resid=resid),
                dict(resid=resids),
                dict(startt=firstSec, intt=shortT),
                dict(startt=firstSec, intt=longT),
                dict(startt=firstSec, intt=obsT),
                dict(startw=wvlStart, stopw=wvlStop),
                dict(resid=resids, startt=firstSec, intt=shortT, startw=wvlStart, stopw=wvlStop)
                ]
RID_QUERY = TEST_QUERIES[0]
MIDT_QUERY = TEST_QUERIES[3]
OBST_QUERY = TEST_QUERIES[4]


def setting_id(sdict):
    return hashlib.md5(str(sdict).encode()).hexdigest()


def recover():
    from glob import glob
    import pickle, os
    from mkidpipeline.tests.h5speed import setting_id
    with open('/scratch/baileyji/mec/speedtest/recovery.pickle', 'rb') as f:
        results = pickle.load(f)

    h5s = glob('/scratch/baileyji/mec/speedtest/out/*.h5')

    lookup = {k:v for k,v in zip(('1567922621_-0x2c4b44c7b43c6b31.h5', '1567922621_-0xef86fbefe6ab9f0.h5', '1567922621_-0x43217a48696f50ba.h5', '1567922621_0x5ba62e77c9b38fe2.h5'),[str(dict(index=('full', 9), chunkshape=None, timesort=False, shuffle=True,
              bitshuffle=False, ndx_bitshuffle=ndx_bshuffle, ndx_shuffle=ndx_shuffle)) for ndx_shuffle in (True, False) for ndx_bshuffle in (True, False)])}

    patches = {}
    for k in results:
        patches[k] =[]

    for h5 in h5s:

        size = os.stat(h5).st_size/1024**2
        for k,v in results.items():
            newh5 = '{}_{}.h5'.format(h5.split('_')[0], setting_id(k))
            if h5 in v.get('file_nfo',''):
                v['file_nfo'] = 'ObsFile: '+newh5
            elif v['size'] == size and lookup.get(os.path.basename(h5),'') == k:
                patches[k].append((h5, newh5))
                of = pipe.ObsFile(h5)
                v.update(dict(nphotons=len(of.photonTable), file_nfo='ObsFile: '+newh5, table_nfo=repr(of.photonTable)))
                del of
            os.rename(h5,newh5)

    with open('/scratch/baileyji/mec/speedtest/recovery_fixed.pickle', 'wb') as f:
        pickle.dump(results, f)


def test_settings(cfg, settings, queries, do_build=True, do_query=True, cleanup=True, recovery=''):

    if recovery:
        try:
            with open(recovery, 'rb') as f:
                results = pickle.load(f)
        except IOError:
            results = {}

    builders = []
    h5s = ['{}_{}.h5'.format(cfg.h5file[:-3], setting_id(s)) for s in settings]

    for s, h5 in zip(settings, h5s):
        builder = bin2hdf.HDFBuilder(copy.copy(cfg))
        builder.cfg.user_h5file = h5
        builder.kwargs = s
        builders.append(builder)

    needed = {b.cfg.h5file: False for b in builders}

    for b in builders:
        if b.cfg.h5file not in results:
            needed[b.cfg.h5file] = do_build
        elif do_query:
            for q in queries:
                if str(q) not in results[b.cfg.h5file]:
                    needed[b.cfg.h5file] = do_build
                    break

    for b in builders:
        if not (needed[b.cfg.h5file] and not os.path.exists(b.cfg.h5file)):
            continue
        tic = time.time()
        b.run()
        toc = time.time()
        results[b.cfg.h5file] = dict(creation=toc - tic)
        of = ObsFile(b.cfg.h5file)
        results[b.cfg.h5file].update(dict(size=os.stat(b.cfg.h5file).st_size/1024**2, nphotons=len(of.photonTable),
                                          file_nfo=str(of), table_nfo=repr(of.photonTable)))
        del of
        if recovery:
            with open(recovery, 'wb') as f:
                pickle.dump(results, f)

    if not do_query:
        queries = []

    for q in queries:
        for b in builders:
            key = b.cfg.h5file
            if key not in results or not os.path.exists(b.cfg.h5file):
                getLogger(__name__).info(f'No result record/file for {b.cfg.h5file}')
                continue
            if str(q) in results[key]:
                continue
            of = ObsFile(b.cfg.h5file)
            tic = time.time()
            result = of.query(**q)
            toc = time.time()
            del of
            results[key][str(q)] = {'time': toc-tic, 'nphotons': len(result)}
            if recovery:
                with open(recovery, 'wb') as f:
                    pickle.dump(results, f)

    if cleanup:
        for f in h5s:
            os.remove(f)

    return results


class SpeedResults:
    def __init__(self, file):

        self.count_images = {}

        with open(file, 'rb') as f:
            self._r = pickle.load(f)

        settings = [dict(index=ndx, chunkshape=cshape, timesort=tsort, shuffle=shuffle,
                         bitshuffle=bshuffle, ndx_bitshuffle=ndx_bshuffle, ndx_shuffle=ndx_shuffle)
                    for ndx in [(k, l) for k in ('full', 'medium', 'ultralight') for l in (3, 6, 9)]
                    for tsort in (False, True)
                    for cshape in (20, 100, 180, 260, 340, 500, 1000, 5000, 10000, None)
                    for ndx_shuffle in (True, False) for ndx_bshuffle in (True, False)
                    for shuffle in (True, False) for bshuffle in (True, False)]

        for k, v in self._r.items():
            for s in settings:
                if setting_id(s) in k:
                    v['kwargs'] = repr(s)
                    break

        for k, v in self._r.items():
            if 'kwargs' not in v:
                print(v['table_nfo'], k)
                raise ValueError

        for k in self._r:
            matches = re.search(r"chunkshape := \((\d+),\)", self._r[k]['table_nfo'])
            self._r[k]['chunkshape'] = int(matches.group(1))

    def determine_mean_photperRID(self):
        ret = {}
        for d in self.datasets:
            if d in self.count_images:
                cnt_img = self.count_images[d]
            else:
                res = list(self.query_iter((MIDT_QUERY, OBST_QUERY), dataset=d))
                i = np.array([r.queryt for r in res]).argmin()
                fast_file = res[i].file
                print('This might take a while: {} s'.format(res[i].queryt))
                of = ObsFile(fast_file)
                cnt_img = of.getPixelCountImage(flagToUse=0xFFFFFFFFFF)['image']
                del of
                self.count_images[d] = cnt_img
            ret[d] = cnt_img.sum() / (cnt_img > 0).sum()
        return ret

    @property
    def all_chunkshapes(self):
        return np.array(list(set([v['chunkshape'] for v in self._r.values()])))

    @property
    def settings(self):
        """A lift of all the H5 creation settings in the results data"""
        return list(map(ast.literal_eval, set([v['kwargs'] for v in self._r.values()])))

    @property
    def datasets(self):
        """A list of all the dataset start times in the results data"""
        return list(set(map(lambda f: int(os.path.basename(f).split('_')[0]), self._r.keys())))

    def query_iter(self, query, settings=None, dataset=None):

        if not isinstance(query, (tuple, list)):
            query = [query]

        if dataset is not None and not isinstance(dataset, (tuple, list)):
            dataset = [dataset]

        if settings is not None and not isinstance(settings, (tuple, list)):
            settings = [settings]

        def use(data):
            """use if a desired dataset and it was made with any of the desired settings"""
            if settings is not None:
                dsetting = ast.literal_eval(data['kwargs'])
                for setting in settings:
                    setting_match = all(dsetting[k] == v for k, v in setting.items())
                    if setting_match:
                        break
            else:
                setting_match = True
            if dataset is not None:
                dataset_match = any([str(d) in data['file_nfo'] for d in dataset])
            else:
                dataset_match = True
            return setting_match and dataset_match

        Result = collections.namedtuple('Result', ('createt', 'nphotons', 'chunkshape', 'size',
                                                   'index', 'timesorted', 'queryt', 'queryn', 'file'))

        for file, data in self._r.items():
            if not use(data):
                continue
            kwargs = ast.literal_eval(data['kwargs'])
            for q in query:
                if str(q) not in data:
                    continue
                r = data[str(q)]
                x = Result(createt=data['creation'], nphotons=data['nphotons'],
                           chunkshape=data['chunkshape'], size=data['size'],
                           timesorted=kwargs['timesort'], index=''.join(map(str, kwargs['index'])),
                           queryt=r['time'], queryn=r['nphotons'], file=file)
                yield x


def report():
    results = SpeedResults('/scratch/baileyji/mec/speedtest/recovery.pickle')
    pipe.logtoconsole()
    pipe.configure_pipeline(resource_filename('mkidpipeline',  os.path.join('tests','h5speed_pipe.yml')))

    mymin = lambda x: np.min(x) if len(x) else np.nan
    mymax = lambda x: np.max(x) if len(x) else np.nan
    array_range = lambda x: (min(x), max(x))

    summary = 'The {dset_name} dataset file ranges from {gblo:.1f}-{gbhi:.1f} GB and contains {nphot:.1f}E6 photons.'
    SERIES_NAMES = ['F9, UL3, UL9, M3, M9', 'BShufM9', 'NShufM9']
    series_settings = [dict(index=('full', 9), timesort=True, shuffle=True, bitshuffle=False,
                            ndx_bitshuffle=False, ndx_shuffle=True),
                       dict(index=('ultralight', 3), timesort=True, shuffle=True, bitshuffle=False,
                            ndx_bitshuffle=False, ndx_shuffle=True),
                       dict(index=('ultralight', 9), timesort=True, shuffle=True, bitshuffle=False,
                            ndx_bitshuffle=False, ndx_shuffle=True),
                       dict(index=('medium', 3), timesort=True, shuffle=True, bitshuffle=False,
                            ndx_bitshuffle=False, ndx_shuffle=True),
                       dict(index=('medium', 9), timesort=True, shuffle=True, bitshuffle=False,
                            ndx_bitshuffle=False, ndx_shuffle=True),
                       dict(index=('medium', 9), timesort=True, shuffle=True, bitshuffle=True,
                            ndx_bitshuffle=False, ndx_shuffle=True),
                       dict(index=('medium', 9), timesort=True, shuffle=False, bitshuffle=False,
                            ndx_bitshuffle=False, ndx_shuffle=True)]

    phot_per_RrID = results.determine_mean_photperRID()
    all_chunkshapes = results.all_chunkshapes
    all_chunkshapes.sort()

    fig, axes = plt.subplots(nrows=3, ncols=5, figsize=(12, 8))
    plt.setp(axes.flat, xlabel='Chunkshape (rows)', ylabel='Query Time')

    # Row & Column Labels
    pad = 5  # in points
    for ax, col in zip(axes[0], QUERY_NAMES):
        ax.annotate(col, xy=(0.5, 1), xytext=(0, pad), xycoords='axes fraction', textcoords='offset points',
                    size='large', ha='center', va='baseline')
    for ax, row in zip(axes[:, 0], DATASET_NAMES):
        ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0), xycoords=ax.yaxis.label,
                    textcoords='offset points', size='large', ha='right', va='center', rotation=90)

    for i, d in results.datasets:
        file_sizes = [v['size'] for k,v in results._r.items() if str(d) in k]
        duration = DURATIONS[d]
        # Fetch the total number of photons
        nphot = 0
        for k in results._r:
            if str(d) in k:
                nphot = results._r[k]['size']
                break
        # Print a summary
        print(summary.format(dset_name=DATASET_NAMES[i], gblo=min(file_sizes)/1024, gbhi=max(file_sizes)/1024,
                             nphot=nphot))

        nqpoht = 0
        for j, q in TEST_QUERIES:
            plt.sca(ax[i, j])
            for s, name in zip(series_settings, SERIES_NAMES):
                res = list(results.query_iter(q, settings=s, dataset=d))
                if not res:
                    continue
                nqpoht = res[0].queryn
                chunkshapes = list(set([r.chunkshape for r in res]))
                chunkshapes.sort()
                queryts = [[r.queryt for r in res if r.chunkshape == cs] for cs in chunkshapes]
                min_queryts = np.array(map(mymin, queryts))
                max_queryts = np.array(map(mymax, queryts))
                mean_queryts = np.array(map(np.mean, queryts))
                n_queryts = np.array(map(len, queryts))
                plt.plot(chunkshapes, mean_queryts, yerr=max_queryts-min_queryts, label=name)
                for n, x, y in zip(n_queryts, chunkshapes, mean_queryts):
                    plt.annotate(str(n), (x, y))

            plt.annotate('{} phot'.format(nqpoht), (0, 10))  # minimum of nqphot/chunkshape chunks

        print('Timesort: {:.1f} ~chunks/s. ResID Sort: {} ~chunks/rid'.format(array_range(nphot / all_chunkshapes / duration),
                                                                    array_range(phot_per_RrID[d] / all_chunkshapes)))

    fig.tight_layout()
    # tight_layout doesn't take these labels into account. We'll need
    # to make some room. These numbers are are manually tweaked.
    # You could automatically calculate them, but it's a pain.
    fig.subplots_adjust(left=0.15, top=0.95)



def query_times(results, k, q=''):
    if q:
        return results[k][q]['time']
    else:
        return np.array([v['time'] for n,v in results[k].items() if n.startswith('{')])


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


if __name__ == '__main__':

    # from mkidpipeline.tests.h5speed import *
    pipe.logtoconsole(file='/scratch/baileyji/mec/speedtest/lastrun-pold.log')
    pipe.configure_pipeline(resource_filename('mkidpipeline',  os.path.join('tests','h5speed_pipe.yml')))
    d = pipe.load_data_description(resource_filename('mkidpipeline',  os.path.join('tests','h5speed_data.yml')))

    # Basic checks
    print(numexpr.get_vml_version())

    b2h_configs = pipe.bin2hdf.gen_configs(d.timeranges)

    # Index shuffling
    # settings = [dict(index=('full', 9), chunkshape=None, timesort=False, shuffle=True,
    #                  bitshuffle=False, ndx_bitshuffle=ndx_bshuffle, ndx_shuffle=ndx_shuffle)
    #             for ndx_shuffle in (True, False) for ndx_bshuffle in (True, False)]
    #
    # for cfg in b2h_configs[0:1]:
    #     results = test_settings(cfg, settings, TEST_QUERIES, do_query=False, cleanup=False,
    #                             recovery='/scratch/baileyji/mec/speedtest/recovery.pickle')
    # Simple shuffling is the clear winner here.

    # Chunk shape
    # settings = [dict(index=('ultralight', 3), chunkshape=cshape, timesort=tsort, shuffle=True,
    #                  bitshuffle=False, ndx_bitshuffle=False, ndx_shuffle=True)
    #             for tsort in (False, True)
    #             for cshape in (20, 100, 180, 260, 340, 500, 1000, 5000, 10000, None)]
    # #
    # for cfg in b2h_configs[::-1]:
    #     results = test_settings(cfg, settings, TEST_QUERIES, do_query=True, cleanup=False,
    #                             recovery='/scratch/baileyji/mec/speedtest/recovery.pickle')

    # Index type. Use ndx_shuffle and ndx_bshuffle from test 1, include that run in the results
    # settings = [dict(index=ndx, chunkshape=cshape, timesort=False, shuffle=True,
    #                  bitshuffle=False, ndx_bitshuffle=False, ndx_shuffle=True)
    #             for ndx in ([('full',9)]+[(k, l) for k in ('ultralight',) for l in (3, 9)])
    #             for cshape in (None,340)]
    # for cfg in b2h_configs:
    #     results = test_settings(cfg, settings, TEST_QUERIES, do_build=False, do_query=True, cleanup=False,
    #                             recovery='/scratch/baileyji/mec/speedtest/recovery.pickle')



    # Shuffling. Should just be a gain or loss at the ideal cs (but also test an extremal value
    settings = [dict(index=('medium', 9), chunkshape=cshape_i, timesort=False, shuffle=shuffle,
                     bitshuffle=bshuffle, ndx_bitshuffle=False, ndx_shuffle=True)
                for shuffle in (True, False) for bshuffle in (True, False)
                for cshape_i in (340, None)]

    for cfg in b2h_configs:
        results = test_settings(cfg, settings, TEST_QUERIES, do_build=True, do_query=True, cleanup=False,
                                recovery='/scratch/baileyji/mec/speedtest/recovery.pickle')
    # shuffle = True
    # bshuffle = False
    #
    # # Apparent Ideal
    # ideal = dict(index=ndx, chunkshape=cshape, timesort=False, shuffle=shuffle,
    #              bitshuffle=bshuffle, ndx_bitshuffle=False, ndx_shuffle=True)

    # results = test_settings(b2h_configs, settings, TEST_QUERIES, cleanup=True)

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