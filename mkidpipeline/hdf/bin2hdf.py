#!/opt/anaconda3/envs/pipeline/bin/python
import tempfile
import subprocess
import os
import tables
import time
import sys
import numpy as np
import psutil
import multiprocessing as mp
import pkg_resources as pkg
from datetime import datetime
from glob import glob
import warnings
from io import StringIO
from mkidcore import pixelflags
from mkidcore.headers import ObsFileCols, ObsHeader
from mkidcore.corelog import getLogger
from mkidcore.config import yaml, yaml_object
import mkidcore.utils
from mkidcore.objects import Beammap

from mkidpipeline.hdf.photontable import ObsFile
import mkidpipeline.config

_datadircache = {}


def _get_dir_for_start(base, start):
    global _datadircache

    if not base.endswith(os.path.sep):
        base = base+os.path.sep

    try:
        nmin = _datadircache[base]
    except KeyError:
        try:
            nights_times = glob(os.path.join(base, '*', '*.bin'))
            with warnings.catch_warnings():  # ignore warning for nights_times = []
                warnings.simplefilter("ignore", UserWarning)
                nights, times = np.genfromtxt(list(map(lambda s: s[len(base):-4], nights_times)),
                                              delimiter=os.path.sep, dtype=int).T
            nmin = {times[nights == n].min(): str(n) for n in set(nights)}
            _datadircache[base] = nmin
        except ValueError:  # for not pipeline oriented bin file storage
            return base

    keys = np.array(list(nmin))
    try:
        return os.path.join(base, nmin[keys[keys < start].max()])
    except ValueError:
        raise ValueError('No directory in {} found for start {}'.format(base, start))


def estimate_ram_gb(directory, start, inttime):
    PHOTON_BIN_SIZE_BYTES = 8
    files = [os.path.join(directory, '{}.bin'.format(t)) for t in range(start-1, start+inttime+1)]
    files = filter(os.path.exists, files)
    n_max_photons = int(np.ceil(sum([os.stat(f).st_size for f in files])/PHOTON_BIN_SIZE_BYTES))
    return n_max_photons*PHOTON_BIN_SIZE_BYTES/1024/1024/1024


def build_pytables(cfg, index=('ultralight', 6), timesort=False, chunkshape=None, shuffle=True, bitshuffle=False,
                   wait_for_ram=3600, ndx_shuffle=True, ndx_bitshuffle=False):
    """wait_for_ram speficies the number of seconds to wait for sufficient ram"""
    from mkidpipeline.hdf.mkidbin import extract

    if cfg.starttime < 1518222559:
        raise ValueError('Data prior to 1518222559 not supported without added fixtimestamps')

    def free_ram_gb():
        mem = psutil.virtual_memory()
        return (mem.free+mem.cached)/1024**3

    ram_est_gb = estimate_ram_gb(cfg.datadir, cfg.starttime, cfg.inttime) + 2  # add some headroom
    if free_ram_gb()<ram_est_gb:
        msg = 'Insufficint free RAM to build {}, {:.1f} vs. {:.1f} GB.'
        getLogger(__name__).warning(msg.format(cfg.h5file, free_ram_gb(), ram_est_gb))
        if wait_for_ram:
            getLogger(__name__).info('Waiting up to {} s for enough RAM'.format(wait_for_ram))
            while wait_for_ram and free_ram_gb()<ram_est_gb:
                sleeptime = np.random.uniform(1,2)
                time.sleep(sleeptime)
                wait_for_ram-=sleeptime
                if wait_for_ram % 30:
                    getLogger(__name__).info('Still waiting (up to {} s) for enough RAM'.format(wait_for_ram))
    if free_ram_gb()<ram_est_gb:
        getLogger(__name__).error('Aborting build due to insufficient RAM.')
        return

    getLogger(__name__).debug('Starting build of {}'.format(cfg.h5file))

    photons = extract(cfg.datadir, cfg.starttime, cfg.inttime, cfg.beamfile, cfg.x, cfg.y)

    getLogger(__name__).debug('Data Extracted for {}'.format(cfg.h5file))

    if timesort:
        photons.sort(order=('Time', 'ResID'))
        getLogger(__name__).warning('Sorting photon data on time for {}'.format(cfg.h5file))
    elif not np.all(photons['ResID'][:-1] <= photons['ResID'][1:]):
        getLogger(__name__).warning('binprocessor.extract returned data that was not sorted on ResID, sorting'
                                    '({})'.format(cfg.h5file))
        photons.sort(order=('ResID', 'Time'))

    h5file = tables.open_file(cfg.h5file, mode="a", title="MKID Photon File")
    group = h5file.create_group("/", 'Photons', 'Photon Information')
    filter = tables.Filters(complevel=1, complib='blosc:lz4', shuffle=shuffle, bitshuffle=bitshuffle, fletcher32=False)
    table = h5file.create_table(group, name='PhotonTable', description=ObsFileCols, title="Photon Datatable",
                                expectedrows=len(photons), filters=filter, chunkshape=chunkshape)
    table.append(photons)

    getLogger(__name__).debug('Table Populated for {}'.format(cfg.h5file))
    if index:
        index_filter = tables.Filters(complevel=1, complib='blosc:lz4', shuffle=ndx_shuffle, bitshuffle=ndx_bitshuffle,
                                      fletcher32=False)

        def indexer(col, index, filter=None):
            if isinstance(index, bool):
                col.create_csindex(filters=filter)
            else:
                col.create_index(optlevel=index[1], kind=index[0], filters=filter)

        indexer(table.cols.Time, index, filter=index_filter)
        getLogger(__name__).debug('Time Indexed for {}'.format(cfg.h5file))
        indexer(table.cols.ResID, index, filter=index_filter)
        getLogger(__name__).debug('ResID Indexed for {}'.format(cfg.h5file))
        indexer(table.cols.Wavelength, index, filter=index_filter)
        getLogger(__name__).debug('Wavelength indexed for {}'.format(cfg.h5file))
        getLogger(__name__).debug('Table indexed ({}) for {}'.format(index, cfg.h5file))
    else:
        getLogger(__name__).debug('Skipping Index Generation for {}'.format(cfg.h5file))

    bmap = Beammap(cfg.beamfile, xydim=(cfg.x, cfg.y))
    group = h5file.create_group("/", 'BeamMap', 'Beammap Information', filters=filter)
    h5file.create_array(group, 'Map', bmap.residmap.astype(int), 'resID map')
    h5file.create_array(group, 'Flag', pixelflags.beammap_flagmap_to_h5_flagmap(bmap.flagmap), 'flag map')
    getLogger(__name__).debug('Beammap Attached to {}'.format(cfg.h5file))

    h5file.create_group('/', 'header', 'Header')
    headerTable = h5file.create_table('/header', 'header', ObsHeader, 'Header')
    headerContents = headerTable.row
    headerContents['isWvlCalibrated'] = False
    headerContents['isFlatCalibrated'] = False
    headerContents['isSpecCalibrated'] = False
    headerContents['isLinearityCorrected'] = False
    headerContents['isPhaseNoiseCorrected'] = False
    headerContents['isPhotonTailCorrected'] = False
    headerContents['timeMaskExists'] = False
    headerContents['startTime'] = cfg.starttime
    headerContents['expTime'] = cfg.inttime
    headerContents['wvlBinStart'] = 700
    headerContents['wvlBinEnd'] = 1500
    headerContents['energyBinWidth'] = 0.1
    headerContents['target'] = ''
    headerContents['dataDir'] = cfg.datadir
    headerContents['beammapFile'] = cfg.beamfile
    headerContents['wvlCalFile'] = ''
    headerContents['fltCalFile'] = ''
    headerContents['metadata'] = ''
    out = StringIO()
    yaml.dump({'flags': mkidcore.pixelflags.FLAG_LIST}, out)
    out = out.getvalue().encode()
    if len(out) > mkidcore.headers.METADATA_BLOCK_BYTES:  # this should match mkidcore.headers.ObsHeader.metadata
        raise ValueError("Too much metadata! {} KB needed, {} allocated".format(len(out) // 1024,
                                                                                mkidcore.headers.METADATA_BLOCK_BYTES // 1024))
    headerContents['metadata'] = out

    headerContents.append()
    getLogger(__name__).debug('Header Attached to {}'.format(cfg.h5file))


    h5file.close()
    getLogger(__name__).debug('Done with {}'.format(cfg.h5file))


def build_bin2hdf(cfg, exc, polltime=.1):
    tfile = tempfile.NamedTemporaryFile('w', suffix='.cfg', delete=False)
    cfg.write(tfile)
    tfile.close()

    getLogger(__name__).debug('Wrote temp file for bin2hdf {}'.format(tfile.name))

    proc = psutil.Popen((exc, tfile.name),
                        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                        shell=False, cwd=None, env=None, creationflags=0)
    tic = time.time()
    while True:
        try:
            out, err = proc.communicate(timeout=polltime)
            if out:
                getLogger(__name__).info(out.decode('utf-8'))
            if err:
                getLogger(__name__).error(err.decode('utf-8'))
        except subprocess.TimeoutExpired:
            if time.time() - tic > 5:
                tic = time.time()
                getLogger(__name__).debug('Waiting on bin2hdf (pid={}) for {}'.format(proc.pid, cfg.h5file))

        if proc.poll() is not None:
            break

    if exc == __file__:
        getLogger(__name__).info('Dummy run done')
        return

    _postprocess(cfg)

    getLogger(__name__).info('Postprocessing complete, cleaning temp files')
    try:
        os.remove(tfile.name)
    except IOError:
        getLogger(__name__).debug('Failed to delete temp file {}'.format(tfile.name))


def _postprocess(cfg, event=None):
    time.sleep(.1)
    _add_header(cfg)
    time.sleep(.1)
    if cfg.starttime < 1518222559:
        fix_timestamp_bug(cfg.h5file)
    time.sleep(.1)
    # Prior to Ben's speedup of bin2hdf.c the consolidatePhotonTablesCmd step would need to be here
    _index_hdf(cfg)
    if event is not None:
        event.set()


def _correct_timestamps(timestamps):
    """
    Corrects errors in timestamps due to firmware bug present through PAL2017b.

    Parameters
    ----------
    timestamps: numpy array of integers
        List of timestamps from photon list. Must be in original, unsorted order.

    Returns
    -------
    Array of corrected timestamps, dtype is uint32
    """
    timestamps = np.array(timestamps, dtype=np.int64)  # convert timestamps to signed values
    photonTimestamps = timestamps % 500
    hdrTimestamps = timestamps - photonTimestamps

    unsortedInds = np.where(np.diff(timestamps) < 0)[0] + 1  # mark locations n where T(n)<T(n-1)

    for ind in unsortedInds:
        indsToIncrement = np.where(hdrTimestamps == hdrTimestamps[ind])[0]
        indsToIncrement = indsToIncrement[indsToIncrement >= ind]
        hdrTimestamps[indsToIncrement] += 500

    correctedTimestamps = hdrTimestamps + photonTimestamps

    if np.any(np.diff(correctedTimestamps) < 0):
        correctedTimestamps = _correct_timestamps(correctedTimestamps)

    return np.array(correctedTimestamps, dtype=np.uint32)


def _index_hdf(cfg):
    hfile = tables.open_file(cfg.h5file, 'a')
    hfile.set_node_attr('/', 'PYTABLES_FORMAT_VERSION', '2.0')
    hfile.format_version = '2.0'
    filterObj = tables.Filters(complevel=0, complib='lzo')
    photonTable = hfile.root.Photons.PhotonTable
    photonTable.cols.Time.create_csindex(filters=filterObj)
    photonTable.cols.ResID.create_csindex(filters=filterObj)
    photonTable.cols.Wavelength.create_csindex(filters=filterObj)
    photonTable.flush()
    hfile.close()


def fix_timestamp_bug(file):
    # which writes the same photonlist twice to certain resIDs
    noResIDFlag = 2 ** 32 - 1
    hfile = tables.open_file(file, mode='a')
    beamMap = hfile.root.BeamMap.Map.read()
    imShape = np.shape(beamMap)
    photonTable = hfile.get_node('/Photons/PhotonTable/')
    photonList = photonTable.read()

    resIDDiffs = np.diff(photonList['ResID'])
    if np.any(resIDDiffs < 0):
        warnings.warn('Photon list not sorted by ResID! This could take a while...')
        photonList = np.sort(photonList, order='ResID',
                             kind='mergsort')  # mergesort is stable, so time order will be preserved
        resIDDiffs = np.diff(photonList['ResID'])

    resIDBoundaryInds = np.where(resIDDiffs > 0)[
                            0] + 1  # indices in masterPhotonList where ResID changes; ie marks boundaries between pixel tables
    resIDBoundaryInds = np.insert(resIDBoundaryInds, 0, 0)
    resIDList = photonList['ResID'][resIDBoundaryInds]
    resIDBoundaryInds = np.append(resIDBoundaryInds, len(photonList['ResID']))
    correctedTimeListMaster = np.zeros(len(photonList))
    for x in range(imShape[0]):
        for y in range(imShape[1]):
            resID = beamMap[x, y]
            resIDInd0 = np.where(resIDList == resID)[0]
            if resID == noResIDFlag or len(resIDInd0) == 0:
                # getLogger(__name__).info('Table not found for pixel', x, ',', y)
                continue
            resIDInd = resIDInd0[0]
            photonList_resID = photonList[resIDBoundaryInds[resIDInd]:resIDBoundaryInds[resIDInd + 1]]
            timeList = photonList_resID['Time']
            timestamps = np.array(timeList, dtype=np.int64)  # convert timestamps to signed values
            repeatTest = np.array(np.where(timestamps == timestamps[0]))
            if len(repeatTest[0]) > 1:
                print(x, y, resID)
                timestamps1 = timestamps[0:repeatTest[0][1]]
                timestamps2 = timestamps[repeatTest[0][1]:len(timestamps)]
                correctedTimeList1 = _correct_timestamps(timestamps1).tolist()
                correctedTimeList2 = _correct_timestamps(timestamps2).tolist()
                correctedTimeList = correctedTimeList1 + correctedTimeList2
                correctedTimeList = np.array(correctedTimeList)
            else:
                correctedTimeList = _correct_timestamps(timeList)
            assert len(photonList_resID) == len(timeList), 'Timestamp list does not match length of photon list!'
            correctedTimeListMaster[resIDBoundaryInds[resIDInd]:resIDBoundaryInds[resIDInd + 1]] = correctedTimeList
    assert len(photonList) == len(correctedTimeListMaster), 'Timestamp list does not match length of photon list!'
    correctedTimeListMaster = np.array(correctedTimeListMaster).flatten()
    photonTable.modify_column(column=correctedTimeListMaster, colname='Time')
    photonTable.flush()
    hfile.close()


def _add_header(cfg, wvlBinStart=700, wvlBinEnd=1500, energyBinWidth=0.1):
    dataDir = cfg.datadir
    firstSec = cfg.starttime
    expTime = cfg.inttime
    beammapFile = cfg.beamfile
    hfile = tables.open_file(cfg.h5file, mode='a')
    hfile.create_group('/', 'header', 'Header')
    headerTable = hfile.create_table('/header', 'header', ObsHeader, 'Header')
    headerContents = headerTable.row
    headerContents['isWvlCalibrated'] = False
    headerContents['isFlatCalibrated'] = False
    headerContents['isSpecCalibrated'] = False
    headerContents['isLinearityCorrected'] = False
    headerContents['isPhaseNoiseCorrected'] = False
    headerContents['isPhotonTailCorrected'] = False
    headerContents['timeMaskExists'] = False
    headerContents['startTime'] = firstSec
    headerContents['expTime'] = expTime
    headerContents['wvlBinStart'] = wvlBinStart
    headerContents['wvlBinEnd'] = wvlBinEnd
    headerContents['energyBinWidth'] = energyBinWidth
    headerContents['target'] = ''
    headerContents['dataDir'] = dataDir
    headerContents['beammapFile'] = beammapFile
    headerContents['wvlCalFile'] = ''
    headerContents.append()
    headerTable.flush()
    hfile.close()


@yaml_object(yaml)
class Bin2HdfConfig(object):
    _template = ('{x} {y}\n'
                 '{datadir}\n'
                 '{starttime}\n'
                 '{inttime}\n'
                 '{beamfile}\n'
                 '1\n'
                 '{outdir}')

    def __init__(self, datadir='./', beamfile='./default.bmap', starttime=None, inttime=None,
                 outdir='./', x=140, y=146, writeto=None, beammap=None):

        self.datadir = datadir
        self.starttime = starttime
        self.inttime = inttime

        self.beamfile = beamfile
        self.x = x
        self.y = y

        if beammap is not None:
            self.beamfile = beammap.file
            self.x = beammap.ncols
            self.y = beammap.nrows

        self.outdir = outdir
        if writeto is not None:
            self.write(writeto)

    @property
    def h5file(self):
        try:
            return self.user_h5file
        except AttributeError:
            return os.path.join(self.outdir, str(self.starttime) + '.h5')

    def write(self, file):
        dir = self.datadir
        if not glob(os.path.join(dir, '*.bin')):
            dir = os.path.join(self.datadir, datetime.utcfromtimestamp(self.starttime).strftime('%Y%m%d'))
        else:
            getLogger(__name__).debug('bin files found in data directory. Will not append YYYMMDD')

        try:
            file.write(self._template.format(datadir=dir, starttime=self.starttime,
                                             inttime=self.inttime, beamfile=self.beamfile,
                                             outdir=self.outdir, x=self.x, y=self.y))
        except AttributeError:
            with open(file, 'w') as wavefile:
                wavefile.write(self._template.format(datadir=dir, starttime=self.starttime,
                                                     inttime=self.inttime, beamfile=self.beamfile,
                                                     outdir=self.outdir, x=self.x, y=self.y))

    def load(self):
        raise NotImplementedError


class HDFBuilder(object):
    def __init__(self, cfg, force=False, executable_path=pkg.resource_filename('mkidpipeline.hdf', 'bin2hdf'),
                 **kwargs):
        self.cfg = cfg
        self.exc = executable_path
        self.done = False
        self.force = force
        self.kwargs = kwargs

    def handle_existing(self):
        """ Handles existing h5 files, deleting them if appropriate"""
        if os.path.exists(self.cfg.h5file):

            if self.force:
                getLogger(__name__).info('Remaking {} forced'.format(self.cfg.h5file))
                done = False
            else:
                try:
                    done = ObsFile(self.cfg.h5file).duration >= self.cfg.inttime
                    if not done:
                        getLogger(__name__).info(('{} does not contain full duration, '
                                                  'will remove and rebuild').format(self.cfg.h5file))
                except:
                    done = False
                    getLogger(__name__).info(('{} presumed corrupt,'
                                              ' will remove and rebuild').format(self.cfg.h5file), exc_info=True)
            if not done:
                try:
                    os.remove(self.cfg.h5file)
                    getLogger(__name__).info('Deleted {}'.format(self.cfg.h5file))
                except FileNotFoundError:
                    pass
            else:
                getLogger(__name__).info('H5 {} already built. Remake not requested. Done.'.format(self.cfg.h5file))
                self.done = True

    def run(self, polltime=0.1, usepytables=True, **kwargs):
        """kwargs is passed on to build_pytables or buildbin2hdf"""
        self.kwargs.update(kwargs)
        self.handle_existing()
        if self.done:
            return

        tic = time.time()
        if usepytables:
            build_pytables(self.cfg, **self.kwargs)
        else:
            build_bin2hdf(self.cfg, self.exc, polltime=polltime)
        self.done = True

        getLogger(__name__).info('Created {} in {:.0f}s'.format(self.cfg.h5file, time.time()-tic))


def runbuilder(b):
    getLogger(__name__).debug('Calling run on {}'.format(b.cfg.h5file))
    try:
        b.run()
    except Exception as e:
        getLogger(__name__).critical('Caught exception during run of {}'.format(b.cfg.h5file),
                                     exc_info=True)


def gen_configs(timeranges, config=None):
    cfg = mkidpipeline.config.config if config is None else config

    timeranges = list(set(timeranges))

    b2h_configs = []
    for start_t, end_t in timeranges:
        bc = Bin2HdfConfig(datadir=_get_dir_for_start(cfg.paths.data, start_t),
                           beammap=cfg.beammap, outdir=cfg.paths.out,
                           starttime=start_t, inttime=end_t - start_t)
        b2h_configs.append(bc)

    return b2h_configs


def buildtables(timeranges, config=None, ncpu=1, asynchronous=False, remake=False, **kwargs):
    timeranges = list(set(timeranges))

    b2h_configs = gen_configs(timeranges, config)

    builders = [HDFBuilder(c, force=remake, **kwargs) for c in b2h_configs]

    if ncpu == 1:
        for b in builders:
            try:
                b.run(**kwargs)
            except MemoryError:
                getLogger(__name__).error('Insufficient memory to process {}'.format(b.h5file))
        return timeranges

    pool = mp.Pool(min(ncpu, mp.cpu_count()))

    if asynchronous:
        getLogger(__name__).debug('Running async on {} builders'.format(len(builders)))
        async_res = pool.map_async(runbuilder, builders)
        pool.close()
        return timeranges, async_res
    else:
        pool.map(runbuilder, builders)
        pool.close()
        pool.join()


def tests(file):
    """tests that should be good on a h5 file"""
    import h5py
    t = h5py.File(file, 'r')
    badrids = np.array(t['BeamMap']['Map'])[np.array(t['BeamMap']['Flag']) > 0].ravel()
    rid = np.array(t['Photons']['PhotonTable']['ResID'])
    assert np.isin(rid, badrids).sum()==0


if __name__ == '__main__':
    """run as a dummy for testing"""
    print('Pretending to run bin2hdf on ' + sys.argv[1])
