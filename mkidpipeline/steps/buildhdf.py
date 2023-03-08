import os
import tables
import time
import numpy as np
import multiprocessing as mp
from mkidcore.corelog import getLogger
from mkidcore.config import yaml
import mkidcore.utils
from mkidcore.objects import Beammap
from mkidcore.instruments import InstrumentInfo


from mkidpipeline.photontable import Photontable
import mkidpipeline.config
from mkidpipeline.utils.memory import PIPELINE_MAX_RAM_GB, free_ram_gb, reserve_ram, release_ram


PHOTON_BIN_SIZE_BYTES = 8


class StepConfig(mkidpipeline.config.BaseStepConfig):
    yaml_tag = u'!buildhdf_cfg'
    REQUIRED_KEYS = (('remake', False, 'Remake H5 even if they exist'),
                     ('include_baseline', False, 'Include the baseline in H5 phase/wavelength column'),
                     ('chunkshape', 250, 'HDF5 Chunkshape to use'),)  # nb propagates to kwargs of build_pytables


mkidcore.config.yaml.register_class(StepConfig)


def estimate_ram_gb(directory, start, inttime):
    files = [os.path.join(directory, f'{t}.bin') for t in
             range(int(start - 1), int(np.ceil(start) + inttime + 1))]
    files = filter(os.path.exists, files)
    n_max_photons = int(np.ceil(sum([os.stat(f).st_size for f in files]) / PHOTON_BIN_SIZE_BYTES))
    return 4.75 * n_max_photons * PHOTON_BIN_SIZE_BYTES / 1024 ** 3  #4.75 is empirical fudge


def _build_pytables(filename, bmap, instrument, datadir, starttime, inttime, include_baseline,
                    index=('ultralight', 6), timesort=False, chunkshape=250, shuffle=True, bitshuffle=False,
                    ndx_shuffle=True, ndx_bitshuffle=False, data=None):

    from mkidcore.binfile.mkidbin import extract
    from mkidpipeline.pipeline import PIPELINE_FLAGS, BEAMMAP_FLAGS    #here to prevent circular imports!
    getLogger(__name__).debug('Starting build of {}'.format(filename))

    if data is not None:
        photons = data
    else:
        photons = extract(datadir, starttime, inttime, bmap.file, bmap.ncols, bmap.nrows,
                          include_baseline=include_baseline)

    getLogger(__name__).debug('Data Extracted for {}'.format(filename))

    if timesort:
        photons.sort(order=('time', 'resID'))
        getLogger(__name__).warning('Sorting photon data on time for {}'.format(filename))
    elif not np.all(photons['resID'][:-1] <= photons['resID'][1:]):
        if data is not None:
            getLogger(__name__).warning('binprocessor.extract returned data that was not sorted on ResID, sorting'
                                        '({})'.format(filename))
        photons.sort(order=('resID', 'time'))

    h5file = tables.open_file(filename, mode="a", title="MKID Photon File")
    group = h5file.create_group("/", 'photons', 'Photon Information')
    filter = tables.Filters(complevel=1, complib='blosc:lz4', shuffle=shuffle, bitshuffle=bitshuffle, fletcher32=False)
    table = h5file.create_table(group, name='photontable', description=Photontable.PhotonDescription, title="Photon Datatable",
                                expectedrows=len(photons), filters=filter, chunkshape=chunkshape)
    table.append(photons)
    if data is not None:
        del photons

    getLogger(__name__).debug('Table populated for {}'.format(filename))
    if index:
        index_filter = tables.Filters(complevel=1, complib='blosc:lz4', shuffle=ndx_shuffle, bitshuffle=ndx_bitshuffle,
                                      fletcher32=False)

        def indexer(col, index, filter=None):
            if isinstance(index, bool):
                col.create_csindex(filters=filter)
            else:
                col.create_index(optlevel=index[1], kind=index[0], filters=filter)

        indexer(table.cols.time, index, filter=index_filter)
        getLogger(__name__).debug('Time Indexed for {}'.format(filename))
        indexer(table.cols.resID, index, filter=index_filter)
        getLogger(__name__).debug('ResID Indexed for {}'.format(filename))
        indexer(table.cols.wavelength, index, filter=index_filter)
        getLogger(__name__).debug('Wavelength indexed for {}'.format(filename))
        getLogger(__name__).debug('Table indexed ({}) for {}'.format(index, filename))
    else:
        getLogger(__name__).debug('Skipping Index Generation for {}'.format(filename))

    group = h5file.create_group("/", 'beammap', 'Beammap Information', filters=filter)
    h5file.create_array(group, 'map', bmap.residmap.astype(int), 'resID map')

    def beammap_flagmap_to_h5_flagmap(flagmap):
        h5map = np.zeros_like(flagmap, dtype=int)
        for i, v in enumerate(flagmap.flat):  # convert each bit to the new bit
            bset = [f'beammap.{f.name}' for f in BEAMMAP_FLAGS.flags.values() if f.bit == int(v)]
            h5map.flat[i] = PIPELINE_FLAGS.bitmask(bset)
        return h5map

    h5file.create_array(group, 'flag', beammap_flagmap_to_h5_flagmap(bmap.flagmap), 'flag map')
    getLogger(__name__).debug('Beammap Attached to {}'.format(filename))

    headerContents = {}
    headerContents['wavecal'] = ''
    headerContents['flatcal'] = ''
    headerContents['speccal'] = ''
    headerContents['flags'] = PIPELINE_FLAGS.names
    headerContents['pixcal'] = False
    headerContents['lincal'] = False
    headerContents['cosmiccal'] = False
    headerContents['dead_time'] = instrument.deadtime_us
    headerContents['UNIXSTR'] = starttime
    headerContents['UNIXEND'] = starttime + inttime
    headerContents['EXPTIME'] = inttime
    headerContents['E_BMAP'] = bmap.file
    headerContents['max_wavelength'] = instrument.maximum_wavelength
    headerContents['min_wavelength'] = instrument.minimum_wavelength
    headerContents['energy_resolution'] = instrument.energy_bin_width_ev
    headerContents['data_path'] = datadir

    # must not be an overlap
    assert set(h5file.root.photons.photontable.attrs._f_list('sys')).isdisjoint(headerContents)
    for k, v in headerContents.items():
        setattr(h5file.root.photons.photontable.attrs, k, v)

    h5file.close()
    del h5file
    getLogger(__name__).debug('Done with {}'.format(filename))


class HDFBuilder(object):
    def __init__(self, force=False, datadir='./', outdir='./', include_baseline=False, instrument='MEC',
                 beammap='MEC', starttime=None, inttime=None, user_h5file='',  **kwargs):
        # self.cfg = cfg
        self.datadir = datadir
        self.starttime = int(starttime)
        self.inttime = int(np.ceil(inttime))
        self.outdir = outdir
        self.include_baseline = include_baseline
        self.beammap = Beammap(beammap) if isinstance(beammap, str) else beammap
        self.beamfile = beammap.file
        self.done = False
        self.force = force
        self.build_kwargs = kwargs
        self.user_h5file=user_h5file
        self.instrument = InstrumentInfo(instrument) if isinstance(instrument, str) else instrument

    @property
    def h5file(self):
        return self.user_h5file if self.user_h5file else os.path.join(self.outdir, str(self.starttime) + '.h5')

    def handle_existing(self):
        """ Handles existing h5 files, deleting them if appropriate"""
        if os.path.exists(self.h5file):

            if self.force:
                getLogger(__name__).info('Remaking {} forced'.format(self.h5file))
                done = False
            else:
                done = False
                try:
                    pt = Photontable(self.h5file)
                    pt.photonTable[0],pt.photonTable[-1]
                    done = pt.duration >= self.inttime
                    if not done:
                        getLogger(__name__).info(f'{self.h5file} does not contain full duration, '
                                                 f'will remove and rebuild')
                except Exception as e:
                    getLogger(__name__).info(f'{self.h5file} presumed corrupt ({e}), will remove and rebuild')
            if not done:
                try:
                    os.remove(self.h5file)
                    getLogger(__name__).info('Deleted {}'.format(self.h5file))
                except FileNotFoundError:
                    pass
            else:
                getLogger(__name__).info('H5 {} already built. Remake not requested. Done.'.format(self.h5file))
                self.done = True

    def build(self, index=('ultralight', 6), timesort=False, chunkshape=250, shuffle=True, bitshuffle=False,
              wait_for_ram=300, ndx_shuffle=True, ndx_bitshuffle=False, data=None):
        """
        wait_for_ram speficies the number of seconds to wait for sufficient ram

        data may be a numpy recarray to bypass extraction
        """
        if data is None:
            if self.starttime < 1518222559:
                raise ValueError('Data prior to 1518222559 not supported without added fixtimestamps')

            ram_est_gb = estimate_ram_gb(self.datadir, self.starttime, self.inttime)
            if PIPELINE_MAX_RAM_GB < ram_est_gb:
                getLogger(__name__).error(f'Pipeline limited to {PIPELINE_MAX_RAM_GB:.0f} GB RAM '
                                          f'need ~{ram_est_gb:.0f} to build file. Aborting')
                return
        try:
            if data is None:
                reserved = reserve_ram(ram_est_gb * 1024 ** 3, timeout=wait_for_ram, id=self.h5file)
            else:
                reserved = 0
            _build_pytables(self.h5file, self.beammap, self.instrument, self.datadir, self.starttime, self.inttime,
                            self.include_baseline,
                            index=index, timesort=timesort, chunkshape=chunkshape, shuffle=shuffle,
                            bitshuffle=bitshuffle, ndx_shuffle=ndx_shuffle, ndx_bitshuffle=ndx_bitshuffle, data=data)
        except TimeoutError:
            reserved = 0
            getLogger(__name__).error(f'Aborting build of {self.h5file} due to insufficient RAM '
                                      f'(req. {ram_est_gb:.1f}, free  {free_ram_gb():.1f} GB) after {wait_for_ram} s.')
            return

        finally:
            release_ram(reserved)

    def run(self, **kwargs):
        """kwargs is passed on to build_pytables"""
        self.build_kwargs.update(kwargs)
        self.handle_existing()
        if self.done:
            return

        tic = time.time()
        self.build(**self.build_kwargs)
        self.done = True
        getLogger(__name__).info('Created {} in {:.0f}s'.format(self.h5file, time.time() - tic))


def _runbuilder(b):
    getLogger(__name__).debug('Calling run on {}'.format(b.h5file))
    try:
        b.run()
    except Exception as e:
        getLogger(__name__).critical('Caught exception during run of {}'.format(b.h5file), exc_info=True)


def buildfromarray(array, config=None, **kwargs):
    cfg = mkidpipeline.config.PipelineConfigFactory(step_defaults=dict(buildhdf=StepConfig()), cfg=config, copy=True)
    b = HDFBuilder(beammap=cfg.beammap, outdir=cfg.paths.out, starttime=array['time'].min()/1e6,
                   inttime=(array['time'].max()-array['time'].min())/1e6, force=True, **kwargs)
    b.run(data=array)


def buildtables(timeranges, config=None, ncpu=None, remake=None, **kwargs):
    """
    timeranges must be an iterable of (start, stop) or objects that have .start, .stop attributes providing the same
    If the pipeline is not configured it will be configured with defaults.
    ncpu and remake will be pulled from config if not specified
    kwargs my be used to pass settings on to pytables, exttra settings in the step config will also be passed to
    pytables
    """
    timeranges = map(lambda x: x if isinstance(x, tuple) else (x.start, x.stop), timeranges)
    timeranges = set(map(lambda x: (int(np.floor(x[0])), int(np.ceil(x[1]))), timeranges))

    cfg = mkidpipeline.config.PipelineConfigFactory(step_defaults=dict(buildhdf=StepConfig()), cfg=config, ncpu=ncpu,
                                                    copy=True)

    remake = mkidpipeline.config.config.buildhdf.get('remake', False) if remake is None else remake
    ncpu = mkidpipeline.config.config.get('buildhdf.ncpu') if ncpu is None else ncpu

    for k in mkidpipeline.config.config.buildhdf.keys():  # This is how chunkshape is propagated
        if k not in kwargs and k not in ('ncpu', 'remake', 'include_baseline'):
            kwargs[k] = mkidpipeline.config.config.buildhdf.get(k)

    builders = [HDFBuilder(datadir=mkidcore.utils.get_bindir_for_time(cfg.paths.data, start_t), beammap=cfg.beammap,
                           instrument=cfg.instrument, outdir=cfg.paths.out, starttime=start_t, inttime=end_t - start_t,
                           include_baseline=cfg.buildhdf.include_baseline, force=remake, **kwargs)
                for start_t, end_t in timeranges]

    if not builders:
        return

    if ncpu == 1 or len(builders) == 1:
        for b in builders:
            try:
                b.run()
            except MemoryError:
                getLogger(__name__).error('Insufficient memory to process {}'.format(b.h5file))
        return timeranges

    pool = mp.Pool(mkidpipeline.config.n_cpus_available(max=cfg.get('buildhdf.ncpu', inherit=True)))
    pool.map(_runbuilder, builders)
    pool.close()
    pool.join()
