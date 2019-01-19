import numpy as np
import tables
import mkidpipeline.hdf.bin2hdf as bin2hdf
import shutil
import time
import timeit

"""Test 1: Creation"""
#make a bin2hdfconfig
# cfg=None
# builder = bin2hdf.HDFBuilder(cfg)
# builder.run(usepytables=True, index=('ultralight', 6))
# shutil.move(cfg.h5file, cfg.h5file[:-3]+'_pytables_ul6.h5')
# builder.done.clear()
# builder.run(usepytables=True, index=True)
# shutil.move(cfg.h5file, cfg.h5file[:-3]+'_pytables_csi.h5')
# builder.done.clear()
# builder.run(usepytables=False)
# shutil.move(cfg.h5file, cfg.h5file[:-3]+'_bin2hdf.h5')


"""Test 2 Query Times"""


def testqueries(fn, xCoord, yCoord, firstSec, intTime, wvlStart, wvlStop, nIters=1):
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


def testqueriesmultires(fn, xCoord, yCoord, firstSec, intTime, wvlStart, wvlStop, nIters=1):
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
    queriename = q.format('res', 'time-', 'time+', 'wave-', 'wave+')
    qtime=0
    qtime = timeit.timeit('doquery(h5, "{}")'.format(q),
                          setup='from __main__ import tables, doquery; h5=tables.open_file("{}", mode="r")'.format(fn),
                          number=nIters,  globals=locals())

    return queriename, qtime



def doquery(h5, query):
    return h5.get_node('/Photons/PhotonTable').read_where(query)

binfile = '/mnt/data0/baileyji/mec/out/1545542180_bin2hdf.h5'
csifile = '/mnt/data0/baileyji/mec/out/1545542180_pytables_csi.h5'
ulifile = '/mnt/data0/baileyji/mec/out/1545542180_pytables_ul6.h5'

# bin = tables.open_file(binfile, mode="r")
# csi = tables.open_file(csifile, mode="r")
# uli = tables.open_file(ulifile, mode="r")


firstSec = 2
intTime = 2
xCoord = 105
xCoord = [105,106,50,30]
yCoord = 55
wvlStart = -90
wvlStop = -80
#wvlStart = None
#wvlStop = None

binres = list(testqueries(binfile, xCoord, yCoord, firstSec, intTime, wvlStart, wvlStop))
csires = list(testqueries(csifile, xCoord, yCoord, firstSec, intTime, wvlStart, wvlStop))
ulires = list(testqueries(ulifile, xCoord, yCoord, firstSec, intTime, wvlStart, wvlStop))

testqueriesmultires(ulifile, xCoord, yCoord, firstSec, intTime, wvlStart, wvlStop)

msg='Query form {}: {:.2f} {:.2f} {:.2f}'
for b,c,u in zip(binres,csires,ulires):
    q,bt = b
    _,ct = c
    _,ut = u
    print(msg.format(q.replace('0',''),bt,ct,ut))

msg='Query form {}: {:.2f} s'
ulires.sort(key=lambda x: x[1])
for q,t in ulires:
    print(msg.format(q,t))
"""
Query form res&(wave-&wave+&(time-&time+)): 0.44 s
Query form res&((wave-&wave+)&(time-&time+)): 0.44 s
Query form res&(time-&time+&(wave-&wave+)): 0.44 s
Query form res&((time-&time+)&(wave-&wave+)): 0.44 s
Query form time-&time+&(wave-&wave+)&res: 0.45 s

Query form time-&time+&wave-&wave+&res: 4.38 s
Query form res&((time-&time+)&wave-&wave+): 4.38 s
Query form res&(time-&time+&wave-&wave+): 4.38 s
Query form (time-&time+)&wave-&wave+&res: 4.39 s
Query form time-&time+&wave-&(wave+&res): 4.39 s
Query form res&((wave-&wave+)&time-&time+): 4.42 s
Query form res&(wave-&wave+&time-&time+): 4.42 s
Query form res&time-&time+&(wave-&wave+): 4.42 s
Query form res&wave-&wave+&(time-&time+): 4.57 s
Query form res&(wave-&(wave+&(time-&time+))): 4.93 s

Query form res&(time-&(time+&(wave-&wave+))): 7.30 s

Query form res&wave-&(wave+&time-)&time+: 8.33 s
Query form wave-&((time-&wave+)&(time+&res)): 8.34 s
Query form (wave-&time-)&wave+&time+&res: 8.34 s
Query form time-&((time+&wave-)&(wave+&res)): 8.35 s
Query form time-&((time+&wave-)&wave+&res): 8.35 s
Query form time-&(time+&wave-&(wave+&res)): 8.35 s
Query form res&(wave-&time-&wave+&time+): 8.35 s
Query form time-&(time+&(wave-&(wave+&res))): 8.35 s
Query form wave-&((time-&wave+)&time+&res): 8.35 s
Query form wave-&(time-&wave+&time+&res): 8.35 s
Query form (res&wave-)&wave+&time-&time+: 8.35 s
Query form res&time-&time+&wave-&wave+: 8.35 s
Query form wave-&(time-&(wave+&(time+&res))): 8.35 s
Query form wave-&time-&(wave+&time+)&res: 8.36 s
Query form wave-&time-&wave+&time+&res: 8.36 s
Query form res&wave-&wave+&time-&time+: 8.36 s
Query form res&wave-&time-&wave+&time+: 8.36 s
Query form wave-&(time-&wave+&(time+&res)): 8.36 s
Query form (res&wave-)&time-&wave+&time+: 8.37 s
Query form time-&(time+&wave-&wave+&res): 8.37 s
Query form res&wave-&(time-&wave+)&time+: 8.37 s
Query form wave-&time-&wave+&(time+&res): 8.37 s
Query form res&time-&(time+&wave-)&wave+: 8.38 s
Query form res&wave-&time-&(wave+&time+): 8.38 s
Query form (res&time-)&time+&wave-&wave+: 8.48 s
Query form res&(wave-&(time-&(wave+&time+))): 8.50 s
Query form res&((wave-&time-)&(wave+&time+)): 8.53 s
Query form res&((wave-&time-)&wave+&time+): 8.54 s
Query form res&(wave-&time-&(wave+&time+)): 8.55 s
"""


def query(startw=None, stopw=None, startt=None, stopt=None, resid=None, intt=None):
    """
    intt takes precedence

    :param startw: number or none
    :param stopw: number or none
    :param startt: number or none
    :param endt: number or none
    :param resid: number, list/array or None
    :return:
    """
    ticksPerSec=1
    try:
        startt = int(startt * ticksPerSec)  # convert to us
    except TypeError:
        pass

    try:
        stopt = int(stopt * ticksPerSec)
    except TypeError:
        pass

    if intt is not None:
        stopt = (startt if startt is not None else 0) + int(intt * ticksPerSec)

    if resid is None:
        resid = tuple()
    elif isinstance(resid, (int, float)):
        resid = [resid]

    res = '|'.join(['(ResID=={})'.format(r) for r in map(int, resid)])
    res = '(' + res + ')' if '|' in res and res else res
    tp = '(Time < stopt)'
    tm = '(Time >= startt)'
    wm = '(Wavelength >= startw)'
    wp = '(Wavelength < stopw)'
    # should follow '{res} & ( ({time}) & ({wave}))'

    if startw is not None:
        if stopw is not None:
            wave = '({} & {})'.format(wm, wp)
        else:
            wave = wm
    elif stopw is not None:
        wave = wp
    else:
        wave = ''

    if startt is not None:
        if stopt is not None:
            time = '({} & {})'.format(tm, tp)
        else:
            time = tm
    elif stopt is not None:
        time = tp
    else:
        time = ''

    query = res + ('&(' if res and (time or wave) else '')
    query += time + ('&' if wave and time else '')
    query += wave + (')' if res else '')
    print(query)