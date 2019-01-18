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
    startTime = 0
    if firstSec > 0:
        startTime = int(firstSec * tickspsec)  # convert to us
    if intTime != -1:
        endTime = startTime + int(intTime * tickspsec)

    tq='(Time<{endt}) & (Time>={startt})'.format(startt=startTime, endt=endTime)
    wq='(Wavelength<{endw}) & (Wavelength>={startw})'.format(startw=wvlStart, endw=wvlStop)
    rq='(ResID == {rid})'.format(rid=resID)
    queries = ['{res} &   ({time}) & ({wave})  ',
               '{res} & ( ({time}) & ({wave}) )',
               '{res} & (  {time}  & ({wave}) )',
               '{res} & (  {time}  &  {wave}  )',  # slow
               '{res} & ( ({time}) &  {wave}  )',  # slow
               '{res} & (  {wave}  & ({time}) )',
               '{res} & ( ({wave}) &  {time}  )',
               '( {wave} & ({time}) ) & {res}',
               '( {wave} & {time} ) & {res}']
    queries = [q.format(res=rq, time=tq, wave=wq) for q in queries]

    qtimes = [timeit.timeit('doquery(h5, "{}")'.format(query),
                            setup='from __main__ import tables, doquery; h5=tables.open_file("{}", mode="r")'.format(fn),
                            number=nIters,  globals=locals()) for query in queries]

    return zip(queries, qtimes)

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
yCoord = 55
wvlStart = -90
wvlStop = -80
#wvlStart = None
#wvlStop = None

binres = list(testqueries(binfile, xCoord, yCoord, firstSec, intTime, wvlStart, wvlStop))
csires = list(testqueries(csifile, xCoord, yCoord, firstSec, intTime, wvlStart, wvlStop))
ulires = list(testqueries(ulifile, xCoord, yCoord, firstSec, intTime, wvlStart, wvlStop))


msg='Query form {}: {:.2f} {:.2f} {:.2f}'
for b,c,u in zip(binres,csires,ulires):
    q,bt = b
    _,ct = c
    _,ut = u
    print(msg.format(q.replace('0',''),bt,ct,ut))