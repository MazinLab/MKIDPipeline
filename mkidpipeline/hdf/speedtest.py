import numpy as np
import tables
import mkidpipeline.hdf.bin2hdf as bin2hdf
import shutil
import time

"""Test 1 creation"""
#make a bin2hdfconfig
cfg=None
builder = bin2hdf.HDFBuilder(cfg)
builder.run(usepytables=True, index=('ultralight', 6))
shutil.move(cfg.h5file, cfg.h5file[:-3]+'_pytables_ul6.h5')
builder.done.clear()
builder.run(usepytables=True, index=True)
shutil.move(cfg.h5file, cfg.h5file[:-3]+'_pytables_csi.h5')
builder.done.clear()
builder.run(usepytables=False)
shutil.move(cfg.h5file, cfg.h5file[:-3]+'_bin2hdf.h5')


"""Test 2 Query Times"""
binfile = '/mnt/data0/baileyji/mec/out/1545542180_bin2hdf.h5'
csifile = '/mnt/data0/baileyji/mec/out/1545542180_pytables_csi.h5'
ulifile = '/mnt/data0/baileyji/mec/out/1545542180_pytables_ul6.h5'

bin = tables.open_file(binfile, mode="r")
csi = tables.open_file(csifile, mode="r")
uli = tables.open_file(ulifile, mode="r")

