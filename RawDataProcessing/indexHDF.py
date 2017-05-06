'''
Author: Seth Meeker        Date: May 03, 2017

Modify HDF obs file made from Bin2HDF.c such that the photon table has key columns indexed.
This will facilitate much faster table searches when opening obs files for analysis
'''

import tables as pt
import time, sys
import numpy as np

if len(sys.argv)<2:
    print "Usage: >python indexHDF.py \"path to obs.h5\""
    sys.exit(0)

#example file for testing
#fn = '/mnt/data0/ScienceData/PAL2017a/20170409/cal_808_1491793638.h5'

fn = sys.argv[1]
f = pt.open_file(fn,'a')
t = f.root.Photons.data
print t

print "Indexing columns Time, ResID, Wavelength..."
t4 = time.time()

'''
try:
    indexrows = t.cols.X.create_index()
    print "Indexed X col"
except ValueError:
    print "X column already indexed"

try:
    indexrows = t.cols.Y.create_index()
    print "Indexed Y col"
except ValueError:
    print "Y column already indexed"

try:
    indexrows = t.cols.Baseline.create_index()
    print "Indexed Baseline col"
except ValueError:
    print "Baseline col already indexed"
'''

try:
    indexrows = t.cols.Time.create_csindex()
    print "Indexed Time col"
except ValueError:
    print "Time col already indexed"

try:
    indexrows = t.cols.ResID.create_csindex()
    print "Indexed ResID col"
except ValueError:
    print "ResID col already indexed"

try:
    indexrows = t.cols.Wavelength.create_index()
    print "Indexed Wvl col"
except ValueError:
    print "Wavelength col already indexed"


t5 = time.time()
print "Time to index = %3.2f"%(t5-t4)
print t.will_query_use_indexing("""(ResID==0) & (X==0) & (Y==0) & (Wavelength<0) & (Baseline<0) & (Time>0)""")

f.close()




