import time

import tables as pt

#fn = '/mnt/data0/ScienceData/PAL2017a/20170409/cal_4lasAllInd_1491793638.h5'
fn = '/mnt/data0/ScienceData/PAL2017a/20170409/cal_maxInd_1491793638.h5'

f = pt.open_file(fn,'a')

t = f.root.Photons.data
print(t)

'''
print "Querying photon table with ITERROWS for all wavelengths in pixel (0,0)..."
print "this may take a minute..."
t0 = time.time()
wvls = [i['Wavelength'] for i in t.iterrows() if i['X']==0 and i['Y']==0]
t1 = time.time()
print "Gathered %i values in %3.2f seconds"%(len(wvls),t1-t0)


print "Now doing same query with WHERE..."
t2 = time.time()
wvlsFast = [i['Wavelength'] for i in t.where("""(X==0) & (Y==0)""")]
t3 = time.time()
print "Gathered %i values in %3.2f seconds"%(len(wvlsFast),t3-t2)
'''
'''
print "Now trying same query with indexed columns..."
print "Indexing columns X, Y, ResID..."
t4 = time.time()
try:
    indexrows = t.cols.X.create_index()
except ValueError:
    print "X column already indexed"
try:
    indexrows = t.cols.Y.create_index()
except ValueError:
    print "Y column already indexed"
try:
    indexrows = t.cols.ResID.create_index()
except ValueError:
    print "ResID col already indexed"
t5 = time.time()
print "Time to index = %3.2f"%(t5-t4)
'''

print("Performing query...")
print(t.will_query_use_indexing("""(X==0) & (Y==0) & (ResID==1)"""))
t6=time.time()
wvlsFaster = [i['Wavelength'] for i in t.where("""(ResID==1)""")]
#wvlsFaster = [i['Wavelength'] for i in t.where("""(X==0) & (Y==0)""")]
#resids = [i['ResID'] for i in t.where("""(X==0) & (Y==0)""")]

t7=time.time()
print("Gathered %i values in %3.2f seconds" % (len(wvlsFaster), t7 - t6))
#print resids[0]

f.close()






