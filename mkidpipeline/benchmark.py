import os
os.environ['NUMEXPR_MAX_THREADS'] = '32'
os.environ['NUMEXPR_NUM_THREADS'] = '16'
os.environ["TMPDIR"] = '/scratch/tmp/'

import tables.index
tables.index.profile = True

from photontable import Photontable



#@profile
def tsectest(f):
    of = Photontable(f)
    # p=LineProfiler(of.photonTable._where)
    # p.enable()
    of.query(start=0, intt=30)

    #All the time is in the tables.tableextension.Row iterator returned by table._where,
    # p.disable()
    # p.dump_stats('benchmark.py.lpstats')
    # p.print_stats()


tsectest('/scratch/baileyji/mec/out/1545544477_slowtest.h5')


# from mkidpipeline.hdf.photontable import Photontable
# import mkidpipeline.config
# mkidpipeline.config.logtoconsole()
# f='/mnt/data0/isabel/highcontrastimaging/Jan2019Run/20190112/Trapezium/trap_ditherwavecalib/1547374834.h5'
# obsfile = Photontable(f)
# obsfile.get_fits(start=0, duration=30, spec_weight=True, noise_weight=True, scaleByEffInt=False)
# #2019-02-15 16:54:41,351 DEBUG Feteched 27438555/27438555 rows in 111.714s using indices ('Time',) for query (Time < stopt)
#
#
# f='/scratch/baileyji/mec/out/1545545212_slowtest.h5'
# obsfile = Photontable(f,mode='w')
# obsfile.get_fits(start=0, duration=30, spec_weight=True, noise_weight=True, scaleByEffInt=False)
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
# obsfile.get_fits(start=0, duration=30, spec_weight=True, noise_weight=True, scaleByEffInt=False)
# #2019-02-15 16:58:02,389 DEBUG Feteched 33126732/70860428 rows in 15.331s using indices () for query (Time < stopt)
# #obsfile.photonTable.cols._g_col('Time').index shows dirty
# obsfile.photonTable.reindex_dirty()
# #2019-02-15 17:50:12,871 DEBUG Feteched 33126732/70860428 rows in 12.114s using indices ('Time',) for query (Time < stopt)
#
