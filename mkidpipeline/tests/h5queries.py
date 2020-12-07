##
from mkidpipeline.hdf.photontable import Photontable
import mkidpipeline as pipe
from pkg_resources import resource_filename
import os

def get_all_data(t):
    """ Return a numpy structured array with all the data in the file """
    return t.photonTable.read()

def long_query(t):
    return t.query(startt=1, stopt=45)

def med_query(t):
    resids=list(range(100))
    return t.query(stopt=10, resid=resids)

def fast_query(t):
    return t.query(resid=1000)

ncpu=10
pipe.logtoconsole(file='query.log')
os.chdir(resource_filename('mkidpipeline',  'tests'))
pipe.configure_pipeline(resource_filename('mkidpipeline',  os.path.join('tests','data', 'h5query_pipe.yml')))
data = pipe.load_data_description(resource_filename('mkidpipeline',  os.path.join('tests','data', 'h5query_data.yml')))

pipe.config.config.update('paths.out', '/work/tmp/out')
pipe.config.config.update('paths.tmp', '/work/tmp')
pipe.config.config.update('paths.data', '/darkdata/ScienceData/Subaru/')

pipe.bin2hdf.buildtables(data.timeranges, ncpu=1, remake=False, chunkshape=250)

pt = Photontable(data.by_name('Vega').h5)

x=pt.query(resid=10000)
x=pt.query(startt=1, stopt=45)


#hog the ram, load all 99GB of uncompressed data into ram.
bigarray = pt.photonTable.read()