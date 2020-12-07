##
from mkidpipeline.hdf.photontable import Photontable
import mkidpipeline as pipe
from pkg_resources import resource_filename
import os

pipe.logtoconsole(file='query.log')
os.chdir(resource_filename('mkidpipeline',  'tests'))
pipe.configure_pipeline(resource_filename('mkidpipeline',  os.path.join('tests','data', 'h5query_pipe.yml')))
data = pipe.load_data_description(resource_filename('mkidpipeline',  os.path.join('tests','data', 'h5query_data.yml')))

pipe.config.config.update('paths.out', '/work/tmp/out')
pipe.config.config.update('paths.tmp', '/work/tmp')
pipe.config.config.update('paths.data', '/darkdata/ScienceData/Subaru/')

pipe.bin2hdf.buildtables(data.timeranges, ncpu=1, remake=False, chunkshape=250)

pt = Photontable(data.by_name('Vega').h5, inmemory=False)

x = pt.query(resid=10000)
x = pt.query(startt=1, stopt=45)

#hog the ram, load all 99GB of uncompressed data into ram.
bigarray = pt.photonTable.read()