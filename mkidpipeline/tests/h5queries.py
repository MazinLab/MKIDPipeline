from mkidpipeline.hdf.photontable import Photontable
import mkidpipeline as pipe
from pkg_resources import resource_filename
import os

h5file='somepath'

t=Photontable(h5file)


def get_all_data():
    """ Return a numpy structured array with all the data in the file """
    return t.photonTable.read()

def long_query():
    return t.query(startt=1, stopt=45)

def med_query():
    resids=list(range(100))
    return t.query(stopt=10, resid=resids)

def fast_query():
    return t.query(resid=1000)

ncpu=10
pipe.logtoconsole(file='query.log')
pipe.configure_pipeline(resource_filename('mkidpipeline',  os.path.join('tests','h5speed_pipe.yml')))
data = pipe.load_data_description(resource_filename('mkidpipeline',  os.path.join('tests','h5query_data.yml')))
pipe.bin2hdf.buildtables(dataset.timeranges, ncpu=ncpu, remake=False, chunkshape=250)

Photontable(data.all_observations[0].h5)