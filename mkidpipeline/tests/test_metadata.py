from mkidpipeline.config import *
import mkidpipeline as pipe
import mkidpipeline.hdf.photontable as pt
from io import StringIO

df = '/scratch/baileyji/mec/data.yml'
pf = '/scratch/baileyji/mec/pipe.yml'
of = '/scratch/baileyji/mec/out.yml'
h5 = '/scratch/baileyji/mec/h5raw.h5'

pipe.logtoconsole()



df='/scratch/steiger/MEC/HD1160/data_HD1160.yml'
pcfg = pipe.configure_pipeline(pf)
dataset = pipe.load_data_description(df)
# out = MKIDOutputCollection(of, df)


md=parse_obslog('/scratch/baileyji/mec/obslog_201905171955.json')
metadata = load_observing_metadata()
ob = dataset.sobs[0]
mdl = select_metadata_for_h5(ob, metadata)
out = StringIO()
extensible_header = {'obs_metadata':mdl}
yaml.dump(extensible_header, out)
emdstr = out.getvalue().encode()
d = yaml.load(emdstr.decode())