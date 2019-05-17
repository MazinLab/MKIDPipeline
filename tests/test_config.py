from mkidpipeline.config import *
import mkidpipeline as pipe

df = '/scratch/baileyji/mec/data.yml'
pf = '/scratch/baileyji/mec/pipe.yml'
of = '/scratch/baileyji/mec/out.yml'

pipe.logtoconsole()

pcfg = pipe.configure_pipeline(pf)
dataset = pipe.load_data_description(df)
out = MKIDOutputCollection(of, df)
