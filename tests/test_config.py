from mkidpipeline.config import *
import mkidpipeline as pipe

df = '/scratch/baileyji/mec/data.yml'
pf = '/scratch/baileyji/mec/pipe.yml'
of = '/scratch/baileyji/mec/out.yml'

pipe.logtoconsole()

pcfg = pipe.configure_pipeline(pf)
dataset = pipe.load_data_description(df)
out = MKIDOutputCollection(of, df)



import mkidcore.config
from datetime import datetime
import json
from mkidcore.config import ConfigThing


def parse_obslog(file):
    with open(file, 'r') as f:
        lines = f.readlines()

    ret = []
    for l in lines:
        ct = ConfigThing(json.loads(l).items())
        ct.register('utc', datetime.strptime(ct.utc, "%Y%m%d%H%M%S"), update=True)
        ret.append(ct)
    return ret

md=parse_obslog('/scratch/baileyji/mec/obslog_201905171955.json')
