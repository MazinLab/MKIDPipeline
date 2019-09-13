import astropy
import json
from datetime import datetime


def from_fits(streamfiles, outfile, timerange=(-np.inf, np.inf)):
    """Rebuild a approximate of an obslog json file from a list of stream fits files.
    Output consists of a file with json-dumped dictionaries of the fits header of each stream image.

    Give a list of stream fits files, an output file, and an optional time range.
    """
    dlist = []
    for f in streamfiles:
        with astropy.io.fits.open(f) as hdul:
            for hdu in hdul[1:]:
                d = dict(hdu.header)
                d.pop('COMMENT')
                dlist.append(d)
    dlist.sort(key=lambda d:d['UTC'])

    with open(outfile,'w') as f:
        for d in dlist:
            timestamp = datetime.strptime(d['UTC'],'%Y%m%d%H%M%S').timestamp()
            if timerange[0] <= timestamp <= timerange[1]:
                f.write(json.dumps(d)+'\n')

# from glob import glob
# fi=glob('/mnt/data0/ScienceData/Subaru/20190908/stream*')
# from_fits(fi, 'dither_recovery_log.json', timerange=(1567930101-5,1567931625+5))