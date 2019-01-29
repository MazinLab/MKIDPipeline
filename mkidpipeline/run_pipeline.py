"""

Convert packetmaster .bin files into h5 files, based on a time window


Wavelength calibrate the dataset, Wavelength calbration solutions are valid for time windows
need a time window  (implicit e.g. full dataset, or explicit), an uncalibrated photon datatable, and all appropriate
mkidpipeline.calibration.wavecal.Solution objects
Note that mkidpipeline.calibration.wavecal.py is capable of creating solutions from raw bin files


Application is presently file based e.g.
photontable.ObsFile().applyWaveCal(calfile)
see mkidpipeline.utils.CalLookupFile as well for ideas, looks like a old version used this


Noise & Linearity Calibraion (noop)
Needs to be pulled from arcons
looks like photontable.ObsFile().applyTPFWeight(weights) was intended for this to limited extent


Flatfield Calibrate the data. Flat calibration solutions are valid for time windows.
need a time window  (implicit e.g. full dataset, or explicit), an uncalibrated photon datatable, and all appropriate
mkidpipeline.calibration.XXX.Solution objects, whihc don't yet exist.
Note that mkidpipeline.calibration.flatcal.py creates the solutions

photontable.ObsFile().applyFlatCal(calsolFile,save_plots=False)
need to move flat debug plots to the flat solution

Detect Cosmic rays and attach flag/correct

Detect Hot Pixels and attach flag
Detect Cold pixels and attach flag
badpix.find_bad_pixels(obsfile, method)

Perform spectrophotometric calibration

Do SSD

Generate image cube, output table, include dither functionality here. Existing dither code
consists of strictly manual alignment followed by:
    for processedIm:
        paddedFrame = irUtils.embedInLargerArray(processedIm,frameSize=padFraction)
        shiftedFrame = irUtils.rotateShiftImage(paddedFrame,0,dXs[i],dYs[i])
        upSampledFrame = irUtils.upSampleIm(shiftedFrame,upSample)
        upSampledFrame /= float(upSample*upSample) #conserve flux.
        shiftedFrames.append(upSampledFrame)
    finalImage = irUtils.medianStack(shiftedFrames)


Attach WCS Info (This is a function of the time and beammap)

"""

#TODO we need a way to retrieve templar/dashboard yml configs that were used based on timestamp
#TODO we need a way to autofetch all the parameters for steps of the pipeline (or at least standardize)
#TODO configure wavecal logging pulling from wavecal.setup_logging as needed



import os
import numpy as np
import mkidpipeline.hdf.bin2hdf as bin2hdf
import mkidpipeline.calibration.wavecal as wavecal
import mkidpipeline.calibration.flatcal as flatcal
import mkidpipeline.badpix as badpix
import mkidpipeline.config
import tables
import tables.parameters
from mkidcore.config import getLogger


os.environ["TMPDIR"] = '/mnt/data0/tmp/'


datafile = '/mnt/data0/baileyji/mec/data.yml'
cfgfile = '/mnt/data0/baileyji/mec/pipe.yml'
mkidpipeline.config.configure_pipeline(cfgfile)
mkidpipeline.config.logtoconsole()
c = dataset = mkidpipeline.config.load_data_description(datafile)
pcfg = mkidpipeline.config.config
bcfgs = bin2hdf.gen_configs(dataset.timeranges)
getLogger('mkidpipeline.calibration.wavecal').setLevel('INFO')
getLogger('mkidpipeline.hdf.photontable').setLevel('INFO')


bin2hdf.buildtables(dataset.timeranges, asynchronous=0, ncpu=6, remake=False)
wavecals = wavecal.fetch(dataset.wavecals, verbose=False)

# inspect/mnt/data0/baileyji/mec/database/2018-12-23 051888891e1e27c48f09da56342a914aed89.npz

raise RuntimeError()
flatcals = flatcal.fetch(dataset.flatcals, async=True)

def wcapply(o, w):
    mkidpipeline.hdf.photontable.ObsFile(mkidpipeline.config.get_h5_path(o)).applyWaveCal(w)

# pool = mp.Pool(4)
#TODO we will eventually need some code/metadata to associate specific wavecals with observations
pool.starmap(wcapply, zip(dataset.observations, [wavecals[0]]*len(dataset.observations)))

#noise.calibrate(table)
# flatcals = flatcal.fetch(input.flatcals, async=True)

raise RuntimeError()


table.applyFlatCal(flatcals)

#cosmic.flag(table)

badpix.find(table, cfg.badpix.method)

#spectralcal.do(table)

imagecube.form(table, wcs=cfg.fitsample.wcs)

filepath = os.path.join(cfg.paths.out, FLATFNAME_TEMPLATE.format(dict(flat.header)))
flat.writeto(filepath)

FLATFNAME_TEMPLATE = os.path.join('{run}','{date}','flat_{timestamp}.h5')