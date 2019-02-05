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




import os
os.environ['NUMEXPR_MAX_THREADS'] = '64'
os.environ['NUMEXPR_NUM_THREADS'] = '32'
os.environ["TMPDIR"] = '/mnt/data0/tmp/'

import numpy as np
import mkidpipeline.hdf.bin2hdf as bin2hdf
import mkidpipeline.calibration.wavecal as wavecal
import mkidpipeline.calibration.flatcal as flatcal
import mkidpipeline.badpix as badpix
import mkidpipeline.config
import mkidpipeline.hdf.photontable
from mkidcore.config import getLogger
import multiprocessing as mp

import tables
import tables.parameters


def wavecal_apply(o):
    of = mkidpipeline.hdf.photontable.ObsFile(mkidpipeline.config.get_h5_path(o), mode='a')
    of.applyWaveCal(wavecal.load_solution(o.wavecal))
    of.file.close()


def batch_apply_wavecals(wavecal_pairs, ncpu=None):
    pool = mp.Pool(ncpu if ncpu is not None else mkidpipeline.config.n_cpus_available())
    pool.map(wavecal_apply, wavecal_pairs)
    pool.close()




datafile = '/mnt/data0/baileyji/mec/data.yml'
cfgfile = '/mnt/data0/baileyji/mec/pipe.yml'
mkidpipeline.config.logtoconsole()
pcfg = mkidpipeline.config.configure_pipeline(cfgfile)
dataset = mkidpipeline.config.load_data_description(datafile)
# pcfg = mkidpipeline.config.config
# bcfgs = bin2hdf.gen_configs(dataset.timeranges)

getLogger('mkidpipeline.calibration.wavecal').setLevel('INFO')
# getLogger('mkidpipeline.hdf.photontable').setLevel('INFO')


# NB dataset.dithers[0].timeranges[0] is sorted on time rest are on resid!!!

# bin2hdf.buildtables([t for wc in dataset.wavecals for t in wc.timeranges], ncpu=6, remake=False, timesort=False)
# bin2hdf.buildtables(dataset.dithers[0].timeranges, ncpu=7, remake=False, timesort=False)

# wavecals = wavecal.fetch(dataset.wavecals, verbose=False)


# # inspect for possible failures in wavecal due to sharing h5 files
# pixels = [p for p in ((x,y) for x in range(140) for y in range(146))
#           if not np.all(sol.has_data(sol.cfg.wavelengths,p)) and sol.beam_map_flags[p]==0]
# wavecal.fetch(dataset.wavecals, verbose=False, parallel=False, pixels=pixels, save='singlethread.npz')
# sol = wavecal.Solution('/mnt/data0/baileyji/mec/database/2018-12-23 051888891e1e27c48f09da56342a914aed89.npz')
# sol2 = wavecal.Solution('/mnt/data0/baileyji/mec/database/singlethread.npz')
# for p in pixels:
#     assert (sol2.has_data(sol2.cfg.wavelengths,p) == sol.has_data(sol.cfg.wavelengths,p)).all()


# batch_apply_wavecals(dataset.wavecalable, 10)




#noise.calibrate(table)

#TODO need to associate and update the flatcals that have wavecals by name with the actual wavecal object
flatcals = flatcal.fetch(dataset.flatcals[:1], async=True, remake=True)



table.applyFlatCal(flatcals)

#cosmic.flag(table)

badpix.find(table, cfg.badpix.method)

#spectralcal.do(table)

imagecube.form(table, wcs=cfg.fitsample.wcs)

filepath = os.path.join(cfg.paths.out, FLATFNAME_TEMPLATE.format(dict(flat.header)))
flat.writeto(filepath)

FLATFNAME_TEMPLATE = os.path.join('{run}','{date}','flat_{timestamp}.h5')