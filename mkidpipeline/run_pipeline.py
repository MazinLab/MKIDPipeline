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


import mkidpipeline.hdf.bin2hdf as bin2hdf
import mkidpipeline.calibration.wavecal as wavecal
import mkidpipeline.calibration.flatcal as flatcal
import mkidpipeline.config
import mkidpipeline.badpix as badpix

datafile = './src/mkidpipeline/mkidpipeline/data.yml'
cfgfile = './src/mkidpipeline/mkidpipeline/pipe.yml'
mkidpipeline.config.configure_pipeline(cfgfile)
c = input = mkidpipeline.config.load_data_description(datafile)

#fetch flat & wave cal for each block of data

#fetch h5 containing the blocks of data

table = bin2hdf.buildtables(input.timeranges, async=True)

wavecals = wavecal.fetch(input.wavecals, async=True)

#noise.calibrate(table)

flatcals = flatcal.fetch(input.flatcals, async=True)

raise RuntimeError()

table.applyWaveCal(wavecals)
table.applyFlatCal(flatcals)

#cosmic.flag(table)

badpix.find(table, cfg.badpix.method)

#spectralcal.do(table)

imagecube.form(table, wcs=cfg.fitsample.wcs)

filepath = os.path.join(cfg.paths.out, FLATFNAME_TEMPLATE.format(dict(flat.header)))
flat.writeto(filepath)

FLATFNAME_TEMPLATE = os.path.join('{run}','{date}','flat_{timestamp}.h5')