# The MKID Pipeline
Data reduction pipeline for Mazinlab MKID instruments - see also 
[The MKID Pipeline paper](https://iopscience.iop.org/article/10.3847/1538-3881/ac5833) for further detail.

## Installation 

### Configure conda + mamba
ask sudo user to run:

```
conda update -n base -c conda-forge -c defaults conda
mamba update -n base -c conda-forge -c defaults mamba
sudo chmod -R g+s /opt/miniconda3 # (if needed because of mamba cache permissions errors)
```
### Create .condarc file, create conda environments, and clone the repositories 
Once your shell and conda + mamba are configured create ~/.condarc with these contents
```shell
#=============
channels:
  - conda-forge
  - http://ssb.stsci.edu/astroconda
  - intel
  - defaults
envs_dirs:
  - /home/<yourusername>/.conda/envs
  - /opt/anaconda3/envs
pkg_dirs:
  - /home/<yourusername>/.conda/pkgs
  - /opt/anaconda3/pkgs
pip_interop_enabled: true
#=============
```
Then make a src directory in your home directory, clone the pipeline repository, and create a pipeline conda environemnt 
```
mkdir ~/src
cd ~/src
git clone https://github.com/mazinlab/mkidpipeline.git
cd mkidpipeline
git checkout <a branch, probably develop or master>
cd ~
#if the following command gives issues try running: conda clean -a
mamba env create --name pipeline -f src/mkidpipeline/condaenv.yml
conda activate pipeline
pip install -e src/mkidpipeline
```
Next install the mkidcore repository which contains supplementary packages needed to run the pipeline such as logging, and
flagging

```
cd ~/src
conda activate pipeline # if not already activated
pip install -e git+https://github.com/mazinlab/mkidcore.git@<whatever branch goes with mkidp>#egg=mkidcore
```
## Pipeline Quick Start Guide

Move to a directory you would like to play around in, activate your `pipeline` environment with

`conda activate pipeline` 

and run: 

`mkidpipe --init`

This will create three YAML config files (NB "_default" will be appended if the file already exists):
1. `pipe.yaml` - The pipeline global configuration file.
1. `data.yaml` - A sample dataset. You'll need to redefine this with your actual data.
1. `out.yaml` - A sample output configuration. You'll need to redefine this as well.

Each of these files contains extensive comments and irrelevant settings (or those for which the defaults are fine) may 
be omitted. More details for `pipe.yaml` are in `mkidpipeline.pipeline` and `mkidpipeline.steps.<stepname>`. More details for 
the other two are in `mkidpipeline.config`. Data and output yaml files can be vetted with helpful errors by running

`mkidpipe --vet` in the directory containing your three files. 

To build and reduce this dataset open the `pipe.yaml` and make sure you are happy with the default `paths`, these should be 
sensible if you are working on GLADoS. On dark you'll want to change the `darkdata` folder to `data`. If the various 
output paths don't exist they will be created, though permissions issues could cause unexpected results. Using a shared 
database location might save you some time and is strongly encouraged at least across all of your pipeline runs 
(consider collaborating even with other users)! Outputs will be placed into a generated directory structure under 
`out` and WILL clobber existing files with the same name.

The `flow` section of the `pipe.yaml` (see also below) lists all of the pipeline steps that will be executed when doing
the reduction. Here you may comment out or delete all steps you do not wish to run. For example to run all steps except
the cosmiccal and speccal, the flow will look like this:

```
flow: 
- buildhdf
- attachmeta
- wavecal
  #- cosmiccal
- pixcal
- flatcal
- wcscal
  #- speccal
```

Note that `buildhdf`, `attachmeta`, and `wavecal` need to be run for all reductions or else you will run into unexpected 
behavior. 
  
To generate all necessary directories as specified in the `paths` section of the `pipe.yaml`, run

`mkidpipe --make-dir`

NOTE: The default values for these `paths` will need to be changed to point to the appropriate location for your machine. 

To run the full calibration pipeline and generate specified outputs, use 

`mkidpipe --make-outputs` in the directory containing the three yaml files.

See `mkidpipe --help` for more options, including how to run a single step or specify yaml files in different directories.

After a while (~TODO hours with the defaults) you should have some outputs to look at. To really get going you'll now 
need to use observing logs to figure out what your `data.yaml` and `out.yaml` should contain for the particular data set
you want to look at. Look for good seeing conditions and note the times of the nearest laser cals.

## Pipeline Flow

When run, the pipeline goes through several steps, some only as needed, and only as needed for the requested outputs, 
so it won't slow you down to have all your data defined in one place (i.e. you do not need multiple `data.yaml` files
for different targets in a given night, they can all go in the same file). The steps are each described briefly in the 
list below with practical notes given for each step following.

1. Photontables (`mkidpipeline.photontable.Photontable`) files _for the defined outputs_ are created as needed by 
`mkidpipeline.steps.buildhdf` and are saved to the `out` path in the form of HDF5 (.h5) files.
1. Observing metadata defined in the data definition and observing logs is attached to tables by 
`mkidpipeline.pipeline.batch_apply_metadata`.
1. Any wavecals not already in the database are generated by `mkidpipeline.steps.wavecal.fetch`. There is some 
intelligence here so if the global config for the step or the start/stop times of the data is the same then the 
solution will not be regenerated. 
1. Photon data is wavelength calibrated by `mkidpipeline.steps.wavecal.apply`.
1. Individual observations (not timeranges) have a bad pixel mask determined and associated by 
   `mkidpipeline.steps.pixcal.apply`.
1. A non-linearity correction is computed and applied `mkidpipeline.steps.lincal.apply`. Note that the maximum correction is <<1% for MECs maximum count rate and this step takes > 1 hour. It is recommended to omit this step.
1. Cosmic-ray events are detected and attached via `mkidpipeline.steps.cosmiccal.apply`
1. Any flatcals not already in the database are generated by `mkidpipeline.steps.flatcal.fetch`. There is some intelligence here so if the global config for the step or the start/stop times of the data  the solution will be regnerated. 
1. Pixels are flat-fielded calibrated `mkidpipeline.steps.flatcal.apply`.
1. Any WCS solutions are generated via `mkidpipeline.steps.wcscal.fetch`, though at present this is a no-op as that code has not been written. All WCS solutions much thus be defined using the wcscal outputs of platescale and so forth (see the example).
1. Requested output products (dithers, stacks, movies, fits cubes, etc) are generated and stored in subdirectories per the output name (`mkidpipeline.steps.output.generate`). 
1. Finally spectral calibration data is generated and saved (speccals often use the output fits files of dithers) and are saved to the database (`mkidpipeline.steps.speccal.fetch`).

### Wavelength Calibration
Takes in a series of laser exposures as inputs with optional dark exposures to remove unwanted backgrounds.
### Cosmic Ray Calibration
Takes a long time to run and is a sub-percent level effect on typical data set. Not recommended for standard reductions.
### Pixel Calibration
Requires no additional calibration data. Parameters defined entirely in the `pipe.yaml` and defaults are typically 
sufficient in most use cases. 
### Flat Fielding
Flat fields are based on either whitelight (e.g. a classical quartz/dome/sky flat) observations or on a set of laser observations used for a wavecal. The whitelight codepath is operational but has not seen appreciable use as MEC flats are generally laser-based.

### WCS Calibration and Your Data

Some general notes:
   MEC observations are always taken with the derotator off (pupil tracking mode) and 
   the telescope target generally w/ a coronagraph. 
   MEC observations sometimes are taken for ADI.
      

Use Cases
1. Getting a FITS image/scube with an appropriate WCS solution
   1. Here WCS makes sense as either the time window midpoint or start
   2. We think implemented correctly
   3. When  these images are to be used by drizzler
      1. timebins are how to break up a dwell, larger than a dwell doesn't make sense
      2. wcs timesteps have an effective lower bound below which the resulting coadded spot size is limited by the PSF, 
   In non adi mode (where you don't want blurring) with timebins
      3. control whether or not you see the fields aligned in the steps or not
      4. control whether or not north aligned


TODO ADD some sort of parallactic angle offset support to build_wcs and teach get_wcs about it. rip out single pa time
then make drizzler compute offsets for each timestep. derotate becomes true always, adi mode turns on computation of 
per bin PA offsets


The end result of this is:
outputs gets an adi mode setting
derotate defaults to true
align start pa vanishes
wcs_timestep leaves the config files and takes its default of the nonblurring value

### Spectral Calibration
Currently not implemented. Converts pixel counts into a flux density. 

## Running From a Shell

Instead of running the pipeline from the command line, MKID data can be manipulated directly from a python shell. 
 
Generally shell operation would consist of something to the effect of

```python
import mkidpipeline.definitions as definitions
import pkg_resources as pkg
import mkidpipeline.config as config
import mkidpipeline.pipeline as pipe
import mkidcore as core
from mkidpipeline.photontable import Photontable 
from mkidcore.corelog import getLogger

#set up logging
lcfg = pkg.resource_filename('mkidpipeline', './utils/logging.yaml')
getLogger('mkidcore', setup=True, logfile=f'mylog.log', configfile=lcfg).setLevel('WARNING')
getLogger('mkidpipeline').setLevel('WARNING')

#To access photon tables directly
pt = Photontable('/path/to/h5_file.h5')

# To load and manipulate the full MKID data sets as defined in the YAMLS
config.configure_pipeline('pipe.yaml')
o = definitions.MKIDOutputCollection('out.yaml', datafile='data.yaml')
print(o.validation_summary())

# ... then playing around

```
