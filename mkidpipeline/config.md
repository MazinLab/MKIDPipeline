`class mkidcore.config.ConfigThing(dict)`  is thread safe


Keys are lowercase, spaces are not permitted (canonical form will replace with underscore)
Values are floats, ints or strings, strings will have paired leading/trailing 
quotes removed. Quotes must be nested to include the mark e.g. '"Some String"'. 

x in config.namespace won't use inheritance
config.namespace.x WILL use inheritance


from mkidpipeline.calibration.flatcal import StepConfig

## Using the pipeline
Set up for reduction by calling `mkidpipe --init [-d]`. This will:
- create a default set of config files in the current directory
-if -d is specified it will create a defult set of all the necessary directories as well.
  