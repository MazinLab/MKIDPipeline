import mkidpipeline.definitions as definitions
import mkidpipeline.samples
import mkidpipeline.config as config
import mkidpipeline.pipeline as pipe
import mkidcore.corelog
import pkg_resources as pkg

mkidcore.corelog.getLogger('mkidcore', setup=True,
                           configfile=pkg.resource_filename('mkidpipeline', './config/logging.yaml'))

data = mkidpipeline.samples.get_sample_data('default')
config.dump_dataconfig(data)
d = definitions.MKIDObservingDataset('data.yaml')

pcfg = config.configure_pipeline('pipe.yaml')
data = mkidpipeline.samples.get_sample_data('default')
config.dump_dataconfig(data)
d = definitions.MKIDObservingDataset('data.yaml')



list(d.all_observations)
list(d.wavecalable)
list(d.flatcalable)
list(d.speccalable)

sample_out = pipe.generate_sample_output()
with open('out.yaml', 'w') as f:
    config.yaml.dump(sample_out, f)
definitions.MKIDOutputCollection('out.yaml')
