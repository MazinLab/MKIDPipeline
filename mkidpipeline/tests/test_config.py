
import mkidpipeline.config as config
import mkidpipeline.pipeline as pipe

config.log_to_console()

data = pipe.generate_sample_data()
config.dump_dataconfig(data)
d = config.MKIDObservingDataset('data.yaml')

pcfg = config.configure_pipeline('pipe.yaml')
data = pipe.generate_sample_data()
config.dump_dataconfig(data)
d = config.MKIDObservingDataset('data.yaml')



list(d.all_observations)
list(d.wavecalable)
list(d.flatcalable)
list(d.speccalable)

sample_out = pipe.generate_sample_output()
with open('out.yaml', 'w') as f:
    config.yaml.dump(sample_out, f)
config.MKIDOutputCollection('out.yaml')