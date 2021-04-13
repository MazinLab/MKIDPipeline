
import mkidpipeline.config as config
import mkidpipeline.pipeline as pipe

config.log_to_console()

data = pipe.generate_sample_data()
config.dump_dataconfig(data)
d = config.MKIDObservingDataset('data.yml')

pcfg = config.configure_pipeline('pipe.yml')
data = pipe.generate_sample_data()
config.dump_dataconfig(data)
d = config.MKIDObservingDataset('data.yml')



list(d.all_observations)
list(d.wavecalable)
list(d.flatcalable)
list(d.speccalable)