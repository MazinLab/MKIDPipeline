import argparse
import os
import shutil
import mkidpipeline.pipeline as pipe
import mkidpipeline.config as config
import sys
import pkg_resources as pkg

if __name__ == "__main__":
    # read in command line arguments
    parser = argparse.ArgumentParser(description='MKID Pipeline CLI')
    parser.add_argument('-o', type=str, help='An output specification file', default='out.yml')
    parser.add_argument('-p', type=str, help='A pipeline config file', default='pipe.yml')
    parser.add_argument('-d', type=str, help='A input config file', default='data.yml')
    parser.add_argument('--vet',  action='store_true', help='Check pipeline configuration', default=False)
    parser.add_argument('--init', action='store_true', help='Setup the pipeline, clobbers files freely')
    parser.add_argument('--verbose', action='store_true', help='Verbose')

    getLogger('mkidcore').setLevel('INFO')
    log = getLogger('mkidpipeline')
    log.setLevel('INFO')

    args = parser.parse_args()
    if args.verbose:
        log.setLevel('DEBUG')
        getLogger('mkidcore').setLevel('DEBUG')

    if args.init:
        shutil.copy2(pkg.resource_filename('mkidpipeline', 'pipe.yml'), args.pipe_cfg)
        shutil.copy2(pkg.resource_filename('mkidpipeline', 'data.yml'), args.data_cfg)
        shutil.copy2(pkg.resource_filename('mkidpipeline', 'out.yml'), args.out_cfg)

    config.configure_pipeline(args.pipe_cfg)
    config.load_data_description(args.data_cfg)
    config.load_output_description(args.out_cfg)

    if args.init:
        config.make_paths()

    if args.vet:
        sys.exit(0)


def generate_default_pipeline_config():
    cfg = mkidcore.cofig.ConfigThing()
    for name, step in pipe.PIPELINE_STEPS.items():
        cfg.register(step.name, step.StepConfig())