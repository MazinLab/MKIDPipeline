import argparse
import os

import sys
import pkg_resources as pkg
import shutil

import mkidcore
import mkidcore.config
from mkidcore.corelog import getLogger
import mkidpipeline.pipeline as pipe
import mkidpipeline.config as config

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
    output = config.load_output_description(args.out_cfg)

    if args.init:
        config.make_paths(output_dirs=[os.path.dirname(o.output_file) for o in output])

    if args.vet:
        sys.exit(0)


