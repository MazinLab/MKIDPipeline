#!/usr/bin/env python3
import argparse
import os
import sys
import pkg_resources as pkg
from datetime import datetime

import mkidpipeline.definitions as definitions
from mkidcore.corelog import getLogger
import mkidpipeline.pipeline as pipe
import mkidpipeline.config as config
import mkidpipeline.steps as steps
import mkidpipeline.samples

def parser():
    # read in command line arguments
    parser = argparse.ArgumentParser(description="MKID Pipeline CLI")
    parser.add_argument(
        "-o",
        type=str,
        help="An output specification file",
        default="out.yaml",
        dest="out_cfg",
    )
    parser.add_argument(
        "-p",
        type=str,
        help="A pipeline config file",
        default="pipe.yaml",
        dest="pipe_cfg",
    )
    parser.add_argument(
        "-d", type=str, help="A input config file", default="data.yaml", dest="data_cfg"
    )
    parser.add_argument(
        "--vet",
        action="store_true",
        help="Check pipeline configuration and exit",
        default=False,
    )
    parser.add_argument(
        "--init",
        type=str,
        default=None,
        dest="instrument",
        help="Setup the pipeline, clobbers _default.yaml as needed",
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose")
    parser.add_argument(
        "--make-dir",
        dest="make_paths",
        help="Create all needed directories",
        action="store_true",
    )
    parser.add_argument(
        "-i",
        "--info",
        dest="info",
        help="Report information about the configuration",
        type=str,
        default="database",
    )
    parser.add_argument(
        "--make-outputs",
        dest="makeout",
        help="Run the pipeline on the outputs",
        action="store_true",
    )
    parser.add_argument(
        "--logcfg",
        dest="logcfg",
        help="Run the pipeline on the outputs",
        type=str,
        default=pkg.resource_filename("mkidpipeline", "./config/logging.yaml"),
    )

    return parser

# Func to create a script run_mkidpipeline.py in the same output directory as out.yaml, pipe.yaml, data.yaml
# run_mkidpipeline.py runs the data through the .yamls with python3 instead of command line 

def create_run_mkidpipeline_script():
    script_content = """
# Run script for mkidpipeline
# Execute in command line with: python3 run_mkidpipeline.py
import subprocess
def run_mkidpipe():
    command = ['mkidpipe', '--make-dir', '--make-outputs']
    try:
        # Run the command
        subprocess.run(command, check=True)
        print("Command executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while running the command: {e}")
if __name__ == '__main__':
    run_mkidpipe()
"""
    with open('run_mkidpipeline.py', 'w') as f: 
        f.write(script_content)

def init_pipeline(args, instrument="MEC"):
    defstr = (
        lambda x: f"{x}_default.yaml" if os.path.exists(f"{x}.yaml") else f"{x}.yaml"
    )
    default_pipe_config = pipe.generate_default_config(instrument=instrument)
    with open(defstr("pipe"), "w") as f:
        config.yaml.dump(default_pipe_config, f)
    create_run_mkidpipeline_script() 
    config.configure_pipeline(args.pipe_cfg)

    sample_data = mkidpipeline.samples.get_sample_data()
    config.dump_dataconfig(sample_data, defstr("data"))

    sample_out = mkidpipeline.samples.get_sample_output()
    with open(defstr("out"), "w") as f:
        config.yaml.dump(sample_out, f)


def run_step(stepname, dataset):
    try:
        pipe.PIPELINE_STEPS[stepname].fetch(dataset)
    except AttributeError:
        pass
    try:
        pipe.PIPELINE_STEPS[stepname].apply(dataset)
    except AttributeError:
        pass


def main(args):
    getLogger(
        "mkidcore",
        setup=True,
        logfile=f'mkidpipe_{datetime.now().strftime("%Y-%m-%d_%H%M")}.log',
        configfile=args.logcfg,
    )
    log = getLogger("mkidpipe")

    if args.verbose:
        getLogger("mkidpipe").setLevel("DEBUG")
        getLogger("mkidpipeline").setLevel("DEBUG")
        getLogger("mkidcore").setLevel("DEBUG")
    getLogger("mkidcore.metadata").setLevel("INFO")

    if args.instrument:
        init_pipeline(args, instrument=args.instrument)
        return 0

    config.configure_pipeline(args.pipe_cfg)
    outputs = definitions.MKIDOutputCollection(args.out_cfg, datafile=args.data_cfg)
    dataset = outputs.dataset

    if args.make_paths:
        config.make_paths(output_collection=outputs)
    missing_paths = config.verify_paths(output_collection=outputs, return_missing=True)
    if missing_paths:
        getLogger("mkidpipeline").critical(
            f"Required paths missing:\n\t" + "\n\t".join(missing_paths)
        )
        return 1

    issue_report = outputs.validation_summary(null_success=True)
    if issue_report:
        getLogger("mkidpipeline").critical(issue_report)
        return 1
    else:
        getLogger("mkidpipeline").info("Validation of output and data successful!")

    if args.vet:
        getLogger("mkidpipeline").info("Vetting YAMLs complete. Done.")
        return 0

    if args.info:
        config.inspect_database(detailed=args.verbose)

    if args.makeout:
        for step in config.config.flow:
            try:
                pipe.PIPELINE_STEPS[step].fetch(getattr(outputs, f"{step}s"))
            except AttributeError:
                pass
            if step == "wavecal":
                steps.wavecal._loaded_solutions = {}  # TODO why is this necessary for Pool to work despite __getstate__
            pipe.batch_applier(step, getattr(outputs, f"to_{step}"))

        steps.output.generate(outputs)
    return 0

# Shim for the new python script standard
def mainmain():
    args = parser().parse_args()
    return main(args)

if __name__ == "__main__":
    sys.exit(mainmain())
