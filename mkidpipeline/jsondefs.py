MKIDReductionJobJSON = {
    'frameid': 'a subaru frame id',  # or None/null string
    'data': [0, 0],  # start and stop timestamps bracketing the relevant logging records
    'kind': 'image',  # 'image', 'dither', or 'wavecal'
    'instrument_config': 'path_to_dashboard.yaml',  #loadable with mkidcore.config.load
    'user_wavecal': 0  # a timestamp of a wavecal or None
}

# whenever a job is recieved the reduction program needs to look at the log directory in case
# the active log has changed and concatenate that log data

# wavecal by user wavecal time, then by first available in same logfile, then by first availabe in night

# path_to_dashboard.yaml contains info about all active log paths, input data directories,
# necessary to configure pipeline, beammap,
