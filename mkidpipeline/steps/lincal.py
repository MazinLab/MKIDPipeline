"""
Author: Sarah Steiger   Data: March 25, 2020

Implementation of a linearity correction to account for photons that may arrive during the dead time of the detector.
"""
import time
import numpy as np
from progressbar import ProgressBar

from mkidcore.corelog import getLogger
import mkidpipeline.config
from mkidpipeline.photontable import Photontable


class StepConfig(mkidpipeline.config.BaseStepConfig):
    yaml_tag = u'!lincal_cfg'
    REQUIRED_KEYS = (('dt', 1000, 'time range over which to calculate the weights (us)'),)


PROBLEM_FLAGS = ('pixcal.hot', 'pixcal.cold', 'pixcal.dead', 'beammap.noDacTone', 'wavecal.bad')


def calculate_weights(times, dt=1000, tau=0.000001):
    """
    Function for calculating the linearity weighting for all of the photons in an h5 file
    :param time_stamps: array of timestamps
    :param dt: time range over which to calculate the weights (microseconds)
    :param tau: detector dead time (microseconds)
    :param pixel: pixel location to calculate the weights for
    :return: numpy ndarray of the linearity weights for each photon
    """
    nphot = np.array([times[((t - dt) < times) & (times < (t + dt))].sum() for t in times])
    weights = 1/(1 - nphot * tau/2/dt)
    return weights


def apply(o: mkidpipeline.config.MKIDTimerange, config=None):
    cfg = mkidpipeline.config.PipelineConfigFactory(step_defaults=dict(lincal=StepConfig()), cfg=config, copy=True)

    of = Photontable(o.h5, mode='w', in_memory=False)
    if of.query_header('lincal'):
        getLogger(__name__).info("H5 {} is already linearity calibrated".format(of.filename))
        return

    of.photonTable.autoindex = False
    tic = time.time()
    bar = ProgressBar(max_value=np.count_nonzero(~of.flagged(PROBLEM_FLAGS, all_flags=False)))

    dead_time = of.query_header('dead_time') * 1e-6
    for resid in bar(of.resonators(exclude=PROBLEM_FLAGS)):
        indices = of.photonTable.get_where_list('resID==resid')
        if not indices.size:
            continue

        # 14s more
        # 12 % (1500 of 11551) |  ##                 | Elapsed Time: 0:04:37 ETA:   2:19:18
        # photons = of.query(resid=resid, column='time')#2rw gw m
        # weights = calculate_weights(photons, cfg.lincal.dt, of.query_header('dead_time')*1e-6)
        # of.multiply_column_weight(resid, weights, 'weight', flush=False)

        if (np.diff(indices) == 1).all():
            # 12 % (1500 of 11551) |  ##                 | Elapsed Time: 0:08:12 ETA:   5:48:36
            # photons = of.photonTable.read(start=indices[0], stop=indices[-1] + 1)
            # new = calculate_weights(photons['time'], cfg.lincal.dt, dead_time) * photons['weight']

            # 12% (1500 of 11551) |##                 | Elapsed Time: 0:04:22 ETA:   2:13:28
            # NB reading this way takes only 53% of the time as above
            times = of.photonTable.read(start=indices[0], stop=indices[-1] + 1, field='time')
            new = of.photonTable.read(start=indices[0], stop=indices[-1] + 1, field='weight')
            new *= calculate_weights(times, cfg.lincal.dt, dead_time)
            of.photonTable.modify_column(start=indices[0], stop=indices[-1] + 1, column=new, colname='weight')
        else:
            getLogger(__name__).warning('Using modify_coordinates, this is very slow')
            photons = of.photonTable.read_coordinates(indices)
            photons['weight'] *= calculate_weights(photons['time'], cfg.lincal.dt, dead_time)
            of.photonTable.modify_coordinates(indices, photons)

    of.photonTable.autoindex = True
    of.photonTable.reindex_dirty()
    of.photonTable.flush()
    of.update_header('lincal', True)
    of.update_header(f'lincal.dt', cfg.lincal.dt)
    getLogger(__name__).info(f'Lincal applied to {of.filename} in {time.time() - tic:.2f}s')
    del of
