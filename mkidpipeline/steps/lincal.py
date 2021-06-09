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


HEADER_KEYS = ('LINCAL.DT', 'LINCAL.TAU')


PROBLEM_FLAGS = ('pixcal.hot', 'pixcal.cold', 'pixcal.dead', 'beammap.noDacTone', 'wavecal.bad',
                 'wavecal.failed_validation', 'wavecal.failed_convergence', 'wavecal.not_monotonic',
                 'wavecal.not_enough_histogram_fits', 'wavecal.no_histograms',
                 'wavecal.not_attempted')


def calculate_weights(time_stamps, dt=1000, tau=0.000001):
    """
    Function for calculating the linearity weighting for all of the photons in an h5 file
    :param time_stamps: array of timestamps
    :param dt: time range over which to calculate the weights (microseconds)
    :param tau: detector dead time (microseconds)
    :param pixel: pixel location to calculate the weights for
    :return: numpy ndarray of the linearity weights for each photon
    """
    # TODO convert away from a for loop scipy.ndimage.general_filter?
    weights = np.zeros(len(time_stamps))
    for i, t in enumerate(time_stamps):
        min_t = t - dt
        max_t = t + dt
        int_t = 2 * dt
        num_phots = np.sum(time_stamps[(min_t < time_stamps) & (time_stamps < max_t)])
        weights[i] = (1 - num_phots * (tau/int_t))**(-1.0)
    return weights


def apply(o: mkidpipeline.config.MKIDTimerange, config=None):
    cfg = mkidpipeline.config.PipelineConfigFactory(step_defaults=dict(lincal=StepConfig()), cfg=config, copy=True)

    of = Photontable(o.h5, mode='w', in_memory=False)
    if of.query_header('lincal'):
        getLogger(__name__).info("H5 {} is already linearity calibrated".format(of.filename))
        return

    tic = time.time()
    bar = ProgressBar(max_value=np.count_nonzero(~of.flagged(PROBLEM_FLAGS, all_flags=False)))
    # for resid in bar(of.resonators(exclude=PROBLEM_FLAGS)):
    #     # TODO as written this performes the same query twice, look at flatcal to improve performance
    #     photons = of.query(resid=resid, column='time')
    #     weights = calculate_weights(photons, cfg.lincal.dt, of.query_header('dead_time')*1e-6)
    #     of.multiply_column_weight(resid, weights, 'weight', flush=False)

    of.photonTable.autoindex = False
    dead_time = of.query_header('dead_time') * 1e-6
    for resid in bar(of.resonators(exclude=PROBLEM_FLAGS)):
        indices = of.photonTable.get_where_list('resID==resid')
        if not indices.size:
            continue

        if (np.diff(indices) == 1).all():
            # times = of.photonTable.read(start=indices[0], stop=indices[-1] + 1, field='time')
            # old = of.photonTable.read(start=indices[0], stop=indices[-1] + 1, field='weight')
            photons = of.photonTable.read(start=indices[0], stop=indices[-1] + 1)
            new = calculate_weights(photons['time'], cfg.lincal.dt, dead_time) * photons['weight']
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
    of.update_header(f'lincal.DT', cfg.lincal.dt)
    getLogger(__name__).info(f'Lincal applied to {of.filename} in {time.time() - tic:.2f}s')
    del of
