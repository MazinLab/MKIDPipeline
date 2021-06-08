"""
Author: Sarah Steiger   Data: March 25, 2020

Implementation of a linearity correction to account for photons that may arrive during the dead time of the detector.
"""
import time
import numpy as np
from progressbar import ProgressBar

from mkidcore.corelog import getLogger
import mkidpipeline.config


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

    of = o.photontable
    if of.query_header('lincal'):
        getLogger(__name__).info("H5 {} is already linearity calibrated".format(of.filename))
        return

    tic = time.time()
    of.enablewrite()
    bar = ProgressBar().start()
    for resid in of.resonators(exclude=PROBLEM_FLAGS):
        #TODO as written this performes the same query twice, look at flatcal to improve performance
        photons = of.query(resid=resid, column='time')
        weights = calculate_weights(photons, cfg.lincal.dt, of.query_header('dead_time')*1e-6)
        of.multiply_column_weight(resid, weights, 'weight', flush=False)
        bar.update()
    of.photonTable.flush()
    bar.finish()
    of.update_header('lincal', True)
    of.update_header(f'LINCAL.DT', cfg.lincal.dt)
    getLogger(__name__).info(f'Lincal applied to {of.filename} in {time.time() - tic:.2f}s')
    of.disablewrite()
