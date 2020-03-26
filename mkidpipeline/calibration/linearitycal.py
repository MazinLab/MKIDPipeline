#!/bin/env python3
"""
Author: Sarah Steiger   Data: March 25, 2020

Implementation of a linearity correction to account for photons that may arrive during the dead time of the detector.
"""
import numpy as np
import mkidpipeline.config
cfg = mkidpipeline.config.config


def calculateWeights(time_stamps, dt=float(), tau=float(), pixel=tuple()):
    '''
    Function for calculating the linearity weighting for all of the photons in an h5 file
    :param file: file to calculate the weights
    :param dt: time range over which to calculate the weights (microseconds)
    :param tau: detector dead time (microseconds)
    :param pixel: pixel location to calculate the weights for
    :return: numpy ndarray of the linearity weights for each photon
    '''
    weights = np.zeros(len(time_stamps))
    for i, t in enumerate(photon_list['Time']):
        min_t = t - dt
        max_t = t + dt
        int_t = 2 * dt
        num_phots = np.sum(time_stamps[min_t < time_stamps < max_t])
        weights[i] = (1 - num_phots * (tau/int_t))**(-1.0)
    return weights


