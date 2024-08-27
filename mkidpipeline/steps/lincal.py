"""
To preserve the phase response of each pixel to an incident photon, a dead time is enforced on each pixel immediately
following a photon event. This prevents a subsequent photon from falling on the decay tail of the previous one, but also
causes a non-linear response at high count rates.

The lincal corrects for this by calculating a weight for each photon based on the instantaneous count rate. This weight
is multiplied into the 'weight' column of the photontable.

"""
import time
import numpy as np
import tqdm  # Got rid of progressbar import in favor of this.

import mkidpipeline.definitions as definitions
from mkidcore.corelog import getLogger
import mkidpipeline.config
from mkidpipeline.photontable import Photontable


class StepConfig(mkidpipeline.config.BaseStepConfig):
    yaml_tag = u'!lincal_cfg'
    REQUIRED_KEYS = (('dt', 1000, 'time range over which to calculate the weights (us)'),)


PROBLEM_FLAGS = ('pixcal.hot', 'pixcal.cold', 'pixcal.dead', 'beammap.noDacTone', 'wavecal.bad')


def calculate_weights(times, dt=1000, tau=0.000010):
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


def apply(o: definitions.MKIDTimerange, config=None):
    cfg = mkidpipeline.config.PipelineConfigFactory(step_defaults=dict(lincal=StepConfig()), cfg=config, copy=True)

    of = Photontable(o.h5, mode='w', in_memory=False)
    if of.query_header('lincal'):
        getLogger(__name__).info("H5 {} is already linearity calibrated".format(of.filename))
        return

    from scipy.stats import poisson
    #The true weight of a photon is given my how many photons arrived in the deadtime interval (which we don't know)
    # e.g. if 2 then 2, 3 then 3 .... The probability that that number is say 3 is
    #  PoissonCDF(3, .05)-PoissonCDF(2, .05) where .05 would be the average event rate in the interval, here the
    # deadtime*max_count_rate. An upper bound on this is simply 2*(1-poisson.cdf(1, 0.05))
    maximum_effect = 2*(1-poisson.cdf(1, cfg.instrument.deadtime_us*cfg.instrument.maximum_count_rate*1e-6))
    getLogger(__name__).info(f'Linearity correction will not exceed {maximum_effect:.1e} and will take ~'
                             f'{2.6e-5*len(of.photonTable)/60:.0f} minutes. Consider setting lincal: False in your '
                             f'output configuration.')
    of.photonTable.autoindex = False
    tic = time.time()

    n_to_do = np.count_nonzero(~of.flagged(PROBLEM_FLAGS, all_flags=False))
    lastpct = 0

    
    if cfg.get('lincal.ncpu') > 1:
        disable_bar = True
        print('\nNo progress bar displayed due to multiple cpus used for lincal\n')

    else:
        disable_bar = False
    

    #Not ram intensive ~250MB peak

    for done, resid in tqdm.tqdm(enumerate(of.resonators(exclude=PROBLEM_FLAGS)), disable = disable_bar):  # Using tqdm here for a progress bar.
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
            new *= calculate_weights(times, cfg.lincal.dt, cfg.instrument.deadtime_us)
            of.photonTable.modify_column(start=indices[0], stop=indices[-1] + 1, column=new, colname='weight')
        else:
            getLogger(__name__).warning('Using modify_coordinates, this is very slow')
            photons = of.photonTable.read_coordinates(indices)
            photons['weight'] *= calculate_weights(photons['time'], cfg.lincal.dt, cfg.instrument.deadtime_us)
            of.photonTable.modify_coordinates(indices, photons)

        pct = np.round(done/n_to_do, 2)
        if pct and lastpct != pct and pct % .1 == 0:
            lastpct = pct
            getLogger(__name__).info(f'Lincal of {o} {pct*100:.0f} % complete')

    of.photonTable.autoindex = True
    of.photonTable.reindex_dirty()
    of.photonTable.flush()
    of.update_header('lincal', True)
    of.update_header(f'lincal.dt', cfg.lincal.dt)
    getLogger(__name__).info(f'Lincal applied to {of.filename} in {time.time() - tic:.2f}s')
    del of
