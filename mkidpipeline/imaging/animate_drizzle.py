"""
Author: Noah Swimmer

Program to allow temporal and spectral drizzles to be animated.

TODO: There is a bug where if you use the entire field ('square_size: ') there is a bug where the colorbar for the
 stacked data matches that of the instantaneous data. If you use a defined square size, this does not happen. FIX!

TODO: Maybe combine this with movies.py?
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.colors as colors
from astropy.io import fits
from astropy.visualization import ImageNormalize, AsinhStretch, LinearStretch, \
    LogStretch, PowerDistStretch, PowerStretch, SinhStretch, SqrtStretch, SquaredStretch
import logging
import mkidpipeline as pipe

VALID_STRETCHES = {'asinh': AsinhStretch(),
                   'linear': LinearStretch(),
                   'log': LogStretch(),
                   'powerdist': PowerDistStretch(),
                   'sinh': SinhStretch(),
                   'sqrt': SqrtStretch(),
                   'squared': SquaredStretch()}

log = logging.getLogger(__name__)


def read_fits(file, wvl_bin=None):
    """
    Reads a fits file into temporal and spectral cubes. Temporal cube has dimensions of [time, x, y] and spectral cube
    has dimensions of [wvl, x, y].
    If wvl_bin is none, it takes the count rates from all of the spectral bins
    """
    fit = fits.open(file)
    fits_data = fit[1].data  # Fits data has axes of [time, wvl, x, y]
    if wvl_bin:
        temporal = fits_data[:, int(wvl_bin), :, :]  # temporal has axes of [time, x, y]
    else:
        temporal = np.sum(fits_data, axis=1)
    spectral = np.average(fits_data, axis=0)  # spectral has axes of [wvl, x, y]
    return {'temporal': temporal, 'spectral': spectral}


def generate_lightcurve(temporal_data, exp_time):
    """
    Generates a light curve from data in a temporal drizzler. Assumes that the data has been converted from count rate
    to counts (N_counts = count_rate * exp_time)
    """
    if type(temporal_data) is dict:
        lc_data = temporal_data['temporal']
    else:
        lc_data = temporal_data

    counts = np.array([np.sum(i) for i in lc_data])
    time_bins = np.linspace(0, (len(lc_data) * exp_time) - exp_time, len(lc_data))

    return {'time': time_bins, 'counts': counts}


def smooth_data(light_curve):
    residuals = []
    stddevs = []
    fits = []
    orders = []
    for i in np.arange(1, 21, 1):
        orders.append(i)
        fit = np.polyfit(light_curve['time'], light_curve['counts'], i)
        fits.append(fit)
        predicted_data = np.poly1d(fit)

        resid = light_curve['counts'] - predicted_data(light_curve['time'])
        residuals.append(resid)
        std = np.std(resid)
        stddevs.append(std)

    residuals = np.array(residuals)
    stddevs = np.array(stddevs)
    fits = np.array(fits)
    orders = np.array(orders)

    best_mask = stddevs == np.min(stddevs)
    log.info(f"Best fit was of order {orders[best_mask]}")
    best_fit_coeffs = fits[best_mask]
    best_fit_residuals = residuals[best_mask]
    smooth_mask = (abs(best_fit_residuals) <= 2 * (stddevs[best_mask]))
    return {'order': orders[best_mask], 'std_dev': np.min(stddevs), 'coeffs': best_fit_coeffs,
            'residuals': best_fit_residuals, 'mask': smooth_mask[0]}


def stack(fits_data):
    temporal_stack = [fits_data['temporal'][0]]
    spectral_stack = [fits_data['spectral'][0]]

    for i in fits_data['temporal'][1:]:
        temporal_stack.append(temporal_stack[-1] + i)
    for i in fits_data['spectral'][1:]:
        spectral_stack.append(spectral_stack[-1] + i)

    # Each frame in the spectral_stack is in CPS, not Counts. To keep the scale consistent (adding CPS values to a
    # 'stack' doesn't make sense). We make sure to average by the number of frames (in the fits data) that went into
    # that stacked frame. In essence, CPS_frameN = (CPS_wvlBin1 + CPS_wvlBin2 + ... + CPS_wvlBinN) / N (this has to be
    # 1-indexed, not 0,since a 1 wvlBin stack or the first wvlBin of every spectral stack would blow up with a
    # DivideByZeroError). *Final note: This assumes that the exposure times for each wvl bin are the same. This should
    # be the case, but if it is not then this next line breaks and must be treated more explicitly with time weighting.
    spectral_stack = [frame/(counter+1) for counter, frame in enumerate(spectral_stack)]

    return {'temporal': np.array(temporal_stack), 'spectral': np.array(spectral_stack)}


def generate_stack_data(fits_file, exp_time, smooth, square_size, time_range, wvl_bin=None):
    data = read_fits(fits_file, wvl_bin=wvl_bin)
    data['temporal'] = (data['temporal'] * exp_time).astype(int)  # Convert count rate to counts
    lightcurve = generate_lightcurve(data['temporal'], exp_time)

    if time_range:
        time_mask = (lightcurve['time'] >= time_range[0]) & (lightcurve['time'] <= time_range[1])
        lightcurve['time'] = lightcurve['time'][time_mask]
        lightcurve['counts'] = lightcurve['counts'][time_mask]
        data['temporal'] = data['temporal'][time_mask]

    if smooth:
        smooth_dict = smooth_data(lightcurve)
        mask = smooth_dict['mask']
        data['temporal'] = data['temporal'][mask]
        lightcurve['time'] = lightcurve['time'][mask]
        lightcurve['counts'] = lightcurve['counts'][mask]

    if square_size:
        max_size = np.min(data['spectral'][0].shape)  # The spectral and temporal data have the same (x,y) sizes.
        if square_size <= max_size:
            half_size = np.ceil(square_size / 2)
            ctr = [np.ceil(data['spectral'].shape[1]/2), np.ceil(data['spectral'].shape[2]/2)]
            data['temporal'] = data['temporal'][:, int(ctr[0]-half_size):int(ctr[0]+half_size), int(ctr[1]-half_size):int(ctr[1]+half_size)]
            data['spectral'] = data['spectral'][:, int(ctr[0]-half_size):int(ctr[0]+half_size), int(ctr[1]-half_size):int(ctr[1]+half_size)]

    stacks = stack(data)
    return {'data': data, 'stack': stacks, 'lightcurve': lightcurve, 'fits': fits_file, 'exp_time': exp_time, 'smoothed': smooth}


def animate(data, outfile=None, target='', stretch='linear', type='temporal', fps=10,
            plot_data=True, plot_stack=True, title=True):
    outfile = f'{target}.mp4' if outfile is None else outfile

    time_range = [data['lightcurve']['time'][0], data['lightcurve']['time'][-1]+data['exp_time']]
    exp_time = data['exp_time']

    FFMpegWriter = anim.writers['ffmpeg']
    writer = FFMpegWriter(fps, metadata=dict(title=f"Dither Movie: {outfile}", artist='Matplotlib'))
    # writer = set_up_writer(dict(title=f"Dither Movie: {outfile}", artist='Matplotlib', fits=data['fits'],
    #                             target=target, fps=fps, exp_time=exp_time, smoothed=data['smoothed'],
    #                             colorbar_stretch=stretch, time_range_in_drizzle=time_range))

    if type == 'temporal':
        anim_data = {'data': data['data']['temporal'], 'stack': data['stack']['temporal']}
    elif type == 'spectral':
        anim_data = {'data': data['data']['spectral'], 'stack': data['stack']['spectral']}
    else:
        log.error(f"Trying to animate an unsupported data type!!")
        return None

    if plot_data and not plot_stack:
        log.info(f"Only plotting real-time data!")
        anim_data = anim_data['data']
        side_by_side = False
    elif plot_stack and not plot_data:
        log.info(f"Only plotting the frames being stacked!")
        anim_data = anim_data['stack']
        side_by_side = False
    elif plot_data and plot_stack:
        log.info(f"Plotting real-time data and stacked frames side-by-side!")
        side_by_side = True
    else:
        log.error(f"Trying not to animate real-time data or image stacking!")
        return None

    if stretch not in VALID_STRETCHES.keys():
        log.warning(f"Invalid colorbar stretch specified: {stretch}. Using linear stretch by default.")
        stretch = 'linear'

    if side_by_side:
        fig, axs = plt.subplots(1, 2, constrained_layout=True)
        if title:
            plt.suptitle(f"{target}, {exp_time} s Frame exposure time, "
                         f"{fps} fps, {time_range[1] - time_range[0]} s of data")

        ims = []
        with writer.saving(fig, outfile, anim_data['data'][0].shape[1] * 2):
            for count, i in enumerate(zip(anim_data['data'], anim_data['stack'])):
                if (count % 10) == 0:
                    print(f"Frame {count} of {len(anim_data['data'])}")
                if count == 0:
                    norm_d = ImageNormalize(vmin=-10, vmax=np.max(anim_data['data']) + 5, stretch=VALID_STRETCHES[stretch])
                    norm_s = ImageNormalize(vmin=-10, vmax=np.max(anim_data['stack']) + 5, stretch=VALID_STRETCHES[stretch])
                    im1 = axs[0].imshow(i[0], cmap='hot', norm=norm_d)
                    im2 = axs[1].imshow(i[1], cmap='hot', norm=norm_s)
                    axs[0].set_title('Instantaneous Data')
                    axs[1].set_title('Integrated Data')
                    ims.append([im1, im2])
                    c1 = fig.colorbar(im1, ax=axs[0], location='bottom', shrink=0.7, ticks=[0, int(np.max(anim_data['data']))])
                    c2 = fig.colorbar(im2, ax=axs[1], location='bottom', shrink=0.7, ticks=[0, int(np.max(anim_data['stack']))])

                else:
                    im1.set_data(i[0])
                    im2.set_data(i[1])
                    ims.append([im1, im2])

                writer.grab_frame()
                
    else:
        fig, axs = plt.subplots(1, 1, constrained_layout=True)
        if title:
            plt.suptitle(f"{target}, {exp_time} s Frame exposure time, "
                         f"{fps} fps, {time_range[1] - time_range[0]} s of data")

        ims = []
        with writer.saving(fig, outfile, anim_data[0].shape[1] * 2):
            for count, i in enumerate(anim_data):
                if (count % 10) == 0:
                    log.debug(f"Frame {count} of {len(anim_data)}")
                if count == 0:
                    norm_d = ImageNormalize(vmin=-10, vmax=np.max(anim_data) + 5, stretch=VALID_STRETCHES[stretch])
                    im1 = axs.imshow(i, cmap='hot', norm=norm_d)
                    if plot_data:
                        axs.set_title('Instantaneous Data')
                    elif plot_stack:
                        axs.set_title('Stacked Data')
                    ims.append([im1])
                    c1 = fig.colorbar(im1, ax=axs, location='bottom', shrink=0.7, ticks=[0, int(np.max(anim_data))])

                else:
                    im1.set_data(i)
                    ims.append([im1])

                writer.grab_frame()


if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(description="Temporal drizzle animation utility")
    parser.add_argument('cfg', type=str, help='YML config file for the drizzle animator.')

    args = parser.parse_args()

    pipe.configure_pipeline(args.cfg)
    config = pipe.config.config

    if config.animation.power:
        VALID_STRETCHES['power'] = PowerStretch(config.animation.power)

    if not config.animation.target:
        log.warning('Target missing! Must be specified in the config for metadata.')

    if not config.data.startt:
        start = 0
    else:
        start = config.data.startt

    if config.data.duration and config.data.stopt:
        end = start + config.data.duration
    elif config.data.duration and not config.data.stopt:
        end = start + config.data.duration
    elif not config.data.duration and config.data.stopt:
        end = config.data.stopt
    else:
        end = np.inf

    time_range = [start, end]

    stacked_data = generate_stack_data(config.paths.fits, config.data.exp_time, config.data.smooth,
                                       config.data.square_size, time_range, config.data.wvl_bin)

    animate(stacked_data, outfile=config.paths.out, target=config.animation.target,
            stretch=config.animation.stretch, type=config.animation.type, fps=config.animation.fps,
            plot_data=config.animation.plot_data, plot_stack=config.animation.plot_stack,
            title=config.animation.show_title)
