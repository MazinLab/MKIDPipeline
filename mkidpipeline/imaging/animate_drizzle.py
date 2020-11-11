"""
Author: Noah Swimmer

Program to allow temporal drizzles to be animated.

TODO: Allow input of matplotlib keywords (maybe)

TODO: Maybe combine this with movies.py?

TODO: Allow ONLY stacked images or ONLY unstacked?
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.colors as colors
from astropy.io import fits
from astropy.visualization import ImageNormalize, SinhStretch, SquaredStretch
import logging


log = logging.getLogger(__name__)


def read_fits(file, wvl_bin=None):
    """
    Reads a fits file into temporal and spectral cubes. Temporal cube has dimensions of [time, x, y] and spectral cube
    has dimensions of [wvl, x, y].

    """
    fit = fits.open(file)
    fits_data = fit[1].data  # Fits data has axes of [time, wvl, x, y]
    if wvl_bin:
        temporal = fits_data[:, int(wvl_bin), :, :]  # temporal has axes of [time, x, y]
    else:
        temporal = np.average(fits_data, axis=1)
    spectral = np.average(fits_data, axis=0)  # spectral has axes of [wvl, x, y]
    return {'temporal': temporal, 'spectral': spectral}


def generate_lightcurve(temporal_data, exp_time):
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
    data['temporal'] = data['temporal'] * exp_time  # Convert from count rate to counts
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
    return {'data': data, 'stacks': stacks, 'lightcurve': lightcurve}


def set_up_writer(metadata: dict):
    FFMpegWriter = anim.writers['ffmpeg']
    if 'fps' not in metadata.keys():
        metadata['fps'] = 10
    return FFMpegWriter(fps=metadata['fps'], metadata=metadata)


def make_movie(fits_file, outfile, target, fps=10, square_size=None, exp_time=None,
                   time_range=None, smooth=True, stretch='linear'):
    writer = set_up_writer(dict(title=f"Dither Movie: {outfile}", artist='Matplotlib', fits=fits_file,
                                target=target, fps=fps, exp_time=exp_time, smoothed=smooth,
                                colorbar_stretch=stretch, time_range_in_drizzle=time_range))

# TODO: Animate function, make more robust, add spectral category, add ability to do side-by-side or alone
def animate(data, writer, target, exp_time, time_range, outfile, stretch):

    fig, axs = plt.subplots(1, 2, constrained_layout=True)
    plt.suptitle(f"{target}, {exp_time} s Frame exposure time, {fps} fps, {time_range[1] - time_range[0]} s of data")

    ims = []
    with writer.saving(fig, outfile, data[0].shape[1] * 3):
        for count, i in enumerate(zip(data, stack)):
            if (count % 10) == 0:
                log.debug(f"Frame {count} of {len(data)}")
            if count == 0:
                # Initialize the subplots
                if stretch == 'linear':
                    im1 = axs[0].imshow(i[0], cmap='hot', vmin=0, vmax=np.max(data))
                    im2 = axs[1].imshow(i[1], cmap='hot', vmin=0, vmax=np.max(stack))
                else:
                    norm_d = ImageNormalize(vmin=-10, vmax=np.max(data) + 5, stretch=SinhStretch())
                    norm_s = ImageNormalize(vmin=-10, vmax=np.max(stack) + 5, stretch=SinhStretch())
                    im1 = axs[0].imshow(i[0], cmap='hot', norm=norm_d)
                    im2 = axs[1].imshow(i[1], cmap='hot', norm=norm_s)
                axs[0].set_title('Instantaneous Data')
                axs[1].set_title('Integrated Data')
                ims.append([im1, im2])
                c1 = fig.colorbar(im1, ax=axs[0], location='bottom', shrink=0.7, ticks=[0, np.max(data)])
                c2 = fig.colorbar(im2, ax=axs[1], location='bottom', shrink=0.7, ticks=[0, np.max(stack)])

            else:
                im1.set_data(i[0])
                im2.set_data(i[1])
                ims.append([im1, im2])

            writer.grab_frame()


if __name__ == "__main__":

    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(description="Temporal drizzle animation utility")
    parser.add_argument('fits', type=str, help='The temporally drizzled FITS file to animate')
    parser.add_argument('out_file', type=str, help='The desired output file name')
    parser.add_argument('--target', type=str, dest='target', default=None, help='The target of the drizzled observation')
    parser.add_argument('-f', '--fps', type=int, dest='fps', default=None,
                        help='The desired number of frames per second in the animation')
    parser.add_argument('-s', '--size', type=int, dest='square_size', default=None,
                        help='Number of pixels around the host star')
    parser.add_argument('-e', '--exp_time', type=float, dest='exp_time', default=None,
                        help='The exposure time of each frame in the temporally drizzled dither.')
    parser.add_argument('--smooth', type=bool, default=True, dest='smooth',
                        help='Try to smooth out any spikes in count rate over the observation')
    parser.add_argument('--stretch', type=str, default='linear', dest='stretch', help='Colorbar stretch')
    parser.add_argument('--start', type=float, default=0, dest='start', help='Start time within the dither')
    parser.add_argument('--stop', type=float, default=None, dest='stop', help='Stop time within the dither')
    parser.add_argument('--duration', type=float, default=None, dest='duration', help='Duration of desired animation')
    parser.add_argument('--wvlbin', '-w', type=int, default=None, dest='wvl_bin', help='The wavelength bin to use (for '
                        'temporal movie). None defaults to all wavelength bins (no need to specify if nWvlBin=1). '
                        'Otherwise 0 is the lowest wvlBin, 1 the 2nd lowest, etc.')

    args = parser.parse_args()

    if args.target:
        log.info(f"Creating a movie from a drizzle of {args.target}")
        target = args.target
    else:
        log.warning(f"Unspecified target, using 'Target Not Specified' for title")
        target = "Target Not Specified"

    if args.fps:
        log.info(f"Movie will be made with {args.fps} fps")
        fps = args.fps
    else:
        log.warning(f"Using default rate of 10 fps (or you've specified that you want 10 fps, nice choice!)")
        fps = 10

    if args.square_size:
        log.info(f"The size of the image will be a {args.square_size}-by-{args.square_size} square (in pixel coords)")
    else:
        log.info(f"No image size specified, using the full array")

    if args.exp_time:
        log.info(f"The input drizzle has {args.exp_time} s frames.")
        exp_time = args.exp_time
    else:
        log.critical(f"No exposure time per frame was specified! Using the default of 1 s/frame. "
                     f"Make sure that this is correct! If not, the number of counts on the array will not be able to be"
                     f" calculated correctly")
        exp_time = 1

    if args.smooth:
        log.info(f"Using smoothing to try to remove any array 'flashes'")
    else:
        log.warning(f"No attempt will be made to correct for any array 'flashes'")

    log.info(f"The colorbar scale used will be '{args.stretch}' from astropy.visualizations module.")

    if args.stop and not args.duration:
        time_range = [args.start, args.stop]
    elif args.duration and not args.stop:
        time_range = [args.start, args.start + args.duration]
    elif args.stop and args.duration:
        time_range = [args.start, args.start + args.duration]
    else:
        log.warning(f"No stop time or duration specified, using the entire drizzle")
        time_range = [args.start, np.inf]
    log.info(f"Movie will run from {time_range[0]}s to {time_range[1]}s.")

    stacked_data = generate_stack_data(args.fits_file, args.exp_time, args.smooth,
                                       args.square_size, time_range, args.wvl_bin)
