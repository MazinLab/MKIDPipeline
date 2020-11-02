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


def moviefy_frames(fits_file, outfile, target, fps=10, square_size=None, exp_time=None,
                   time_range=None, smooth=True, stretch='linear'):

    # Set up writer to actually make the movie
    FFMpegWriter = anim.writers['ffmpeg']
    metadata = dict(title=f"Dither Movie: {outfile}", artist='Matplotlib', fits=fits_file, target=target, fps=fps,
                    exp_time=exp_time, smoothed=smooth, colorbar_stretch=stretch, time_range_in_drizzle=time_range)
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    # Read and format the data. At this point the only available option of data to look at is count rates. With the
    # exposure time given, convert those count rates to counts (for integrating the image. It doesn't make sense to add
    # count rates, unless we play with some averaging, but that is a different visualization)
    fit = fits.open(fits_file)
    data = fit[1].data[:, 0, :, :]  # This is in cts/sec
    data = data * exp_time
    counts = np.array([np.sum(i) for i in data])


    # Create an appropriate time range. This began as being useful for debugging, but can also be used to make a
    # lightcurve or timeseries data.
    # TODO: Add a lightcurve option.
    # times = np.arange(0, exp_time * len(data) - exp_time, exp_time)
    times = np.arange(0, exp_time * len(data), exp_time)
    if time_range:
        time_mask = (times >= time_range[0]) & (times <= time_range[1])

        times = times[time_mask]
        data = data[time_mask]
        counts = counts[time_mask]

    # By default, the entire array is used. If desired, we can 'zoom in' to a smaller square around the center.
    if square_size:
        max_size = np.min(data[0].shape)
        if square_size <= max_size:
            half_size = np.ceil(square_size / 2)
            ctr = [np.ceil(data.shape[1]/2), np.ceil(data.shape[2]/2)]
            data = data[:, int(ctr[0]-half_size):int(ctr[0]+half_size), int(ctr[1]-half_size):int(ctr[1]+half_size)]

    # Empirically, we've seen that there's some artifact from the drizzle code that results in flashes from the array.
    # What follows is an attempt to remedy that and remove those flashes.
    if smooth:
        data_fit = np.polyfit(times, counts, 8)
        predicted_data = np.poly1d(data_fit)

        residuals = counts - predicted_data(times)
        std = np.std(residuals)
        smooth_mask = (abs(residuals) <= 1.5 * std)

        data = data[smooth_mask]
        times = times[smooth_mask]

    # Create a set of data where each successive frame is a sum of the data before it and the current frame. This is for
    # animating the stacking up of each successive image
    stack = []
    stack.append(data[0])
    for i in data[1:]:
        stack.append(stack[-1] + i)

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
    parser.add_argument('fits', type=str, dest='fits_file', help='The temporally drizzled FITS file to animate')
    parser.add_argument('outfile', type=str, dest='out_file', help='The desired output file name')
    parser.add_argument('--target', type=str, dest='target', help='The target of the drizzled observation')
    parser.add_argument('-f', '--fps', type=int, dest='fps', default=None,
                        help='The desired number of frames per second in the animation')
    parser.add_argument('-s', '--size', type=int, dest='square_size', default=None,
                        help='Number of pixels around the host star')
    parser.add_argument('-e', '--exposure', type=float, dest='exp_time', default=None,
                        help='The exposure time of each frame in the temporally drizzled dither.')
    parser.add_argument('--smooth', type=bool, default=True, dest='smooth',
                        help='Try to smooth out any spikes in count rate over the observation')
    parser.add_argument('--stretch', type=str, default='linear', dest='stretch', help='Colorbar stretch')
    parser.add_argument('--start', type=float, default=0, dest='tStart', help='Start time within the dither')
    parser.add_argument('--stop', type=float, default=None, dest='tStop', help='Stop time within the dither')
    parser.add_argument('--duration', type=float, default=None, dest='duration', help='Duration of desired animation')

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

    moviefy_frames(fits_file=args.fits, outfile=args.outfile, target=target, fps=10, square_size=args.square_size,
                   exp_time=exp_time, time_range=time_range, smooth=args.smooth, stretch=args.stretch)
