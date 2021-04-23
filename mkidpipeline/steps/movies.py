import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from astropy.wcs import WCS
import astropy.units as u
from skimage.restoration import inpaint

from mkidcore.corelog import getLogger
import mkidpipeline.config
import mkidpipeline.photontable as photontable


class StepConfig(mkidpipeline.config.BaseStepConfig):
    yaml_tag = u'!movies_cfg'
    REQUIRED_KEYS = (('inpaint', False, 'Inpaint?'),
                     ('colormap', 'viridis', 'Colormap to use'),
                     ('rate_cut', None, 'Count (rate) cutoff, None=none'),
                     ('axes', True, 'Show the axes'))


def make_movie(out):
    """Make a movie of a single observation"""
    config = mkidpipeline.config.PipelineConfigFactory(step_defaults=dict(movies=StepConfig()), cfg=None, copy=True)

    if isinstance(out.data, mkidpipeline.config.MKIDDitherDescription):
        ob = out.data.obs_for_time(out.startt)
    else:
        ob = out.obs[0]

    _make_movie(ob.h5, out.output_file, out.timestep, movie_duration=out.movie_length,
                title=out.name, usewcs=out.usewcs, startw=out.min_wave, stopw=out.max_wave, startt=out.startt,
                stopt=out.stopt, showaxes=config.movies.axes, inpainting=config.movies.inpaint,
                cps_cutoff=config.movies.cps_cutoff, maskbad=out.pixcal, colormap=config.movies.colormap,
                noise_weight=out.noise, spec_weight=out.photom, **out.extra)


def _make_movie(h5file, outfile, timestep, movie_duration=None, title='', usewcs=False, startw=None, stopw=None, startt=None,
                stopt=None, fps=15, showaxes=False, inpainting=False, cps_cutoff=5000, maskbad=False,
                colormap='Blue', dpi=400, noise_weight=False, spec_weight=True):
    """returns the movie frames"""
    global _cache
    movietype = os.path.splitext(outfile)[1].lower()
    if movietype not in ('.mp4', '.gif'):
        raise ValueError('Only mp4 and gif movies are supported')

    try:
        hdul, wcs, nfo = _cache
        if (h5file, timestep, startt, stopt, usewcs, startw, stopw, spec_weight, noise_weight) != nfo:
            raise ValueError
    except (TypeError, ValueError):
        getLogger(__name__).info('Fetching temporal cube from {}'.format(h5file))
        of = photontable.Photontable(h5file)
        hdul = of.get_fits(start=startt, duration=stopt, bin_width=timestep, wave_start=startw,
                           wave_stop=stopw, spec_weight=spec_weight, noise_weight=noise_weight,
                           cube_type='time', rate=False)
        del of
        _cache = hdul, (h5file, timestep, startt, stopt, usewcs, startw, stopw, spec_weight, noise_weight)
        getLogger(__name__).info(f"Retrieved a temporal cube of shape {hdul['SCIENCE'].data.shape}")

    frames, times = hdul['SCIENCE'].data
    times = hdul['BIN_EDGES'].data * u.Quantity(f"1 {hdul['BIN_EDGES'].header['UNIT']}").to('s').value
    wcs = WCS(hdul['SCIENCE'].header) if usewcs else None

    if not len(frames):
        getLogger(__name__).warning(f'No frames in {startt}-{stopt} s for {h5file}')
        return

    if inpainting:
        getLogger(__name__).info('Inpainting requested ')
        masks = frames < cps_cutoff*timestep #TODO is this the right inequality
        inpainted = np.zeros_like(frames)
        getLogger(__name__).warning('Inpainting presently inpaints both ')
        for i, (frame, mask) in enumerate(zip(frames, masks)):
            inpainted[i] = inpaint.inpaint_biharmonic(frame, mask, multichannel=False)
        getLogger(__name__).info('Completed inpainting!')
        frames = inpainted

    if movie_duration:
        fps = frames.shape[2]/movie_duration

    #Load the writer
    Writer = manimation.writers['ffmpeg'] if movietype is 'mp4' else manimation.writers['imagemagick']
    comment = 't0={:.0f} dt={:.1f}s {:.0f} - {:.0f} nm'.format(times[0], timestep,
                                                               startw if startw is not None else 0,
                                                               stopw if stopw is not None else np.inf)
    metadata = dict(title=title, artist=__name__, genre='Astronomy', comment=comment)
    writer = Writer(fps=fps, metadata=metadata, bitrate=-1)

    fig = plt.figure()
    if usewcs:
        plt.subplot(projection=wcs)
    if maskbad:
        exclude = hdul['BAD'].data
        frames[exclude[np.newaxis,:,:]]=np.nan
    im = plt.imshow(frames[0], interpolation='none', origin='lower', vmin=0, vmax=cps_cutoff*timestep,
                    cmap=plt.get_cmap(colormap))
    im.cmap.set_bad('black')
    cbar = plt.colorbar()
    ticks = cbar.get_ticks()
    cbar.set_label('photons/s')
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(list(map('{:.0f}'.format, ticks/timestep)))

    if not showaxes:
        fig.patch.set_visible(False)
        plt.gca().axis('off')
    else:
        if usewcs:
            plt.xlabel('RA')
            plt.ylabel('Dec')
        else:
            plt.xlabel('pixel')
            plt.ylabel('pixel')
        plt.tight_layout()

    with writer.saving(fig, outfile, dpi):
        for a in frames:
            im.set_array(a)
            writer.grab_frame()

    return frames


def test_writers(out='garbage.gif',showaxes=False, fps=5):
    import os
    frames = np.random.uniform(0,100,size=(100,140,146))

    movietype = os.path.splitext(out)[1].lower()
    if movietype not in ('.mp4', '.gif'):
        raise ValueError('Only mp4 and gif movies are supported')

    Writer = manimation.writers['ffmpeg'] if movietype is 'mp4' else manimation.writers['imagemagick']
    metadata = dict(title='test', artist=__name__, genre='Astronomy', comment='comment')
    writer = Writer(fps=fps, metadata=metadata, bitrate=-1)

    fig = plt.figure()
    im = plt.imshow(frames[0], interpolation='none')
    if not showaxes:
        fig.patch.set_visible(False)
        plt.gca().axis('off')

    plt.tight_layout()
    with writer.saving(fig, out, frames.shape[0]):
        for a in frames:
            im.set_array(a)
            writer.grab_frame()



#+++++++++


class StepConfig(mkidpipeline.config.BaseStepConfig):
    yaml_tag = u'!animation_cfg'
    REQUIRED_KEYS = (('plots', 'all', 'none|data|stack|all'),
                     ('fps', 24, 'framerate'),
                     ('stretch', 'linear', 'linear | asinh | log | power[power=5] | powerdist | sinh | sqrt | squared'),
                     ('title', True, 'Display the title at the top of the animation'))


def generate_stack_data(fits_file, exp_time, smooth_sigma=None, region=None, time_range, wvl_bin=None):
    hdul = fits.open(fits_file)

    cps = hdul['SCIENCE'].header['UNIT'] == 'photons/s'

    temporal = hdul['SCIENCE'].data * (exp_time if cps else 1)
    counts = hdul['SCIENCE'].data.sum(0)
    time = hdul['BIN_EDGES'].data * u.Quantity(f"1 {hdul['BIN_EDGES'].header['UNIT']}").to('s').value

    if time_range:
        time_mask = (time >= time_range[0]) & (time <= time_range[1])
        time = time[time_mask]
        counts = counts[time_mask]
        temporal = temporal[time_mask]

    if smooth_sigma:
        n_frames_smooth = 6
        smooth_power = 3
        from scipy.signal import savgol_filter
        curve = np.median(np.median(temporal, 1), 1)
        mask = np.abs(savgol_filter(curve, n_frames_smooth, smooth_power) - curve) < smooth_sigma * curve.std()

        temporal = temporal[mask]
        time = time[mask]
        counts = counts[mask]

    if region is not None:
        try:
            x0, y0, x1, y1 = region
            region = (slice(x0, x1), slice(y0, y1))
        except TypeError:
            ctr = temporal.shape[-2:] // 2
            hwid = region
            region = (slice(ctr[0] - hwid, ctr[0] + hwid), slice(ctr[1] - hwid, ctr[1] + hwid))
        except ValueError:
            region = (slice(None), slice(None))
        temporal = temporal[(slice(None),) + region]
        spectral = spectral[(slice(None),) + region]

    cumulative = False
    if cumulative:
        temporal_stack = np.cumsum(temporal, axis=0)
        spectral_stack = np.cumsum(spectral, axis=0)
        if cps:
            # Each frame in the spectral_stack is in CPS, not counts. To keep the scale consistent average by the
            # number of frames that went into that stacked frame
            temporal_stack /= (np.arange(temporal_stack.shape[0]) + 1)
            spectral_stack /= (np.arange(spectral_stack.shape[0]) + 1)


def animate(data, outfile=None, target='', stretch='linear', type='temporal', fps=10,
            plot_data=True, plot_stack=True, title=True):
    outfile = f'{target}.mp4' if outfile is None else outfile

    time_range = [data['lightcurve']['time'][0], data['lightcurve']['time'][-1] + data['exp_time']]
    exp_time = data['exp_time']

    FFMpegWriter = anim.writers['ffmpeg']
    writer = FFMpegWriter(fps, metadata=dict(title=f"Dither Movie: {outfile}", artist='Matplotlib'))

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

    if stretch not in VALID_STRETCHES:
        log.warning(f"Invalid stretch: {stretch}. Defaulting to linear.")
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
                    norm_d = ImageNormalize(vmin=-10, vmax=np.max(anim_data['data']) + 5,
                                            stretch=VALID_STRETCHES[stretch])
                    norm_s = ImageNormalize(vmin=-10, vmax=np.max(anim_data['stack']) + 5,
                                            stretch=VALID_STRETCHES[stretch])
                    im1 = axs[0].imshow(i[0], cmap='hot', norm=norm_d)
                    im2 = axs[1].imshow(i[1], cmap='hot', norm=norm_s)
                    axs[0].set_title('Instantaneous Data')
                    axs[1].set_title('Integrated Data')
                    ims.append([im1, im2])
                    c1 = fig.colorbar(im1, ax=axs[0], location='bottom', shrink=0.7,
                                      ticks=[0, int(np.max(anim_data['data']))])
                    c2 = fig.colorbar(im2, ax=axs[1], location='bottom', shrink=0.7,
                                      ticks=[0, int(np.max(anim_data['stack']))])
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


if config.animation.power:
    VALID_STRETCHES['power'] = PowerStretch(config.animation.power)

if not config.animation.target:
    log.warning('Target missing! Must be specified in the config for metadata.')

stacked_data = generate_stack_data(config.paths.fits, config.data.exp_time, config.data.smooth,
                                   config.data.square_size, (start, end), config.data.wvl_bin)

animate(stacked_data, outfile=config.paths.out, target=config.animation.target,
        stretch=config.animation.stretch, type=config.animation.type, fps=config.animation.fps,
        plot_data=config.animation.plot_data, plot_stack=config.animation.plot_stack,
        title=config.animation.show_title)

#+++++++++

# from mkidpipeline.hdf.photontable import Photontable
# h5file='/scratch/steiger/MEC/DeltaAnd/output/1567930101.h5'
# of = Photontable('/scratch/steiger/MEC/DeltaAnd/output/1567930101.h5')
# # 1567930101, 1567931601
# cube = of.getTemporalCube(None, 10, timeslice=.1, startw=None, stopw=None,
#                           spec_weight=True, noise_weight=True)
# # cube,times=cube['cube'],cube['timeslices'][:-1]
# import mkidpipeline.imaging.movies import _make_movie
if __name__ == '__main__':
    data = _make_movie('/scratch/steiger/MEC/DeltaAnd/output/1567930101.h5', 'dand.gif', .1, 5, title='dAnd',
                       startt=None, stopt=10, usewcs=False, startw=None, stopw=None,  showaxes=False)
