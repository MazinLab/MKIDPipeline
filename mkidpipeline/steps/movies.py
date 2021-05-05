import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from astropy.wcs import WCS
import astropy.units as u
from skimage.restoration import inpaint
from astropy.io import fits

from mkidcore.corelog import getLogger
import mkidpipeline.config
import mkidpipeline.photontable as photontable


class StepConfig(mkidpipeline.config.BaseStepConfig):
    yaml_tag = u'!movies_cfg'
    REQUIRED_KEYS = (('inpaint', False, 'Inpaint?'),
                     ('colormap', 'viridis', 'Colormap to use'),
                     ('rate_cut', None, 'Count (rate) cutoff, None=none'),
                     ('axes', True, 'Show the axes'))


class StepConfig(mkidpipeline.config.BaseStepConfig):
    yaml_tag = u'!animation_cfg'
    REQUIRED_KEYS = (('plots', 'all', 'none|data|stack|all'),
                     ('fps', 24, 'framerate'),
                     ('stretch', 'linear', 'linear | asinh | log | power[power=5] | powerdist | sinh | sqrt | squared'),
                     ('title', True, 'Display the title at the top of the animation'))



def _fetch_data(file, startt,stopt):
    global _cache
    try:
        hdul, wcs, nfo = _cache
        if (file, timestep, startt, stopt, usewcs, startw, stopw, spec_weight, noise_weight) != nfo:
            raise ValueError
    except ValueError:
        if file.endswith('fits'):
            hdul = fits.open(file)
        else:
            getLogger(__name__).info('Fetching temporal cube from {}'.format(h5file))
            of = photontable.Photontable(file)
            hdul = of.get_fits(start=startt, duration=stopt, bin_width=timestep, wave_start=min_wave,
                               wave_stop=max_wave, spec_weight=use_weight, noise_weight=use_weight,
                               cube_type='time', rate=cps)
            del of
            getLogger(__name__).info(f"Retrieved a temporal cube of shape {hdul['SCIENCE'].data.shape}")

        _cache = hdul, file, (timestep, startt, stopt, usewcs, startw, stopw, spec_weight, noise_weight)

    # fetch what is needed from the fits
    try:
        wcs = WCS(hdul['SCIENCE'].header)
        cps = hdul['SCIENCE'].header['UNIT'] == 'photons/s'
        exp_time = hdul['BIN_EDGES'].data * u.Quantity(f"1 {hdul['BIN_EDGES'].header['UNIT']}").to('s').value
        frames = hdul['SCIENCE'].data * (exp_time if cps else 1)
        time = exp_time
        time_mask = (time >= startt) & (time <= stopt)
        frames = frames[time_mask]
        time = time[time_mask]
    except KeyError:
        getLogger(__name__).error('FITS data in invalid format')

    return frames, time, wcs


def process(frames, time, cps=True, region=None, smooth_sigma=None, n_frames_smooth=6, smooth_power=3,
            cumulative=False, inpaint=False, inpaint_limit=np.inf):

    if not smooth_sigma:
        smooth_mask = np.ones_like(time,dtype=True)
    else:
        from scipy.signal import savgol_filter
        curve = np.median(np.median(frames, axis=1), axis=1)
        smooth_mask = np.abs(savgol_filter(curve, n_frames_smooth, smooth_power) - curve) < smooth_sigma * curve.std()

    if region is None:
        region = (slice(None), slice(None))
    else:
        try:
            x0, y0, x1, y1 = region
            region = (slice(x0, x1), slice(y0, y1))
        except TypeError:
            ctr = frames.shape[-2:] // 2
            hwid = region
            region = (slice(ctr[0] - hwid, ctr[0] + hwid), slice(ctr[1] - hwid, ctr[1] + hwid))
        except ValueError:
            region = (slice(None), slice(None))

    region_slice = (slice(None),) + region

    frames = frames[smooth_mask][region_slice]
    time = time[smooth_mask]

    if cumulative:
        frames = np.cumsum(frames, axis=0)
        if cps:
            # Each frame in the spectral_stack is in CPS, not counts. To keep the scale consistent average by the
            # number of frames that went into that stacked frame
            frames /= np.arange(frames.shape[0]) + 1

    if inpaint:
        getLogger(__name__).info('Inpainting requested')
        inpainted = np.array([inpaint.inpaint_biharmonic(frame, frame < inpaint_limit, multichannel=False)
                              for frame in frames])
        getLogger(__name__).info('Completed inpainting!')
        frames = inpainted

    return frames, time


def _save_frames(frames, times, title='', outfile='file.mp4', wcs=None, mask=None, showaxes=True, description='',
                colormap='viridis', movie_duration=None, **kwargs):



    stretch='linear'
    wcs=None
    target='foo'
    exp_time=0
    stop=0
    start=0
    bad_color = 'black'
    interpolation='none'
    rate = True
    dpi=None
    stack=data
    side_by_side = True
    timestep = 1.0
    origin = 'lower'
    plot_integrated = True
    units = 'photons/s' if rate else 'photons'

    if movie_duration:
        fps = frames.shape[0]/movie_duration

    if mask:
        frames[mask[np.newaxis, :, :]] = np.nan


    # Create the writer
    Writer = manimation.writers['ffmpeg'] if 'mp4' in outfile else manimation.writers['imagemagick']
    metadata = dict(title=title, artist=__name__, genre='Astronomy', comment=description)
    writer = Writer(fps=fps, metadata=metadata, bitrate=-1)

    try:
        stretch = STRETCHES[stretch]
    except KeyError:
        getLogger(__name__).warning(f"Invalid stretch: {stretch}. Defaulting to linear.")
        stretch = STRETCHES['linear']

    norm_d = ImageNormalize(vmin=-10, vmax=data.max() + 5, stretch=stretch)
    norm_s = ImageNormalize(vmin=-10, vmax=stack.max() + 5, stretch=stretch)

    fig, axs = plt.subplots(1, 2 if side_by_side else 1, constrained_layout=True, projection=wcs)
    if title:
        plt.suptitle(f"{target}: {exp_time} s/frame @ {fps:.0f} fps, {stop-start} s total")

    if showaxes:
        for ax in axs:
            plt.sca(ax)
            plt.xlabel('RA' if wcs else 'pixel')
            plt.ylabel('Dec' if wcs else 'pixel')
    else:
        fig.patch.set_visible(False)
        for ax in axs:
            ax.axis('off')

    #fetch the colormap
    try:
        cmap = plt.get_cmap(colormap)
    except Exception:
        getLogger(__name__).warning(f"Invalid colormap ({colormap}), using 'hot'.")
        cmap = plt.get_cmap('hot')


    with writer.saving(fig, outfile, data[0].shape[1] * 2, dpi=dpi):
        if side_by_side:

            for count, (data_i, stack_i)  in enumerate(zip(data, stack)):
                if not (count % 10):
                    getLogger(__name__).debug(f"Frame {count} of {len(data)}")
                if count == 0:
                    im1 = axs[0].imshow(data_i, cmap=cmap, norm=norm_d, interpolation=interpolation, origin=origin)
                    im1.cmap.set_bad(bad_color)
                    axs[0].set_title('Frame')
                    cbar = fig.colorbar(im1, ax=axs[0], location='bottom', shrink=0.7, ticks=[0, int(data.max())])
                    cbar.set_label(units)
                    ticks = cbar.get_ticks()
                    cbar.set_ticks(ticks)
                    cbar.set_ticklabels(list(map('{:.0f}'.format, ticks / timestep)))

                    im2 = axs[1].imshow(stack_i, cmap=cmap, norm=norm_s, interpolation=interpolation, origin=origin)
                    im2.cmap.set_bad(bad_color)
                    axs[1].set_title('Integration')
                    cbar = fig.colorbar(im2, ax=axs[1], location='bottom', shrink=0.7, ticks=[0, int(stack.max())])
                    cbar.set_label(units)
                    ticks = cbar.get_ticks()
                    cbar.set_ticks(ticks)
                    cbar.set_ticklabels(list(map('{:.0f}'.format, ticks / timestep)))

                else:
                    im1.set_data(data_i)
                    im2.set_data(stack_i)
                writer.grab_frame()

        else:
            for count, data_i in enumerate(data):
                if not (count % 10):
                    getLogger(__name__).debug(f"Frame {count} of {len(data)}")
                if count == 0:
                    im1 = axs.imshow(data_i, cmap=cmap, norm=norm_s if plot_integrated else norm_d,
                                     interpolation=interpolation, origin=origin)
                    im1.cmap.set_bad(bad_color)
                    axs.set_title('Integration' if plot_integrated else 'Frame')
                    cbar = fig.colorbar(im1, ax=axs, location='bottom', shrink=0.7, ticks=[0, int(data.max())])
                    cbar.set_label(units)
                    ticks = cbar.get_ticks()
                    cbar.set_ticks(ticks)
                    cbar.set_ticklabels(list(map('{:.0f}'.format, ticks / timestep)))
                else:
                    im1.set_data(data_i)
                writer.grab_frame()





STRETCHES = {}


def fetch(out, **kwargs):
    """Make a movie of a single observation"""
    config = mkidpipeline.config.PipelineConfigFactory(step_defaults=dict(movies=StepConfig()), cfg=None, copy=True)

    STRETCHES['power'] = PowerStretch(config.animation.power)

    ob = out.data
    if isinstance(ob, mkidpipeline.config.MKIDDitherDescription):
        ob = ob.obs_for_time(ob.start + out.start_offset)

    file=ob.h5
    outfile=out.filename
    timestep=out.timestep
    movie_duration=out.movie_length

                # title=out.name, usewcs=out.usewcs, startw=out.min_wave, stopw=out.max_wave, startt=out.start_offset,
                # stopt=out.duration, showaxes=config.movies.axes, inpainting=config.movies.inpaint,
                # cps_cutoff=config.movies.cps_cutoff, colormap=config.movies.colormap,
                # use_weight=out.use_weights, maskbad=out.exclude_flags


    global _cache

    movietype = os.path.splitext(outfile)[1].lower()
    if movietype not in ('.mp4', '.gif'):
        raise ValueError('Only mp4 and gif movies are supported')

    frames, times, wcs, hdul = _fetch_data(file, **kwargs)


    if not len(frames):
        getLogger(__name__).warning(f'No frames in {startt}-{stopt} s for {h5file}')
        return

    frames, times = _process(frames, times, **kwargs)

    if not len(frames):
        getLogger(__name__).warning(f'No frames in {startt}-{stopt} s after processing for {h5file}')
        return

    description = 't0={:.0f} dt={:.1f}s {:.0f} - {:.0f} nm'.format(times[0], timestep,
                                                               startw if startw is not None else 0,
                                                               stopw if stopw is not None else np.inf)
    _save_frames(frames, times, wcs, mask=hdul['BAD'].data if maskbad else None, description=description, **kwargs)






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



if __name__ == '__main__':
    data = _make_movie('/scratch/steiger/MEC/DeltaAnd/output/1567930101.h5', 'dand.gif', .1, 5, title='dAnd',
                       startt=None, stopt=10, usewcs=False, startw=None, stopw=None,  showaxes=False)
