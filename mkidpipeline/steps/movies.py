import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from astropy.wcs import WCS
import astropy.units as u
import skimage.restoration as restoration
from astropy.io import fits
from astropy.visualization import ImageNormalize, AsinhStretch, LinearStretch, \
    LogStretch, PowerDistStretch, PowerStretch, SinhStretch, SqrtStretch, SquaredStretch

import mkidpipeline.config

from mkidcore.corelog import getLogger
import mkidpipeline.config
import mkidpipeline.photontable as photontable

STRETCHES = {'asinh': AsinhStretch, 'linear': LinearStretch, 'log': LogStretch, 'powerdist': PowerDistStretch,
             'sinh': SinhStretch, 'sqrt': SqrtStretch, 'squared': SquaredStretch, 'power': PowerStretch}


class StepConfig(mkidpipeline.config.BaseStepConfig):
    yaml_tag = u'!movies_cfg'
    REQUIRED_KEYS = (('type', 'simple', 'simple|upramp|both'),
                     ('colormap', 'viridis', 'Colormap to use'),
                     ('rate_cut', None, 'Count (rate) cutoff, None=none'),
                     ('axes', True, 'Show the axes'),
                     ('wcs', True, 'Use the WCS solution'),
                     ('dpi', None, 'Image writer DPI'),
                     ('mask_bad', True, 'Mask bad pixels'),
                     ('fps', 24, 'framerate'),
                     ('smoothing.n', None, 'Smooth over N frames'),
                     ('smoothing.sigma', None, '>n-sigma outliers will be smoothed'),
                     ('smoothing.power', None, 'Power for Savitgy-Golay filter'),
                     ('inpaint_below', 0, 'Counts below limit will be inpainted (0=off)'),
                     ('stretch.name', 'linear',
                      'linear | asinh | log | power[power=5] | powerdist | sinh | sqrt | squared'),
                     ('stretch.args', tuple(), 'see matplotlib docs'),
                     ('stretch.kwargs', dict(), 'see matplotlib docs'),
                     ('title', True, 'Display the title at the top of the animation'),
                     ('movie_format', 'gif', 'The format of the movie. Imagemagik if gif else ffmpeg'))

    def __init__(self, *args,**kwargs):
        super(StepConfig, self).__init__(*args,**kwargs)


def _fetch_data(file, timestep, start=0, stop=np.inf, min_wave=None, max_wave=None, use_weights=False, cps=True,
                exclude_flags=None):
    """
    cps = true each frame consists of the count rate in that timestep, if false each frame consist of the counts in
    that frame. cps is ignored and the unit from the file is used if a fits file is provided.
    """
    global _cache

    if file.endswith('fits'):
        hdul = fits.open(file)
    else:
        try:
            hdul, nfo = _cache
            if (file, timestep, start, stop, min_wave, max_wave, use_weights, cps, exclude_flags) != nfo:
                raise ValueError
        except (ValueError, TypeError, NameError):
            getLogger(__name__).info('Fetching temporal cube from {}'.format(file))
            of = photontable.Photontable(file)
            hdul = of.get_fits(start=start, duration=stop, bin_width=timestep, wave_start=min_wave,
                               wave_stop=max_wave, weight=use_weights, cube_type='time', rate=cps,
                               exclude_flags=exclude_flags)
            del of
            getLogger(__name__).info(f"Retrieved a temporal cube of shape {hdul['SCIENCE'].data.shape}")

        _cache = hdul, (file, timestep, start, stop, min_wave, max_wave, use_weights, cps, exclude_flags)

    # fetch what is needed from the fits
    try:
        cps = hdul['SCIENCE'].header['UNIT'] == 'photons/s'
        exp_time = hdul['CUBE_EDGES'].data.edges * u.Quantity(f"1 {hdul['CUBE_EDGES'].header['UNIT']}").to('s').value
        frames = hdul['SCIENCE'].data * (exp_time if cps else 1)
        time_mask = (exp_time >= start) & (exp_time <= stop)
        assert time_mask.size == frames.shape[0] + 1
    except KeyError:
        getLogger(__name__).error('FITS data in invalid format')
        raise IOError('Invalid data format')

    return frames[time_mask[1:] & time_mask[:-1]], exp_time[time_mask], hdul


def _process(frames, time, region=None, smooth_sigma=None, n_frames_smooth=6, smooth_power=3, inpaint_below=0):
    """ Apply smoothing, region selection, inpainting and compute the cumsum of the frames"""

    if not smooth_sigma:
        smooth_mask = np.ones(frames.shape[0], dtype=bool)
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

    # pull out the axed integration time
    tmp = np.diff(time)
    tmp[smooth_mask] = 0
    tmp = np.cumsum(tmp)
    itime = (time[1:] - time[0] - np.cumsum(tmp))[smooth_mask]  # bin integration time
    times = (time[1:] + time[:-1])[smooth_mask] / 2  # bin midpoint

    if inpaint_below > 0:
        getLogger(__name__).info('Inpainting requested')
        inpainted = np.array([restoration.inpaint.inpaint_biharmonic(frame, frame < inpaint_below, multichannel=False)
                              for frame in frames])
        getLogger(__name__).info('Completed inpainting!')
        frames = inpainted

    return frames, times, itime


def _save_frames(frames, movie_duration, units, suptitle='', movie_type='simple', outfile='file.mp4', wcs=None,
                 mask=None, showaxes=True, description='', colormap='viridis', interpolation='none', dpi=None,
                 bad_color='black', stretch='linear', format='gif'):
    """
    RA is assumed along x and Dec along y
    movie_type = simple|upramp|both
    units must have '/s' in it if data is in per second form
    """
    origin = 'lower'
    movie_type = movie_type.lower()

    fps = frames.shape[0] / movie_duration

    if mask is not None:
        frames[:, mask > 0] = np.nan

    # Create the writer
    writerkey = 'ffmpeg' if 'gif' not in format else 'imagemagick'
    metadata = dict(title=suptitle, artist=__name__, genre='Astronomy', comment=description)
    writer = manimation.writers[writerkey](fps=fps, metadata=metadata, bitrate=-1)

    if isinstance(stretch, str):
        stretch = STRETCHES[stretch]()

    # fetch the colormap
    try:
        cmap = plt.get_cmap(colormap)
    except Exception:
        getLogger(__name__).warning(f"Invalid colormap ({colormap}), using 'hot'.")
        cmap = plt.get_cmap('hot')

    fig, axs = plt.subplots(1, 2 if movie_type == 'both' else 1, constrained_layout=True,
                            subplot_kw={'projection': wcs[:, :, 0]}) #need to slice off the temporal wcs
    if suptitle:
        plt.suptitle(suptitle)

    if showaxes:
        if movie_type == 'both':
            for ax in axs:
                plt.sca(ax)
                plt.xlabel('RA' if wcs else 'pixel')
                plt.ylabel('Dec' if wcs else 'pixel')
        else:
            plt.sca(axs)
            plt.xlabel('RA' if wcs else 'pixel')
            plt.ylabel('Dec' if wcs else 'pixel')
    else:
        fig.patch.set_visible(False)
        for ax in axs:
            ax.axis('off')

    stack = np.cumsum(frames, axis=0)
    if '/s' in units:
        # When in counts per second we build a progressively more accurate average not a sum!
        stack /= np.arange(frames.shape[0]) + 1

    nframes = frames.shape[0]
    frame_max = np.nanmax(frames)
    stack_max = np.nanmax(stack)
    cbmaxs = (int(np.ceil(frame_max)), int(np.ceil(stack_max)))
    if movie_type == 'both':
        image_names = ('Frame', 'Integration')
        norms = (ImageNormalize(vmin=-10, vmax=frame_max + 5, stretch=stretch),
                 ImageNormalize(vmin=-10, vmax=stack_max + 5, stretch=stretch))
        frame_data = np.stack((frames, stack), axis=1)
    elif movie_type == 'upramp':
        image_names = ('Integration',)
        cbmaxs = cbmaxs[1:]
        norms = (ImageNormalize(vmin=-10, vmax=stack_max + 5, stretch=stretch),)
        frame_data = stack
    else:
        image_names = ('Frame',)
        cbmaxs = cbmaxs[:1]
        norms = (ImageNormalize(vmin=-10, vmax=frame_max + 5, stretch=stretch),)
        frame_data = frames

    with writer.saving(fig, outfile, dpi):#, frames[0].shape[1] * 2): #TODO
        img = []
        if movie_type == 'both':
            for ax, frame_im, name, cbmax, norm in zip(axs, frame_data[0], image_names, cbmaxs, norms):
                img.append(ax.imshow(frame_im, cmap=cmap, norm=norm, interpolation=interpolation, origin=origin))
                img[-1].cmap.set_bad(bad_color)
                ax.set_title(name)
                cbar = fig.colorbar(img[-1], ax=ax, location='bottom', shrink=0.9, pad=0.2, ticks=[0, cbmax])
                cbar.set_label(units)
                # ticks = cbar.get_ticks()
                # cbar.set_ticks(ticks)
                # cbar.set_ticklabels(list(map('{:.0f}'.format, ticks)))  #TODO
        else:
            for frame_im, name, cbmax, norm in zip(frame_data, image_names, cbmaxs, norms):
                img.append(axs.imshow(frame_im, cmap=cmap, norm=norm, interpolation=interpolation, origin=origin))
                img[-1].cmap.set_bad(bad_color)
                axs.set_title(name)
                for im in img:
                    cbar = fig.colorbar(im, ax=axs, location='bottom', shrink=0.9, pad=0.2, ticks=[0, cbmax])
                cbar.set_label(units)

        # plt.tight_layout(pad=1.0)
        writer.grab_frame()

        for i, frame_im in enumerate(frame_data[1:]):
            if not (i + 1 % 10):
                getLogger(__name__).debug(f"Frame {i + 1} of {nframes}")
            try:
                for im, data in zip(img, frame_im):
                    im.set_data(data)
            except TypeError:
                for im in zip(img):
                    im[0].set_data(frame_im)
            writer.grab_frame()


def fetch(out, **kwargs):
    """Make a movie of a single observation"""
    config = mkidpipeline.config.PipelineConfigFactory(step_defaults=dict(movies=StepConfig()), cfg=None, copy=True)

    ob = out.data
    if isinstance(ob, mkidpipeline.config.MKIDDitherDescription):
        ob = ob.obs_for_time(ob.obs[0].start + out.start_offset)

    file = ob.h5
    outcfg = out.output_settings_dict

    movietype = os.path.splitext(out.filename)[1].lower()
    if movietype not in ('.mp4', '.gif'):
        raise ValueError('Only mp4 and gif movies are supported')

    frames, times, hdul = _fetch_data(file, outcfg.time_bin_width, start=outcfg.start, stop=outcfg.start+outcfg.duration,
                                      min_wave=outcfg.wave_start, max_wave=outcfg.wave_stop, use_weights=outcfg.weight,
                                      cps=outcfg.rate, exclude_flags=outcfg.exclude_flags)

    suptitle = f"{out.name}: {np.diff(times[:2])[0]} s/frame, {hdul['SCIENCE'].header['EXPTIME']} s total"

    wavestr = ''
    if np.isinf(outcfg.wave_start) | np.isinf(outcfg.wave_stop):
        wavestr = f" {outcfg.wave_start} - {outcfg.max_wave:.0f} nm"

    description = f't0={times[0]:.0f} dt={outcfg.time_bin_width:.1f}s{wavestr}'

    if not frames.size:
        getLogger(__name__).warning(f'No frames in {outcfg.start}-{outcfg.stop} for {file}')
        return

    try:
        region = out.region
    except AttributeError:
        region=None

    frames, times, itimes = _process(frames, times, region=region, smooth_sigma=config.movies.smoothing.sigma,
                                     n_frames_smooth=config.movies.smoothing.n,
                                     smooth_power=config.movies.smoothing.power,
                                     inpaint_below=config.movies.inpaint_below)

    if not frames.size:
        getLogger(__name__).warning(f'No frames in {outcfg.start}-{outcfg.stop} for {file}')
        return
    try:
        stretch = STRETCHES[config.movies.stretch.name](*config.movies.stretch.args, **config.movies.stretch.kwargs)
    except Exception:
        getLogger(__name__).warning(f"Error parsing stretch args stretch: {config.movies.stretch.name}. "
                                    f"Defaulting to linear.")
        stretch = STRETCHES['linear']()

    _save_frames(frames, out.movie_runtime, hdul['SCIENCE'].header['UNIT'], suptitle=suptitle, stretch=stretch,
                 description=description, mask=hdul['BAD'].data if config.movies.mask_bad else None,
                 outfile=out.filename, movie_type=out.movie_type,
                 wcs=WCS(hdul['SCIENCE'].header) if config.movies.wcs else None,
                 colormap=config.movies.colormap, showaxes=config.movies.axes, dpi=config.movies.dpi,
                 bad_color='black', interpolation='none', format=out.movie_format)


def test_writers(out='garbage.gif'):
    frames = np.random.uniform(0, 100, size=(100, 140, 146))
    movietype = os.path.splitext(out)[1].lower()
    if movietype not in ('.mp4', '.gif'):
        raise ValueError('Only mp4 and gif movies are supported')
    writer = manimation.writers['ffmpeg' if 'mp4' in out else 'imagemagick'](fps=24, bitrate=-1)
    fig = plt.figure()
    im = plt.imshow(frames[0], interpolation='none')
    plt.tight_layout()
    with writer.saving(fig, out, frames.shape[0]):
        writer.grab_frame()
        for a in frames[1:]:
            im.set_array(a)
            writer.grab_frame()
