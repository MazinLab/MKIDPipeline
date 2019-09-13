import numpy as np
import matplotlib
import os
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

import mkidpipeline.hdf.photontable
from mkidcore.corelog import getLogger


def make_movie(out, usewcs=False, showaxes=True, **kwargs):
    title = out.name
    outfile = out.filename
    h5file = out.data.h5
    timestep = out.timestep
    movietype = os.path.splitext(outfile)[1].lower()
    if movietype not in ('.mp4', '.gif'):
        raise ValueError('Only mp4 and gif movies are supported')
    try:
        out.frameduration
    except AttributeError:
        try:
            out.movieduration
        except AttributeError:
            getLogger(__name__).error('Neither a frame nor movie duration specified')
            raise ValueError
    try:
        stopw = out.stopw
    except AttributeError:
        stopw = None
    try:
        startw = out.startw
    except AttributeError:
        startw = None
    try:
        startt = out.startt
    except AttributeError:
        startt = None
    try:
        stopt = out.stopt
    except AttributeError:
        stopt = None

    of = mkidpipeline.hdf.photontable.ObsFile(h5file)
    cube = of.getTemporalCube(startt=startt, stopt=stopt, timeslice=timestep, startw=startw, stopw=stopw,
                              applyWeight=True, applyTPFWeight=True)
    wcs = of.get_wcs(timestep=startt)
    del of

    frames, times = cube['cube'], cube['timeslices']

    try:
        fps = 1/out.frameduration
    except AttributeError:
        fps = frames.shape[0]/out.movieduration

    FFMpegWriter = manimation.writers['ffmpeg']
    comment = 't0={:.0f} dt={:.1f}s {:.0f} - {:.0f} nm'.format(times[0], timestep,
                                                               startw if startw is not None else 0,
                                                               stopw if stopw is not None else np.inf)
    metadata = dict(title=title, artist=__name__, genre='Astronomy', comment=comment)
    writer = FFMpegWriter(fps=fps, metadata=metadata, bitrate=-1, copyright='UCSB')

    fig = plt.figure()
    if usewcs:
        plt.subplot(projection=wcs)
    im = plt.imshow(frames[0], interpolation='none')

    if not showaxes:
        fig.patch.set_visible(False)
        plt.gca().axis('off')
    else:
        if usewcs:
            plt.label('RA')
            plt.ylabel('RA')
        else:
            plt.xlabel('Pixel')
            plt.ylabel('Pixel')
        plt.tight_layout()

    with writer.saving(fig, outfile, frames.shape[0]):
        for a in frames:
            im.set_array(a)
            writer.grab_frame()


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

