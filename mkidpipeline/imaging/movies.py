import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

from mkidcore.corelog import getLogger
import mkidpipeline.hdf.photontable
from skimage import data
from skimage.restoration import inpaint

def make_movie(out, usewcs=False, showaxes=True, inpainting=False, **kwargs):
    title = out.name
    outfile = out.output_file
    h5file = out.data.h5
    timestep = out.timestep
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

    try:
        fps = True
        duration = 1/out.frameduration
    except AttributeError:
        fps = False
        duration = 1/out.movieduration

    _make_movie(h5file, outfile, timestep, duration, title=title, usewcs=usewcs,
                startw=startw, stopw=stopw, startt=startt, stopt=stopt,
                fps=fps, showaxes=showaxes, inpainting=inpainting)


def _make_movie(h5file, outfile, timestep, duration, title='', usewcs=False, startw=None, stopw=None, startt=None, stopt=None,
                fps=False, showaxes=False, inpainting=False, cps_cutoff=50):
    """returns the movie frames"""
    global _cache
    movietype = os.path.splitext(outfile)[1].lower()
    if movietype not in ('.mp4', '.gif'):
        raise ValueError('Only mp4 and gif movies are supported')

    try:
        cube,wcs,nfo = _cache
        if (h5file, timestep, startt,stopt, usewcs,startw,stopw)!=nfo:
            raise ValueError
    except Exception:
        getLogger(__name__).info('Fetching temporal cube from {}'.format(h5file))
        of = mkidpipeline.hdf.photontable.ObsFile(h5file)
        cube = of.getTemporalCube(firstSec=startt, integrationTime=stopt, timeslice=timestep, startw=startw,
                                  stopw=stopw, applyWeight=True, applyTPFWeight=True)
        wcs = of.get_wcs(wcs_timestep=startt) if usewcs else None
        del of
        _cache = cube, wcs, (h5file, timestep, startt, stopt, usewcs, startw, stopw)
        getLogger(__name__).info('Retrieved a temporal cube of shape {}'.format(str(cube['cube'].shape)))
    if inpainting:
        getLogger(__name__).info('Inpainting requested - note this will significantly slow down movie creation')
        full_frames, times = cube['cube'].T, cube['timeslices']
        input_frames = full_frames[:, 10:140, 80:122]
        masked_array = np.ma.masked_where(input_frames < cps_cutoff*timestep, input_frames)
        count_masks = np.ma.getmaskarray(masked_array)
        masks = count_masks
        frames = np.zeros(full_frames.shape)
        for i, mask in enumerate(masks):
            frames[i, 10:140, 80:122] = inpaint.inpaint_biharmonic(input_frames[i], mask, multichannel=False)
        getLogger(__name__).info('Completed inpainting process!')
    else:
        frames, times = cube['cube'].T, cube['timeslices']
    if len(frames) == 0:
        getLogger(__name__).info('No frames in the specified timerange - check your stop and start times')
        raise ValueError

    if not fps:
        fps = frames.shape[2]/duration

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
    im = plt.imshow(frames[0], interpolation='none', origin='lower', vmin=frames.min(),
                    vmax=np.percentile(frames, 98), cmap=plt.get_cmap('Blues'))
    cbar = plt.colorbar()
    ticks = cbar.get_ticks()
    cbar.set_label('Photons/s')
    cbar.set_ticks(ticks)
    cbar.set_ticklabels(list(map('{:.0f}'.format, ticks/timestep)))

    if not showaxes:
        fig.patch.set_visible(False)
        plt.gca().axis('off')
    else:
        if usewcs:
            plt.xlabel('RA')
            plt.ylabel('RA')
        else:
            plt.xlabel('Pixel')
            plt.ylabel('Pixel')
        plt.tight_layout()

    with writer.saving(fig, outfile, frames.shape[2]):
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




# from mkidpipeline.hdf.photontable import ObsFile
# h5file='/scratch/steiger/MEC/DeltaAnd/output/1567930101.h5'
# of = ObsFile('/scratch/steiger/MEC/DeltaAnd/output/1567930101.h5')
# # 1567930101, 1567931601
# cube = of.getTemporalCube(None, 10, timeslice=.1, startw=None, stopw=None,
#                           applyWeight=True, applyTPFWeight=True, flagToUse=0xffffff)
# # cube,times=cube['cube'],cube['timeslices'][:-1]
# import mkidpipeline.imaging.movies import _make_movie
if __name__ == '__main__':
    data = _make_movie('/scratch/steiger/MEC/DeltaAnd/output/1567930101.h5', 'dand.gif', .1, 5, title='dAnd', startt=None, stopt=10, usewcs=False,
                       startw=None, stopw=None,  showaxes=False)