import matplotlib.pyplot as plt
import matplotlib.animation as manimation
from astropy.wcs import WCS
from mkidpipeline.config import *
from mkidcore.corelog import getLogger
import mkidpipeline.photontable as photontable
from skimage.restoration import inpaint


def make_movie(out, usewcs=False, showaxes=True, inpainting=False, **kwargs):
    title = out.name
    outfile = out.output_file
    try:
        h5file = out.data.h5
    except AttributeError:
        try:
            idx = int(out.startt/out.data.inttime)
            h5file = out.data.obs[idx].h5
        except AttributeError:
            h5file = out.data.obs[0].h5
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
    if isinstance(out.data, MKIDDitherDescription):
        try:
            startt = out.startt % out.data.inttime
        except AttributeError:
            startt = None
        try:
            stopt = out.stopt % out.data.inttime
            if stopt == 0:
                stopt = out.data.inttime
        except AttributeError:
            stopt = None
    else:
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

    try:
        cps_cutoff = out.cps_cutoff
    except AttributeError:
        cps_cutoff = 100

    try:
        maskbadpix = out.maskbadpix
    except AttributeError:
        maskbadpix = False

    try:
        colormap = out.colormap
    except AttributeError:
        colormap = 'viridis'

    _make_movie(h5file, outfile, timestep, duration, title=title, usewcs=usewcs,
                startw=startw, stopw=stopw, startt=startt, stopt=stopt,
                fps=fps, showaxes=showaxes, inpainting=inpainting, cps_cutoff=cps_cutoff, 
                maskbadpix=maskbadpix, colormap=colormap)


def _make_movie(h5file, outfile, timestep, duration, title='', usewcs=False, startw=None, stopw=None, startt=None, stopt=None,
                fps=False, showaxes=False, inpainting=False, cps_cutoff=5000, maskbadpix=False, colormap='Blue', dpi=400):
    """returns the movie frames"""
    global _cache
    movietype = os.path.splitext(outfile)[1].lower()
    if movietype not in ('.mp4', '.gif'):
        raise ValueError('Only mp4 and gif movies are supported')

    try:
        cube, wcs, nfo = _cache
        if (h5file, timestep, startt, stopt, usewcs, startw, stopw) != nfo:
            raise ValueError
    except Exception:
        getLogger(__name__).info('Fetching temporal cube from {}'.format(h5file))
        of = photontable.Photontable(h5file)
        hdul = of.get_fits(start=startt, duration=stopt, bin_width=timestep, wave_start=startw,
                           wave_stop=stopw, spec_weight=True, noise_weight=True, cube_type='time', rate=False)
        wcs = WCS(hdul[0].header) if usewcs else None
        cube = hdul['SCIENCE'].data
        del of
        _cache = cube, wcs, (h5file, timestep, startt, stopt, usewcs, startw, stopw)
        getLogger(__name__).info('Retrieved a temporal cube of shape {}'.format(str(cube['cube'].shape)))

    frames, times = cube.T, hdul['BIN_EDGES'] / 1e6

    if not len(frames):
        getLogger(__name__).info('No frames in the specified timerange - check your stop and start times')
        raise ValueError

    if inpainting:
        getLogger(__name__).info('Inpainting requested - note this will significantly slow down movie creation')
        masked_array = np.ma.masked_where(frames[:, 10:140, 80:122] < cps_cutoff*timestep, frames[:, 10:140, 80:122])
        inpainted = np.zeros(frames.shape)
        for i, mask in enumerate(np.ma.getmaskarray(masked_array)):
            inpainted[i, 10:140, 80:122] = inpaint.inpaint_biharmonic(frames[i, 10:140, 80:122], mask,
                                                                      multichannel=False)
        getLogger(__name__).info('Completed inpainting process!')
        frames = inpainted

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
    if maskbadpix:
        of = photontable.Photontable(h5file)
        for i in range(frames.shape[0]):
            frames[i][of.bad_pixel_mask.T] = np.nan
    im = plt.imshow(frames[0], interpolation='none', origin='lower', vmin=0,
                    vmax=cps_cutoff*timestep, cmap=plt.get_cmap(colormap))
    im.cmap.set_bad('black')
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


def makeMovie(listOfFrameObj, frameTitles=None, outName='Test_movie',
              delay=0.1, listOfPixelsToMark=None, pixelMarkColor='red',
              **plotArrayKeys):
    """
    Neelay ca 2017

    Makes a movie out of a list of frame objects (2-D arrays). If you
    specify other list inputs, these all need to be the same length as
    the list of frame objects.

    listOfFrameObj is a list of 2d arrays of numbers

    frameTitles is a list of titles to put on the frames

    outName is the file name to write, .gif will be appended

    delay in seconds between frames

    listOfPixelsToMark is a list.  Each entry is itself a list of
    pixels to mark pixelMarkColor is the color to fill in the marked
    pixels

    """
    # Looks like theres some sort of bug when normMax != None.
    # Causes frame to pop up in a window as gif is made.
    if len(listOfFrameObj) == 1:
        raise ValueError("I cannot make movie out of a list of one object!")

    if frameTitles != None:
        assert len(frameTitles) == len(listOfFrameObj), "Number of Frame titles\
        must equal number of frames"

    if os.path.exists("./.tmp_movie"):
        os.system("rm -rf .tmp_movie")

    os.mkdir(".tmp_movie")
    iFrame = 0
    print('Making individual frames ...')

    for frame in listOfFrameObj:

        if frameTitles != None:
            plotTitle = frameTitles[iFrame]
        else:
            plotTitle = ''

        if listOfPixelsToMark != None:
            pixelsToMark = listOfPixelsToMark[iFrame]
        else:
            pixelsToMark = []
        pfn = '.tmp_movie/mov_' + repr(iFrame + 10000) + '.png'
        fp = plotArray(frame, showMe=False, plotFileName=pfn,
                       plotTitle=plotTitle, pixelsToMark=pixelsToMark,
                       pixelMarkColor=pixelMarkColor, **plotArrayKeys)
        iFrame += 1
        del fp

    os.chdir('.tmp_movie')

    if outName[-4:-1] + outName[-1] != '.gif':
        outName += '.gif'

    delay *= 100
    delay = int(delay)
    print('Making Movie ...')

    if '/' in outName:
        os.system('convert -delay %s -loop 0 mov_* %s' % (repr(delay), outName))
    else:
        os.system('convert -delay %s -loop 0 mov_* ../%s' % (repr(delay), outName))
    os.chdir("../")
    os.system("rm -rf .tmp_movie")
    print('done.')


# from mkidpipeline.hdf.photontable import Photontable
# h5file='/scratch/steiger/MEC/DeltaAnd/output/1567930101.h5'
# of = Photontable('/scratch/steiger/MEC/DeltaAnd/output/1567930101.h5')
# # 1567930101, 1567931601
# cube = of.getTemporalCube(None, 10, timeslice=.1, startw=None, stopw=None,
#                           spec_weight=True, noise_weight=True)
# # cube,times=cube['cube'],cube['timeslices'][:-1]
# import mkidpipeline.imaging.movies import _make_movie
if __name__ == '__main__':
    data = _make_movie('/scratch/steiger/MEC/DeltaAnd/output/1567930101.h5', 'dand.gif', .1, 5, title='dAnd', startt=None, stopt=10, usewcs=False,
                       startw=None, stopw=None,  showaxes=False)
