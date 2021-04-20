import numpy as np
import numpy.ma as ma
import matplotlib
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib as mpl
"""
Plotting tools for quick array viewing
Adapted from tutorials from python4astronomers.
Credit to authors: Tom Aldcroft, Tom Robitaille, Brian Refsdal, Gus Muench 
and Smithsonian Astrophysical Observatory.
"""

class ElasticColorbar(object):
    def __init__(self, cbar, mappable):
        self.cbar = cbar
        self.mappable = mappable
        self.press = None
        self.cycle = sorted([i for i in dir(plt.cm) if hasattr(getattr(plt.cm, i), 'N')])
        self.index = self.cycle.index(cbar.get_cmap().name)

    def connect(self):
        """Set up a connection between mouse and colorbar"""
        self.cidpress = self.cbar.patch.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.cbar.patch.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.cbar.patch.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
        self.keypress = self.cbar.patch.figure.canvas.mpl_connect(
            'key_press_event', self.key_press)

    def on_press(self, event):
        """Get ready to respond to a mouse click if mouse is over the colorbar"""
        if event.inaxes != self.cbar.ax: return
        self.press = event.x, event.y

    def key_press(self, event):
        if event.key == 'down':
            self.index += 1
        elif event.key == 'up':
            self.index -= 1
        if self.index < 0:
            self.index = len(self.cycle)
        elif self.index >= len(self.cycle):
            self.index = 0
        cmap = self.cycle[self.index]
        self.cbar.set_cmap(cmap)
        self.cbar.draw_all()
        self.mappable.set_cmap(cmap)
        self.mappable.get_axes().set_title(cmap)
        self.cbar.patch.figure.canvas.draw()

    def on_motion(self, event):
        """Rescale the colorbar if the mouse is dragged over the colorbar"""
        if self.press is None: return
        if event.inaxes != self.cbar.ax: return
        xprev, yprev = self.press
        dx = event.x - xprev
        dy = event.y - yprev
        self.press = event.x, event.y
        scale = self.cbar.norm.vmax - self.cbar.norm.vmin
        perc = 0.03
        if event.button == 1:
            self.cbar.norm.vmin -= (perc * scale) * np.sign(dy)
            self.cbar.norm.vmax -= (perc * scale) * np.sign(dy)
        elif event.button == 3:
            self.cbar.norm.vmin -= (perc * scale) * np.sign(dy)
            self.cbar.norm.vmax += (perc * scale) * np.sign(dy)
        self.cbar.draw_all()
        self.mappable.set_norm(self.cbar.norm)
        self.cbar.patch.figure.canvas.draw()

    def on_release(self, event):
        """When mouse is released, reset the colorbar to the new scale"""
        self.press = None
        self.mappable.set_norm(self.cbar.norm)
        self.cbar.patch.figure.canvas.draw()

    def disconnect(self):
        """Disconnect the mouse from the colorbar"""
        self.cbar.patch.figure.canvas.mpl_disconnect(self.cidpress)
        self.cbar.patch.figure.canvas.mpl_disconnect(self.cidrelease)
        self.cbar.patch.figure.canvas.mpl_disconnect(self.cidmotion)


class AstNormalize(Normalize):
    """
    A Normalize class for imshow that allows different stretching functions
    for astronomical images.
    """

    def __init__(self, stretch='linear', exponent=5, vmid=None, vmin=None,
                 vmax=None, clip=False):
        """
        Initalize an APLpyNormalize instance.
        :param vmin:           Float.   Minimum pixel value to use for the scaling.
        :param vmax:           Float.   Maximum pixel value to use for the scaling.
        :param stretch:        String.  ('linear', 'log', 'sqrt', 'arcsinh', 'power')  The stretch function to use (default is 'linear').
        :param vmid:           Float.   Mid-pixel value used for the log and arcsinh stretches. If set to None, a default value is picked.
        :param exponent:       Float.   If self.stretch is set to 'power', this is the exponent to use.
        :param clip:           Bool.    If clip is True and the given value falls outside the range, the returned value will be 0 or 1, whichever is closer.
        """
        self.stretch = stretch
        self.exponent = exponent

        if vmax < vmin:
            raise Exception("vmax should be larger than vmin")

        # Call original initalization routine
        Normalize.__init__(self, vmin=vmin, vmax=vmax, clip=clip)

        if stretch == 'power' and self.exponent == 'None':
            raise Exception("For stretch=='power', an exponent should be specified")

        if vmid == 'None':
            if stretch == 'log':
                if vmin > 0:
                    self.midpoint = vmax / vmin
                else:
                    raise Exception("When using a log stretch, if vmin < 0, then vmid has to be specified")
            elif stretch == 'arcsinh':
                self.midpoint = -1. / 30.
            else:
                self.midpoint = None
        else:
            if stretch == 'log':
                if vmin < vmid:
                    raise Exception("When using a log stretch, vmin should be larger than vmid")
                self.midpoint = (vmax - vmid) / (vmin - vmid)
            elif stretch == 'arcsinh':
                self.midpoint = (vmid - vmin) / (vmax - vmin)
            else:
                self.midpoint = None

    def __call__(self, value, clip=None):

        method = self.stretch
        exponent = self.exponent
        midpoint = self.midpoint

        if clip is None:
            clip = self.clip

        if cbook.iterable(value):
            vtype = 'array'
            val = ma.asarray(value).astype(np.float)
        else:
            vtype = 'scalar'
            val = ma.array([value]).astype(np.float)

        self.autoscale_None(val)
        vmin, vmax = self.vmin, self.vmax
        if vmin > vmax:
            raise ValueError("minvalue must be less than or equal to maxvalue")
        elif vmin == vmax:
            return 0.0 * val
        else:
            if clip:
                mask = ma.getmask(val)
                val = ma.array(np.clip(val.filled(vmax), vmin, vmax),
                               mask=mask)
            result = (val - vmin) * (1.0 / (vmax - vmin))

            negative = result < 0.
            if self.stretch == 'linear':
                pass
            elif self.stretch == 'log':
                result = ma.log10(result * (self.midpoint - 1.) + 1.) \
                         / ma.log10(self.midpoint)
            elif self.stretch == 'sqrt':
                result = ma.sqrt(result)
            elif self.stretch == 'arcsinh':
                result = ma.arcsinh(result / self.midpoint) \
                         / ma.arcsinh(1. / self.midpoint)
            elif self.stretch == 'power':
                result = ma.power(result, exponent)
            else:
                raise Exception("Unknown stretch in APLpyNormalize: %s" %
                                self.stretch)
            result[negative] = -np.inf

        if vtype == 'scalar':
            result = result[0]
        return result

    def inverse(self, value):

        if not self.scaled():
            raise ValueError("Not invertible until scaled")

        vmin, vmax = self.vmin, self.vmax

        if cbook.iterable(value):
            val = ma.asarray(value)
        else:
            val = value

        if self.stretch == 'linear':
            pass
        elif self.stretch == 'log':
            val = (ma.power(10., val * ma.log10(self.midpoint)) - 1.) / (self.midpoint - 1.)
        elif self.stretch == 'sqrt':
            val = val * val
        elif self.stretch == 'arcsinh':
            val = self.midpoint * \
                  ma.sinh(val * ma.arcsinh(1. / self.midpoint))
        elif self.stretch == 'power':
            val = ma.power(val, (1. / self.exponent))
        else:
            raise Exception("Unknown stretch in APLpyNormalize: %s" %
                            self.stretch)
        return vmin + val * (vmax - vmin)


def plot_array(image, title='', xlabel='', ylabel='', cbar_stretch='linear', vmid=None, norm_nsigma=3,
               show_colorbar=True, **kwargs):
    """The main plotting function"""
    if not 'vmax' in kwargs:
        good_image = image[np.isfinite(image)]
        kwargs['vmax'] = np.mean(good_image) + norm_nsigma * np.std(good_image)
    if not 'vmin' in kwargs:
        good_image = image[np.isfinite(image)]
        kwargs['vmin'] = good_image.min()
    if not 'cmap' in kwargs:
        default_cmap = matplotlib.cm.magma
        default_cmap.set_bad('0.15')
        kwargs['cmap'] = default_cmap
    if not 'origin' in kwargs:
        kwargs['origin'] = 'lower'

    img = plt.imshow(image, **kwargs)
    if show_colorbar:
        cbar = plt.colorbar(format='%05.2f')
        cbar.set_norm(AstNormalize(vmin=kwargs['vmin'], vmax=kwargs['vmax'], vmid=vmid, stretch=cbar_stretch))
        cbar = ElasticColorbar(cbar, img)
        cbar.connect()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def plotArray(array, colormap=mpl.cm.gnuplot2,
              norm_min=None, norm_max=None, showMe=True,
              cbar=False, cbarticks=None, cbarlabels=None,
              plotFileName='arrayPlot.png',
              plotTitle='', sigma=None,
              pixelsToMark=[], pixelMarkColor='red',
              fignum=1, pclip=None):
    """
    Plots the 2D array to screen or if showMe is set to False, to
    file.  If norm_min and norm_max are None, the norm is just set to
    the full range of the array.

    array is the array to plot

    colormap translates from a number in the range [0,1] to an rgb color,
    an existing matplotlib.cm value, or create your own

    norm_min minimum used for normalizing color values

    norm_max maximum used for normalizing color values

    showMe=True to show interactively; false makes a plot

    cbar to specify whether to add a colorbar

    cbarticks to specify whether to add ticks to the colorbar

    cbarlabels lables to put on the colorbar

    plotFileName where to write the file

    plotTitle put on the top of the plot

    sigma calculate norm_min and norm_max as this number of sigmas away
    from the mean of positive values

    pixelsToMark a list of pixels to mark in this image

    pixelMarkColor is the color to fill in marked pixels

    fignum - to specify which window the figure should be plotted in.
             Default is 1. If None, automatically selects a new figure number.
            Added 2013/7/19 2013, JvE

    pclip - set to percentile level (in percent) for setting the upper and
            lower colour stretch limits (overrides sigma).

    """
    if sigma != None:
        # Chris S. does not know what accumulatePositive is supposed to do
        # so he changed the next two lines.
        # mean_val = np.mean(accumulatePositive(array))
        # std_val = np.std(accumulatePositive(array))
        mean_val = np.nanmean(array)
        std_val = np.nanstd(array)
        norm_min = mean_val - sigma * std_val
        norm_max = mean_val + sigma * std_val
    if pclip != None:
        norm_min = np.percentile(array[np.isfinite(array)], pclip)
        norm_max = np.percentile(array[np.isfinite(array)], 100. - pclip)
    if norm_min == None:
        norm_min = array.min()
    if norm_max == None:
        norm_max = array.max()
    norm = mpl.colors.Normalize(vmin=norm_min, vmax=norm_max)

    figWidthPt = 550.0
    inchesPerPt = 1.0 / 72.27  # Convert pt to inch
    figWidth = figWidthPt * inchesPerPt  # width in inches
    figHeight = figWidth * 1.0  # height in inches
    figSize = [figWidth, figHeight]
    params = {'backend': 'ps',
              'axes.labelsize': 10,
              'axes.titlesize': 12,
              'text.fontsize': 10,
              'legend.fontsize': 10,
              'xtick.labelsize': 10,
              'ytick.labelsize': 10,
              'figure.figsize': figSize}

    fig = plt.figure(fignum)  ##JvE - Changed fignum=1 to allow caller parameter
    plt.clf()
    plt.rcParams.update(params)
    plt.matshow(array, cmap=colormap, origin='lower', norm=norm, fignum=False)

    for ptm in pixelsToMark:
        box = mpl.patches.Rectangle((ptm[0] - 0.5, ptm[1] - 0.5), \
                                    1, 1, color=pixelMarkColor)
        # box = mpl.patches.Rectangle((1.5,2.5),1,1,color=pixelMarkColor)
        fig.axes[0].add_patch(box)

    if cbar:
        if cbarticks == None:
            cbar = plt.colorbar(shrink=0.8)
        else:
            cbar = plt.colorbar(ticks=cbarticks, shrink=0.8)
        if cbarlabels != None:
            cbar.ax.set_yticklabels(cbarlabels)

    plt.ylabel('Row Number')
    plt.xlabel('Column Number')
    plt.title(plotTitle)

    if showMe == False:
        plt.savefig(plotFileName)
    else:
        plt.show()


def ds9Array(array, colormap='B', norm_min=None, norm_max=None, sigma=None, scale=None, frame=None):
    """
    Display a 2D array as an image in DS9 if available. Similar to 'plotArray()'

    array is the array to plot

    colormap - string, takes any value in the DS9 'color' menu.

    norm_min minimum used for normalizing color values

    norm_max maximum used for normalizing color values

    sigma calculate norm_min and norm_max as this number of sigmas away
    from the mean of positive values

    scale - string, can take any value allowed by ds9 xpa interface.
    Allowed values include:
        linear|log|pow|sqrt|squared|asinh|sinh|histequ
        mode minmax|<value>|zscale|zmax
        limits <minvalue> <maxvalue>
    e.g.: scale linear
        scale log 100
        scale datasec yes
        scale histequ
        scale limits 1 100
        scale mode zscale
        scale mode 99.5
        ...etc.
    For more info see:
        http://hea-www.harvard.edu/saord/ds9/ref/xpa.html#scale

    ## Not yet implemented: pixelsToMark a list of pixels to mark in this image

    ## Not yet implemented: pixelMarkColor is the color to fill in marked pixels

    frame - to specify which DS9 frame number the array should be displayed in.
             Default is None.

    """
    if sigma != None:
        # Chris S. does not know what accumulatePositive is supposed to do
        # so he changed the next two lines.
        # mean_val = np.mean(accumulatePositive(array))
        # std_val = np.std(accumulatePositive(array))
        mean_val = np.mean(array)
        std_val = np.std(array)
        norm_min = mean_val - sigma * std_val
        norm_max = mean_val + sigma * std_val

    d = ds9.ds9()  # Open a ds9 instance
    if type(frame) is int:
        d.set('frame ' + str(frame))

    d.set_np2arr(array)
    # d.view(array, frame=frame)
    d.set('zoom to fit')
    d.set('cmap ' + colormap)
    if norm_min is not None and norm_max is not None:
        d.set('scale ' + str(norm_min) + ' ' + str(norm_max))
    if scale is not None:
        d.set('scale ' + scale)

    # plt.matshow(array, cmap=colormap, origin='lower',norm=norm, fignum=False)

    # for ptm in pixelsToMark:
    #    box = mpl.patches.Rectangle((ptm[0]-0.5,ptm[1]-0.5),\
    #                                    1,1,color=pixelMarkColor)
    #    #box = mpl.patches.Rectangle((1.5,2.5),1,1,color=pixelMarkColor)
    #    fig.axes[0].add_patch(box)