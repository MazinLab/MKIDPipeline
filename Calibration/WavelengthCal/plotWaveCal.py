import os
import warnings
import numpy as np
import tables as tb
import matplotlib as mpl
from matplotlib import cm
from matplotlib import lines
from astropy.constants import h, c
from matplotlib import pyplot as plt
from configparser import ConfigParser
from matplotlib.backends.backend_pdf import PdfPages
from Headers import pipelineFlags
from Calibration.WavelengthCal.WaveCal import fitModels


def plotEnergySolution(file_name, res_id=None, pixel=[]):
    '''
    Plot the phase to energy solution for a pixel from the wavlength calibration solution
    file 'file_name'. Provide either the pixel location pixel=(row, column) or the res_id
    for the resonator.

    Args:
        file_name: the wavecal solution file including the path (string)
        res_id: the resonator ID for the plotted pixel. Can use pixel keyword-arg
                instead. (integer)
        pixel: the pixel row and column for the plotted pixel. Can use res_id keyword-args
               instead. (length 2 list of integers)
    '''
    # load file_name
    wave_cal = tb.open_file(file_name, mode='r')
    wavelengths = wave_cal.root.header.wavelengths.read()[0]
    info = wave_cal.root.header.info.read()
    model_name = info['model_name'][0].decode('utf-8')
    calsoln = wave_cal.root.wavecal.calsoln.read()
    debug = wave_cal.root.debug.debug_info.read()
    beamImage = wave_cal.root.header.beamMap.read()

    # parse inputs
    if len(pixel) != 2 and res_id is None:
        wave_cal.close()
        raise ValueError('please supply resonator location or res_id')
    if len(pixel) == 2 and res_id is None:
        row = pixel[0]
        column = pixel[1]
        res_id = beamImage[row][column]
        index = np.where(res_id == np.array(calsoln['resid']))
    elif res_id is not None:
        index = np.where(res_id == np.array(calsoln['resid']))
        if len(index[0]) != 1:
            wave_cal.close()
            raise ValueError("res_id must exist and be unique")
        row = calsoln['pixel_row'][index][0]
        column = calsoln['pixel_col'][index][0]

    # load data
    poly = calsoln['polyfit'][index][0]
    flag = calsoln['wave_flag'][index][0]
    centers = []
    errors = []
    energies = []
    for ind, wavelength in enumerate(wavelengths):
        hist_fit = debug['hist_fit' + str(ind)][index][0]
        hist_cov = debug['hist_cov' + str(ind)][index][0]
        hist_flag = debug['hist_flag'][index, ind][0]
        if hist_flag == 0:
            if model_name == 'gaussian_and_exp':
                energies.append(h.to('eV s').value * c.to('nm/s').value /
                                np.array(wavelength))

                centers.append(hist_fit[3])
                errors.append(np.sqrt(hist_cov.reshape((5, 5))[3, 3]))
            else:
                raise ValueError("{0} is not a valid fit model name"
                                 .format(model_name))
    wave_cal.close()
    energies = np.array(energies)
    centers = np.array(centers)
    errors = np.array(errors)

    if len(energies) == 0:
        print('pixel has no data')
        return

    # plot data
    fig, axis = plt.subplots()
    axis.set_xlabel('phase [deg]')
    axis.set_ylabel('energy [eV]')
    axis.errorbar(centers, energies, xerr=errors, linestyle='--', marker='o',
                  markersize=5, markeredgecolor='black', markeredgewidth=0.5,
                  ecolor='black', capsize=3, elinewidth=0.5)

    ylim = [0.95 * min(energies), max(energies) * 1.05]
    axis.set_ylim(ylim)
    xlim = [1.05 * min(centers - errors), 0.92 * max(centers + errors)]
    axis.set_xlim(xlim)

    if poly[0] != -1:
        xx = np.arange(-180, 0, 0.1)
        axis.plot(xx, np.polyval(poly, xx), color='orange')

    xmax = xlim[1]
    ymax = ylim[1]
    dx = xlim[1] - xlim[0]
    dy = ylim[1] - ylim[0]
    flag_dict = pipelineFlags.waveCal
    if poly[0] == -1:
        axis.text(xmax - 0.05 * dx, ymax - 0.1 * dy, flag_dict[flag],
                  color='red', ha='right', va='top')
    else:
        axis.text(xmax - 0.05 * dx, ymax - 0.1 * dy, flag_dict[flag],
                  ha='right', va='top')
    axis.text(xmax - 0.05 * dx, ymax - 0.05 * dy,
              '{0} : ({1}, {2})'.format(res_id, row, column), ha='right', va='top')
    plt.show(block=False)


def plotHistogramFits(file_name, res_id=None, pixel=[]):
    '''
    Plot the histogram fits for a pixel from the wavlength calibration solution
    file 'file_name'. Provide either the pixel location pixel=(row, column) or the res_id
    for the resonator.

    Args:
        file_name: the wavecal solution file including the path (string)
        res_id: the resonator ID for the plotted pixel. Can use pixel keyword-arg
                instead. (integer)
        pixel: the pixel row and column for the plotted pixel. Can use res_id keyword-args
               instead. (length 2 list of integers)
    '''
    # load file_name
    wave_cal = tb.open_file(file_name, mode='r')
    wavelengths = wave_cal.root.header.wavelengths.read()[0]
    info = wave_cal.root.header.info.read()
    model_name = info['model_name'][0].decode('utf-8')
    debug = wave_cal.root.debug.debug_info.read()
    beamImage = wave_cal.root.header.beamMap.read()

    # parse inputs
    if len(pixel) != 2 and res_id is None:
        wave_cal.close()
        raise ValueError('please supply resonator location or res_id')
    if len(pixel) == 2 and res_id is None:
        row = pixel[0]
        column = pixel[1]
        res_id = beamImage[row][column]
        index = np.where(res_id == np.array(debug['resid']))
    elif res_id is not None:
        index = np.where(res_id == np.array(debug['resid']))
        if len(index[0]) != 1:
            wave_cal.close()
            raise ValueError("res_id must exist and be unique")
        row = debug['pixel_row'][index][0]
        column = debug['pixel_col'][index][0]

    fit_function = fitModels(model_name)
    x_num = int(np.ceil(len(wavelengths) / 2))
    y_num = 3

    fig = plt.figure(figsize=(4 * x_num, 8))
    fig.text(0.01, 0.5, 'Counts', va='center', rotation='vertical')
    fig.text(0.5, 0.01, 'Phase [degrees]', ha='center')
    fig.text(0.85, 0.95, '{0} : ({1}, {2})'.format(res_id, row, column))

    axes = []
    for x_ind in range(x_num):
        for y_ind in range(y_num - 1):
            axes.append(plt.subplot2grid((y_num, x_num), (y_ind, x_ind)))
    axes.append(plt.subplot2grid((y_num, x_num), (y_num - 1, 0), colspan=x_num))

    fit_accepted = lines.Line2D([], [], color='green', label='fit accepted')
    fit_rejected = lines.Line2D([], [], color='red', label='fit rejected')
    gaussian = lines.Line2D([], [], color='orange',
                            linestyle='--', label='gaussian')
    noise = lines.Line2D([], [], color='purple',
                         linestyle='--', label='noise')
    axes[0].legend(handles=[fit_accepted, fit_rejected, gaussian, noise],
                   loc=3, bbox_to_anchor=(0, 1.02, 1, .102), ncol=2)
    counter = 0
    for ind, wavelength in enumerate(wavelengths):
        # path = '/debug/res' + str(res_id) + '/wvl' + str(ind)
        hist_fit = debug['hist_fit' + str(ind)][index][0]
        centers = debug['phase_centers' + str(ind)][index][0]
        centers = centers[centers <= 0]
        # centers = wave_cal.get_node(path + '/phase_centers').read()[0]
        counts = debug['phase_counts' + str(ind)][index][0]
        counts = counts[counts >= 0]
        # counts = wave_cal.get_node(path + '/phase_counts').read()[0]
        bin_width = debug['bin_width'][index, ind][0][0]
        flag = debug['hist_flag'][index, ind][0]
        if flag == 0:
            color = 'green'
        else:
            color = 'red'
        axes[ind].bar(centers, counts, align='center', width=bin_width)
        if ind == 0:
            xlim = axes[ind].get_xlim()
        if model_name == 'gaussian_and_exp':
            if len(hist_fit) > 0 and len(centers) != 0:
                g_func = fitModels('gaussian')
                e_func = fitModels('exp')
                phase = np.arange(np.min(centers), np.max(centers), 0.1)
                axes[ind].plot(phase, e_func(phase, *hist_fit[:2]),
                                 color='purple', linestyle='--')
                axes[ind].plot(phase, g_func(phase, *hist_fit[2:]),
                                 color='orange', linestyle='--')
                axes[ind].plot(phase, fit_function(phase, *hist_fit),
                                 color=color)
                ylim = axes[ind].get_ylim()
                xmin = xlim[0]
                ymax = ylim[1]
                axes[ind].set_xlim(xlim)
            else:
                ylim = axes[ind].get_ylim()
                axes[ind].set_xlim(xlim)
                xmin = xlim[0]
                ymax = ylim[1]
                if len(hist_fit) <= 0:
                    axes[ind].text(xmin * 0.98, ymax * 0.5, 'Fit Error',
                                   color='red')
                else:
                    axes[ind].text(xmin * 0.98, ymax * 0.5, 'No Data',
                                   color='red')
        dx = xlim[1] - xlim[0]
        dy = ylim[1] - ylim[0]
        axes[ind].text(xmin + 0.05 * dx, ymax - 0.05 * dy,
                         str(wavelength) + ' nm', va='top', ha='left')
        if len(centers) != 0:
            axes[-1].plot(centers, counts, drawstyle='steps-mid',
                          label=str(wavelength) + ' nm')
        else:
            counter += 1
    wave_cal.close()
    if counter == len(wavelengths):
        print('pixel has no data')
        plt.close(fig)
        return
    axes[-1].set_xlim(xlim)
    axes[-1].legend()
    plt.tight_layout(rect=[0.03, 0.03, 1, 0.95])
    plt.show(block=False)


def plotRHistogram(file_name, mask=None, axis=None):
    '''
    Plot a histogram of the energy resolution, R, for each wavelength in the wavlength
    calibration solution file 'file_name'.

    Args:
        file_name: the wavecal solution file including the path (string)
        mask: a list of booleans which masks which wavelengths to include in the plot
              False entries are not displayed.
        axis: matplotlib axis object on which to display the plot. If no axis is provided
              a new figure will be made.
    '''
    # load file_name
    wave_cal = tb.open_file(file_name, mode='r')
    wavelengths = wave_cal.root.header.wavelengths.read()[0]
    if mask is None:
        mask = np.ones(np.shape(wavelengths), dtype=bool)
    elif np.shape(mask) != np.shape(wavelengths):
        wave_cal.close()
        raise ValueError('mask must be a list of booleans the same length as the ' +
                         'number of wavelengths in the solution file')
    calsoln = wave_cal.root.wavecal.calsoln.read()
    R = calsoln['R'][:, mask]

    show = False
    if axis is None:
        fig, axis = plt.subplots()
        show = True
    axis.set_xlabel(r'R [E/$\Delta$E]')
    axis.set_ylabel('counts')
    cmap = cm.get_cmap('viridis')
    max_counts = []
    for index, wavelength in enumerate(wavelengths[mask]):
        r = R[:, index]
        r = r[r != -1]
        counts, edges = np.histogram(r, bins=30, range=(0, 12))
        bws = np.diff(edges)
        cents = edges[:-1] + bws[0] / 2.0
        bins = cents
        if len(r) > 0:
            median = np.round(np.median(r), 2)
        else:
            median = np.nan

        label = "{0} nm, Median R = {1}".format(wavelength, median)
        color = cmap(index / len(wavelengths))
        axis.step(bins, counts, color=color, linewidth=2, label=label, where="mid")
        axis.axvline(x=median, ymin=0, ymax=1000, linestyle='--', color=color,
                     linewidth=2)
        max_counts.append(np.max(counts))
    axis.set_ylim([0, 1.2 * np.max(max_counts)])
    plt.tight_layout()
    axis.legend(fontsize=6)
    if show:
        plt.show(block=False)
    else:
        return axis


def plotRvsF(file_name, config_name, axis=None, verbose=True):
    '''
    Plot the energy resolution averaged over all wavelengths against the resonance
    frequency.

    Args:
        file_name: the wavecal solution file including the path (string)
        config_name: the templar configuration file, including the path, associated with
                     the data (string)
        axis: matplotlib axis object on which to display the plot. If no axis is provided
              a new figure will be made.
        verbose: determines whether information about loading the frequency files is
                 printed to the terminal (boolean)
    '''
    freqs = loadFrequencyFile(config_name, verbose=verbose)
    wave_cal = tb.open_file(file_name, mode='r')
    wavelengths = wave_cal.root.header.wavelengths.read()[0]
    calsoln = wave_cal.root.wavecal.calsoln.read()
    R0 = calsoln['R']
    R0[R0 == -1] = np.nan
    with warnings.catch_warnings():
        # rows with all nan values will give an unnecessary RuntimeWarning
        warnings.simplefilter("ignore", category=RuntimeWarning)
        R = np.nanmean(R0, axis=1)
    res_id = calsoln['resid']
    f = []
    r0 = []
    for id_ in res_id:
        index = np.where(id_ == freqs[:, 0])
        if len(index[0]) == 1 and not np.isnan(R[index]):
            f.append(freqs[index, 1])
            r0.append(R[index])
    f = np.ndarray.flatten(np.array(f))
    r0 = np.ndarray.flatten(np.array(r0))
    args = np.argsort(f)
    f = f[args]
    r0 = r0[args]
    window = 0.3e9  # 200 MHz
    r = np.zeros(r0.shape)
    for index, _ in enumerate(r0):
        points = np.where(np.logical_and(f > f[index] - window / 2,
                          f < f[index] + window / 2))
        if len(points[0]) > 0:
            r[index] = np.median(r0[points])
        else:
            r[index] = 0
    show = False
    if axis is None:
        fig, axis = plt.subplots()
        show = True
    axis.plot(f / 1e9, r, color='k', label='median')
    axis.scatter(f / 1e9, r0, s=3)
    axis.set_xlabel('resonance frequency [GHz]')
    axis.set_ylabel(r'R [E/$\Delta$E]')
    axis.legend(fontsize=6)
    wave_cal.close()
    plt.tight_layout()
    if show:
        plt.show(block=False)
    else:
        return axis


def plotCenterHist(file_name, mask=None, axis=None):
    '''
    Plot a histogram of the fitted gaussian centers for the solution file (file_name).
    Args:
        file_name: the wavecal solution file including the path (string)
        mask: a list of booleans which masks which wavelengths to include in the plot
              False entries are not displayed.
        axis: matplotlib axis object on which to display the plot. If no axis is provided
              a new figure will be made.
    '''
    # load file_name
    wave_cal = tb.open_file(file_name, mode='r')
    info = wave_cal.root.header.info.read()
    model_name = info['model_name'][0].decode('utf-8')
    wavelengths = wave_cal.root.header.wavelengths.read()[0]
    if mask is None:
        mask = np.ones(np.shape(wavelengths), dtype=bool)
    elif np.shape(mask) != np.shape(wavelengths):
        wave_cal.close()
        raise ValueError('mask must be a list of booleans the same length as the ' +
                         'number of wavelengths in the solution file')
    debug = wave_cal.root.debug.debug_info.read()
    calsoln = wave_cal.root.wavecal.calsoln.read()
    show = False
    if axis is None:
        fig, axis = plt.subplots()
        show = True
    axis.set_xlabel('gaussian center [degrees]')
    axis.set_ylabel('counts')
    cmap = cm.get_cmap('viridis')
    max_counts = []
    for index, wavelength in enumerate(wavelengths):
        if not mask[index]:
            continue
        if model_name == 'gaussian_and_exp':
            centers = debug["hist_fit" + str(index)][:, 3]
        else:
            raise ValueError("{0} is not a valid fit model name"
                             .format(model_name))
        hist_flag = debug["hist_flag"][:, index]
        wave_flag = calsoln["wave_flag"]
        condition = np.logical_and(hist_flag == 0,
                                   np.logical_or(wave_flag == 4, wave_flag == 5))
        centers = centers[condition]
        counts, edges = np.histogram(centers, bins=30, range=(-150, -20))
        bws = np.diff(edges)
        cents = edges[:-1] + bws[0] / 2.0
        bins = cents
        if len(centers) > 0:
            median = np.round(np.median(centers), 2)
        else:
            median = np.nan
        label = "{0} nm, Median = {1}".format(wavelength, median)
        color = cmap(index / len(wavelengths))
        axis.step(bins, counts, color=color, linewidth=2, where="mid", label=label)
        axis.axvline(x=median, ymin=0, ymax=1000, linestyle='--', color=color,
                     linewidth=2)
        max_counts.append(np.max(counts))
    axis.set_ylim([0, 1.2 * np.max(max_counts)])

    wave_cal.close()
    plt.tight_layout()
    axis.legend(fontsize=6)
    if show:
        plt.show(block=False)
    else:
        return axis


def plotSummary(file_name, config_name='', save_pdf=False, save_name=None, verbose=True):
    '''
    Plot one page summary pdf of the wavelength calibration solution file 'file_name'.

    Args:
        file_name: the wavecal solution file including the path (string)
        config_name: the templar configuration file, including the path, associated with
                     the data (string)
        save_pdf: save a pdf of the plot to the directory of 'file_name' instead of
                  displaying a matplotlib figure (boolean)
        save_name: name of the pdf that's saved. Needed only when save_pdf=True.
        verbose: determines whether information about loading the frequency files is
                 printed to the terminal (boolean)
    '''

    # load file_name
    wave_cal = tb.open_file(file_name, mode='r')
    wavelengths = wave_cal.root.header.wavelengths.read()[0]
    obsFiles = wave_cal.root.header.obsFiles.read()[0]
    info = wave_cal.root.header.info.read()
    model_name = info['model_name'][0].decode('utf-8')
    calsoln = wave_cal.root.wavecal.calsoln.read()
    debug = wave_cal.root.debug.debug_info.read()
    beamImage = wave_cal.root.header.beamMap.read()
    res_id = debug['resid']
    R = calsoln['R']
    data = []
    flags = debug['hist_flag']
    has_data = debug['has_data']
    fit_flags = calsoln['wave_flag']
    text = np.zeros((3, 4))
    text[0, 0] = round(np.sum(flags == 0) / len(wavelengths), 2)
    text[0, 1] = round(text[0, 0] / beamImage.size * 100, 2)
    text[0, 2] = round(text[0, 0] / np.sum(res_id != 2**23 - 1) * 100, 2)
    text[0, 3] = round(text[0, 0] * len(wavelengths) /
                       np.sum(has_data) * 100, 2)

    text[1, 0] = np.sum(fit_flags == 4) + np.sum(fit_flags == 5)
    text[1, 1] = round(text[1, 0] / beamImage.size * 100, 2)
    text[1, 2] = round(text[1, 0] / np.sum(res_id != 2**23 - 1) * 100, 2)
    text[1, 3] = round(text[1, 0] * len(wavelengths) /
                       np.sum(np.array(has_data) == 1) * 100, 2)

    text[2, 0] = np.sum(fit_flags == 5)
    text[2, 1] = np.sum(fit_flags == 4)
    wave_cal.close()

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6.95, 9))
    row_labels = ['histogram fits', 'energy solution', '(linear, quadratic)']
    col_labels = ["total pixels sucessful", "total \% sucessful",
                  "total \% out of read-out pixels",
                  "total \% out of photosensitive pixels"]
    mpl.rcParams['text.latex.preamble'] = [r'\usepackage{array}']
    table = r'''\begin{{tabular}}{{>{{\raggedright}}p{{1.1in}} | ''' + \
            r''' >{{\raggedright}}p{{1in}} | >{{\raggedright}}p{{1in}} | ''' + \
            r''' >{{\raggedright}}p{{1in}} | p{{1.1in}}}}''' + \
            r''' & {0} & {1} & {2} & {3} \\''' + \
            r'''\hline {4} & {5} & {6} & {7} & {8} \\''' + \
            r'''{9} & {10} & {11} & {12} & {13} \\''' + \
            r'''{14} & ({15}, {16}) & - & - & - \end{{tabular}}'''
    table = table.format(col_labels[0], col_labels[1], col_labels[2], col_labels[3],
                         row_labels[0], text[0, 0], text[0, 1], text[0, 2], text[0, 3],
                         row_labels[1], text[1, 0], text[1, 1], text[1, 2], text[1, 3],
                         row_labels[2], text[2, 0], text[2, 1])

    axes[0, 0] = plotRHistogram(file_name, axis=axes[0, 0])
    ylim = axes[0, 0].get_ylim()
    xlim = axes[0, 0].get_xlim()
    axes[0, 0].text(xlim[0] - np.mean(xlim) / 5, ylim[1] * 1.4,
                    "Wavelength Calibration Solution Summary:", fontsize=15)
    switch = False
    if not mpl.rcParams['text.usetex']:
        mpl.rc('text', usetex=True)
        switch = True
    axes[0, 0].text(xlim[0] - np.mean(xlim) / 5, ylim[1] * 1.1, table)
    mpl.rc('text', usetex=False)
    axes[1, 1].axis('off')
    axes[1, 1].text(-0.1, 0.97, "Wavelength Calibration File Name:")
    axes[1, 1].text(-0.1, 0.92, file_name.split('/')[-1])
    axes[1, 1].text(-0.1, 0.82, "Fit Model: {0}".format(model_name))
    axes[1, 1].text(-0.1, 0.72, "ObsFile Names:")
    for ind, obs in enumerate(obsFiles):
        axes[1, 1].text(-0.1, 0.67 - ind / 20, obs.decode("utf-8"))
    axes[1, 1].text(-0.1, 0.57 - (len(wavelengths) - 1) / 20, "Wavelengths [nm]:")
    for ind, wavelength in enumerate(wavelengths):
        axes[1, 1].text(-0.1, 0.57 - ind / 20 - len(wavelengths) / 20, wavelength)
    good_hist = np.sum(flags == 0, axis=1)
    perc = round(np.sum(fit_flags == 7) / np.sum(good_hist >= 3) * 100, 2)
    axes[1, 1].text(-0.1, 0.37 - 2 * len(wavelengths) / 20,
                    'Percent of pixels that failed the \ncalibration because the' +
                    ' centers were not \nmonotonic (out of pixels with > 2 good' +
                    ' \nhistogram fits): {0}%'.format(perc))

    if not switch:
        mpl.rc('text', usetex=True)

    axes[1, 0] = plotCenterHist(file_name, axis=axes[1, 0])
    axes[0, 1] = plotRvsF(file_name, config_name, axis=axes[0, 1], verbose=verbose)

    plt.tight_layout(rect=[0.03, 0.03, 1, 0.85])
    if save_pdf:
        if save_name is None:
            raise ValueError('define key-value pair save_name to be a string')
        out_directory = os.path.dirname(file_name)
        pdf = PdfPages(os.path.join(out_directory, save_name))
        pdf.savefig(fig)
        pdf.close()
        if verbose:
            print("plot saved")
        plt.close()
    else:
        plt.show(block=False)


def loadFrequencyFile(config_file, verbose=True):
    '''
    Returns a dictionary mapping res_id to resonance frequency for a particular templar
    configuration file.

    Args:
        config_file: full path and file name of the templar configuration file. (string)
        verbose: determines whether information about loading the frequency files is
                 printed to the terminal (boolean)

    Returns:
        a numpy array of the frequency files that could be loaded from the templar config
        file vertically stacked. The first column is the res_id and the second is the
        frequency. a 1 by 2 array with -1 entries is returned if no files could be loaded.
    '''
    config = ConfigParser()
    config.read(config_file)
    freqs = []
    for roach in config.keys():
        if roach[0] == 'R':
            try:
                freq_file = config[roach]['freqfile']
                if verbose:
                    print('loading frequency file: {0}'.format(freq_file))
                freq = np.loadtxt(freq_file)
                freqs.append(freq)
            except Exception as error:
                if verbose:
                    print(error)
    if len(freqs) == 0:
        if verbose:
            print('Warning: no frequency files could be loaded')
        return np.ones((1, 2)) * -1
    freqs = np.vstack(freqs)
    return freqs
