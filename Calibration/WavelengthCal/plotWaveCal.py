import os
import warnings
import numpy as np
import tables as tb
import matplotlib as mpl
from matplotlib import cm
from matplotlib import lines
from astropy.constants import h, c
from matplotlib import pyplot as plt
from matplotlib import image as mpimg
from configparser import ConfigParser
from matplotlib.backends.backend_pdf import PdfPages
from Headers import pipelineFlags


def plotEnergySolution(file_name, res_id=None, pixel=[], axis=None):
    '''
    Plot the phase to energy solution for a pixel from the wavlength calibration solution
    file 'file_name'. Provide either the pixel location pixel=(row, column) or the res_id
    for the resonator.

    Args:
        file_name: the wavecal solution file including the path (string)
        res_id: the resonator ID for the plotted pixel. Can use pixel keyword-arg
                instead. (integer)
        pixel: the pixel row and column for the plotted pixel. Can use res_id keyword-arg
               instead. (length 2 list of integers)
        axis: matplotlib axis object on which to display the plot. If no axis is provided
              a new figure will be made.
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
    if len(calsoln['polyfit'][index]) == 0:
        print('pixel has no data')
        return
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

    R0 = calsoln['R'][index]
    R0[R0 == -1] = np.nan
    with warnings.catch_warnings():
        # rows with all nan values will give an unnecessary RuntimeWarning
        warnings.simplefilter("ignore", category=RuntimeWarning)
        R = np.nanmedian(R0)

    # plot data
    show = False
    if axis is None:
        fig, axis = plt.subplots()
        show = True
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
    axis.text(xmax - 0.05 * dx, ymax - 0.05 * dy,
              '{0} : ({1}, {2})'.format(res_id, row, column), ha='right', va='top')
    if poly[0] == -1:
        axis.text(xmax - 0.05 * dx, ymax - 0.1 * dy, flag_dict[flag],
                  color='red', ha='right', va='top')
    else:
        if flag == 9:
            axis.text(xmax - 0.05 * dx, ymax - 0.1 * dy, flag_dict[flag], color='red',
                      ha='right', va='top')
        else:
            axis.text(xmax - 0.05 * dx, ymax - 0.1 * dy, flag_dict[flag],
                      ha='right', va='top')
    axis.text(xmax - 0.05 * dx, ymax - 0.15 * dy, "Median R = {0}".format(round(R, 2)),
              ha='right', va='top')
    if show:
        plt.show(block=False)
    else:
        return axis


def plotHistogramFits(file_name, res_id=None, pixel=[], axis=None):
    '''
    Plot the histogram fits for a pixel from the wavlength calibration solution
    file 'file_name'. Provide either the pixel location pixel=(row, column) or the res_id
    for the resonator.

    Args:
        file_name: the wavecal solution file including the path (string)
        res_id: the resonator ID for the plotted pixel. Can use pixel keyword-arg
                instead. (integer)
        pixel: the pixel row and column for the plotted pixel. Can use res_id keyword-arg
               instead. (length 2 list of integers)
        axis: matplotlib axis object on which to display the plot (will be an embedded
              png). If no axis is provided a new figure will be made.
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
        if len(debug['pixel_row'][index]) == 0:
            print('pixel has no data')
            return
        row = debug['pixel_row'][index][0]
        column = debug['pixel_col'][index][0]

    if len(debug['pixel_row'][index]) == 0:
        print('pixel has no data')
        return

    cmap = cm.get_cmap('viridis')
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
    y_list = []
    if x_num > 5:
        for ind, wavelength in enumerate(wavelengths):
            counts = debug['phase_counts' + str(ind)][index][0]
            counts = counts[counts >= 0]
            if len(counts) > 0:
                y_list.append(np.max(counts))
        if len(y_list) > 0:
            largest_y = np.max(y_list)
        else:
            largest_y = 1
    for ind, wavelength in enumerate(wavelengths):
        hist_fit = debug['hist_fit' + str(ind)][index][0]
        centers = debug['phase_centers' + str(ind)][index][0]
        centers = centers[centers <= 0]
        counts = debug['phase_counts' + str(ind)][index][0]
        counts = counts[counts >= 0]
        bin_width = debug['bin_width'][index, ind][0][0]
        flag = debug['hist_flag'][index, ind][0]
        if flag == 0:
            color = 'green'
        else:
            color = 'red'
        axes[ind].bar(centers, counts, align='center', width=bin_width)
        if ind == 0:
            xlim = axes[ind].get_xlim()
        if x_num > 5:
            axes[ind].set_ylim([0, 1.1 * largest_y])
            ymax = 1.1 * largest_y
            ylim = axes[ind].get_ylim()
            if ind > 1:
                axes[ind].get_yaxis().set_ticks([])
        else:
            ylim = axes[ind].get_ylim()
            ymax = ylim[1]
        if model_name == 'gaussian_and_exp':
            if len(hist_fit) > 0 and len(centers) != 0 and flag != 3 and flag != 10:
                g_func = fitModels('gaussian')
                e_func = fitModels('exp')
                phase = np.arange(np.min(centers), np.max(centers), 0.1)
                axes[ind].plot(phase, e_func(phase, *hist_fit[:2]),
                                 color='purple', linestyle='--')
                axes[ind].plot(phase, g_func(phase, *hist_fit[2:]),
                                 color='orange', linestyle='--')
                axes[ind].plot(phase, fit_function(phase, *hist_fit),
                                 color=color)
                if x_num > 3:
                    axes[ind].tick_params(axis='both', which='major', labelsize=8)
                xmin = xlim[0]
                axes[ind].set_xlim(xlim)
                dx = xlim[1] - xlim[0]
                dy = ylim[1] - ylim[0]
            else:
                ylim = axes[ind].get_ylim()
                axes[ind].set_xlim(xlim)
                xmin = xlim[0]
                ymax = ylim[1]
                dx = xlim[1] - xlim[0]
                dy = ylim[1] - ylim[0]
                if len(hist_fit) == 0:
                    axes[ind].text(xmin + 0.05 * dx, ymax - 0.15 * dy, 'Fit Error',
                                   color='red', fontsize=10, va='top', ha='left')
                elif flag == 3:
                    axes[ind].text(xmin + 0.05 * dx, ymax - 0.15 * dy,
                                   'Not Enough \nData', color='red', fontsize=10,
                                   va='top', ha='left')
                elif flag == 10:
                    axes[ind].text(xmin + 0.05 * dx, ymax - 0.15 * dy,
                                   'Too Much \nData (Hot)', color='red', fontsize=10,
                                   va='top', ha='left')
                else:
                    axes[ind].text(xmin + 0.05 * dx, ymax - 0.15 * dy, 'No Data',
                                   color='red', fontsize=10, va='top', ha='left')
        axes[ind].text(xmin + 0.05 * dx, ymax - 0.05 * dy,
                         str(wavelength) + ' nm', va='top', ha='left')
        if len(centers) != 0:
            color = cmap(ind / len(wavelengths))
            axes[-1].plot(centers, counts, color=color, drawstyle='steps-mid',
                          label=str(wavelength) + ' nm')
        else:
            counter += 1
    wave_cal.close()
    if ind < x_num * 2 + 1:
        axes[ind + 1].get_yaxis().set_ticks([])
        if x_num > 3:
            axes[ind + 1].tick_params(axis='both', which='major', labelsize=8)
    if counter == len(wavelengths):
        print('pixel has no data')
        plt.close(fig)
        return
    axes[-1].set_xlim(xlim)
    axes[-1].legend(ncol=int(np.ceil(2 * x_num / 7)))
    plt.tight_layout(rect=[0.04, 0.03, 1, 0.95])
    if axis is None:
        plt.show(block=False)
    else:
        fig.savefig('temp.png', dpi=500)
        img = mpimg.imread('temp.png')
        axis.imshow(img, aspect='auto')
        axis.axes.get_yaxis().set_visible(False)
        axis.axes.get_xaxis().set_visible(False)
        axis.set_frame_on(False)
        os.remove('temp.png')
        plt.close(fig)
        return axis


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
    if len(wavelengths) >= 10:
        Z = [[0, 0], [0, 0]]
        levels = np.arange(min(wavelengths), max(wavelengths), 1)
        c = axis.contourf(Z, levels, cmap=cmap)
        plt.colorbar(c, ax=axis, label='wavelength [nm]', aspect=50)
        axis.clear()
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
        color = cmap(index / (len(wavelengths) - 1))
        axis.step(bins, counts, color=color, linewidth=2, label=label, where="mid")
        axis.axvline(x=median, ymin=0, ymax=1000, linestyle='--', color=color,
                     linewidth=2)
        max_counts.append(np.max(counts))
    if np.max(max_counts) != 0:
        axis.set_ylim([0, 1.2 * np.max(max_counts)])
    plt.tight_layout()
    if len(wavelengths) < 10:
        axis.legend(fontsize=6)
    if show:
        plt.show(block=False)
    else:
        return axis


def plotRvsF(file_name, config_name, axis=None, verbose=True):
    '''
    Plot the median energy resolution over all wavelengths against the resonance
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
        R = np.nanmedian(R0, axis=1)
    res_id = calsoln['resid']
    f = []
    r0 = []
    for id_index, id_ in enumerate(res_id):
        index = np.where(id_ == freqs[:, 0])
        if len(index[0]) == 1 and not np.isnan(R[id_index]):
            f.append(freqs[index, 1])
            r0.append(R[id_index])
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
    cmap = cm.get_cmap('viridis')
    if len(wavelengths) >= 10:
        Z = [[0, 0], [0, 0]]
        levels = np.arange(min(wavelengths), max(wavelengths), 1)
        c = axis.contourf(Z, levels, cmap=cmap)
        plt.colorbar(c, ax=axis, label='wavelength [nm]', aspect=50)
        axis.clear()
    axis.set_xlabel('gaussian center [degrees]')
    axis.set_ylabel('counts')
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
        condition = np.logical_and(hist_flag == 0, np.logical_or(wave_flag == 4,
                                   np.logical_or(wave_flag == 5, wave_flag == 9)))
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
        color = cmap(index / (len(wavelengths) - 1))
        axis.step(bins, counts, color=color, linewidth=2, where="mid", label=label)
        axis.axvline(x=median, ymin=0, ymax=1000, linestyle='--', color=color,
                     linewidth=2)
        max_counts.append(np.max(counts))
    if np.max(max_counts) != 0:
        axis.set_ylim([0, 1.2 * np.max(max_counts)])
    wave_cal.close()
    plt.tight_layout()
    if len(wavelengths) < 10:
        axis.legend(fontsize=6)
    if show:
        plt.show(block=False)
    else:
        return axis


def plotFitParameters(file_name):
    '''
    Plots histograms of the fit parameters for the solution file (file_name).
    Args:
        file_name: the wavecal solution file including the path (string)
    '''
    wave_cal = tb.open_file(file_name, mode='r')
    wavelengths = wave_cal.root.header.wavelengths.read()[0]
    info = wave_cal.root.header.info.read()
    model_name = info['model_name'][0].decode('utf-8')
    debug = wave_cal.root.debug.debug_info.read()
    hist_flags = debug['hist_flag']
    wave_cal.close()
    cmap = cm.get_cmap('viridis')

    if model_name == 'gaussian_and_exp':
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6.95, 9))
        if len(wavelengths) >= 10:
            Z = [[0, 0], [0, 0]]
            levels = np.arange(min(wavelengths), max(wavelengths), 1)
            c = axes[0, 1].contourf(Z, levels, cmap=cmap)
            plt.colorbar(c, ax=axes[0, 1], label='wavelength [nm]', aspect=50)
            axes[0, 1].clear()
            c = axes[1, 1].contourf(Z, levels, cmap=cmap)
            plt.colorbar(c, ax=axes[1, 1], label='wavelength [nm]', aspect=50)
            axes[1, 1].clear()
        for index, wavelength in enumerate(wavelengths):
            fit = debug["hist_fit" + str(index)]
            # first histogram
            a = fit[:, 0]
            a = a[np.logical_and(a != -1, hist_flags[:, index] == 0)]
            counts, edges = np.histogram(a, bins=30, range=(-1, 1e8), density=True)
            bws = np.diff(edges)
            cents = edges[:-1] + bws[0] / 2.0
            bins = cents
            color = cmap(index / (len(wavelengths) - 1))
            label = "{0} nm".format(wavelength)
            axes[0, 0].step(bins, counts, color=color, linewidth=2, where="mid",
                            label=label)
            axes[0, 0].set_xlabel('parameter a')
            axes[0, 0].set_ylabel('probability density')
            if len(wavelengths) < 10:
                axes[0, 0].legend(fontsize=6)

            # second histogram
            b = fit[:, 1]
            b = b[np.logical_and(b != -1, hist_flags[:, index] == 0)]
            counts, edges = np.histogram(b, bins=30, range=(-0.5, 1), density=True)
            bws = np.diff(edges)
            cents = edges[:-1] + bws[0] / 2.0
            bins = cents
            color = cmap(index / (len(wavelengths) - 1))
            label = "{0} nm".format(wavelength)
            axes[0, 1].step(bins, counts, color=color, linewidth=2, where="mid",
                            label=label)
            axes[0, 1].set_xlabel('parameter b')
            if len(wavelengths) < 10:
                axes[0, 1].legend(fontsize=6)

            # third histogram
            c = fit[:, 2]
            c = c[np.logical_and(c != -1, hist_flags[:, index] == 0)]
            counts, edges = np.histogram(c, bins=30, range=(-1, np.max(c) * 0.7),
                                         density=True)
            bws = np.diff(edges)
            cents = edges[:-1] + bws[0] / 2.0
            bins = cents
            color = cmap(index / (len(wavelengths) - 1))
            label = "{0} nm".format(wavelength)
            axes[1, 0].step(bins, counts, color=color, linewidth=2, where="mid",
                            label=label)
            axes[1, 0].set_xlabel('parameter c')
            axes[1, 0].set_ylabel('probability density')
            if len(wavelengths) < 10:
                axes[1, 0].legend(fontsize=6)

            # fourth histogram
            f = fit[:, 4]
            f = f[np.logical_and(f != -1, hist_flags[:, index] == 0)]
            counts, edges = np.histogram(f, bins=30, range=(-1, 35), density=True)
            bws = np.diff(edges)
            cents = edges[:-1] + bws[0] / 2.0
            bins = cents
            color = cmap(index / (len(wavelengths) - 1))
            label = "{0} nm".format(wavelength)
            axes[1, 1].step(bins, counts, color=color, linewidth=2, where="mid",
                            label=label)
            axes[1, 1].set_xlabel('parameter f')
            if len(wavelengths) < 10:
                axes[1, 1].legend(fontsize=6)

        axes[0, 0].set_ylim(bottom=0)
        axes[0, 1].set_ylim(bottom=0)
        axes[1, 0].set_ylim(bottom=0)
        axes[1, 1].set_ylim(bottom=0)

        ylim = axes[0, 0].get_ylim()
        xlim = axes[0, 0].get_xlim()
        title = "Wavelength Calibration Histogram Fit Model: \n'{0}'".format(model_name)
        axes[0, 0].text(xlim[0] - np.mean(xlim) / 4.5,
                        ylim[1] + (ylim[1] - ylim[0]) * 0.45, title, fontsize=15,
                        va='top')
        equation = r"histogram = $a e^{-b \phi} + c e^{-\frac{(\phi - d)^2}{2 f^2}}$"
        axes[0, 0].text(xlim[0] - np.mean(xlim) / 4.5,
                        ylim[1] + (ylim[1] - ylim[0]) * 0.25, equation, fontsize=12,
                        va='top')

    plt.tight_layout(rect=[0.03, 0.03, 0.95, 0.85])
    plt.show(block=False)


def plotSummary(file_name, config_name='', save_name=None, verbose=True):
    '''
    Plot one page summary pdf of the wavelength calibration solution file 'file_name'.

    Args:
        file_name: the wavecal solution file including the path (string)
        config_name: the templar configuration file, including the path, associated with
                     the data (string)
        save_name: name of the pdf that's saved. No pdf is saved if set to None.
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
    ind = np.where(R[:, 0] > 0)
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

    text[1, 0] = np.sum(fit_flags == 4) + np.sum(fit_flags == 5) + np.sum(fit_flags == 9)
    text[1, 1] = round(text[1, 0] / beamImage.size * 100, 2)
    text[1, 2] = round(text[1, 0] / np.sum(res_id != 2**23 - 1) * 100, 2)
    text[1, 3] = round(text[1, 0] * len(wavelengths) /
                       np.sum(np.array(has_data) == 1) * 100, 2)

    text[2, 0] = np.sum(fit_flags == 5) + np.sum(fit_flags == 9)
    text[2, 1] = np.sum(fit_flags == 4)
    wave_cal.close()

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6.95, 9))
    row_labels = ['histogram fits', 'energy solution', '(linear, quadratic)']
    col_labels = ["total pixels sucessful", "total \\% sucessful",
                  "total \\% out of read-out pixels",
                  "total \\% out of photosensitive pixels"]
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
    axes[0, 0].text(xlim[0] - np.mean(xlim) / 5, ylim[1] + (ylim[1] - ylim[0]) * 0.4,
                    "Wavelength Calibration Solution Summary:", fontsize=15)
    switch = False
    if not mpl.rcParams['text.usetex']:
        mpl.rc('text', usetex=True)
        switch = True
    axes[0, 0].text(xlim[0] - np.mean(xlim) / 5,
                    ylim[1] + (ylim[1] - ylim[0]) * 0.1, table)
    mpl.rc('text', usetex=False)
    axes[1, 1].axis('off')
    large_number = 10
    if len(wavelengths) > large_number:
        fontstyle = {'fontsize': 6.5, 'weight': 'normal'}
        dz = 0.04
    else:
        fontstyle = {'fontsize': 10, 'weight': 'normal'}
        dz = 0.05
    axes[1, 1].text(-0.1, 0.97, "Wavelength Calibration File Name:", **fontstyle)
    axes[1, 1].text(-0.1, 0.97 - dz, file_name.split('/')[-1], **fontstyle)
    axes[1, 1].text(-0.1, 0.97 - 3 * dz, "Fit Model: {0}".format(model_name), **fontstyle)
    axes[1, 1].text(-0.1, 0.97 - 5 * dz, "ObsFile Names:", **fontstyle)
    if len(wavelengths) > large_number:
        for ind in range(int(np.floor(len(obsFiles) / 3))):
            axes[1, 1].text(-0.1, 0.97 - 6 * dz - ind * dz,
                            obsFiles[3 * ind].decode("utf-8") + ', ' +
                            obsFiles[3 * ind + 1].decode("utf-8") + ', ' +
                            obsFiles[3 * ind + 2].decode("utf-8"), **fontstyle)
        if len(obsFiles) - (3 * ind + 2) == 3:
            axes[1, 1].text(-0.1, 0.97 - 6 * dz - (ind + 1) * dz,
                            obsFiles[3 * ind + 3].decode("utf-8") + ', ' +
                            obsFiles[3 * ind + 4].decode("utf-8"), **fontstyle)
        elif len(obsFiles) - (3 * ind + 2) == 2:
            axes[1, 1].text(-0.1, 0.97 - 6 * dz - (ind + 1) * dz,
                            obsFiles[3 * ind + 3].decode("utf-8"), **fontstyle)
        axes[1, 1].text(-0.1, 0.97 - 8 * dz - (np.ceil(len(obsFiles) / 3) - 1) * dz,
                        "Wavelengths [nm]:", **fontstyle)
        for ind in range(int(np.floor(len(obsFiles) / 3))):
            axes[1, 1].text(-0.1, 0.97 - 8 * dz - ind * dz -
                            np.ceil(len(obsFiles) / 3) * dz,
                            str(wavelengths[3 * ind]) + ", " +
                            str(wavelengths[3 * ind + 1]) + ", " +
                            str(wavelengths[3 * ind + 2]), **fontstyle)
        if len(obsFiles) - (3 * ind + 2) == 3:
            axes[1, 1].text(-0.1, 0.97 - 8 * dz - (ind + 1) * dz -
                            np.ceil(len(obsFiles) / 3) * dz,
                            str(wavelengths[3 * ind + 3]) + ", " +
                            str(wavelengths[3 * ind + 4]), **fontstyle)
        elif len(obsFiles) - (3 * ind + 2) == 2:
            axes[1, 1].text(-0.1, 0.97 - 8 * dz - (ind + 1) * dz -
                            np.ceil(len(obsFiles) / 3) * dz,
                            str(wavelengths[3 * ind + 3]), **fontstyle)
        good_hist = np.sum(flags == 0, axis=1)

        if np.sum(good_hist >= 3) > 0:
            perc = round(np.sum(fit_flags == 7) / np.sum(good_hist >= 3) * 100, 2)
        else:
            perc = 'N/A'
        axes[1, 1].text(-0.1, 0.97 - 8 * dz - 2 * np.ceil(len(obsFiles) / 3) * dz,
                        'Percent of pixels that failed the calibration because the'
                        ' \ncenters were not monotonic (out of pixels with \n> 2 good'
                        ' histogram fits): {0}%'.format(perc), va='top', **fontstyle)
    else:
        for ind in range(int(np.floor(len(obsFiles) / 2))):
            axes[1, 1].text(-0.1, 0.97 - 6 * dz - ind * dz,
                            obsFiles[2 * ind].decode("utf-8") + ', ' +
                            obsFiles[2 * ind + 1].decode("utf-8"), **fontstyle)
        if len(obsFiles) - (2 * ind + 1) == 2:
            axes[1, 1].text(-0.1, 0.97 - 6 * dz - (ind + 1) * dz,
                            obsFiles[2 * ind + 2].decode("utf-8"), **fontstyle)
        axes[1, 1].text(-0.1, 0.97 - 8 * dz - (np.ceil(len(obsFiles) / 2) - 1) * dz,
                        "Wavelengths [nm]:", **fontstyle)
        for ind in range(int(np.floor(len(obsFiles) / 2))):
            axes[1, 1].text(-0.1, 0.97 - 8 * dz - ind * dz -
                            np.ceil(len(obsFiles) / 2) * dz,
                            str(wavelengths[2 * ind]) + ", " +
                            str(wavelengths[2 * ind + 1]), **fontstyle)
        if len(obsFiles) - (2 * ind + 1) == 2:
            axes[1, 1].text(-0.1, 0.97 - 8 * dz - (ind + 1) * dz -
                            np.ceil(len(obsFiles) / 2) * dz,
                            str(wavelengths[2 * ind + 2]), **fontstyle)
        good_hist = np.sum(flags == 0, axis=1)
        if np.sum(good_hist >= 3) > 0:
            perc = round(np.sum(fit_flags == 7) / np.sum(good_hist >= 3) * 100, 2)
        else:
            perc = 'N/A'
        axes[1, 1].text(-0.1, 0.97 - 8 * dz - 2 * np.ceil(len(obsFiles) / 2) * dz,
                        'Percent of pixels that failed the \ncalibration because the' +
                        ' centers were not \nmonotonic (out of pixels with > 2 good' +
                        ' \nhistogram fits): {0}%'.format(perc), va='top', **fontstyle)

    if not switch:
        mpl.rc('text', usetex=True)

    axes[1, 0] = plotCenterHist(file_name, axis=axes[1, 0])
    axes[0, 1] = plotRvsF(file_name, config_name, axis=axes[0, 1], verbose=verbose)

    plt.tight_layout(rect=[0.03, 0.03, 0.95, 0.85])
    if save_name is not None:
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


def fitModels(model_name):
    '''
    Returns the specified fit model from a library of possible functions
    '''
    if model_name == 'gaussian_and_exp':
        fit_function = lambda x, a, b, c, d, f: \
            a * np.exp(b * x) + c * np.exp(-1 / 2.0 * ((x - d) / f)**2)
    elif model_name == 'gaussian':
        fit_function = lambda x, c, d, f: c * np.exp(-1 / 2.0 * ((x - d) / f)**2)
    elif model_name == 'exp':
        fit_function = lambda x, a, b: a * np.exp(b * x)
    elif model_name == 'quadratic':
        fit_function = lambda p, x: p['a'] * x**2 + p['b'] * x + p['c']
    elif model_name == 'linear':
        fit_function = lambda p, x: p['b'] * x + p['c']
    elif model_name == 'linear_zero':
        fit_function = lambda p, x: p['b'] * x

    return fit_function
