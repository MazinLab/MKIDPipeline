import os
import warnings
from configparser import ConfigParser

import matplotlib as mpl
import numpy as np
import tables as tb
from astropy.constants import c, h
from matplotlib import cm, image as mpimg, lines, pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.widgets import Button, Slider
from mpl_toolkits.axes_grid1 import axes_size, make_axes_locatable

from mkidcore import pixelflags


def plot_energy_solution(solution, res_id=None, pixel=None, axis=None):
    """
    Plot the phase to energy solution for a pixel from the wavelength calibration solution
    object. Provide either the pixel location pixel=(row, column) or the res_id for the
    resonator.

    Args:
        solution: the wavecal solution object
        res_id: the resonator ID for the plotted pixel. Can use pixel keyword-arg
                instead. (integer)
        pixel: the pixel row and column for the plotted pixel. Can use res_id keyword-arg
               instead. (length 2 list of integers)
        axis: matplotlib axis object on which to display the plot. If no axis is provided
              a new figure will be made.
    """
    # parse inputs
    if pixel is not None and len(pixel) != 2 and res_id is None:
        raise ValueError('please supply resonator location or res_id')
    if pixel is not None and len(pixel) == 2 and res_id is None:
        row = pixel[0]
        column = pixel[1]
        res_id = solution.beam_map[row, column]
    elif res_id is not None:
        index = np.where(res_id == solution.res_ids)
        if len(index[0]) != 1:
            raise ValueError("res_id must exist and be unique")
        row = solution.rows[index][0]
        column = solution.columns[index][0]

    # load data
    centers, energies, errors = solution.energies(res_id=res_id)
    flag = solution.energy_fit_flag(res_id=res_id)
    if len(energies) == 0:
        print('pixel has no data')
        return
    R0 = solution.resolving_power(res_id=res_id)
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

    if not np.isnan(solution.energy_fit_coefficients(res_id=res_id)).any():
        xx = np.arange(-180, 0, 0.1)
        fit_function = solution.energy_fit_function(res_id=res_id)
        axis.plot(xx, fit_function(xx), color='orange')

    xmax = xlim[1]
    ymax = ylim[1]
    dx = xlim[1] - xlim[0]
    dy = ylim[1] - ylim[0]
    flag_dict = pixelflags.waveCal
    axis.text(xmax - 0.05 * dx, ymax - 0.05 * dy,
              '{0} : ({1}, {2})'.format(res_id, row, column), ha='right', va='top')
    if not solution.has_good_energy_solution(res_id=res_id):
        axis.text(xmax - 0.05 * dx, ymax - 0.1 * dy, flag_dict[flag],
                  color='red', ha='right', va='top')
    else:
        axis.text(xmax - 0.05 * dx, ymax - 0.1 * dy, flag_dict[flag],
                  ha='right', va='top')
    axis.text(xmax - 0.05 * dx, ymax - 0.15 * dy, "Median R = {0}".format(round(R, 2)),
              ha='right', va='top')
    if show:
        plt.show(block=False)
    else:
        return axis


def plot_histogram_fits(solution, res_id=None, pixel=None, axis=None):
    """
    Plot the histogram fits for a pixel from the wavlength calibration solution
    object. Provide either the pixel location pixel=(row, column) or the res_id for the
    resonator.

    Args:
        solution: the wavecal solution object
        res_id: the resonator ID for the plotted pixel. Can use pixel keyword-arg
                instead. (integer)
        pixel: the pixel row and column for the plotted pixel. Can use res_id keyword-arg
               instead. (length 2 list of integers)
        axis: matplotlib axis object on which to display the plot (will be an embedded
              png). If no axis is provided a new figure will be made.
    """

    # parse inputs
    if pixel is not None and len(pixel) != 2 and res_id is None:
        raise ValueError('please supply resonator location or res_id')
    if pixel is not None and len(pixel) == 2 and res_id is None:
        row = pixel[0]
        column = pixel[1]
        res_id = solution.beam_map[row, column]
    elif res_id is not None:
        index = np.where(res_id == solution.histogram_res_ids)
        if len(index[0]) != 1:
            raise ValueError("res_id must exist and be unique")
        row = solution.histogram_rows[index][0]
        column = solution.histogram_columns[index][0]

    wavelengths = solution.wavelengths

    cmap = cm.get_cmap('viridis')
    fit_function = fitModels(solution.model_name)
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
            _, counts = solution.histogram(wavelength, res_id=res_id)
            if len(counts) > 0:
                y_list.append(np.max(counts))
        if len(y_list) > 0:
            largest_y = np.max(y_list)
        else:
            largest_y = 1
    for ind, wavelength in enumerate(wavelengths):
        hist_fit = solution.histogram_fit_coefficients(wavelength, res_id=res_id)
        centers, counts = solution.histogram(wavelength, res_id=res_id)
        bin_width = solution.bin_width(wavelength, res_id=res_id)
        flag = solution.histogram_fit_flag(wavelength, res_id=res_id)
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
        if solution.model_name == 'gaussian_and_exp':
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


def plot_R_histogram(solution, wavelengths=None, axis=None):
    """
    Plot a histogram of the energy resolution, R, for each wavelength in the wavlength
    calibration solution object.

    Args:
        solution: the wavecal solution object
        wavelengths: a list of wavelengths to include in the plot.
        axis: matplotlib axis object on which to display the plot. If no axis is provided
              a new figure will be made.
    """

    if wavelengths is None:
        wavelengths = solution.wavelengths
    if isinstance(wavelengths, (int, float)):
        wavelengths = [wavelengths]

    R, _ = solution.resolving_powers(wavelengths=wavelengths)

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
    max_counts = []
    for index, wavelength in enumerate(wavelengths):
        r = R[:, index]
        r = r[np.logical_not(np.isnan(r))]
        counts, edges = np.histogram(r, bins=30, range=(0, 12))
        bws = np.diff(edges)
        cents = edges[:-1] + bws[0] / 2.0
        bins = cents
        if len(r) > 0:
            median = np.round(np.median(r), 2)
        else:
            median = np.nan

        label = "{0} nm, Median R = {1}".format(wavelength, median)
        color = cmap(index / (max([len(wavelengths) - 1, 1])))
        axis.step(bins, counts, color=color, linewidth=2, label=label, where="mid")
        axis.axvline(x=median, ymin=0, ymax=1000, linestyle='--', color=color,
                     linewidth=2)
        max_counts.append(np.max(counts))
    if np.max(max_counts) != 0:
        axis.set_ylim([0, 1.2 * np.max(max_counts)])

    axis.set_xlabel(r'R [E/$\Delta$E]')
    axis.set_ylabel('counts')
    plt.tight_layout()
    if len(wavelengths) < 10:
        axis.legend(fontsize=6)
    if show:
        plt.show(block=False)
    else:
        return axis


def plot_R_vs_F(solution, config_name, axis=None, verbose=True):
    """
    Plot the median energy resolution over all wavelengths against the resonance
    frequency.

    Args:
        solution: the wavecal solution object
        config_name: the templar configuration file, including the path, associated with
                     the data (string)
        axis: matplotlib axis object on which to display the plot. If no axis is provided
              a new figure will be made.
        verbose: determines whether information about loading the frequency files is
                 printed to the terminal (boolean)
    """
    freqs = loadFrequencyFile(config_name, verbose=verbose)
    R0, _ = solution.resolving_powers()
    with warnings.catch_warnings():
        # rows with all nan values will give an unnecessary RuntimeWarning
        warnings.simplefilter("ignore", category=RuntimeWarning)
        R = np.nanmedian(R0, axis=1)
    res_id = solution.res_ids
    f = []
    r0 = []
    for id_index, id_ in enumerate(res_id):
        index = np.where(id_ == freqs[:, 0])
        if (len(index[0]) == 1 and not np.isnan(R[id_index]) and
           solution.has_good_energy_solution(res_id=id_)):
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
    plt.tight_layout()
    if show:
        plt.show(block=False)
    else:
        return axis


def plot_center_histogram(solution, wavelengths=None, axis=None):
    """
    Plot a histogram of the fitted gaussian centers for the solution object.
    Args:
        solution: the wavecal solution object
        wavelengths: a list of wavelengths to include in the plot.
        axis: matplotlib axis object on which to display the plot. If no axis is provided
              a new figure will be made.
    """
    model_name = solution.model_name
    if wavelengths is None:
        wavelengths = solution.wavelengths
    if isinstance(wavelengths, (int, float)):
        wavelengths = [wavelengths]

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
        if model_name == 'gaussian_and_exp':
            centers = solution.histogram_fit_coefficients(wavelength)[:, 3]
        else:
            raise ValueError("{0} is not a valid fit model name"
                             .format(model_name))

        condition = np.logical_and(solution.has_good_energy_solution(),
                                   solution.has_good_histogram_solution(wavelength))
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
        color = cmap(index / max([len(wavelengths) - 1, 1]))
        axis.step(bins, counts, color=color, linewidth=2, where="mid", label=label)
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


def plot_fit_parameters(solution):
    """
    Plots histograms of the fit parameters for the solution object.
    Args:
        solution: the wavecal solution object
    """
    wavelengths = solution.wavelengths
    model_name = solution.model_name

    cmap = cm.get_cmap('viridis')

    calibrated_pixels = solution.has_good_energy_solution()

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
            fit = solution.histogram_fit_coefficients(wavelength)
            # first histogram
            a = fit[:, 0]
            logic = np.logical_and(solution.has_good_histogram_solution(wavelength),
                                   calibrated_pixels)
            a = a[logic]
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
            b = b[logic]
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
            c = c[logic]
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
            f = f[logic]
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


def plot_resolution_image(solution):
    """
    Plots an image of the array with the energy resolution as a color for the solution.
    Args:
        solution: the wavecal solution object
    """
    beam_map = solution.beam_map
    wavelengths = solution.wavelengths
    R0, _ = solution.resolving_powers()
    R0[np.isnan(R0)] = 0
    rows = solution.rows
    columns = solution.columns
    res_id = solution.res_ids

    R = np.zeros((len(wavelengths) + 1, *beam_map.shape))

    for pixel_index, res_id in enumerate(res_id):
        # add good fits to the image
        if solution.has_good_energy_solution(res_id=res_id):
            row = rows[pixel_index]
            col = columns[pixel_index]
            for wave_index, _ in enumerate(wavelengths):
                R[wave_index, row, col] = R0[pixel_index, wave_index]
            R[-1, row, col] = 1
    R = np.transpose(R, (0, 2, 1))

    fig, ax = plt.subplots(figsize=(8, 8))
    image = ax.imshow(R[0])
    divider = make_axes_locatable(ax)
    width = axes_size.AxesY(ax, aspect=1. / 20)
    pad = axes_size.Fraction(0.5, width)
    cax = divider.append_axes("right", size=width, pad=pad)
    maximum = np.max(R)
    cbar_ticks = np.linspace(0., maximum, num=11)
    cbar = fig.colorbar(image, cax=cax, ticks=cbar_ticks)
    cbar.set_clim(vmin=0, vmax=maximum)
    cbar.draw_all()

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    position = ax.get_position()
    middle = position.x0 + 3 * position.width / 4
    ax_prev = plt.axes([middle - 0.18, 0.05, 0.15, 0.03])
    ax_next = plt.axes([middle + 0.02, 0.05, 0.15, 0.03])
    ax_slider = plt.axes([position.x0, 0.05, position.width / 2, 0.03])

    class Index(object):
        def __init__(self, ax_slider, ax_prev, ax_next):
            self.ind = 0
            self.num = len(wavelengths)
            self.bnext = Button(ax_next, 'Next')
            self.bnext.on_clicked(self.next)
            self.bprev = Button(ax_prev, 'Previous')
            self.bprev.on_clicked(self.prev)
            self.slider = Slider(ax_slider,
                                 "Energy Resolution: {:.2f} nm".format(wavelengths[0]), 0,
                                 self.num, valinit=0, valfmt='%d')
            self.slider.valtext.set_visible(False)
            self.slider.label.set_horizontalalignment('center')
            self.slider.on_changed(self.update)

            position = ax_slider.get_position()
            self.slider.label.set_position((0.5, -0.5))
            self.slider.valtext.set_position((0.5, -0.5))

        def next(self, event):
            i = (self.ind + 1) % (self.num + 1)
            self.slider.set_val(i)

        def prev(self, event):
            i = (self.ind - 1) % (self.num + 1)
            self.slider.set_val(i)

        def update(self, i):
            self.ind = int(i)
            image.set_data(R[self.ind])
            if self.ind != len(wavelengths):
                self.slider.label.set_text("Energy Resolution: {:.2f} nm"
                                           .format(wavelengths[self.ind]))
            else:
                self.slider.label.set_text("Calibrated Pixels")
            if self.ind != len(wavelengths):
                number = 11
                cbar.set_clim(vmin=0, vmax=maximum)
                cbar_ticks = np.linspace(0., maximum, num=number, endpoint=True)
            else:
                number = 2
                cbar.set_clim(vmin=0, vmax=1)
                cbar_ticks = np.linspace(0., 1, num=number)
            cbar.set_ticks(cbar_ticks)
            cbar.draw_all()
            plt.draw()

    indexer = Index(ax_slider, ax_prev, ax_next)
    plt.show(block=True)


def plot_summary(solution, config_name='', save_name=None, verbose=True):
    """
    Plot one page summary pdf of the wavelength calibration solution.

    Args:
        solution: the wavecal solution object
        config_name: the templar configuration file, including the path, associated with
                     the data (string)
        save_name: name of the pdf that's saved. No pdf is saved if set to None.
        verbose: determines whether information about loading the frequency files is
                 printed to the terminal (boolean)
    """

    wavelengths = solution.wavelengths
    obsFiles = solution.h5_files
    model_name = solution.model_name
    beam_map = solution.beam_map
    res_id = solution.histogram_res_ids
    R, _ = solution.resolving_powers()
    file_name = solution.file_name


    has_data = solution.histogram_has_data()
    fit_flags = solution.energy_fit_flag()
    text = np.zeros((3, 4))
    text[0, 0] = np.round(np.sum(solution.has_good_histogram_solution()) /
                          len(wavelengths), 2)
    text[0, 1] = np.round(text[0, 0] / beam_map.size * 100, 2)
    text[0, 2] = np.round(text[0, 0] / np.sum(res_id != 2**23 - 1) * 100, 2)
    text[0, 3] = np.round(text[0, 0] * len(wavelengths) /
                          np.sum(has_data) * 100, 2)

    text[1, 0] = np.sum(solution.has_good_energy_solution())
    text[1, 1] = np.round(text[1, 0] / beam_map.size * 100, 2)
    text[1, 2] = np.round(text[1, 0] / np.sum(res_id != 2**23 - 1) * 100, 2)
    text[1, 3] = np.round(text[1, 0] * len(wavelengths) /
                          np.sum(np.array(has_data) == 1) * 100, 2)

    text[2, 0] = np.sum(fit_flags == 5)
    text[2, 1] = np.sum(fit_flags == 4)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(6.95, 9))
    row_labels = ['histogram fits', 'energy solution', '(linear, quadratic)']
    col_labels = ["total pixels sucessful", "total \\% sucessful",
                  "total \\% out of read-out pixels",
                  "total \\% out of photosensitive pixels"]
    mpl.rcParams['text.latex.preamble'] = [r'\usepackage{array}']
    table = r"""\begin{{tabular}}{{>{{\raggedright}}p{{1.1in}} | """ + \
            r""" >{{\raggedright}}p{{1in}} | >{{\raggedright}}p{{1in}} | """ + \
            r""" >{{\raggedright}}p{{1in}} | p{{1.1in}}}}""" + \
            r""" & {0} & {1} & {2} & {3} \\""" + \
            r"""\hline {4} & {5} & {6} & {7} & {8} \\""" + \
            r"""{9} & {10} & {11} & {12} & {13} \\""" + \
            r"""{14} & ({15}, {16}) & - & - & - \end{{tabular}}"""
    table = table.format(col_labels[0], col_labels[1], col_labels[2], col_labels[3],
                         row_labels[0], text[0, 0], text[0, 1], text[0, 2], text[0, 3],
                         row_labels[1], text[1, 0], text[1, 1], text[1, 2], text[1, 3],
                         row_labels[2], text[2, 0], text[2, 1])

    axes[0, 0] = plot_R_histogram(solution, axis=axes[0, 0])
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
                            obsFiles[3 * ind] + ', ' +
                            obsFiles[3 * ind + 1] + ', ' +
                            obsFiles[3 * ind + 2], **fontstyle)
        if len(obsFiles) - (3 * ind + 2) == 3:
            axes[1, 1].text(-0.1, 0.97 - 6 * dz - (ind + 1) * dz,
                            obsFiles[3 * ind + 3] + ', ' +
                            obsFiles[3 * ind + 4], **fontstyle)
        elif len(obsFiles) - (3 * ind + 2) == 2:
            axes[1, 1].text(-0.1, 0.97 - 6 * dz - (ind + 1) * dz,
                            obsFiles[3 * ind + 3], **fontstyle)
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
        good_hist = np.sum(solution.has_good_histogram_solution(), axis=1)

        if np.sum(good_hist >= 3) > 0:
            perc = np.round(np.sum(fit_flags == 7) / np.sum(good_hist >= 3) * 100, 2)
        else:
            perc = 'N/A'
        axes[1, 1].text(-0.1, 0.97 - 8 * dz - 2 * np.ceil(len(obsFiles) / 3) * dz,
                        'Percent of pixels that failed the calibration because the'
                        ' \ncenters were not monotonic (out of pixels with \n> 2 good'
                        ' histogram fits): {0}%'.format(perc), va='top', **fontstyle)
    else:
        for ind in range(int(np.floor(len(obsFiles) / 2))):
            axes[1, 1].text(-0.1, 0.97 - 6 * dz - ind * dz,
                            obsFiles[2 * ind] + ', ' +
                            obsFiles[2 * ind + 1], **fontstyle)
        if len(obsFiles) - (2 * ind + 1) == 2:
            axes[1, 1].text(-0.1, 0.97 - 6 * dz - (ind + 1) * dz,
                            obsFiles[2 * ind + 2], **fontstyle)
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
        good_hist = np.sum(solution.has_good_histogram_solution(), axis=1)
        if np.sum(good_hist >= 3) > 0:
            perc = np.round(np.sum(fit_flags == 7) / np.sum(good_hist >= 3) * 100, 2)
        else:
            perc = 'N/A'
        axes[1, 1].text(-0.1, 0.97 - 8 * dz - 2 * np.ceil(len(obsFiles) / 2) * dz,
                        'Percent of pixels that failed the \ncalibration because the' +
                        ' centers were not \nmonotonic (out of pixels with > 2 good' +
                        ' \nhistogram fits): {0}%'.format(perc), va='top', **fontstyle)

    if not switch:
        mpl.rc('text', usetex=True)

    axes[1, 0] = plot_center_histogram(solution, axis=axes[1, 0])
    axes[0, 1] = plot_R_vs_F(solution, config_name, axis=axes[0, 1], verbose=verbose)

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
    """
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
    """
    #TODO Move this elsewhere if it is general and make part of a templar config file class
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
    """
    Returns the specified fit model from a library of possible functions
    """
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
    else:
        raise ValueError("{} is not a valid fit model".format(model_name))

    return fit_function
