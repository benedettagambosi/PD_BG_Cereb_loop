import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq
from scipy import signal
from nest_utils import utils
from fooof import FOOOF
from scipy.ndimage import gaussian_filter
import seaborn as sns


import pickle as p
import matplotlib.pyplot as plt
import numpy as np
import math

''' Plot membrane potential '''
iter = 101
palette = list(reversed(sns.color_palette("viridis", iter).as_hex()))
sm = plt.cm.ScalarMappable(cmap="viridis_r", norm=plt.Normalize(vmin=0, vmax=iter))



''' Plot a raster plot '''


def raster_plot(times, neurons_idx, compartment_name, ax_plt=None, start_stop_times=None, t_start=0., n_joints=None, ctx_in=None):
    if ax_plt is None:
        fig, ax = plt.subplots()  # create a new figure
    else:
        ax = ax_plt
    if start_stop_times is not None:
        scale_xy_axes(ax, xlim=start_stop_times)  # x range
    if len(neurons_idx) > 0:  # is there are spikes
        if compartment_name == 'dcn': compartment_name = 'dcnp'
        elif compartment_name == 'dcnp': compartment_name = 'dcni'

        if compartment_name == 'IO':
            if n_joints is not None:
                if n_joints == 1:
                    neg_idxs = [n in range(6373, 6409) for n in neurons_idx]
                    pos_idxs = [n in range(6409, 6445) for n in neurons_idx]
                elif n_joints == 2:
                    neg_idxs = [n in list(range(6373, 6391)) + list(range(6409, 6427)) for n in neurons_idx]
                    pos_idxs = [n in list(range(6391, 6409)) + list(range(6427, 6445)) for n in neurons_idx]
                ax.scatter(times[neg_idxs], neurons_idx[neg_idxs], c='tab:orange', s=4)
                ax.scatter(times[pos_idxs], neurons_idx[pos_idxs], c='tab:blue', s=4)
        elif compartment_name == 'io':
            neg_idxs = [n in range(96756, 96762) for n in neurons_idx]
            ax.scatter(times[neg_idxs], neurons_idx[neg_idxs], c='tab:orange', s=4)
            pos_idxs = [n in range(96762, 96768) for n in neurons_idx]
            ax.scatter(times[pos_idxs], neurons_idx[pos_idxs], c='tab:blue', s=4)


        elif compartment_name == 'DCN':
            if n_joints is not None:
                if n_joints == 1:
                    neg_idxs = [n in range(6445, 6463) for n in neurons_idx]
                    pos_idxs = [n in range(6463, 6481) for n in neurons_idx]
                elif n_joints == 2:
                    neg_idxs = [n in list(range(6445, 6454)) + list(range(6463, 6472)) for n in neurons_idx]
                    pos_idxs = [n in list(range(6454, 6463)) + list(range(6472, 6481)) for n in neurons_idx]
                ax.scatter(times[neg_idxs], neurons_idx[neg_idxs], c='tab:orange', s=4)
                ax.scatter(times[pos_idxs], neurons_idx[pos_idxs], c='tab:blue', s=4)
        # print(times)
        elif compartment_name == 'dcn old':
            neg_idxs = [n in [96735, 96736, 96737, 96741, 96742, 96743] for n in neurons_idx]
            ax.scatter(times[neg_idxs], neurons_idx[neg_idxs], c='tab:orange', s=4)
            pos_idxs = [n in [96732, 96733, 96734, 96738, 96739, 96740] for n in neurons_idx]
            ax.scatter(times[pos_idxs], neurons_idx[pos_idxs], c='tab:blue', s=4)
        # print(times)
        elif compartment_name == 'MF' and ctx_in is not None:
            ax.scatter(times, neurons_idx, c='tab:blue', s=4)
            ax.plot(np.linspace(0, len(ctx_in)*10, len(ctx_in)), min(neurons_idx) + ctx_in / max(ctx_in) * (max(neurons_idx) - min(neurons_idx)), c='tab:orange', linewidth=2., alpha=0.8)
        else:
            ax.scatter(times, neurons_idx, c='tab:blue', s=4)
            if compartment_name == 'purkinje':
                mylist = sorted(neurons_idx)
                mylist2 = list(dict.fromkeys(mylist))
                occ = []
                for key in mylist2:
                    occr = mylist.count(key)
                    occ += [occr]
                # print(compartment_name, len(mylist), mylist)
        max_idx = max(neurons_idx)
        min_idx = min(neurons_idx)
        if max_idx - min_idx >= 4:
            scale_xy_axes(ax, ylim=[min_idx, max_idx])  # y range
            number_of_ticks = 5  # with min and max too
            yticks = [min_idx + int((max_idx - min_idx) * i / (number_of_ticks - 1)) for i in range(number_of_ticks)]
        else:
            yticks = list(dict.fromkeys(neurons_idx))  # select elements without repets
        ax.set_yticks(yticks)
    else:
        scale_xy_axes(ax, ylim=[0, 1])
        ax.set_yticks([0, 1])
    ax.axvline(x=t_start, color='tab:red')
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel(f'{compartment_name} neuron idx')
    # ax.set_title(f"{compartment_name} raster plot")
    if ax_plt is None:
        res = fig, ax
    else:
        res = ax
    return res


''' Plot a multiple rasterplot in 2 columns'''


def raster_plots_multiple(rasters_list, start_stop_times=None, clms=2, t_start=0., n_joints=None,  ctx_in=None):
    keywords = {'x_data': 'times', 'y_data': 'neurons_idx', 'name': 'compartment_name'}
    fig, axs = multiple_plots(rasters_list, raster_plot, keywords, clms=clms, start_stop_times=start_stop_times,
                              t_start=t_start, n_joints=n_joints,  ctx_in=ctx_in)
    return fig, axs


''' UTILS '''

''' Scale x and y axes '''


def scale_xy_axes(ax, xlim=None, ylim=None):
    if xlim is not None:
        if xlim != [None, None]:
            range_norm = xlim[1] - xlim[0]
            border = range_norm * 5 / 100  # leave a 5% of blank space at borders
            ax.set_xlim(xlim[0] - border, xlim[1] + border)
    if ylim is not None:
        if ylim != [None, None]:
            range_norm = ylim[1] - ylim[0]
            border = range_norm * 5 / 100  # leave a 5% of blank space at borders
            ax.set_ylim(ylim[0] - border, ylim[1] + border)


''' Multiple plots framework '''


def multiple_plots(plot_list, plot_function, keywords, clms=2, ext_axs=None, **kwargs):
    n_plots = len(plot_list)
    rows = int(np.ceil(n_plots / clms))
    fig_width = 8.0
    plot_height = 3.0
    if ext_axs is None:
        fig, axs = plt.subplots(rows, clms, figsize=(fig_width * clms, plot_height * rows))
    else:
        axs = ext_axs  # if provided, use external axes
    if n_plots % clms != 0:  # some empty plots at the and
        for idx in range(clms - 1):
            if ext_axs is None:
                fig.delaxes(axs[rows - 1, clms - 1 - idx])  # delete last plots
    for i, plot in enumerate(plot_list):
        if clms == 1:
            index = i
        else:
            index = int(i / clms), i % clms  # if not 1 column, manage 2 indexes
        ax = plot_function(plot[keywords['x_data']], plot[keywords['y_data']], plot[keywords['name']],
                           ax_plt=axs[index], **kwargs)
        for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
            item.set_fontsize(plot_height * 5)
        for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(plot_height * 4)
        if (n_plots - 1 - i) >= clms:  # not the last #clms plots
            ax.set_xlabel(None)
        # if int(i % clms) != 0:          # not the first column
        #     ax.set_ylabel(None)
    if ext_axs is None:
        fig.tight_layout()
        return fig, axs
    else:
        ax.set_xlabel(None)
        return axs


def plot_my_histogram(ax, width, value_list, name_list, target_list, y_label):
    x = list(range(len(value_list)))
    t = []
    tval = []

    if target_list is not None:
        for idx, val in enumerate(target_list):
            if val != 0:
                t = t + [x[idx] + width]  # place it the right
                tval = tval + [np.round(val, 2)]

    ax.set_xticks(np.array(x))
    ax.set_xticklabels(name_list)

    width = width * 0.9
    bars1 = ax.bar(x, value_list, width, label='Simulation')
    ax.bar_label(bars1, rotation=60)

    if target_list is not None:
        bars2 = ax.bar(t, tval, width, label='Reference')
        ax.bar_label(bars2, rotation=60)

    if target_list is not None:
        ax.set_ylim(0, max(np.max(value_list), np.max(target_list)) * 1.4)
    else:
        ax.set_ylim(0, np.max(value_list) * 1.4)

    ax.legend(loc='upper right')   # , bbox_to_anchor=(0.14, 1.0))

    ax.set_ylabel(y_label)

    ax.tick_params(bottom=False)


def firing_rate_histogram(fr_list, name_list, CV_list=None, target_fr=None, target_CV=None):
    ''' Bar plot of the avarage firing rates of the population rasters '''

    fig_width = 6.0
    width = 0.4  # columns width
    plot_height = 2.5

    if CV_list == None:
        rows = 1  # one for fr
    else:
        rows = 2  # one for fr and onw fo CV
    fig, axes = plt.subplots(rows, 1, figsize=(fig_width, plot_height * rows))
    if rows == 1: axes = [axes]

    # plot afr
    plot_my_histogram(axes[0], width, fr_list, name_list, target_fr, 'Firing rate [Hz]')

    if CV_list != None:
        # plot CV
        plot_my_histogram(axes[1], width, CV_list, name_list, target_CV, 'CV []')

    fig.suptitle(f'Average firing rate')  # in mode: {mode}')

    fig.tight_layout()

    return fig, axes


def plot_fourier_transform(fr_array, T_sample, legend_labels, mean=None, sd=None, t_start=0.):
    """ Plot the discrete fourier transform of the mass models """
    fig_width = 9.
    plot_height = 3.5
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, plot_height))

    ax = axes[0]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][1:]  # don't use blue color as in Cereb
    ax.set_prop_cycle(color=colors)

    T = T_sample / 10.
    y = fr_array[int(t_start / T):, :]  # calculate tf after the t_start

    T = T_sample  # resample to 1 ms
    y = y[::10]  # select one time sample every 10
    N = y.shape[0]

    yf = fft(y, axis=0)
    xf = fftfreq(N, d=T/1000)[:N // 2]     # take just pos freq
    y_plot = 2.0 / N * np.abs(yf[0:N // 2])

    freq_p = np.linspace(-10, 10, 21, endpoint=True)
    kernel = utils.gaussian(freq_p, 0., 2.)
    for i in range(yf.shape[1]):
        y_plot[:, i] = np.convolve(y_plot[:, i], kernel, 'same')

    fourier_idx = utils.calculate_fourier_idx(xf, [1, 60])
    # print(f'In plot: considering frequencies in the range {[xf[fourier_idx[0]], xf[fourier_idx[1]-1]]}')

    fm_list = []
    ax.set_prop_cycle(color=colors)
    peak_width_limits = [[2, 8], [2, 8], [2, 12]]
    for idx in range(len(y_plot[0, :])):
        fm = FOOOF(peak_width_limits=peak_width_limits[idx])  # aperiodic_mode='knee')
        fm.fit(xf[fourier_idx[0]:fourier_idx[1]], y_plot[:, idx][fourier_idx[0]:fourier_idx[1]])
        # ax.plot(freq[wavelet_idx[0]:wavelet_idx[1]], np.log10(y_val[:, idx]), label=legend_labels[idx], alpha=0.5)
        # diff = np.log10(y_plot[:, idx][fourier_idx[0]:fourier_idx[1]]) - np.log10(
        #     1 / xf[fourier_idx[0]:fourier_idx[1]] ** fm.aperiodic_params_[1]) - \
        #        fm.aperiodic_params_[0]
        diff = fm.fooofed_spectrum_ - np.log10(
            1 / xf[fourier_idx[0]:fourier_idx[1]] ** fm.aperiodic_params_[1]) - \
               fm.aperiodic_params_[0]
        ax.plot(xf[fourier_idx[0]:fourier_idx[1]], diff, label=legend_labels[idx])
        fm_list += [fm]
        # fm.report(freq[wavelet_idx[0]:wavelet_idx[1]], y_val[:, idx], [1, 60])

    # ax.set_prop_cycle(color=colors)
    # for idx in range(len(y_plot[0, :])):
    #     ax.plot(xf[0], np.log10(y_plot[:, idx][0]), alpha=1, label=legend_labels[idx])
    #
    # ax.set_prop_cycle(color=colors)
    # for idx in range(len(y_plot[0, :])):
    #     ax.plot(xf[fourier_idx[0]:fourier_idx[1]], np.log10(y_plot[:, idx][fourier_idx[0]:fourier_idx[1]]), alpha=.5) # , label=legend_labels[idx])
    #
    #
    # ax.set_prop_cycle(color=colors)
    # for idx in range(len(y_plot[0, :])):
    #     ax.plot(xf[fourier_idx[0]:fourier_idx[1]], fm_list[idx].fooofed_spectrum_, '-.') # , label=f'{legend_labels[idx]} fooof')
    #
    # ax.plot(xf[fourier_idx[0]:fourier_idx[1]], np.log10(1/xf[fourier_idx[0]:fourier_idx[1]]**fm_list[0].aperiodic_params_[1]) + fm_list[idx].aperiodic_params_[0], ':', c='tab:grey', label='aperiodic') # , label=f'{legend_labels[idx]} aperiodic')
    #
    # ax.set_prop_cycle(color=colors)
    # for idx in range(len(y_plot[0, :])):
    #     ax.plot(xf[fourier_idx[0]:fourier_idx[1]], np.log10(1/xf[fourier_idx[0]:fourier_idx[1]]**fm_list[idx].aperiodic_params_[1]) + fm_list[idx].aperiodic_params_[0], ':') # , label=f'{legend_labels[idx]} aperiodic')

    if mean is not None:
        ax.axvspan(mean - sd, mean + sd, alpha=0.5, color='tab:blue')

    ax.grid(linestyle='-.')
    # scale_xy_axes(ax, ylim=[0., 0.9])
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel(f'Log Power [Activity^2]')

    ax.set_title('Fourier transform')
    ax.legend(legend_labels, fontsize=14) # loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)

    for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
        item.set_fontsize(12)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(10)

    ax = axes[1]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][1:]  # don't use blue color as in Cereb
    ax.set_prop_cycle(color=colors)

    T = T_sample / 10.
    y = fr_array[int(t_start / T):, :]  # calculate tf after the t_start

    T = T_sample  # resample to 1 ms
    y = y[::10]  # select one time sample every 10

    fs = 1000. / T  # [Hz], sampling time
    w = 15.  # [adim], "omega0", in the definition of Morlet wavelet: pi**-0.25 * exp(1j*w*x) * exp(-0.5*(x**2))
    freq = np.linspace(1, fs / 2, 2 * int(fs / 2 - 1) + 1)  # frequency range, until half fs
    widths = w * fs / (
                2 * freq * np.pi)  # [adim] reduce time widths for higher frequencies. Widhts / sample_freq = time

    y_plot = np.zeros((len(freq), y.shape[1]))
    for idx in range(y.shape[1]):
        cwtm = signal.cwt(y[:, idx], signal.morlet2, widths, w=w)
        y_plot[:, idx] = np.abs(cwtm).sum(axis=1)

    wavelet_idx = utils.calculate_fourier_idx(freq, [1, 60])
    # print(f'In plot: considering frequencies in the range {[freq[wavelet_idx[0]], freq[wavelet_idx[1]-1]]}')

    y_val = y_plot[wavelet_idx[0]:wavelet_idx[1], :]

    fm_list = []
    ax.set_prop_cycle(color=colors)
    peak_width_limits = [[2, 8], [2, 8], [3, 20]]

    if mean is not None:
        ax.axvspan(mean - sd, mean + sd, alpha=0.5, color='tab:blue')

    diff = np.zeros(y_val.shape)
    for idx in range(len(y_val[0, :])):
        fm = FOOOF(peak_width_limits=peak_width_limits[idx])  # aperiodic_mode='knee')
        fm.fit(freq[wavelet_idx[0]:wavelet_idx[1]], y_val[:, idx])
        # diff[:, idx] = np.log10(y_val[:, idx]) - np.log10(1/freq[wavelet_idx[0]:wavelet_idx[1]]**fm.aperiodic_params_[1]) - fm.aperiodic_params_[0]
        diff[:, idx] = fm.fooofed_spectrum_ - np.log10(1/freq[wavelet_idx[0]:wavelet_idx[1]]**fm.aperiodic_params_[1]) - fm.aperiodic_params_[0]
        ax.plot(freq[wavelet_idx[0]:wavelet_idx[1]], diff[:, idx], label=legend_labels[idx])
        fm_list += [fm]
        # fm.report(freq[wavelet_idx[0]:wavelet_idx[1]], y_val[:, idx], [1, 60])

    # ax.set_prop_cycle(color=colors)
    # for idx in range(len(y_val[0, :])):
    #     ax.plot(freq[wavelet_idx[0]:wavelet_idx[1]], np.log10(y_val[:, idx]), alpha=.5, label=legend_labels[idx])
    #
    # # ax.plot(freq[wavelet_idx[0]:wavelet_idx[1]], fm_list[idx].fooofed_spectrum_, '-.', c='tab:grey',
    # #         label='fooof interp')  # , label=f'{legend_labels[idx]} fooof')
    #
    # ax.set_prop_cycle(color=colors)
    # for idx in range(len(y_val[0, :])):
    #     ax.plot(freq[wavelet_idx[0]:wavelet_idx[1]], fm_list[idx].fooofed_spectrum_,
    #             '-.')  # , label=f'{legend_labels[idx]} fooof')
    #
    # ax.plot(freq[wavelet_idx[0]:wavelet_idx[1]],
    #         np.log10(1 / freq[wavelet_idx[0]:wavelet_idx[1]] ** fm_list[idx].aperiodic_params_[1]) +
    #         fm_list[idx].aperiodic_params_[0], ':', c='tab:grey',
    #         label='aperiodic')  # , label=f'{legend_labels[idx]} aperiodic')
    #
    # ax.set_prop_cycle(color=colors)
    # for idx in range(len(y_val[0, :])):
    #     ax.plot(freq[wavelet_idx[0]:wavelet_idx[1]],
    #             np.log10(1 / freq[wavelet_idx[0]:wavelet_idx[1]] ** fm_list[idx].aperiodic_params_[1]) +
    #             fm_list[idx].aperiodic_params_[0], ':')  # , label=f'{legend_labels[idx]} aperiodic')

    ax.grid(linestyle='-.')
    # scale_xy_axes(ax, ylim=[0., 0.58])
    ax.set_xlabel('Frequency [Hz]')
    # ax.set_ylabel(f'Log Power [Activity^2]')

    # ax.legend(fontsize=16)  # loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)
    ax.set_title('Wavelet transform')
    ax.legend(fontsize=14) # loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)

    for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
        item.set_fontsize(12)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(10)

    fig.tight_layout()




    return fig, ax


def plot_wavelet_transform(mass_models_sol, T_sample, legend_labels, mean=None, sd=None, t_start=0., y_range=None, dopa_depl=None,
                           times_key='mass_frs_times', data_key="mass_frs"):
    """ Plot the discrete fourier transform of the mass models """
    fig_width = 6.
    plot_height = 4.
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, plot_height))

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][1:]  # don't use blue color as in Cereb
    ax.set_prop_cycle(color=colors)

    if not isinstance(mass_models_sol, list):
        mass_models_sol = [mass_models_sol]

    T = mass_models_sol[0][times_key][1] - mass_models_sol[0][times_key][0]
    sel_mass_times1 = mass_models_sol[0][times_key] > t_start
    sel_mass_times2 = mass_models_sol[0][times_key] % T_sample == 0
    y_list = [mms[data_key][np.logical_and(sel_mass_times1, sel_mass_times2), :] for mms in mass_models_sol]  # calculate tf after the t_start

    # T = T_sample    # resample to 1 ms
    # y = y[::10]     # select one time sample every 10

    fs = 1000./T_sample    # [Hz], sampling time
    w = 15.         # [adim], "omega0", in the definition of Morlet wavelet: pi**-0.25 * exp(1j*w*x) * exp(-0.5*(x**2))
    freq = np.linspace(1, fs / 2, 2*int(fs/2-1)-1)      # frequency range, until half fs
    widths = w * fs / (2 * freq * np.pi)    # [adim] reduce time widths for higher frequencies. Widhts / sample_freq = time

    y_plot_list = [np.zeros((len(freq), y.shape[1])) for y in y_list]
    for y, y_plot in zip(y_list, y_plot_list):
        for idx in range(y.shape[1]):
            cwtm = signal.cwt(y[:, idx], signal.morlet2, widths, w=w)
            y_plot[:, idx] = np.abs(cwtm).sum(axis=1)

    wavelet_idx = utils.calculate_fourier_idx(freq, [1, 60])
    # print(f'In plot: considering frequencies in the range {[freq[wavelet_idx[0]], freq[wavelet_idx[1]-1]]}')

    y_val_list = [y_plot[wavelet_idx[0]:wavelet_idx[1], :] for y_plot in y_plot_list]

    fm_list = []
    ax.set_prop_cycle(color=colors)
    # peak_width_limits = [[1, 8], [1, 8], [1, 8]]
    peak_width_limits = [[2, 8], [2, 8], [2, 8]]

    if mean is not None:
        ax.axvspan(mean - sd, mean + sd, alpha=0.5, color='tab:blue')

    diff_list = [np.zeros(y_val.shape) for y_val in y_val_list]
    fm_list_list = []
    for y_val, diff in zip(y_val_list, diff_list):
        fm_list = []
        for idx in range(len(y_val[0, :])):
            fm = FOOOF(peak_width_limits=peak_width_limits[idx])  # aperiodic_mode='knee')
            fm.fit(freq[wavelet_idx[0]:wavelet_idx[1]], y_val[:, idx])
            fm_list += [fm]
            # diff[:, idx] = np.log10(y_val[:, idx]) - np.log10(1/freq[wavelet_idx[0]:wavelet_idx[1]]**fm.aperiodic_params_[1]) - fm.aperiodic_params_[0]
            diff[:, idx] = fm.fooofed_spectrum_ - np.log10(1/freq[wavelet_idx[0]:wavelet_idx[1]]**fm.aperiodic_params_[1]) - fm.aperiodic_params_[0]
        fm_list_list += [fm_list]

    diff_array = np.array(diff_list)
    diff_mean = diff_array.mean(axis=0)
    diff_std = diff_array.std(axis=0)
    for idx in range(diff_mean.shape[1]):
        ax.fill_between(freq[wavelet_idx[0]:wavelet_idx[1]], diff_mean[:, idx] - diff_std[:, idx], diff_mean[:, idx] + diff_std[:, idx], alpha=0.5)
        ax.plot(freq[wavelet_idx[0]:wavelet_idx[1]], diff_mean[:, idx], label=legend_labels[idx])

    # fm.report(freq[wavelet_idx[0]:wavelet_idx[1]], y_val[:, idx], [1, 60])

    # if mean is not None:
    #     ax.axvspan(mean - sd, mean + sd, alpha=0.5, color='tab:blue')
    #
    # ax.set_prop_cycle(color=colors)
    # for idx in range(len(y_val_list[0][0, :])):
    #     ax.plot(freq[wavelet_idx[0]:wavelet_idx[1]], np.log10(np.array(y_val_list).mean(axis=0)[:, idx]), alpha=.5, label=legend_labels[idx])
    #
    #     fooofed = [fm_list_list[k][idx].fooofed_spectrum_ for k in range(len(mass_models_sol))]
    #     ax.plot(freq[wavelet_idx[0]:wavelet_idx[1]], np.array(fooofed).mean(axis=0), '-.', c='tab:grey', label='fooof interp') # , label=f'{legend_labels[idx]} fooof')

    # ax.set_prop_cycle(color=colors)
    # for idx in range(len(y_val[0, :])):
    #     ax.plot(freq[wavelet_idx[0]:wavelet_idx[1]], fm_list[idx].fooofed_spectrum_, '-.') # , label=f'{legend_labels[idx]} fooof')
    #
    # ax.plot(freq[wavelet_idx[0]:wavelet_idx[1]], np.log10(1/freq[wavelet_idx[0]:wavelet_idx[1]]**fm_list[idx].aperiodic_params_[1]) + fm_list[idx].aperiodic_params_[0], ':', c='tab:grey', label='aperiodic') # , label=f'{legend_labels[idx]} aperiodic')
    #
    # ax.set_prop_cycle(color=colors)
    # for idx in range(len(y_val[0, :])):
    #     ax.plot(freq[wavelet_idx[0]:wavelet_idx[1]], np.log10(1/freq[wavelet_idx[0]:wavelet_idx[1]]**fm_list[idx].aperiodic_params_[1]) + fm_list[idx].aperiodic_params_[0], ':') # , label=f'{legend_labels[idx]} aperiodic')

    ax.grid(linestyle='-.')
    scale_xy_axes(ax, ylim=[-0.0, 1.1])
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel(f'Log Power [Activity^2]')

    ax.legend(fontsize=12) # loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)
    ax.set_title(f'Wavelet transform with dopa. depl. = {dopa_depl}')

    for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
        item.set_fontsize(12)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(10)

    fig.tight_layout()


    return fig, ax, diff_mean


def plot_wavelet_transform_and_mass(mass_models_sol, T_sample, legend_labels, mean=None, sd=None, t_start=0., t_end=0., y_range=None):
    """ Plot the discrete fourier transform of the mass models """
    fig_width = 10.
    plot_height = 3.
    # fig, ax = plt.subplots(1, 1, figsize=(fig_width, plot_height))
    fig, axes = plt.subplots(1, 2, figsize=(fig_width, plot_height))

    ax = axes[0]
    plot_mass_frs(mass_models_sol, legend_labels, ext_ax=ax)

    for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
        item.set_fontsize(12)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(10)

    ax = axes[1]

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][1:]  # don't use blue color as in Cereb
    ax.set_prop_cycle(color=colors)

    T = T_sample / 10.
    sel_mass_times = mass_models_sol["mass_frs_times"] > t_start
    y = mass_models_sol["mass_frs"][sel_mass_times, :]  # calculate tf after the t_start

    T = T_sample    # resample to 1 ms
    y = y[::10]     # select one time sample every 10

    fs = 1000./T    # [Hz], sampling time
    w = 15.         # [adim], "omega0", in the definition of Morlet wavelet: pi**-0.25 * exp(1j*w*x) * exp(-0.5*(x**2))
    freq = np.linspace(1, fs / 2, 2*int(fs/2-1)+1)      # frequency range, until half fs
    widths = w * fs / (2 * freq * np.pi)    # [adim] reduce time widths for higher frequencies. Widhts / sample_freq = time

    y_plot = np.zeros((len(freq), y.shape[1]))
    for idx in range(y.shape[1]):
        cwtm = signal.cwt(y[:, idx], signal.morlet2, widths, w=w)
        y_plot[:, idx] = np.abs(cwtm).sum(axis=1)

    wavelet_idx = utils.calculate_fourier_idx(freq, [1, 60])
    # print(f'In plot: considering frequencies in the range {[freq[wavelet_idx[0]], freq[wavelet_idx[1]-1]]}')

    y_val = y_plot[wavelet_idx[0]:wavelet_idx[1], :]

    fm_list = []
    ax.set_prop_cycle(color=colors)
    peak_width_limits = [[2, 8], [2, 8], [3, 20]]

    if mean is not None:
        ax.axvspan(mean - sd, mean + sd, alpha=0.5, color='tab:blue')

    diff = np.zeros(y_val.shape)
    for idx in range(len(y_val[0, :])):
        fm = FOOOF(peak_width_limits=peak_width_limits[idx])  # aperiodic_mode='knee')
        fm.fit(freq[wavelet_idx[0]:wavelet_idx[1]], y_val[:, idx])
        # diff[:, idx] = np.log10(y_val[:, idx]) - np.log10(1/freq[wavelet_idx[0]:wavelet_idx[1]]**fm.aperiodic_params_[1]) - fm.aperiodic_params_[0]
        diff[:, idx] = fm.fooofed_spectrum_ - np.log10(1/freq[wavelet_idx[0]:wavelet_idx[1]]**fm.aperiodic_params_[1]) - fm.aperiodic_params_[0]
        ax.plot(freq[wavelet_idx[0]:wavelet_idx[1]], diff[:, idx], label=legend_labels[idx])
        fm_list += [fm]
        # fm.report(freq[wavelet_idx[0]:wavelet_idx[1]], y_val[:, idx], [1, 60])

    # if mean is not None:
    #     ax.axvspan(mean - sd, mean + sd, alpha=0.5, color='tab:blue')
    #
    # ax.set_prop_cycle(color=colors)
    # for idx in range(len(y_val[0, :])):
    #     ax.plot(freq[wavelet_idx[0]:wavelet_idx[1]], np.log10(y_val[:, idx]), alpha=.5, label=legend_labels[idx])
    #
    # ax.plot(freq[wavelet_idx[0]:wavelet_idx[1]], fm_list[idx].fooofed_spectrum_, '-.', c='tab:grey', label='fooof interp') # , label=f'{legend_labels[idx]} fooof')
    #
    # ax.set_prop_cycle(color=colors)
    # for idx in range(len(y_val[0, :])):
    #     ax.plot(freq[wavelet_idx[0]:wavelet_idx[1]], fm_list[idx].fooofed_spectrum_, '-.') # , label=f'{legend_labels[idx]} fooof')
    #
    # ax.plot(freq[wavelet_idx[0]:wavelet_idx[1]], np.log10(1/freq[wavelet_idx[0]:wavelet_idx[1]]**fm_list[idx].aperiodic_params_[1]) + fm_list[idx].aperiodic_params_[0], ':', c='tab:grey', label='aperiodic') # , label=f'{legend_labels[idx]} aperiodic')
    #
    # ax.set_prop_cycle(color=colors)
    # for idx in range(len(y_val[0, :])):
    #     ax.plot(freq[wavelet_idx[0]:wavelet_idx[1]], np.log10(1/freq[wavelet_idx[0]:wavelet_idx[1]]**fm_list[idx].aperiodic_params_[1]) + fm_list[idx].aperiodic_params_[0], ':') # , label=f'{legend_labels[idx]} aperiodic')

    ax.grid(linestyle='-.')
    # scale_xy_axes(ax, ylim=[-0.2, 0.7])
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel(f'Log Power [Activity^2]')

    ax.legend(fontsize=12) # loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)
    ax.set_title('b) Wavelet transform with dopa. depl. = -0.4')

    for item in [ax.title, ax.xaxis.label, ax.yaxis.label]:
        item.set_fontsize(12)
    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(10)

    fig.tight_layout()


    return fig, ax, diff

def plot_mass_frs(mass_models_sol, legend_labels, u_array=False, xlim=None, ylim=None, ext_ax=None, title=None):
    """ Plot the equivalent firing rate of the mass models """
    fig_width = 8.0
    plot_height = 4.0
    if ext_ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(fig_width, plot_height))
    else:
        ax = ext_ax

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][1:]  # don't use blue color as in Cereb
    ax.set_prop_cycle(color=colors)

    # fr_array = fr_array[15000:]
    ax.grid(linestyle='-.')
    ax.plot(mass_models_sol['mass_frs_times'], mass_models_sol['mass_frs'])
    if u_array:
        ax.plot(mass_models_sol['mass_frs_times'], mass_models_sol['in_frs'])
    scale_xy_axes(ax, xlim, ylim)
    ax.set_xlabel('Time [ms]')
    ax.set_ylabel(f'Activity [sp/(neu*s)]')

    if title is not None:
        ax.set_title(f'{title}')

    # print(f'Final mass f.r. value = {fr_array[-1, :]}')
    if ext_ax is None:
        ax.legend(legend_labels, fontsize=16) # loc='center left', bbox_to_anchor=(1, 0.5), fontsize=16)
        fig.tight_layout()
        return fig, ax
    else:
        ax.set_title(f'a) Average activity with dopa. depl. = -0.4')
        # ax.legend(legend_labels, fontsize=16)
        return ax

def sdf(cell, trial, rster,trials_start,burst_dur):
    for i in range(len(rster)):
        if rster[i]["compartment_name"] == cell:
            pass
            ids = rster[i]["neurons_idx"].reshape(-1,1)
            times = rster[i]["times"].reshape(-1,1)
            spk = np.concatenate((ids, times), axis=1)

    if cell == 'dcn':
        g_size = 10
    else:
        g_size = 20

    if cell=="glomerulus":
        neurons = np.unique(spk[:,0])[:200]
    else:
        neurons = np.unique(spk[:,0])

    spk_first = spk[(spk[:,1]>=trials_start[trial]-50) & (spk[:,1]<trials_start[trial]+burst_dur+50)]
    spk_first[:,1] -= trials_start[trial]-50
    dur = burst_dur+100

    sdf_full = np.empty([len(neurons),int(dur)])
    sdf = []
    for neu in range(len(neurons)):
        spike_times_first = spk_first[spk_first[:,0]==neurons[neu],1]
        for t in range(int(dur)):
            tau_first = t-spike_times_first
            sdf_full[neu,t] = sum(1/(math.sqrt(2*math.pi)*g_size)*np.exp(-np.power(tau_first,2)/(2*(g_size**2))))*(10**3)

        sdf.append(sdf_full[neu])#[50:330])

    return(sdf)

def sdf_baseline( cell, rster,trials_start,burst_dur, between_start):
    for i in range(len(rster)):
        if rster[i]["compartment_name"] == cell:
            pass
            ids = rster[i]["neurons_idx"].reshape(-1,1)
            times = rster[i]["times"].reshape(-1,1)
            spk = np.concatenate((ids, times), axis=1)

    if cell == 'dcn':
        g_size = 10
    else:
        g_size = 20

    neurons = np.unique(spk[:,0])

    spk_first = spk[(spk[:,1]>trials_start[0]+burst_dur) & (spk[:,1]<=trials_start[1])]
    spk_first[:,1] -= trials_start[0]+burst_dur

    sdf = np.empty([len(neurons),int(between_start-burst_dur)])

    for neu in range(len(neurons)):
        spike_times_first = spk_first[spk_first[:,0]==neurons[neu],1]
        for t in range(int(between_start-burst_dur)):
            tau_first = t-spike_times_first
            sdf[neu,t] = sum(1/(math.sqrt(2*math.pi)*g_size)*np.exp(-np.power(tau_first,2)/(2*(g_size**2))))*(10**3)
    sdf = np.mean(sdf, axis=1)

    return(sdf)

def sdf_baseline_trial(cell, trial, rster,trials_start,burst_dur, between_start):
    for i in range(len(rster)):
        if rster[i]["compartment_name"] == cell:
            pass
            ids = rster[i]["neurons_idx"].reshape(-1,1)
            times = rster[i]["times"].reshape(-1,1)
            spk = np.concatenate((ids, times), axis=1)

    if cell == 'dcn':
        g_size = 10
    else:
        g_size = 20

    neurons = np.unique(spk[:,0])

    spk_first = spk[(spk[:,1]>trials_start[trial]+burst_dur) & (spk[:,1]<=trials_start[trial])]
    spk_first[:,1] -= trials_start[trial]+burst_dur

    sdf = np.empty([len(neurons),int(between_start-burst_dur)])

    for neu in range(len(neurons)):
        spike_times_first = spk_first[spk_first[:,0]==neurons[neu],1]
        for t in range(int(between_start-burst_dur)):
            tau_first = t-spike_times_first
            sdf[neu,t] = sum(1/(math.sqrt(2*math.pi)*g_size)*np.exp(-np.power(tau_first,2)/(2*(g_size**2))))*(10**3)
    sdf = np.mean(sdf, axis=1)

    return(sdf)

def sdf_mean(sdf):
    sdf_mean = np.mean(sdf, axis=0)

    return(sdf_mean)

def sdf_maf(sdf,  step):
    sdf_maf = np.convolve(sdf_mean(sdf), np.ones(step), 'valid') / step
    return(sdf_maf)

def plot_cell_sdf(cell, selected_trials, maf_step):
    sdf_mean_cell = []
    sdf_maf_cell = []
    for trial in selected_trials:
        sdf_cell = sdf(cell, trial)
        sdf_mean_cell.append(sdf_mean(sdf_cell))
        sdf_maf_cell.append(sdf_maf(sdf_cell, maf_step))

    fig = plt.figure()
    for i in selected_trials[0:-1]:
        plt.plot(sdf_mean_cell[i], palette[i])
    plt.title("Second half of " + cell +" pop")
    plt.title(cell)
    plt.xlabel("Time [ms]")
    plt.ylabel("SDF [Hz]")
    plt.axvline(50, label = "CS start", c = "grey")
    plt.axvline(300, label = "US start", c = "black")
    plt.axvline(330, label = "CS & US end ", c = "red")

    plt.xticks(np.arange(0,351,50), np.arange(50,401,50))
    plt.legend()
    plt.colorbar(sm, label="Trial")
    #plt.ylim([100,200])
    return fig

def plot_cell_sdf_MA(cell, selected_trials, maf_step):

    sdf_mean_cell = []
    sdf_maf_cell = []
    for trial in selected_trials:
        sdf_cell = sdf( cell, trial)
        sdf_mean_cell.append(sdf_mean(sdf_cell))
        sdf_maf_cell.append(sdf_maf(sdf_cell, maf_step))

    fig = plt.figure()
    for i in selected_trials[:-1]:
        plt.plot(sdf_maf_cell[i], palette[i])
    plt.title(cell)
    # plt.ylim([25,55])
    plt.xticks(np.arange(0,251,50), np.arange(100,351,50))
    plt.axvline(0, label = "CS start", c = "grey")
    plt.axvline(250, label = "US start", c = "black")
    plt.axvline(280, label = "CS & US end ", c = "red")
    plt.legend()
    plt.colorbar(sm, label="Trial")
    plt.show()

    return fig

def sdf_maf_max_dcn(burst_dur, maf_step, burst_dur_cs, n_trials):
    sdf_maf_ratio = (burst_dur-maf_step)/burst_dur
    isi_start = int(100*sdf_maf_ratio) #+25
    isi_end = int(burst_dur_cs*sdf_maf_ratio-1)

    baseline = np.mean(sdf_baseline("", 'dcn'))
    sdf_maf_max_all = []
    for j in range(1,n_trials):
        sdf_f = sdf("dcn", i)
        sdf_maf_f = sdf_maf(sdf_f,  maf_step)
        sdf_maf_f -= baseline
        sdf_maf_f = sdf_maf[isi_start:isi_end]
        sdf_maf_max = np.max(sdf_maf_f)
        sdf_maf_max_all.append(sdf_maf_max)
    sdf_maf_max_all = np.split(np.asarray(sdf_maf_max_all), 10)

    return(sdf_maf_max_all)

def plot_CR(CR):
    CR_fig = plt.figure()
    plt.plot(CR)
    plt.title("Complex responces")
    plt.ylabel("%")
    plt.xlabel("10 trial set")
    return CR_fig

def cr_thr(thr, ratio_f, selected_trials, maf_step, threshold, plot = False):
    over_threshold = []
    if plot:
        fig = plt.figure()
    for j in selected_trials:
        
        sdf_f = sdf( "dcn", j)
        sdf_maf_f = sdf_maf(sdf_f, maf_step)
        baseline_f = np.mean(sdf_baseline_trial('dcn', j))
        if plot:
            plt.plot(sdf_maf_f - baseline_f, palette[j])

        indx = sdf_maf_f[100:250]> thr
        if indx.any():   
            t_cr = np.where(sdf_maf_f[100:250]>threshold)[0][0]

            ratio = sdf_maf_f[100+t_cr]/np.mean(sdf_maf_f[100:100+t_cr])
            if ratio > ratio_f:
            # ratio = max(sdf_maf_f[100+t_cr:250])/sdf_maf_f[100+t_cr]
            
            # if ratio > 1.25:
                if plot:
                    plt.scatter(100+t_cr, sdf_maf_f[100+t_cr], c = palette[j])
                over_threshold.append(1)
            else: over_threshold.append(0)
        else: over_threshold.append(0)

    
    over_threshold = np.split(np.asarray(over_threshold), 10)
    CR = np.sum(over_threshold,axis=1)*10
        # return(over_threshold)
    if plot:
        plt.axhline(threshold, label="threshold")
        plt.title("dcn")
        plt.xticks(np.arange(0,251,50), np.arange(100,351,50))
        plt.axvline(0, label = "CS start", c = "grey")
        plt.axvline(250, label = "US start", c = "black")
        plt.axvline(280, label = "CS & US end ", c = "red")
        plt.legend()
    
    if plot:
        return CR, fig
    else:
        return CR

def get_cell_sdf_MA(cell, selected_trials, rster, trials_start, burst_dur, maf_step):
    sdf_mean_cell = []
    sdf_maf_cell = []
    for trial in selected_trials:
        sdf_cell = sdf(cell, trial, rster, trials_start, burst_dur)
        sdf_mean_cell.append(sdf_mean(sdf_cell))
        sdf_maf_cell.append(sdf_maf(sdf_cell, maf_step))
    return sdf_mean_cell, sdf_maf_cell

def cr_isi(thr, selected_trials, maf_step, threshold, burst_dur, burst_dur_cs, trials_start, rster, between_start,plot = False):

    sdf_maf_ratio = (burst_dur-maf_step)/burst_dur
    isi_start = int(100*sdf_maf_ratio)
    isi_end = int(burst_dur_cs*sdf_maf_ratio)
    baseline = np.mean(sdf_baseline('dcn',  rster,trials_start,burst_dur, between_start))

    over_threshold = []
    if plot:
        fig = plt.figure()
    for j in selected_trials:
        
        sdf_f = sdf("dcn", j, rster,trials_start,burst_dur)
        sdf_maf_f = sdf_maf(sdf_f, maf_step)
        sdf_maf_f -= baseline
        if plot:
            plt.plot(sdf_maf_f, palette[j])
        sdf_maf_pre_cs = sdf_maf_f[:isi_start]
        sdf_maf_cs = sdf_maf_f[isi_start:isi_end]

        sdf_maf_pre_cs_over = sdf_maf_pre_cs[sdf_maf_pre_cs >= threshold]
        if len(sdf_maf_pre_cs_over) > 0:
            over_threshold.append(0)
        elif len(sdf_maf_pre_cs_over) == 0:
            sdf_maf_win_over = sdf_maf_cs[sdf_maf_cs >= threshold]
            if len(sdf_maf_win_over) == 0:
                over_threshold.append(0)
            elif len(sdf_maf_win_over) > 0:
                for i in range(len(sdf_maf_cs)):
                    if sdf_maf_cs[i] >= thr:
                        onset_index = i
                        break
                sdf_maf_cs_onset = sdf_maf_cs[onset_index:]
                if len(sdf_maf_win_over) >= len(sdf_maf_cs_onset)*0.75:
                    over_threshold.append(1)
                else:
                    over_threshold.append(0)

    over_threshold = np.split(np.asarray(over_threshold), 10)
    CR = np.sum(over_threshold,axis=1)*10
    if plot:
        plt.title("dcn")
        # plt.ylim([25,55])
        plt.axvline(isi_start, label="isi start")
        plt.axvline(isi_end, label="isi end")
        plt.xticks(np.arange(0,251,50), np.arange(100,351,50))
        plt.axvline(0, label = "CS start", c = "grey")
        plt.axvline(250, label = "US start", c = "black")
        plt.axvline(280, label = "CS & US end ", c = "red")
        plt.ylabel("SDF Norm")
        plt.xlabel("Time [ms]")
        plt.legend()
        plt.colorbar(sm, label="Trial")

    if plot:
        return CR, fig
    else:
        return CR


def norm_sdf(cell, selected_trials, rster, trials_start, burst_dur, burst_dur_cs, between_start, maf_step):
    sdf_maf_ratio = (burst_dur - maf_step) / burst_dur
    isi_start = int(100 * sdf_maf_ratio)
    isi_end = int(burst_dur_cs * sdf_maf_ratio)
    baseline = np.mean(sdf_baseline(cell, rster, trials_start, burst_dur, between_start))
    sdf_maf_list = []
    sdf_mean_list = []
    for j in selected_trials:
        sdf_f = sdf(cell, j, rster, trials_start, burst_dur)
        sdf_maf_f = sdf_maf(sdf_f, maf_step)
        sdf_mean_f = sdf_mean(sdf_f)
        sdf_maf_f -= baseline
        sdf_mean_f -= baseline
        sdf_maf_list.append(sdf_maf_f)
        sdf_mean_list.append(sdf_mean_f)
    return sdf_maf_list, sdf_mean_list
