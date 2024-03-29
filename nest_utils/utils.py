from random import randint
import numpy as np
from scipy import signal
from scipy import io
from scipy.fft import fft, fftfreq
from nest_utils import visualizer as vsl
from pathlib import Path
import pickle
from scipy.ndimage import gaussian_filter


def attach_spikedetector(nest, pop_list, pop_list_to_ode=None, sd_list_to_ode=None):
    """ Function to attach a spike_detector to all populations in list
        Returns a list of vms coherent with passed population    """
    sd_list = []

    if pop_list_to_ode is not None:     # if there is no spike detector already set, load it
        pop_list_to_ode_pointer = 0
        for pop in pop_list:
            if pop_list_to_ode[pop_list_to_ode_pointer] == pop:
                sd_list = sd_list + [sd_list_to_ode[pop_list_to_ode_pointer]]
                if pop_list_to_ode_pointer < len(pop_list_to_ode):
                    pop_list_to_ode_pointer += 1
            else:
                sd = nest.Create('spike_detector', params={'to_file': False})
                nest.Connect(pop, sd)
                sd_list = sd_list + [sd]

    else:       # if there is no spike detector already set
        for pop in pop_list:
            sd = nest.Create('spike_detector', params={'to_file': False})
            nest.Connect(pop, sd)
            sd_list = sd_list + [sd]
    return sd_list

def get_spike_values(nest, sd_list, pop_names):
    """ Function to select spike idxs and times from spike_det events
        Returns a list of dictionaries with spikes and times  """
    dic_list = []
    for sd, name in zip(sd_list, pop_names):
        spikes = nest.GetStatus(sd, "events")[0]["senders"]
        times = nest.GetStatus(sd, "events")[0]["times"]
        dic = {'times': times, 'neurons_idx': spikes, 'compartment_name': name}
        dic_list = dic_list + [dic]
    return dic_list

def create_model_dictionary(N_neurons, pop_names, pop_ids, sim_time, sample_time=None, settling_time=None, trials=None, b_c_params=None):
    """ Function to create a dictionary containing model parameters  """
    dic = {}
    dic['N_neurons'] = N_neurons
    dic['pop_names_list'] = pop_names
    dic['pop_ids'] = pop_ids
    dic['simulation_time'] = sim_time
    dic['sample_time'] = sample_time
    dic['settling_time'] = settling_time
    dic['trials'] = trials
    dic['b_c_params'] = b_c_params
    return dic

def calculate_fr_stats(raster_list, pop_dim_ids, t_start=0., t_end=None, multiple_trials=False):
    """ Function to evaluate the firing rate and the
    coefficient of variation of the inter spike interval"""
    if not multiple_trials:     # the raster list corresponds to only 1 trial
        fr_list, CV_list, name_list = calculate_fr(raster_list, pop_dim_ids, t_start, t_end, return_CV_name=True)
    else:                       # raster_list is a list of list. Multiple trials are applied
        fr_list_list = []
        CV_list_list = []
        name_list_list = []
        for raster in raster_list:
            fr_list, CV_list, name_list = calculate_fr(raster, pop_dim_ids, t_start, t_end, return_CV_name=True)
            fr_list_list += [np.array(fr_list)]
            CV_list_list += [np.array(CV_list)]
        fr_list = np.array(fr_list_list).mean(axis=0)
        fr_list_sd = np.array(fr_list_list).std(axis=0)
        CV_list = np.array(CV_list_list).mean(axis=0)
        CV_list_sd = np.array(CV_list_list).mean(axis=0)

    pop_list_dim = get_pop_dim_from_ids(name_list, pop_dim_ids)
    # expand with average values for GPe and MSN if there are
    if 'MSND1' in name_list and 'GPeTA' in name_list:
        average_MSN = lambda list: round(
            (list[1] * pop_list_dim[1] + list[2] * pop_list_dim[2]) / (pop_list_dim[1] + pop_list_dim[2]), 2)
        average_GPe = lambda list: round(
            (list[3] * pop_list_dim[3] + list[4] * pop_list_dim[4]) / (pop_list_dim[3] + pop_list_dim[4]), 2)
        fr_list = fr_list[0:3] + [average_MSN(fr_list)] + fr_list[3:5] + [average_GPe(fr_list)] + fr_list[5:7]
        CV_list = CV_list[0:3] + [average_MSN(CV_list)] + CV_list[3:5] + [average_GPe(CV_list)] + CV_list[5:7]
        name_list = name_list[0:3] + ['MSN'] + name_list[3:5] + ['GPe'] + name_list[5:7]

    # elif 'GPeTA' in name_list:
    #     average_GPe = lambda list: round(
    #         (list[3] * pop_list_dim[3] + list[4] * pop_list_dim[4]) / (pop_list_dim[3] + pop_list_dim[4]), 2)
    #     fr_list = fr_list[0:3] + fr_list[3:5] + [average_GPe(fr_list)] + fr_list[5:7]
    #     CV_list = CV_list[0:3] + CV_list[3:5] + [average_GPe(CV_list)] + CV_list[5:7]
    #     name_list = name_list[0:3] + name_list[3:5] + ['GPe'] + name_list[5:7]

    if not multiple_trials:
        ret = {'fr': fr_list, 'CV': CV_list, 'name': name_list}
    else:
        ret = {'fr': fr_list, 'fr_sd': fr_list_sd, 'CV': CV_list, 'CV_sd': CV_list_sd, 'name': name_list}
    return ret

def calculate_fr(raster_list, pop_dim_ids, t_start=0., t_end=None, return_CV_name=False):
    """ Function to evaluate the firing rate and the
    coefficient of variation of the inter spike interval"""
    fr_list = []
    if return_CV_name:
        CV_list = []
        name_list = []
    min_idx = 0  # useful to process neurons indexes

    if t_end == None:
        t_end = np.inf

    for raster in raster_list:
        pop_name = raster['compartment_name']
        pop_dim = pop_dim_ids[pop_name][1] - pop_dim_ids[pop_name][0] + 1
        t_prev = -np.ones(pop_dim)  # to save the last spike time for idx-th neuron
        ISI_list = [[] for _ in range(pop_dim)]  # list of list, will contain the ISI for each neuron
        for tt, idx in zip(raster['times'], raster['neurons_idx'] - pop_dim_ids[pop_name][0] - 1):
            if tt > t_start:  # consider just element after t_start
                if tt < t_end:
                    if t_prev[idx] == -1:  # first spike of the neuron
                        t_prev[idx] = tt
                    else:
                        ISI = (tt - t_prev[idx])  # inter spike interval
                        if ISI != 0:
                            ISI_list[idx] = ISI_list[idx] + [ISI]
                            t_prev[idx] = tt  # update the last spike time
        # we calculate the average ISI for each neuron, comprehends also neurons with fr = 0
        inv_mean_ISI = np.array([1000. / (sum(elem) / len(elem)) if len(elem) != 0 else 0. for elem in ISI_list])
        fr = inv_mean_ISI.mean()
        fr_list = fr_list + [round(fr, 2)]
        if return_CV_name:
            CV_el = np.array([np.array(sublist).std() / np.array(sublist).mean() if len(sublist) != 0 else 0. for sublist in ISI_list])
            CV_list = CV_list + [round(CV_el.mean(), 2)]
            # ISI_array = np.array([item for sublist in ISI_list for item in sublist])  # flat the ISI array
            # CV_list = CV_list + [round(ISI_array.std() / ISI_array.mean(), 2) if len(ISI_array) != 0 else 0.]   # calculate CV between all of the ISI of that population
            name_list = name_list + [raster['compartment_name']]

    if return_CV_name:  # return also ISI array as flatten np.array
        return fr_list, CV_list, name_list
    else:
        return fr_list

def get_pop_dim_from_ids(pop_list, pop_ids):
    ''' Given a dictionary of pop indexes, return a list of pop_dimention '''
    dim_list = []

    for pop in pop_list:
        dim = pop_ids[pop][1] - pop_ids[pop][0] + 1
        dim_list = dim_list + [dim]

    return dim_list

def gaussian(x, mu, sig):
    g = 1. / (np.sqrt(2. * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.) / 2)
    return g/g.sum()

def calculate_fourier_idx(xf, range):
    lower = range[0]
    upper = range[1]

    xf_lower = np.abs(xf - lower)
    # min_idx = xf_lower.where(min(np.abs(xf_lower)))
    min_idx = np.where(xf_lower <= min(xf_lower+0.01))
    xf_upper = np.abs(xf - upper)
    # max_idx = xf_upper.where(max(np.abs(xf_upper)))
    max_idx = np.where(xf_upper <= min(xf_upper))
    return [min_idx[0][0], max_idx[0][0]+1]

def fitness_function(fr, fr_target, mass_fr, T_sample, filter_range, filter_sd, t_start=0., fr_weights=None):
    '''
    Evaluate fitness function
    :param fr: average firing rate calculated from simulation raster plots
    :param fr_target: the desired value of the average firing rate
    :param mass_fr: the mass model firing rate over time, will be analyzed in frequencies
    :param T_sample: sampling time of the mass models, necessary for Fourier transform
    :param mean: of the gaussian filter applied to extract Fourier fitness
    :param sd: of the gaussian filter applied to extract Fourier fitness
    :param t_start: if you want to neglect the first simulation instants
    :param fr_weights: weights of the fr distances
    :return: return teh fitness
    '''

    # evaluate wavelet transform
    T = T_sample / 10.
    y = mass_fr[int(t_start / T):, :]  # calculate tf after the t_start
    T = T_sample  # resample to 1 ms
    y = y[::10]  # select one time sample every 10

    fs = 1000. / T  # Hz
    w = 15  # []
    freq = np.linspace(17, fs / 2, 2 * int(fs / 2 - 17 + 1))  # frequency range
    widths = w * fs / (2 * freq * np.pi)  # reduce time widths for higher frequencies

    wt = np.zeros((len(freq), y.shape[1]))
    for idx in range(y.shape[1]):
        cwtm = signal.cwt(y[:, idx], signal.morlet2, widths, w=w)
        wt[:, idx] = np.abs(cwtm).sum(axis=1)  # /(np.abs(cwtm)).sum()

    wavelet_idx = calculate_fourier_idx(freq, filter_range)
    print(f'In fitness: considering frequencies in the range {[freq[wavelet_idx[0]], freq[wavelet_idx[1] - 1]]}')

    dim = wavelet_idx[1] - wavelet_idx[0]
    freq_p = np.linspace(-10, 10, dim, endpoint=True)
    kernel_f = gaussian(freq_p, 0., filter_sd)
    conv = (np.dot(wt[wavelet_idx[0]:wavelet_idx[1], :].T, kernel_f)).sum()

    # evaluate fr accuracy
    dist = np.array(fr) - fr_target
    if fr_weights is not None:
        dist = dist * fr_weights
    dist = np.power(dist, 2)
    fitness_Cereb = np.sum(dist)
    fitness_fourier = conv / 2
    fitness = - fitness_Cereb + fitness_fourier

    print(f'fitness_firing_rate = {"%.2f" % - fitness_Cereb}, fitness_fourier = {"%.2f" % fitness_fourier}, fitness = {"%.2f" % fitness}')

    return fitness
