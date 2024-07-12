import numpy as np
import pandas as pd
from scipy import signal
from fooof import FOOOF
from scipy.ndimage import gaussian_filter
import concurrent.futures
import itertools
import pickle


def read_reference_rasters(
    shared_dir="./", areas=["BGs", "cereb"], n_trials=10, filtering=True
):
    # Reconstruct desired paths
    folders = []
    trial_numbers = []
    for area in areas:
        folders.extend(
            [f"{shared_dir}/{area}_active_trial_{i}" for i in range(n_trials)]
        )
        trial_numbers.extend(list(range(10)))

    trials_data = []
    indices_maps = []
    corresponding_trials = []
    # NOTE: these lists are meaningless here, but allow to re-use some functions
    corresponding_levels = []
    corresponding_modes = []

    for folder, trial in zip(folders, trial_numbers):
        with open(f"{folder}/rasters", "rb") as raster_file:
            data = pickle.load(raster_file)
        with open(f"{folder}/model_dic", "rb") as model_file:
            model_dict = pickle.load(model_file)
        # Filter o
        # ut invalid areas:
        trials_data.append(data)
        available_populations = [d["compartment_name"] for d in data]
        indices_maps.append(
            [model_dict["pop_ids"][pop] for pop in available_populations]
        )  # list(itertools.chain(model_dict["pop_ids"].values())))
        corresponding_levels.append(
            [-999] * len(data)
        )  # Setting unsignificant value as flag
        corresponding_modes.append(["reference"] * len(data))
        corresponding_trials.append([trial] * len(data))
    trials_data = list(itertools.chain(*trials_data))
    indices_maps = list(itertools.chain(*indices_maps))
    corresponding_trials = list(itertools.chain(*corresponding_trials))
    corresponding_levels = list(itertools.chain(*corresponding_levels))
    corresponding_modes = list(itertools.chain(*corresponding_modes))

    return (
        trials_data,
        indices_maps,
        corresponding_levels,
        corresponding_modes,
        corresponding_trials,
    )


def read_saved_rasters(
    depletion_modes,
    n_trials=5,
    shared_dir="results/complete_3000ms_x_1_sol17",
    dopa_levels=[1, 2, 4, 8],
    physiological_modes=["both", "external", "internal"],
):
    # Reconstruct desired paths
    folders = []
    depl_modes = []
    dopa_lvls = []
    trial_numbers = []
    for mode in depletion_modes:
        base_dir = f"{shared_dir}_{mode}_dopa_active"
        if mode in physiological_modes:
            folders.extend([f"{base_dir}_trial_{i}" for i in range(1, n_trials + 1)])
            trial_numbers.extend(list(range(1, n_trials + 1)))
            depl_modes.extend([mode] * n_trials)
            dopa_lvls.extend([[0] * n_trials])
        # Expand the list of directories to include the different dopamine depletion levels
        if len(dopa_levels) > 0:
            depl_modes.extend([mode] * (n_trials * len(dopa_levels)))
            dopa_dirs = [f"{base_dir}_dopadepl_{i}" for i in dopa_levels]
            folders.extend(
                [f"{dopa_dir}_trial_{i}" for dopa_dir in dopa_dirs for i in range(1, 6)]
            )
            trial_numbers.extend(list(range(1, 6)) * len(dopa_dirs))
            dopa_lvls.extend([lvl] * n_trials for lvl in [*dopa_levels])
    # Flatten the nested list
    dopa_lvls = list(itertools.chain(*dopa_lvls))

    # Extract quantities related to all levels of dopamine depletion
    # Dopamine depletion = 0
    trials_data = []
    indices_maps = []
    corresponding_levels = []
    corresponding_modes = []
    corresponding_trials = []

    # Load the results
    for folder, level, mode, trial in zip(
        folders, dopa_lvls, depl_modes, trial_numbers
    ):
        with open(f"{folder}/rasters", "rb") as raster_file:
            data = pickle.load(raster_file)
        with open(f"{folder}/model_dic", "rb") as model_file:
            model_dict = pickle.load(model_file)
        trials_data.append(data)
        indices_maps.append(list(itertools.chain(model_dict["pop_ids"].values())))
        corresponding_levels.append([level] * len(data))
        corresponding_modes.append([mode] * len(data))
        corresponding_trials.append([trial] * len(data))
    trials_data = list(itertools.chain(*trials_data))
    indices_maps = list(itertools.chain(*indices_maps))
    corresponding_levels = list(itertools.chain(*corresponding_levels))
    corresponding_modes = list(itertools.chain(*corresponding_modes))
    corresponding_trials = list(itertools.chain(*corresponding_trials))

    return (
        trials_data,
        indices_maps,
        corresponding_levels,
        corresponding_modes,
        corresponding_trials,
    )


def read_mm_outputs(
    depletion_modes,
    n_trials=5,
    shared_dir="results/complete_3000ms_x_1_sol17",
    dopa_levels=[],
):
    # Reconstruct desired paths
    folders = []
    depl_modes = []
    dopa_lvls = []
    trial_numbers = []
    for mode in depletion_modes:
        base_dir = f"{shared_dir}_{mode}_dopa_active"
        # NOTE: add `if mode=="both"` if some depletion modes do not have the 0 case
        folders.extend([f"{base_dir}_trial_{i}" for i in range(1, 6)])
        trial_numbers.extend(list(range(1, 6)))
        depl_modes.extend([mode] * n_trials)
        dopa_lvls.extend([[0] * n_trials])
        # Expand the list of directories to include the different dopamine depletion levels
        if len(dopa_levels) > 0:
            depl_modes.extend([mode] * (n_trials * len(dopa_levels)))
            dopa_dirs = [f"{base_dir}_dopadepl_{i}" for i in dopa_levels]
            folders.extend(
                [f"{dopa_dir}_trial_{i}" for dopa_dir in dopa_dirs for i in range(1, 6)]
            )
            trial_numbers.extend(list(range(1, 6)) * len(dopa_dirs))
            dopa_lvls.extend([lvl] * n_trials for lvl in [*dopa_levels])
    # Flatten the nested list
    dopa_lvls = list(itertools.chain(*dopa_lvls))

    # Extract quantities related to all levels of dopamine depletion
    # Dopamine depletion = 0
    mass_models_data = {dm: {} for dm in depletion_modes}
    for folder, _depl_mode, lvl in zip(folders, depl_modes, dopa_lvls):
        with open(f"{folder}/mass_models_sol", "rb") as pickle_file:
            mass_frs = pickle.load(pickle_file)
            if lvl in mass_models_data[_depl_mode]:
                mass_models_data[_depl_mode][lvl] += [mass_frs]
            else:
                mass_models_data[_depl_mode][lvl] = [mass_frs]

    return mass_models_data

def load_and_process_wavelet(path, region, wavelet_dict, wavelet_dict_norm):
    file_types = ['both', 'external', 'internal']
    wavelet_dict[region] = {}
    wavelet_dict_norm[region] = {}

    for file_type in file_types:
        with open(f"{path}/{file_type}_wavelet_per_trial_list_{region}.p", "rb") as f:
            data = p.load(f)
        
        data = np.array(data)
        wavelet_dict[region][file_type] = data
        wavelet_dict_norm[region][file_type] = data - data[0, :, :]

def reference_fr_dataframe_from_literature(
    control_reference_values, dopa_reference_values
):

    # Expand the reference quantities in dataframes
    names = []
    frs = []
    numerosities = []
    dopamine_lvls = []
    modes = []
    n_trials = []
    for k, v in control_reference_values.items():
        frs.extend([value for value in v])
        names.extend([k] * len(v))
        numerosities.extend([None] * len(v))
        dopamine_lvls.extend(["Ref0"] * len(v))
        n_trials.extend([None] * len(v))
        modes.extend([None] * len(v))
    for k, v in dopa_reference_values.items():
        frs.extend([value for value in v])
        names.extend([k] * len(v))
        numerosities.extend([None] * len(v))
        dopamine_lvls.extend(["Reference"] * len(v))
        n_trials.extend([None] * len(v))
        modes.extend([None] * len(v))

    ref_df = convert_to_dataframe(
        names,
        numerosities,
        frs,
        dopamine_lvls,
        modes,
        n_trials,
        normalize_dop=False,
        reference=True,
    )

    return ref_df
 
def collect_firing_rates_dataframe(
    raster_data,
    pop_indices,
    dopa_lvls,
    depl_modes,
    trial_numbers,
    return_CV=False,
    t_start=0.0,
):
    firing_rates = []
    names = []
    numerosities = []
    dopamine_levels = []
    modes = []
    trials = []
    # Launch multiple jobs
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(process_population_data, data, t_start): data
            for data in zip(
                raster_data, pop_indices, dopa_lvls, depl_modes, trial_numbers
            )
        }
        # finished_jobs = concurrent.futures.wait(futures)[0]
        for future in concurrent.futures.as_completed(futures):
            name, pop_dim, fr, lvl, mode, trial_num = future.result()
            names.append(name)
            numerosities.append(pop_dim)
            firing_rates.append(fr)
            dopamine_levels.append(lvl)
            modes.append(mode)
            trials.append(trial_num)

    df = convert_to_dataframe(
        names, numerosities, firing_rates, dopamine_levels, modes, trials
    )

    return df


def collect_mass_models_dataframe(
    freq_array, periodic_spectra_values, ordered_areas=["ctx", "thal", "nrt"]
):
    # NOTE: not used due to bug in plotting wavelets with sns

    freq_values = np.zeros((0, 1))
    spectral_values = np.zeros((0, 1))
    areas = []
    trials = []
    # Iterate over the trials
    for i in range(periodic_spectra_values.shape[0]):
        # For each area, fill the matrix with the different values over trials
        for j, area in enumerate(ordered_areas):
            freq_values = np.vstack([freq_values, freq_array.reshape(-1, 1)])
            spectral_components = periodic_spectra_values[i, :, j].reshape(-1, 1)
            spectral_values = np.vstack([spectral_values, spectral_components])
            areas.extend([area] * len(spectral_components))
        trials.extend([i] * (len(spectral_components) * len(ordered_areas)))

    areas = np.array(areas).reshape(-1, 1)
    trials = np.array(trials).reshape(-1, 1)

    data = np.hstack([freq_values, spectral_values, areas, trials])

    df = pd.DataFrame(data=data, columns=["freq", "spectrum", "area", "trial_num"])
    return df


def raster_to_trace(
    raster_data,
    pop_indices,
    dopa_lvls,
    depl_modes,
    trial_numbers,
    tot_sim_time,
    window_width=1.0,
    step=1.0,
    t_start=0.0,
    # TODO: filter unwanted areas
    desired_areas=["GPeTA", "STN", "SNr"],
    desired_mode=None,
    desired_lvl=None,
):
    names = []
    firing_rates = []
    times = []
    dopamine_levels = []
    modes = []
    # Launch multiple jobs
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = {
            executor.submit(
                compute_windowed_fr,
                data,
                tot_sim_time,
                window_width,
                step,
                t_start,
                desired_areas,
            ): data
            for data in zip(
                raster_data, pop_indices, dopa_lvls, depl_modes, trial_numbers
            )
        }
        # finished_jobs = concurrent.futures.wait(futures)[0]
        for future in concurrent.futures.as_completed(futures):
            (
                name,
                fr,
                central_windows_time,
                lvl,
                mode,
            ) = future.result()
            if name is not None:
                names.append(name)
                firing_rates.append(fr)
                times.append(central_windows_time)
                dopamine_levels.append(lvl)
                modes.append(mode)

    if desired_mode is not None and desired_lvl is not None:
        return _apply_gaussian_filter(
            names,
            times,
            firing_rates,
            dopamine_levels,
            modes,
            desired_mode,
            desired_lvl,
        )
    return names, times, firing_rates, dopamine_levels, modes


def _apply_gaussian_filter(
    names,
    times,
    firing_rates,
    dopamine_levels,
    modes,
    target_mode,
    target_lvl,
    n_trials=5,
):
    out_tensor = np.empty((n_trials, len(times[0]), len(set(names))), dtype=float)
    _dict = {}
    for name, t, fr, lvl, mode in zip(
        names, times, firing_rates, dopamine_levels, modes
    ):
        if mode == target_mode and lvl == target_lvl:
            gaussian_fr = gaussian_filter(fr, [0, 2]).sum(axis=0) / 1000.0
            if name in _dict:
                _dict[name] += [gaussian_fr]
            else:
                _dict[name] = [gaussian_fr]

    for j, name in enumerate(set(names)):
        for i, data in enumerate(_dict[name]):
            out_tensor[i, :, j] = data

    return times[0], out_tensor


"""
def fr_window_step(raster_list, pop_dim_ids, sim_time, window, step, start_time):
    # calculate punctual fr per neuron
    # sim_time, window, step in [ms]
    # in start_time will be centered the first window

    ret_list = []

    for raster in raster_list:
        #
        pop_name = raster["compartment_name"]
        t = raster["times"]

        i = raster["neurons_idx"] - pop_dim_ids[pop_name][0]
        pop_dim = pop_dim_ids[pop_name][1] - pop_dim_ids[pop_name][0] + 1

        elem_len = int((sim_time - start_time) / step) + 1
        firing = np.zeros((pop_dim, elem_len))

        for index, time in zip(i, t):
            last_window_index = int(
                (time - start_time + window / 2.0) / step
            )  # index in the array for the last window containing the firing
            dt = time - (
                start_time + last_window_index * step - window / 2.0
            )  # time from t and the beginning of the last window
            how_many_wind_before = int(
                (window - dt) / step
            )  # how many windows before contain the firing
            for j in range(how_many_wind_before + 1):
                elem = last_window_index - j
                if (
                    0 <= elem < elem_len
                ):  # if spike at the last ms, do not count following "out" window
                    firing[int(index), elem] += 1

        # print(firing[0,:])
        fr = firing / (window / 1000.0)

        central_windows_time = np.linspace(
            start_time, start_time + step * (elem_len - 1), elem_len
        )

        ret = {"times": central_windows_time, "instant_fr": fr, "name": pop_name}

        ret_list = ret_list + [ret]

    return ret_list
"""


def compute_windowed_fr(
    data, tot_sim_time, window_width, step, start_time, desired_areas
):
    raster_data, population_indices, level, mode, trial_num = data
    spike_times = raster_data["times"]
    name = raster_data["compartment_name"]
    if name not in desired_areas:
        return [None] * 5
    start_index, end_index = (
        population_indices[0],
        population_indices[1],
    )
    # Get the total dimension of the area
    numerosity = end_index - start_index + 1
    neurons_ids = raster_data["neurons_idx"] - start_index
    # Initialize an array holding the firing occurences
    num_total_windows = int((tot_sim_time - start_time) / step) + 1
    firing = np.zeros((numerosity, num_total_windows))
    for spike_time, neuron_id in zip(spike_times, neurons_ids):
        # Retrieve the index of the window the spike being processed belongs to
        window_idx = int((spike_time - start_time + window_width / 2.0) / step)
        # Compute the time offset from the beginning of the  window and the actual spike time
        dt = spike_time - (start_time + window_idx * step - window_width / 2.0)
        # Compute the number of windows preceeding the current one
        num_past_windows = int((window_width - dt) / step)
        for j in range(num_past_windows + 1):
            window = window_idx - j
            # Check that the spike does not exceed the limit
            if 0 <= window < num_total_windows:
                firing[int(neuron_id), window] += 1

    # Rescale the firing units to Hz
    fr = firing / (window_width / 1000.0)
    # Center the first window around the start time
    central_windows_time = np.linspace(
        start_time, start_time + step * (num_total_windows - 1), num_total_windows
    )
    return (name, fr, central_windows_time, level, mode)


def process_population_data(data, return_CV=False, t_start=0.0):
    raster_data, population_indices, level, mode, trial_num = data
    spike_times = raster_data["times"]
    neurons_ids = raster_data["neurons_idx"]
    name = raster_data["compartment_name"]
    start_index, end_index = (
        population_indices[0],
        population_indices[1],
    )
    # Get the total dimension of the area
    numerosity = end_index - start_index + 1
    # Filter spike events before the starting time
    filter = spike_times > t_start
    filtered_spike_times = spike_times[filter]
    # Scale the spike ids to use them as population pointers
    filtered_neurons_ids = neurons_ids[filter] - start_index - 1
    # Initialize an array holding the first spiking occurence
    last_spike = -np.ones(numerosity)
    spike_intervals = [[]] * numerosity
    for spike_time, neuron_id in zip(filtered_spike_times, filtered_neurons_ids):
        if last_spike[neuron_id] == -1:  # Fist spike of the neuron detected
            last_spike[neuron_id] = spike_time
        else:
            inter_spike_interval = spike_time - last_spike[neuron_id]
            # If a valid interval is found
            if inter_spike_interval != 0:
                # Add it to the intervals list and update last spike time
                spike_intervals[neuron_id] = spike_intervals[neuron_id] + [
                    inter_spike_interval
                ]
                last_spike[neuron_id] = spike_time
    # Compute each neuron firing rate as the inverse of the average ISI
    firing_rates = np.array(
        [
            1 / (sum(spike_intervals) / len(spike_intervals))
            if len(spike_intervals) != 0
            else 0.0
            for spike_intervals in spike_intervals
        ]
    )
    # Note: the firing rates are in  [1/ms], the mean
    # firing rate need to be converted to Hz [1/s] multiplying by 1000
    mean_firing_rate = (firing_rates).mean() * 1000
    return (name, numerosity, mean_firing_rate, level, mode, trial_num)


def compute_population_average(subpopulation_firing_rates, subpopulation_dimensions):
    subpopulation_firing_rates = np.array(subpopulation_firing_rates)
    subpopulation_dimensions = np.array(subpopulation_dimensions)
    # Multiply each quantity for the corresponding subpopulation dimension
    cumulative_firing_rates = np.einsum(
        "ij,i-> ij", subpopulation_firing_rates, subpopulation_dimensions
    )
    # Get the toal dimension
    total_dim = np.sum(subpopulation_dimensions)
    return np.sum(cumulative_firing_rates, axis=0) / total_dim


def convert_to_dataframe(
    names,
    numerosities,
    firing_rates,
    dopamine_levels,
    modes,
    trial_numbers,
    normalize_dop=True,
    reference=False,
):
    names = np.array(names).reshape(-1, 1)
    numerosities = np.array(numerosities).reshape(-1, 1)
    firing_rates = np.array(firing_rates).reshape(-1, 1)
    dopamine_levels = np.array(dopamine_levels).reshape(-1, 1)
    if not reference:
        dopamine_levels = dopamine_levels.astype(float)
    if normalize_dop:
        dopamine_levels /= 10
    depl_modes = np.array(modes).reshape(-1, 1)
    n_trials = np.array(trial_numbers).reshape(-1, 1)
    data = np.hstack(
        (names, numerosities, firing_rates, dopamine_levels, depl_modes, n_trials)
    )
    df = pd.DataFrame(
        data=data,
        columns=[
            "name",
            "numerosity",
            "firing_rate",
            "dopa_lvl",
            "depletion_mode",
            "n_trial",
        ],
    )
    # Convert the data types
    if not reference:
        df["numerosity"] = pd.to_numeric(df["numerosity"])
        df["dopa_lvl"] = pd.to_numeric(df["dopa_lvl"])
        df["n_trial"] = pd.to_numeric(df["n_trial"])
    df["firing_rate"] = pd.to_numeric(df["firing_rate"])
    return df


def normalize_firing_rates(
    df, populations_list=None, modes=None, reference=False, ref_lvl=0.0
):
    if modes is None and not reference:
        modes = [m for m in np.unique(df["depletion_mode"].values)]
    if populations_list is None:
        populations_list = [n for n in np.unique(df["name"].values)]
    if reference:
        ref_lvl = "Ref0"
    for name in populations_list:
        if reference:
            mean_value = df.loc[
                (df["name"] == name) & (df["dopa_lvl"] == ref_lvl),
                "firing_rate",
            ].mean()
            df.loc[df["name"] == name, "firing_rate"] -= mean_value
            df.loc[df["name"] == name, "firing_rate"] /= mean_value
        else:
            for mode in modes:
                mean_value = df.loc[
                    (df["name"] == name)
                    & (df["dopa_lvl"] == ref_lvl)
                    & (df["depletion_mode"] == mode),
                    "firing_rate",
                ].mean()
                df.loc[
                    (df["name"] == name) & (df["depletion_mode"] == mode), "firing_rate"
                ] -= mean_value
                df.loc[
                    (df["name"] == name) & (df["depletion_mode"] == mode), "firing_rate"
                ] /= mean_value
    return df


def normalize_references(df, populations_list=None):
    if populations_list is None:
        populations_list = [n for n in np.unique(df["name"].values)]
    for name in populations_list:
        mean_value = df.loc[
            (df["name"] == name) & (df["dopa_lvl"] == "ref0"), "firing_rate"
        ].mean()
        df.loc[df["name"] == name, "firing_rate"] -= mean_value
        df.loc[df["name"] == name, "firing_rate"] /= mean_value
    return df


def average_subpopulations(
    df, populations_subsets, merged_populations, desired_modes, n_trials=5
):
    for subset, target in zip(populations_subsets, merged_populations):
        sub_df = df.loc[df["name"].isin(subset)]
        # Initialize lists that will be filled for the merged df
        names, nums, frs, dopa_lvls, modes, trial_nums = [[] for i in range(6)]
        # Compute new df here
        for mode in desired_modes:
            for lvl in np.unique(sub_df["dopa_lvl"].values):
                # Only consider the current dopamine depletion level and the target mode
                dopa_df = sub_df.loc[
                    np.logical_and(
                        sub_df["dopa_lvl"] == lvl, sub_df["depletion_mode"] == mode
                    )
                ]
                # Initialize firing rates and numerosities arrays
                firing_rates = np.empty((n_trials, 0))
                numerosities = np.empty((n_trials, 0))
                for population in subset:
                    pop_df = dopa_df.loc[sub_df["name"] == population]
                    num = pop_df["numerosity"].values.reshape(-1, 1)
                    fr = pop_df["firing_rate"].values.reshape(-1, 1)
                    # Create the matricies to be multiplied
                    firing_rates = np.hstack((firing_rates, fr))
                    numerosities = np.hstack((numerosities, num))
                # Merged firing rates
                cumulative_frs = np.einsum("ij,ij->ij", firing_rates, numerosities)
                merged_frs = np.sum(cumulative_frs, axis=1) / np.sum(
                    numerosities, axis=1
                )
                # Fill the lists with the new values
                names.extend([target] * len(merged_frs))
                nums.extend((np.sum(numerosities, axis=1)).tolist())
                frs.extend(merged_frs.tolist())
                dopa_lvls.extend([lvl] * len(merged_frs))
                modes.extend([mode] * len(merged_frs))
                trial_nums.extend([i for i in range(1, len(merged_frs) + 1)])
        # Create the new df to be concatenated
        new_df = convert_to_dataframe(
            names, nums, frs, dopa_lvls, modes, trial_nums, normalize_dop=False
        )
        # Remove unused rows
        df = df.drop(df[df["name"].isin(subset)].index)
        # Concat new ones
        df = pd.concat([df, new_df], ignore_index=True)

    return df


def compute_periodic_spectra(mass_models_sol, t_sample, t_start, from_snn=False):
    if not from_snn:
        if not isinstance(mass_models_sol, list):
            mass_models_sol = [mass_models_sol]
        times = mass_models_sol[0]["mass_frs_times"]
        # Construct boolean masks to only retrieve values after t_start and each t_sample
        after_start = times > t_start
        discrete_samples = times % t_sample == 0
        # NOTE: `mass_fr_values` is a list holding 5 values, one for each trial
        # Each values is an array of shape (1500, 3): representing 1500 samples of the 3 different mass models
        mass_fr_values = [
            mms["mass_frs"][after_start * discrete_samples, :]
            for mms in mass_models_sol
        ]
    else:
        # Data must be prepared to extract the power spectrum
        time, data = mass_models_sol
        after_start = time > t_start
        discrete_samples = time % t_sample == 0
        mass_fr_values = [d[after_start * discrete_samples, :] for d in data]

    sampling_freq = 1000.0 / t_sample  # [Hz], sampling time
    w = 15.0  # [adim], "omega0", in the definition of Morlet wavelet: pi**-0.25 * exp(1j*w*x) * exp(-0.5*(x**2))
    # Get array of frequencies, ranging until half the sampling_frequency
    freq = np.linspace(1, sampling_freq / 2, 2 * int(sampling_freq / 2 - 1) - 1)
    widths = (
        w * sampling_freq / (2 * freq * np.pi)
    )  # [adim] reduce time widths for higher frequencies. Widhts / sample_freq = time

    # Initialize a list to hold the y-axis (wavelet transform) values of the three mass models,
    # one item for each trial
    wavelet_transforms = [
        np.zeros((len(freq), mass_fr.shape[1])) for mass_fr in mass_fr_values
    ]
    # Iterate the data for each trial
    for mass_fr, wavelet_t in zip(mass_fr_values, wavelet_transforms):
        # Iterate over the three mass models
        for idx in range(mass_fr.shape[1]):
            # Extract the continuous wavelet transform of each
            cwtm = signal.cwt(mass_fr[:, idx], signal.morlet2, widths, w=w)
            # Store the values in the list
            wavelet_t[:, idx] = np.abs(cwtm).sum(axis=1)

    # Clip the frequencies between 1-60 Hz
    lower_freq_id, upper_freq_id = calculate_fourier_idx(freq, [1, 60])
    freq = freq[lower_freq_id:upper_freq_id]
    wavelets = [
        wavelet_t[lower_freq_id:upper_freq_id, :] for wavelet_t in wavelet_transforms
    ]
    # if mean is not None:
    #    ax.axvspan(mean - sd, mean + sd, alpha=0.5, color="tab:blue")

    peak_width_limits = [[2, 8], [2, 8], [2, 8]]
    #peak_width_limits = [[1, 8], [1, 8], [1, 8]]
    # Empty list for the 5 trials, each item is a zero matrix of shape (N, 3)
    periodic_spectra = [np.zeros(wavelet.shape) for wavelet in wavelets]
    for wavelet, periodic_spectrum in zip(wavelets, periodic_spectra):
        for idx in range(wavelet.shape[1]):
            # Initialize FOOOF to extract the periodic and a-periodic spectral components
            fm = FOOOF(peak_width_limits=peak_width_limits[idx])
            # Fit the full spectrum as a combination of periodic and a-periodic components
            fm.fit(freq, wavelet[:, idx])
            # Remove the a-periodic components
            periodic_spectrum[:, idx] = (
                fm.fooofed_spectrum_
                - np.log10(1 / freq ** fm.aperiodic_params_[1])
                - fm.aperiodic_params_[0]
            )

    periodic_spectra = np.array(periodic_spectra)

    return freq, periodic_spectra


def calculate_fourier_idx(xf, range):
    lower = range[0]
    upper = range[1]

    xf_lower = np.abs(xf - lower)
    # min_idx = xf_lower.where(min(np.abs(xf_lower)))
    min_idx = np.where(xf_lower <= min(xf_lower + 0.01))
    xf_upper = np.abs(xf - upper)
    # max_idx = xf_upper.where(max(np.abs(xf_upper)))
    max_idx = np.where(xf_upper <= min(xf_upper))
    return [min_idx[0][0], max_idx[0][0] + 1]


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
