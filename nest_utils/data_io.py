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


def reference_fr_dataframe_from_literature(
    control_reference_values, dopa_reference_values
):
    from data_processing import convert_to_dataframe

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
