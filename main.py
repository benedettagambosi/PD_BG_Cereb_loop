#!/usr/bin/env python3
"""
"""

__author__ = 'marco'

path = "/home/docker/packages/tvb-multiscale/my_tests/PD_test/"

import time
import numpy as np
import scipy.io
from datetime import datetime
from nest_utils import utils, visualizer as vsl
from multiprocessing import cpu_count
import os
import pickle
from pathlib import Path
import sys

# NEST modules
import nest

### USER PARAMS ###
load_from_file = 0      # load results from directory or simulate and save

dopa_depl_level_list = [0.,-0.1,-0.2,-0.4,-0.8]     # between 0. and -0.8

mode_list = ['external_dopa', 'internal_dopa', 'both_dopa']     # external = only BGs dopa depl, internal = only Cereb dopa depl
experiment_list = ['active', 'EBCC']
mode_i = int(sys.argv[2])
experiment_i = int(sys.argv[3])
dopa_depl_level_i = int(sys.argv[4])
mode = mode_list[mode_i]                 # dopa depl location
experiment = experiment_list[experiment_i]     # cortical activation or EBCC
dopa_depl_level = dopa_depl_level_list[dopa_depl_level_i]      # between 0. and -0.8



MODULE_PATH = 'ml_module'
nest.Install(MODULE_PATH)  # Import my_BGs module
MODULE_PATH = 'cerebmodule'
nest.Install(MODULE_PATH)  # Import CerebNEST

# path to h5py spatial distribution
hdf5_path = path + 'Cereb_nest3/scaffold_full_IO_400.0x400.0_microzone.hdf5'

# my modules
import sys
sys.path.append(path)

from Cereb_nest.Cereb import Cereb_class as C_c
from BGs_nest.BGs import BGs_class as B_c
from nest_multiscale.nest_multiscale import sim_handler, generate_ode_dictionary
from experiments import conditioning

print(f'CPU = {cpu_count()}')

# simulation parameters
CORES = 24
run_on_vm = True

# IO stimulation every trial
t_start_MF = 100
t_start_IO = t_start_MF + 250
t_end = t_start_IO + 30
stimulation_frequency = 500  # [sp/s]

N_BGs = 20000
N_Cereb = 96767

sol_n = 18
if dopa_depl_level != 0.:
    dopa_depl = True
else:
    dopa_depl = False


if experiment == 'active':
    settling_time = 1000.
    sim_time = 3000.
    start_time = 0.  # starting time for histograms data
    sim_period = 1.  # ms
    trials = 1
    RESOLUTION = 0.1
    n_wind = 1
elif experiment == 'EBCC':
    settling_time = 0.
    sim_time = t_end + 200  #    1760.
    start_time = 0.  # starting time for histograms data
    sim_period = 10.  # ms
    trials = 101
    RESOLUTION = 1.
    n_wind = 28

else:
    assert False, 'Select a correct experiment'

# defines where the dopamine is depleted
dopa_depl_cereb = 0.
dopa_depl_BGs = 0.
if mode != 'external_dopa':
    dopa_depl_cereb = dopa_depl_level
if mode != 'internal_dopa':
    dopa_depl_BGs = dopa_depl_level

tests_dict = {8: [-0.00008, 0.00008]}

tests_dict = {0: [3e-05, 2e-05]}

for key in tests_dict.keys():
# set number of kernels
    nest.ResetKernel()
    nest.SetKernelStatus({"total_num_virtual_procs": CORES, "resolution": RESOLUTION})
    nest.set_verbosity("M_ERROR")  # reduce plotted info

    LTD = - tests_dict[key][0]
    LTP = tests_dict[key][1]
    savings_dir = f'last_results/complete_{int(sim_time)}ms_x_{trials}_sol{sol_n}_{mode}_{experiment}'  # f'savings/{date_time}'
    if dopa_depl: savings_dir = savings_dir + f'_dopadepl_{(str(int(-dopa_depl_level*10)))}'
    
    if len(sys.argv) > 1:
        n_trial = int(sys.argv[1])
        savings_dir = savings_dir + f'_trial_{n_trial}'

    # create folder if not present
    if not load_from_file:
        if not os.path.exists(savings_dir):
            os.makedirs(savings_dir)
            print(f'\nWriting to {savings_dir}\n')
        else:
            print(f'\nATTENTION: subscribing to {savings_dir}\n')

    print("LTP "+str(LTP))
    print("LTD "+str(LTD))
    ''' Set up multi-scale simulation: order is important| '''

    # Define all the NEST populations:
    Cereb_pop_names = ['golgi', 'glomerulus', 'granule', 'purkinje', 'basket', 'stellate', 'dcn', 'dcnp', 'io']
    BGs_pop_names = ['FSN', 'MSND1', 'MSND2', 'GPeTA', 'GPeTI', 'STN', 'SNr']
    # Select the NEST populations you will attach to a spike detector:
    if experiment == 'active':
        Cereb_recorded_names = Cereb_pop_names
        BGs_recorded_names = BGs_pop_names
    elif experiment == 'EBCC':
        Cereb_recorded_names = ['glomerulus', 'purkinje', 'dcn', 'io']
        BGs_recorded_names = ['FSN', 'MSND1', 'MSND2', 'GPeTA', 'GPeTI','STN', 'SNr']
    recorded_names = Cereb_recorded_names + BGs_recorded_names
    # Define the names of the mass-model populations:
    ode_names = ['CTX', 'thalamus', 'nRT']

    # Nest populations connected to mass models:
    Cereb_pop_names_to_ode = ['dcn']
    BGs_pop_names_to_ode = ['SNr']
    # Mass model population connected to NEST:
    if n_wind == 1:
        Cereb_pop_names_to_nest= ['CTX']
    else:
        Cereb_pop_names_to_nest= ['CTX_'+str(i) for i in range(n_wind)]
    # Cereb_pop_names_to_nest = ['CTX_1', "CTX_2"]
    BGs_pop_names_to_nest = ['CF', 'CM1_1', 'CM1_2', 'CM2_1', 'CM2_2', 'CS_1', 'CS_2']

    # Mass models connections:
    w = [20., 5., 8., 25., 15., 5., 19., 5., 15., 20., 20.]
    if sol_n == 18: b_c_params = [162.095694, 88.93865742, 107.52074467, 127.63904076]

    # with bground
    b1 = w[3] / b_c_params[0]     # DCN -> Thal  # 2900
    b2 = -w[4] / b_c_params[1]     # SNr -> Thal
    c1 = b_c_params[2]             # CTX -> MF
    c5 = b_c_params[3]            # CTX -> STN
    # maintain the same oscillation ratio of Lindahl
    c2 = c5 / 175. * 173.14     # CTX -> FSN
    c3 = c5 / 175. * 120.12     # CTX -> MSN1
    c4 = c5 / 175. * 158.84     # CTX -> MSN2
    c5 = c5                     # CTX -> STN

    # Define ODEs matrices
    # follow the order in ode_names = ['CTX', 'thalamus', 'nRT']
    A_mat = np.array([[0, w[0], 0],
                    [w[1], 0, -w[2]],
                    [w[5], 0, 0]])
    # follow also the order in Cereb_pop_names_to_ode = ['dcn'] + BGs_pop_names_to_ode = ['SNr']
    B_mat = np.array([[0, 0],
                    [b1, b2],
                    [0, 0]])
    # follow also the order in Cereb_pop_names_to_nest = ['CTX'] + BGs_pop_names_to_nest
    C_mat = np.array([[c1, 0, 0],
                    [c2, 0, 0],
                    [c3, 0, 0],
                    [c3, 0, 0],
                    [c4, 0, 0],
                    [c4, 0, 0],
                    [c5, 0, 0],
                    [c5, 0, 0]])

    # ODE params from Yousif et al. 2020
    b_e = 1.3
    b_i = 2.0
    theta_e = 4.0
    theta_i = 3.7
    k_e = 1 - 1/(1+np.exp(b_e*theta_e))
    k_i = 1 - 1/(1+np.exp(b_i*theta_i))

    # define sigmoid params
    theta = np.array([theta_e, theta_e, theta_i])
    lambda_max = lambda x: np.array([k_e, k_e, k_i]) - x
    a = np.array([b_e, b_e, b_i])  # []
    q_exp = lambda b, th: -1. / (1 + np.exp(b * th))
    q = q_exp(a, theta)

    # define characteristic time
    tau = 10.  # [ms]

    # create a dictionary with all the defined prams
    params_dic = generate_ode_dictionary(A_matrix=A_mat, B_matrix=B_mat, C_matrix=C_mat, theta_vec=theta,
                                        lambda_max_vec=lambda_max, a_vec=a, q_vec=q, tau_val=tau, const_in=None)

    if __name__ == "__main__":
        if not load_from_file:
            # create an instance of the populations and inputs
            
            if experiment == "active":
                cortex_type = "spike_generator"
            else:
                cortex_type = ""

            Cereb_class = C_c(nest, hdf5_path, cortex_type=cortex_type, n_spike_generators=500,
                            mode=mode, experiment=experiment, dopa_depl=dopa_depl_cereb, LTD=LTD, LTP=LTP, n_wind = n_wind)
            BGs_class = B_c(nest, N_BGs, 'active', 'BGs_nest/default_params.csv', dopa_depl=dopa_depl_BGs,
                            cortex_type='spike_generator', in_vitro=False,
                            n_spike_generators={'FS': 250, 'M1': 1250, 'M2': 1250, 'ST': 50})

            if experiment == 'active':
                additional_classes = []
            if experiment == 'EBCC':
                cond_exp = conditioning(nest, Cereb_class, t_start_MF=t_start_MF, t_start_IO=t_start_IO, t_end=t_end,
                                        stimulation_IO=stimulation_frequency, resolution=RESOLUTION)
                additional_classes = [cond_exp]

                ct = Cereb_class.create_ctxinput(nest,in_spikes="spike_generator_ebcc", 
                                        experiment='EBCC', CS ={"start":float(t_start_MF), "end":float(t_end), "freq":36.}, US ={"start":float(t_start_IO), "end":float(t_end), "freq":500.}, tot_trials = trials, len_trial = sim_time)
                # Cereb_class.CTX_pops = ct
                
            recorded_list = [Cereb_class.Cereb_pops[name] for name in Cereb_recorded_names] + \
                            [BGs_class.BGs_pops[name] for name in BGs_recorded_names]
            pop_list_to_ode = [Cereb_class.Cereb_pops[name] for name in Cereb_pop_names_to_ode] + \
                            [BGs_class.BGs_pops[name] for name in BGs_pop_names_to_ode]
            pop_list_to_nest = [Cereb_class.CTX_pops[name] for name in Cereb_pop_names_to_nest] + \
                            [BGs_class.CTX_pops[name] for name in BGs_pop_names_to_nest]

            # initiate the simulation handler
            if experiment == "EBCC":
                s_h = sim_handler(nest, pop_list_to_ode, pop_list_to_nest,
                                params_dic, sim_time, sim_period_=sim_period, resolution=RESOLUTION, additional_classes=additional_classes, CS_stim = ct, n_wind=n_wind)
            else:
                s_h = sim_handler(nest, pop_list_to_ode, pop_list_to_nest,
                                params_dic, sim_time, sim_period_=sim_period, resolution=RESOLUTION, additional_classes=additional_classes, n_wind=n_wind)


            # record membrane potential from the first neuron of the population
            # MF parrots neurons cannot be connected to vm
            # vm_list = utils.attach_voltmeter(nest, recorded_list[1:], sampling_resolution=2., target_neurons=0)

            # record spikes neurons
            sd_list = utils.attach_spikedetector(nest, recorded_list,
                                                pop_list_to_ode=pop_list_to_ode,   # possibility to load sd from s_h
                                                sd_list_to_ode=s_h.sd_list)

            # min and max index for every population
            pop_ids = {**Cereb_class.Cereb_pop_ids, **BGs_class.BGs_pop_ids}
            # dictionary of the population params
            model_dic = utils.create_model_dictionary(N_BGs+N_Cereb, recorded_names, pop_ids, sim_time,
                                                    sample_time=sim_period, settling_time=settling_time,
                                                    trials=trials, b_c_params=b_c_params)
            print('Starting the simulation ...')
            tic = time.time()
            s_h.simulate(tot_trials=trials, pre_sim_time=settling_time)
            toc = time.time()
            print(f'Elapsed simulation time with {cpu_count()} cores: {int((toc - tic) / 60)} min, {(toc - tic) % 60:.0f} sec')

            # potentials = utils.get_voltage_values(nest, vm_list, recorded_names[1:])
            rasters = utils.get_spike_values(nest, sd_list, recorded_names)
            # load mass models states and inputs
            mass_models_sol = {'mass_frs': s_h.ode_sol,
                            'mass_frs_times': s_h.ode_sol_t,
                            'in_frs': s_h.u_sol}

            with open(f'{savings_dir}/model_dic', 'wb') as pickle_file:
                pickle.dump(model_dic, pickle_file)
            with open(f'{savings_dir}/rasters', 'wb') as pickle_file:
                pickle.dump(rasters, pickle_file)
            with open(f'{savings_dir}/mass_models_sol', 'wb') as pickle_file:
                pickle.dump(mass_models_sol, pickle_file)

        elif load_from_file:
            print(f'Simulation results loaded from files')

            with open(f'{savings_dir}/model_dic', 'rb') as pickle_file:
                model_dic = pickle.load(pickle_file)
            with open(f'{savings_dir}/rasters', 'rb') as pickle_file:
                rasters = pickle.load(pickle_file)
            with open(f'{savings_dir}/mass_models_sol', 'rb') as pickle_file:
                mass_models_sol = pickle.load(pickle_file)

        print(f'Showing results obtained from {model_dic["b_c_params"]}')
        import matplotlib.pyplot as plt
        fig3, ax3 = vsl.plot_mass_frs(mass_models_sol, ode_names, u_array=None, # xlim=[0, settling_time+sim_time*trials],
                                      ylim=[None, None])
        if load_from_file:
            plt.show()

        if load_from_file:
            plt.show()

        fr_stats = utils.calculate_fr_stats(rasters, model_dic['pop_ids'], t_start=start_time)
        Cereb_target = np.array([23.538, 151.228,  43.043])

        BGs_target = np.array([12.092, 24.402])     # ['STN', 'SNr']
        fr_target = np.concatenate((Cereb_target, BGs_target))
        fr_target = None

        fr_weights = np.array([1. / 0.4398, 1. / 0.3276, 1. / 0.6918, 1 / 0.276, 1 / 0.242])
        # print the fitness
        filter_range = [30, 50]     # [Hz]
        filter_sd = 6               # [Hz]
        fig6, ax6 =vsl.firing_rate_histogram(fr_stats['fr'], fr_stats['name'], CV_list=fr_stats['CV'],
                                  target_fr=fr_target)
        if load_from_file:
            plt.show()

        fig7, ax7 = vsl.plot_fourier_transform(mass_models_sol["mass_frs"][:, :], sim_period, ode_names,
                                               mean=sum(filter_range)/2, sd=filter_sd, t_start=start_time)
        if load_from_file:
            plt.show()

    experiment = None
    if experiment == 'EBCC':


        n_trials = model_dic["trials"]
        sim_time = model_dic["simulation_time"]
        set_time = model_dic["settling_time"]
        len_trial = int(sim_time + set_time)
        len_trial = int(sim_time)

        first = 100#set_time#all_data['simulations']['DCN_update']['devices']['CS']['parameters']['start_first']
        n_trials = n_trials#all_data['simulations']['DCN_update']['devices']['CS']['parameters']['n_trials']
        between_start = 580 #all_data['simulations']['DCN_update']['devices']['CS']['parameters']['between_start']
        last = first + between_start*(n_trials-1)
        burst_dur = 280#all_data['simulations']['DCN_update']['devices']['CS']['parameters']['burst_dur']
        burst_dur_us = 30#all_data['simulations']['DCN_update']['devices']['US']['parameters']['burst_dur']
        burst_dur_cs = burst_dur- burst_dur_us
        trials_start = np.arange(first, last+between_start, between_start)

        selected_trials = np.linspace(1,n_trials-1,n_trials-1).astype(int) #Can specify trials to be analyzed

        maf_step = 100 #selected step for moving average filter when computing motor output from DCN SDF

        for threshold in range(0,6):
        
            CR, fig = vsl.cr_isi(float(threshold), selected_trials, maf_step, threshold, burst_dur, burst_dur_cs, trials_start, rasters, between_start, plot = True)
            name_fig = "threshold_"+str(threshold)
            fig.savefig(f'{savings_dir}/dcn_{name_fig}.png')

            CR_fig = vsl.plot_CR(CR)
            CR_fig.savefig(f'{savings_dir}/CR_{name_fig}.png')