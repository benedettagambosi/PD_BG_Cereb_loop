# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import h5py
from copy import deepcopy
import sys
sys.path.append("/home/docker/packages/tvb-multiscale/my_tests/PD_test")


tot_trials = 1

pf_pc = 0.4
ratio = 1/18.67
pc_dcn = 0.55
pc_dcnp = 0.03
ratio_pc_dcn = 34/26
ratio_pc_dcnp = 11.5/26
# Synapse parameters: in E-GLIF, 3 synaptic receptors are present: the first is always associated to exc, the second to inh, the third to remaining synapse type
Erev_exc = 0.0  # [mV]	#[Cavallari et al, 2014]
Erev_inh = -80.0  # [mV]
tau_exc = {'golgi': 0.23, 'granule': 5.8, 'purkinje': 1.1, 'basket': 0.64, 'stellate': 0.64, 'dcn': 1.0, 'dcnp': 3.64,
           'io': 1.0}  # tau_exc for pc is for pf input; tau_exc for goc is for mf input; tau_exc for mli is for pf input
tau_inh = {'golgi': 10.0, 'granule': 13.61, 'purkinje': 2.8, 'basket': 2.0, 'stellate': 2.0, 'dcn': 0.7, 'dcnp': 1.14,
           'io': 60.0}
tau_exc_cfpc = 0.4
tau_exc_pfgoc = 0.5
tau_exc_cfmli = 1.2

# Single neuron parameters:
neuron_param = {'golgi': {'t_ref': 2.0, 'C_m': 145.0,'tau_m': 44.0,'V_th': -55.0,'V_reset': -75.0,'Vinit': -62.0,'E_L': -62.0,'V_min':-150.0,
                         'lambda_0':1.0, 'tau_V':0.4,'I_e': 16.214,'kadap': 0.217,'k1': 0.031, 'k2': 0.023,'A1': 259.988,'A2':178.01,
                         'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc,'tau_syn1': tau_exc['golgi'], 'tau_syn2': tau_inh['golgi'], 'tau_syn3': tau_exc_pfgoc},
               'granule': {'t_ref': 1.5, 'C_m': 7.0,'tau_m': 24.15,'V_th': -41.0,'V_reset': -70.0,'Vinit': -62.0,'E_L': -62.0,'V_min': -150.0,
                           'lambda_0':1.0, 'tau_V':0.3,'I_e': -0.888,'kadap': 0.022,'k1': 0.311, 'k2': 0.041,'A1': 0.01,'A2':-0.94,
                           'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc,'tau_syn1': tau_exc['granule'], 'tau_syn2': tau_inh['granule'], 'tau_syn3': tau_exc['granule']},
               'purkinje': {'t_ref': 0.5, 'C_m': 334.0,'tau_m': 47.0,'V_th': -43.0,'V_reset': -69.0,'Vinit': -59.0,'E_L': -59.0,
                            'lambda_0':4.0, 'tau_V':3.5,'I_e': 742.54,'kadap': 1.492,'k1': 0.1950, 'k2': 0.041,'A1': 157.622,'A2':172.622,
                            'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc,'tau_syn1': tau_exc['purkinje'], 'tau_syn2': tau_inh['purkinje'], 'tau_syn3': tau_exc_cfpc},
               'basket': {'t_ref': 1.59, 'C_m': 14.6,'tau_m': 9.125,'V_th': -53.0,'V_reset': -78.0,'Vinit': -68.0,'E_L': -68.0,
                          'lambda_0':1.8, 'tau_V':1.1,'I_e': 3.711,'kadap': 2.025,'k1': 1.887, 'k2': 1.096,'A1': 5.953,'A2':5.863,
                          'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc,'tau_syn1': tau_exc['basket'], 'tau_syn2': tau_inh['basket'], 'tau_syn3': tau_exc_cfmli},
               'stellate': {'t_ref': 1.59, 'C_m': 14.6,'tau_m': 9.125,'V_th': -53.0,'V_reset': -78.0,'Vinit': -68.0,'E_L': -68.0,
                            'lambda_0':1.8, 'tau_V':1.1,'I_e': 3.711,'kadap': 2.025,'k1': 1.887, 'k2': 1.096,'A1': 5.953,'A2':5.863,
                            'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc,'tau_syn1': tau_exc['basket'], 'tau_syn2': tau_inh['basket'], 'tau_syn3': tau_exc_cfmli},
                'dcn': {'t_ref': 0.8, 'C_m': 142.0,'tau_m': 33.0,'V_th': -36.0,'V_reset': -55.0,'Vinit': -45.0,'E_L': -45.0,
                       'lambda_0':3.5, 'tau_V':3.0,'I_e': 75.385,'kadap': 0.408,'k1': 0.697, 'k2': 0.047,'A1': 13.857,'A2':3.477,
                       'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc,'tau_syn1': tau_exc['dcn'], 'tau_syn2': tau_inh['dcn'], 'tau_syn3': tau_exc['dcn']},
               'dcnp': {'t_ref': 0.8, 'C_m': 56.0,'tau_m': 56.0,'V_th': -39.0,'V_reset': -55.0,'Vinit': -40.0,'E_L': -40.0,
                        'lambda_0':0.9, 'tau_V':1.0,'I_e': 2.384,'kadap': 0.079,'k1': 0.041, 'k2': 0.044,'A1': 176.358,'A2':176.358,
                        'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc,'tau_syn1': tau_exc['dcnp'], 'tau_syn2': tau_inh['dcnp'], 'tau_syn3': tau_exc['dcnp']},
               'io': {'t_ref': 1.0, 'C_m': 189.0,'tau_m': 11.0,'V_th': -35.0,'V_reset': -45.0,'Vinit': -45.0,'E_L': -45.0,
                      'lambda_0':1.2, 'tau_V':0.8,'I_e': -18.101,'kadap': 1.928,'k1': 0.191, 'k2': 0.091,'A1': 1810.93,'A2':1358.197,
                      'E_rev1': Erev_exc, 'E_rev2': Erev_inh, 'E_rev3': Erev_exc,'tau_syn1': tau_exc['io'], 'tau_syn2': tau_inh['io'], 'tau_syn3': tau_exc['io']},
                'death_purkinje': {'t_ref': 0.5, 'C_m': 1000.0, 'tau_m': 47.0, 'V_th': 100.0, 'V_reset': -80.0, 'Vinit': -80.0,
                 'E_L': -80.0,
                 'lambda_0': 4.0, 'tau_V': 3.5, 'I_e': 0., 'kadap': 0., 'k1': 1., 'k2': 1., 'A1': 0.,
                 'A2': 0.,
                 'E_rev1': Erev_inh, 'E_rev2': Erev_inh, 'E_rev3': Erev_inh, 'tau_syn1': tau_exc['purkinje'],
                 'tau_syn2': tau_inh['purkinje'], 'tau_syn3': tau_exc_cfpc},}


# Connection weights
conn_weights = {'pc_dcn': 0.4, 'pc_dcnp': 0.12, 'pf_bc': 0.015, 'pf_goc': 0.05,'pf_pc': pf_pc*ratio, \
                'pf_sc': 0.015, 'sc_pc': 0.3, 'aa_goc': 1.2, 'aa_pc': 0.7, 'bc_pc': 0.3, 'dcnp_io': 3.0, 'gj_bc': 0.2, 'gj_sc': 0.2, 'glom_dcn': 0.05,\
                'glom_goc': 1.5, 'glom_grc': 0.15, 'goc_glom': 0.0, 'gj_goc': 0.3,'goc_grc': 0.6, 'io_dcn': 0., 'io_dcnp': 0.,\
                 'io_pc': 0.0, 'io_bc': .0,'io_sc': .0,}##'io_dcn': 0.1, 'io_dcnp': 0.2,'io_pc': 40.0, 'io_bc': 1.0,'io_sc': 1.0,

# Connection delays
conn_delays = {'aa_goc': 2.0, 'aa_pc': 2.0, 'bc_pc': 4.0, 'dcnp_io': 20.0, 'gj_bc': 1.0, 'gj_sc': 1.0, 'glom_dcn': 4.0,
               'glom_goc': 4.0, 'glom_grc': 4.0, 'goc_glom': 0.5, 'gj_goc': 1.0, 'goc_grc': 2.0, 'io_dcn': 4.0, 'io_dcnp': 5.0,
               'io_bc': 70.0,'io_sc': 70.0, 'io_pc': 4.0, 'pc_dcn': 4.0, 'pc_dcnp': 4.0, 'pf_bc': 5.0, 'pf_goc': 5.0,'pf_pc': 5.0,
               'pf_sc': 5.0, 'sc_pc':5.0}

sd_iomli = 10.0          # IO-MLI delayes are set as normal distribution to reproduce the effect of spillover-based transmission
min_iomli = 40.0

# Connection receptors
conn_receptors = {'aa_goc': 3, 'aa_pc': 1, 'bc_pc': 2, 'dcnp_io': 2, 'gj_bc': 2, 'gj_sc': 2, 'glom_dcn': 1,
               'glom_goc': 1, 'glom_grc': 1, 'goc_glom': 1, 'gj_goc': 2, 'goc_grc': 2, 'io_dcn': 1, 'io_dcnp': 1,
               'io_bc': 3,'io_sc': 3, 'io_pc': 3, 'pc_dcn': 2, 'pc_dcnp': 2, 'pf_bc': 1, 'pf_goc': 3,'pf_pc': 1,
               'pf_sc': 1, 'sc_pc': 2}

# Receiver plastic (name of post-synaptic neurons for heterosynaptic plastic connections)
receiver = {'pf_pc': 'purkinje', 'pf_bc': 'basket', 'pf_sc': 'stellate', 'glom_dcn': 'dcn', "io_bc":"basket","io_pc":"purkinje", "io_sc":"stellate"}

# Plasticity parameters
LTD_PFPC = -0.02
LTP_PFPC = 0.002
LTD_PFMLI = -0.01
LTP_PFMLI = 0.001

mli = False
LTD_MFDCN = -0.001
LTP_MFDCN = 0.0001
LTD_PCDCN = -0.001
LTP_PCDCN = 0.0001

PC_DCN_alpha = LTD_PCDCN/LTP_PCDCN
PC_DCN_lambda = LTP_PCDCN

class Cereb_class:
    def __init__(self, nest, hdf5_file_name, cortex_type = "", n_spike_generators='n_glomeruli',
                 mode='external_dopa', experiment='active', dopa_depl=0, LTD=LTD_PFPC, LTP=LTP_PFPC, n_wind = 1):
        # create Cereb neurons and connections
        # Create a dictionary where keys = nrntype IDs, values = cell names (strings)
        # Cell type ID (can be changed without constraints)
        self.cell_type_ID = {'golgi': 1,
                             'glomerulus': 2,
                             'granule': 3,
                             'purkinje': 4,
                             'basket': 5,
                             'stellate': 6,
                             'dcn': 7,  # this project to cortex
                             'dcnp': 8,  # while this project to IO (there is dcnp_io connection) -> opposite to paper!!
                              'io': 9}

        self.hdf5_file_name = hdf5_file_name
        self.n_wind = n_wind
        self.Cereb_pops, self.Cereb_pop_ids, self.WeightPFPC, self.PF_PC_conn = self.create_Cereb(nest, hdf5_file_name,mode, experiment, dopa_depl, LTD, LTP)
        
        background_pops = self.create_ctxinput(nest, pos_file=None, in_spikes='background')

        if not cortex_type:                                                                                          
            self.CTX_pops = background_pops
        else:
            self.CTX_pops = self.create_ctxinput(nest, pos_file=None, in_spikes=cortex_type, n_spike_generators=n_spike_generators)

    def create_Cereb(self, nest_, pos_file, mode, experiment, dopa_depl, LTD, LTP):
        ### Load neuron positions from hdf5 file and create them in NEST:
        with h5py.File(pos_file, 'r') as f:
            positions = np.array(f['positions'])

        if experiment == 'EBCC':
            plasticity = False #%SET BACK TO TRUE
        else:
            plasticity = False

        id_2_cell_type = {val: key for key, val in self.cell_type_ID.items()}
        # Sort nrntype IDs
        sorted_nrn_types = sorted(list(self.cell_type_ID.values()))
        # Create a dictionary; keys = cell names, values = lists to store neuron models
        neuron_models = {key: [] for key in self.cell_type_ID.keys()}

        # All cells are modelled as E-GLIF models;
        # with the only exception of Glomeruli (not cells, just modeled as
        # relays; i.e., parrot neurons)
        for cell_id in sorted_nrn_types:
            cell_name = id_2_cell_type[cell_id]
            if cell_name != 'glomerulus':
                if cell_name not in nest_.Models():
                    nest_.CopyModel('eglif_cond_alpha_multisyn', cell_name)
                    nest_.SetDefaults(cell_name, neuron_param[cell_name])
            else:
                if cell_name not in nest_.Models():
                    nest_.CopyModel('parrot_neuron', cell_name)

            cell_pos = positions[positions[:, 1] == cell_id, :]
            n_cells = cell_pos.shape[0]
            neuron_models[cell_name] = nest_.Create(cell_name, n_cells)

            # delete death PCs
            if cell_name == 'purkinje':
                if mode == 'internal_dopa' or mode == 'both_dopa':
                    n_PC_alive = int(cell_pos.shape[0] * (1. - 0.5 * (-dopa_depl) / 0.8))  # number of PC still alive
                else:
                    n_PC_alive = cell_pos.shape[0]

                all_purkinje = list(neuron_models['purkinje'])
                np.random.shuffle(all_purkinje)
                selected_purkinje = all_purkinje[:n_PC_alive]      # indexes of PC still alive
                death_purkinje = all_purkinje[n_PC_alive:]
                for PC in death_purkinje:
                    nest_.SetStatus(PC, neuron_param['death_purkinje'])

        
        with h5py.File(pos_file, 'r') as f:
            vt = {}
            for conn in conn_weights.keys():
                # pre_model = conn_names[conn][0]
                # post_model = conn_names[conn][1]
                # connection = np.array(f['connections/' + conn])
                # pre_start = nest.GetNodes({"model":pre_model})[0].tolist()
                # post_start = nest.GetNodes({"model":post_model})[0].tolist()
                # pre = [int(x + 1 -pre_start) for x in connection[:, 0]]  # pre and post may contain repetitions!
                # post = [int(x + 1-post_start) for x in connection[:, 1]]
                connection = np.array(f['connections/' + conn])
                pre = [int(x + 1) for x in connection[:, 0]]  # pre and post may contain repetitions!
                post = [int(x + 1) for x in connection[:, 1]]
                
                if "pf_pc" in conn and plasticity:
                    vt[receiver[conn]] = nest_.Create("volume_transmitter_alberto",len(np.unique(post)))
                    print("Created vt for ", conn, " connections")
                    for n,vti in enumerate(vt[receiver[conn]]):
                        nest_.SetStatus([vti],{"vt_num" : n})
                    
                    # Set plastic connection parameters for stdp_synapse_sinexp synapse model
                    name_plast = 'plast_'+conn
                    nest_.CopyModel('stdp_synapse_sinexp', name_plast)
                    nest_.SetDefaults(name_plast,{"A_minus": LTD,   # double - Amplitude of weight change for depression
                                                "A_plus": LTP,   # double - Amplitude of weight change for facilitation
                                                "Wmin": 0.0,    # double - Minimum synaptic weight
                                                "Wmax": 4000.0,     # double - Maximum synaptic weight
                                                "vt": vt[receiver[conn]][0]})
                        
                    syn_param = {"model": name_plast, "weight": conn_weights[conn], "delay": conn_delays[conn], "receptor_type": conn_receptors[conn]}

                    # Create connection and associate a volume transmitter to them
                    for vt_num, post_cell in enumerate(np.unique(post)):
                                        syn_param["vt_num"] = float(vt_num)
                                        indexes = np.where(post == post_cell)[0]
                                        pre_neurons = np.array(pre)[indexes]
                                        post_neurons = np.array(post)[indexes]
                                        nest_.Connect(pre_neurons,post_neurons, {"rule": "one_to_one"}, syn_param)

                elif (conn == "pf_bc" and mli) or (conn =="pf_sc" and mli):
                    # Create 1 volume transmitter for each post-synaptic neuron
                    vt[receiver[conn]] = nest_.Create("volume_transmitter_alberto",len(np.unique(post)))
                    print("Created vt for ", conn, " connections")
                    for n,vti in enumerate(vt[receiver[conn]]):
                        nest_.SetStatus([vti],{"vt_num" : n})
                    
                    # Set plastic connection parameters for stdp_synapse_alpha synapse model
                    name_plast = 'plast_'+conn
                    nest_.CopyModel('stdp_synapse_alpha', name_plast)
                    nest_.SetDefaults(name_plast,{"A_minus": LTD_PFMLI,   # double - Amplitude of weight change for depression
                                                    "A_plus": LTP_PFMLI,   # double - Amplitude of weight change for facilitation
                                                    "Wmin": 0.0,    # double - Minimum synaptic weight
                                                    "Wmax": 4000.0,     # double - Maximum synaptic weight
                                                    "vt": vt[receiver[conn]][0]})
                        
                    syn_param = {"model": name_plast, "weight": conn_weights[conn], "delay": conn_delays[conn], "receptor_type": conn_receptors[conn]}

                    # Create connection and associate a volume transmitter to them
                    for vt_num, post_cell in enumerate(np.unique(post)):
                                            syn_param["vt_num"] = float(vt_num)
                                            indexes = np.where(post == post_cell)[0]
                                            pre_neurons = np.array(pre)[indexes]
                                            post_neurons = np.array(post)[indexes]
                                            nest_.Connect(pre_neurons,post_neurons, {"rule": "one_to_one"}, syn_param)
                
                # Static connections with distributed delay                                
                elif conn == "io_bc" or conn == "io_sc":
                    from scipy import stats
                    sample =stats.truncnorm.rvs(a = (min_iomli-conn_delays[conn])/sd_iomli,b = np.inf, loc=conn_delays[conn], scale=sd_iomli, size=len(pre))

                    syn_param = {"synapse_model": "static_synapse", "weight": np.ones(len(pre))*conn_weights[conn], \
                                # "delay": {'distribution': 'normal_clipped', 'low': min_iomli, 'mu': conn_delays[conn],'sigma': sd_iomli},
                                "delay":sample,
                                "receptor_type":np.ones(len(pre))*conn_receptors[conn]}
                    nest_.Connect(np.array(pre),np.array(post), {"rule": "one_to_one"}, syn_param)
                                                    
                # Static connections with constant delay
                else:
                # elif conn in ["glom_grc", "pf_pc","pc_dcn"]:
                    syn_param = {"synapse_model": "static_synapse", "weight": np.ones(len(pre))*conn_weights[conn], "delay": np.ones(len(pre))*conn_delays[conn],"receptor_type": np.ones(len(pre))*conn_receptors[conn]}
                    nest_.Connect(np.array(pre),np.array(post), {"rule": "one_to_one"}, syn_param)
    
                # If a connection is a teaching one, also the corresponding volume transmitter should be connected
                if (conn == "io_bc" or conn == "io_sc") and mli:                                     
                    post_n = np.array(post)-neuron_models[receiver[conn]][0] +vt[receiver[conn]][0]
                    nest_.Connect(np.asarray(pre, int), np.asarray(post_n, int), {"rule": "one_to_one"},{"model": "static_synapse", "weight": 1.0, "delay": 1.0})
                
                if conn == "io_pc" and plasticity:                                     
                    post_n = np.array(post)-neuron_models[receiver[conn]][0] +vt[receiver[conn]][0]
                    nest_.Connect(np.asarray(pre, int), np.asarray(post_n, int), {"rule": "one_to_one"},{"model": "static_synapse", "weight": 1.0, "delay": 1.0})
        

                    print("Connections ", conn, " done!")
    
        Cereb_pops = neuron_models
        pop_ids = {key: (min(neuron_models[key].tolist()), max(neuron_models[key].tolist())) for key, _ in self.cell_type_ID.items()}
        WeightPFPC = None
        PF_PC_conn = None
        return Cereb_pops, pop_ids, WeightPFPC, PF_PC_conn


    def create_ctxinput(self, nest_, pos_file=None, in_spikes='poisson', n_spike_generators='n_glomeruli',
                        experiment='active', CS ={"start":500., "end":760., "freq":36.}, US ={"start":750., "end":760., "freq":500.}, tot_trials = None, len_trial = None):

        glom_id, _ = self.get_glom_indexes(self.Cereb_pops['glomerulus'], "EBCC")
        id_stim = np.sort(list(set(glom_id)))
        n = len(id_stim)
        IO_id = self.Cereb_pops['io']

        if in_spikes == "background":
        # Background as Poisson process, always present

            CTX = nest_.Create('poisson_generator', len(self.Cereb_pops['glomerulus']),params={'rate': 4.0, 'start': 0.0})
            nest_.Connect(CTX, self.Cereb_pops['glomerulus'], {"rule":"one_to_one"})  # connected to all of them

        elif in_spikes == 'spike_generator':
            print('The cortex input is a spike generator')

            if n_spike_generators == 'n_glomeruli':
                n_s_g = n  # create one spike generator for each input population
            else:
                n_s_g = n_spike_generators  # create n_s_g, randomly connected to the input population

            # create a cortex input
            CTX = nest_.Create("spike_generator", n_s_g)  # , params=generator_params)
            syn_param = {"delay": 2.0}

            # connect
            if n_spike_generators == 'n_glomeruli':
                nest_.Connect(CTX, id_stim, {'rule': 'one_to_one'}, syn_param)
            else:
                np.random.shuffle(id_stim)
                n_targets = len(id_stim) / n_s_g
                for i in range(n_s_g - 1):
                    post = id_stim[round(i * n_targets):round((i + 1) * n_targets)]
                    nest_.Connect(CTX[i], np.sort(post), {'rule': 'all_to_all'})
                post = id_stim[round((n_s_g - 1) * n_targets):]
                nest_.Connect(CTX[n_s_g - 1], np.sort(post), {'rule': 'all_to_all'}, syn_param)

        elif in_spikes == 'spike_generator_ebcc':
            print('The cortex input is a spike generator')

            if n_spike_generators == 'n_glomeruli':
                n_s_g = n  # create one spike generator for each input population
            else:
                n_s_g = n_spike_generators  # create n_s_g, randomly connected to the input population

            # create a cortex input
            CTX = nest_.Create("spike_generator", n_s_g)  # , params=generator_params)
            syn_param = {"delay": [2.0]*n}

            # connect
            if n_spike_generators == 'n_glomeruli':
                nest_.Connect(CTX, id_stim, {'rule': 'one_to_one'}, syn_param)
            else:
                np.random.shuffle(id_stim)
                n_targets = len(id_stim) / n_s_g
                for i in range(n_s_g - 1):
                    post = id_stim[round(i * n_targets):round((i + 1) * n_targets)]
                    nest_.Connect(CTX[i], np.sort(post), {'rule': 'all_to_all'})
                post = id_stim[round((n_s_g - 1) * n_targets):]
                nest_.Connect(CTX[n_s_g - 1], np.sort(post), {'rule': 'all_to_all'}, syn_param)

            # Create an empty dictionary 
            split_dict = {}  
            size_chunks = int(len(CTX)/self.n_wind)
            # Split the original list into chunks of size n
            k= 0 
            for i in range(0, len(CTX)-size_chunks, size_chunks): 
                split_list = CTX[i:i+size_chunks] 
                key = "CTX_" + str(k) 
                split_dict[key] = split_list 
                k+=1
            self.CTX_pops = split_dict

            IO_id = self.Cereb_pops['io']
            US_matrix = np.concatenate(
                            [
                                np.arange(US["start"], US["end"] + 2, 2)
                                + len_trial * t
                                for t in range(tot_trials)
                            ]
                        )
            
            US_stim = nest_.Create("spike_generator", len(IO_id), {"spike_times":US_matrix})
            
            nest_.Connect(US_stim, IO_id, "all_to_all", {"receptor_type": 1, "delay":1.,"weight":10.}) #10.

            self.US = US_stim

        elif in_spikes == 'dynamic_poisson':
            print('The cortex input is a poissonian process')

            CS_FREQ = 36.
            # Simulate a conscious stimulus
            CTX = nest_.Create('poisson_generator', params={'rate': CS_FREQ})
            CTX_id = np.array(CTX.tolist())
            nest_.Connect(CTX_id, np.array(id_stim))

        else:
            print("ATTENTION! no cortex input generated")
            CTX = []
            pass

        return {'CTX': CTX}

    def get_glom_positions_xz(self):
        _, idx = self.get_glom_indexes(self.Cereb_pops['glomerulus'])
        with h5py.File(self.hdf5_file_name, 'r') as f:
            positions = np.array(f['positions'])
            gloms_pos = positions[positions[:, 1] == self.cell_type_ID['glomerulus'], :]
            gloms_pos_xz = gloms_pos[:, [2, 4]]
        return gloms_pos_xz[idx, :]

    def get_glom_indexes(self, glom_pop, experiment):
        with h5py.File(self.hdf5_file_name, 'r') as f:
            positions = np.array(f['positions'])
            glom_posi = positions[positions[:, 1] == self.cell_type_ID['glomerulus'], :]
            glom_xz = glom_posi[:, [2, 4]]

            if experiment == 'EBCC' or experiment == 'active':
                x_c, z_c = 200., 200.

                RADIUS = 150.  # [um] - radius of glomeruli stimulation cylinder to avoid border effects
                # Connection to glomeruli falling into the selected volume, i.e. a cylinder in the Granular layer
                bool_idx = np.sum((glom_xz - np.array([x_c, z_c])) ** 2, axis=1).__lt__(
                    RADIUS ** 2)  # lt is less then, <
                target_gloms = glom_posi[bool_idx, 0] + 1
                id_stim = list(set([glom for glom in glom_pop.tolist() if glom in target_gloms]))

            elif experiment == 'robot':
                x_high_bool = np.array(glom_xz[:, 0].__gt__(200 - 150))      # (200 - 120))  # z > 200 (left in paper)
                x_low_bool = np.array(glom_xz[:, 0].__lt__(200 + 150))     # (200 + 120))  # z > 200 (left in paper)
                z_high_bool = np.array(glom_xz[:, 1].__gt__(200 - 150))       # (200 - 20))  # 180 < z < 220 (right in paper)
                z_low_bool = np.array(glom_xz[:, 1].__lt__(200 + 150))      # (200 + 20))
                bool_idx = x_low_bool & x_high_bool & z_low_bool & z_high_bool# 180 < z < 220 (right in paper)
                idx = glom_posi[bool_idx, 0] + 1
                id_stim = list(set([glom for glom in glom_pop if glom in idx]))

        return id_stim, bool_idx

def plot_nest_results_raster(raster, model_dic, SIMULATION_LENGTH):

    import plotly.graph_objs as go
    from plotly.subplots import make_subplots

    # ######################### PLOTTING PSTH AND RASTER PLOTS ########################

    CELL_TO_PLOT = ['glomerulus',  "granule",  "basket", "stellate", 'purkinje', 'dcn','dcnp', 'io']

    cells = {'glomerulus': [raster[0]["times"], raster[0]["neurons_idx"]],
             'granule': [raster[1]["times"], raster[1]["neurons_idx"]],
             'basket': [raster[2]["times"], raster[2]["neurons_idx"]],
             'stellate': [raster[3]["times"], raster[3]["neurons_idx"]],
             'purkinje': [raster[4]["times"], raster[4]["neurons_idx"]],
             'dcn': [raster[5]["times"], raster[5]["neurons_idx"]],
             'dcnp': [raster[6]["times"], raster[6]["neurons_idx"]],
             'io': [raster[7]["times"], raster[7]["neurons_idx"]],}

    color = {'granule': '#E62214',  # 'rgba(255, 0, 0, .8)',
             'golgi': '#332EBC',  # 'rgba(0, 255, 0, .8)',
             'glomerulus': '#0E1030',  # rgba(0, 0, 0, .8)',
             'purkinje': '#0F8944',  # 'rgba(64, 224, 208, .8)',
             'stellate': '#FFC425',  # 'rgba(234, 10, 142, .8)',
             'basket': '#F37735',
             'io': 'rgba(75, 75, 75, .8)',
             'dcn': 'rgba(100, 100, 100, .8)',
             'dcnp': '#080808'}  # 'rgba(234, 10, 142, .8)'}

    # PSTH

    neuron_number = {}
    for cell in CELL_TO_PLOT:
        neuron_number[cell] = model_dic["pop_ids"][cell][1] - model_dic["pop_ids"][cell][0]

    def metrics(spikeData, TrialDuration, cell, figure_handle, sel_row):
        id_spikes = np.sort(np.unique(spikeData, return_index=True))
        bin_size = 5  # [ms]
        n_bins = int(TrialDuration / bin_size) + 1
        psth, tms = np.histogram(spikeData, bins=n_bins, range=(0, TrialDuration))

        # absolute frequency
        abs_freq = np.zeros(id_spikes[0].shape[0])
        for idx, i in enumerate(id_spikes[0]):
            count = np.where(spikeData == i)[0]
            abs_freq[idx] = count.shape[0]

        # mean frequency
        m_f = (id_spikes[0].shape[0]) / ((TrialDuration / 1000))

        layout = go.Layout(
            scene=dict(aspectmode='data'),
            xaxis={'title': 'time (ms)'},
            yaxis={'title': 'number of spikes'}
        )

        n_neurons = neuron_number[cell]
        if cell == "granule":
            n_neurons = int(np.round(n_neurons/10))
        figure_handle.add_trace(go.Bar(
            x=tms[0:len(tms) - 1],
            y=psth / ((bin_size * 0.001) * n_neurons),
            width=4.0,
            marker=dict(
                color=color[cell])
        ), row=sel_row, col=1)


        return tms

    # RASTER
    def raster(times, cell_ids, cell, fig_handle, sel_row):
        trace0 = go.Scatter(
            x=times,
            y=cell_ids,
            name='',
            mode='markers',
            marker=dict(
                size=4,
                color=color[cell],
                line=dict(
                    width=.2,
                    color='rgb(0, 0, 0)'
                )
            )
        )
        fig_handle.add_trace(trace0, row=sel_row, col=1)

    fig_psth = make_subplots(rows=len(CELL_TO_PLOT), cols=1, subplot_titles=CELL_TO_PLOT, x_title='Time [ms]',
                             y_title='Frequency [Hz]')
    fig_raster = make_subplots(rows=len(CELL_TO_PLOT), cols=1, subplot_titles=CELL_TO_PLOT, x_title='Time [ms]',
                               y_title='# cells')
    num = 1
    for c in CELL_TO_PLOT:
        times = cells[c][0]
        cell_ids = cells[c][1]
        metrics(times, SIMULATION_LENGTH, c, fig_psth, num)
        raster(times, cell_ids, c, fig_raster, num)
        num += 1
    fig_psth.update_xaxes(range=[0, SIMULATION_LENGTH * 1.1])
    fig_raster.update_xaxes(range=[0, SIMULATION_LENGTH * 1.1])
    fig_psth.update_layout(showlegend=False)
    fig_raster.update_layout(showlegend=False)
    
    fig_psth.show()
    fig_raster.show()
    
    return fig_psth, fig_raster

if __name__ == "__main__":
    #pass 
    import sys
    sys.path += ['/home/docker/packages/tvb-multiscale/my_test/']
    import nest
    from pathlib import Path
    from nest_utils import utils
    import pickle
    import os
    CORES = 30
    VIRTUAL_CORES = 1
    RESOLUTION = 1.
    run_on_vm = False
# set number of kernels
    nest.ResetKernel()
    nest.SetKernelStatus({"total_num_virtual_procs": CORES, "resolution": RESOLUTION})
    nest.set_verbosity("M_ERROR")  # reduce plotted info
    # MODULE_PATH = str(Path.home()) + '/nest/lib/nest/ml_module'
    nest.Install("ml_module")  # Import my_BGs module
    # MODULE_PATH = str(Path.home()) + '/nest/lib/nest/cerebmodule'
    nest.Install("cerebmodule")  # Import CerebNEST

    hdf5_file_name = "/home/docker/packages/tvb-multiscale/tvb_data/mouse/rising-net/scaffold_full_IO_400.0x400.0_microzone.hdf5"
    Cereb_recorded_names = ['glomerulus',  "granule",  "basket", "stellate", 'purkinje', 'io','dcn','dcnp',]
    

    len_trial = 1000.
    set_time = 0.

    for j in range(1):
        nest.ResetKernel()
        nest.SetKernelStatus({'rng_seed': 100 * j + 1,
                            # 'rng_seeds': [100 * j + k for k in range(2,26)],
                            'local_num_threads': CORES, 'total_num_virtual_procs': CORES})
        nest.set_verbosity("M_ERROR")  # reduce plotted info
        savings_dir = f'/home/docker/packages/tvb-multiscale/my_test/Cereb_nest3/savings/cereb_active_trial_{j}'
        if not os.path.exists(savings_dir): os.makedirs(savings_dir)  # create folder if not present


        #nest.ResetKernel()
        cereb = Cereb_class(nest, hdf5_file_name,cortex_type="dynamic_poisson", n_spike_generators=500,
                    mode='external_dopa', experiment='active', dopa_depl=0)
    
        recorded_list = [cereb.Cereb_pops[name] for name in Cereb_recorded_names]
        sd_list = utils.attach_spikedetector(nest, recorded_list)
        
        model_dict = utils.create_model_dictionary(0, Cereb_recorded_names, {**cereb.Cereb_pop_ids}, len_trial,
                                                    sample_time=1., settling_time=set_time,
                                                    trials=tot_trials, b_c_params=[])
        

        print("Simulating settling time: " + str(set_time) )

        nest.Simulate(set_time)

        
        for trial in range(tot_trials):
            
            print("Simulating trial: " + str(trial +1) +" of "+ str(tot_trials))
            
            nest.Simulate(len_trial)

        
        rasters = utils.get_spike_values(nest, sd_list, Cereb_recorded_names)
        
        plot_nest_results_raster(rasters, model_dict, len_trial)
        
        with open(f'{savings_dir}/rasters', 'wb') as pickle_file:
            pickle.dump(rasters, pickle_file)


        with open(f'{savings_dir}/model_dic', 'wb') as pickle_file:
            pickle.dump(model_dict, pickle_file)


        import matplotlib.pyplot as plt
        from nest_utils import utils, visualizer as vsl
        fr_stats = utils.calculate_fr_stats(rasters, model_dict['pop_ids'], t_start=10.)

        with open(f'{savings_dir}/fr_stats', 'wb') as pickle_file:
            pickle.dump(fr_stats, pickle_file)
        print(fr_stats['fr'])
        print(fr_stats['name'])
        
        # fig3, ax3 = vsl.firing_rate_histogram(fr_stats['fr'], fr_stats['name'], CV_list=fr_stats['CV'],
        #                                 target_fr=np.array([0.,0.,0.,0.,0.]))
        # plt.show()