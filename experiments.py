# -*- coding: utf-8 -*-
"""
"""

from nest_utils import utils
import numpy as np
from time import time

from nest_multiscale.nest_multiscale import set_poisson_fr

class conditioning():
    def __init__(self, nest_, cereb_class, t_start_MF=100, t_start_IO=350, t_end=380, stimulation_IO=500, resolution=0.1, t_sim= 1000., tot_trials = 1):
        self.t_start_MF = t_start_MF
        self.t_start_IO = t_start_IO
        self.t_end = t_end
        self.nest_ = nest_
        self.stimulation_IO = stimulation_IO
        self.n_wind = cereb_class.n_wind

        self.resolution = resolution

        self.rng = np.random.default_rng(round(time() * 1000))


    def start(self, sim_handler, Sim_time, T_sample):
        self.T = T_sample

    def before_loop(self, sim_handler):
        ...

    def beginning_loop(self, sim_handler, trial_time, total_time, in_spikes):
        if trial_time >= self.t_start_IO and trial_time < self.t_end:
            set_poisson_fr(sim_handler.nest, self.stimulation_IO, self.US, total_time,
                           self.T, self.rng, self.resolution, sin_weight=1., in_spikes=in_spikes, n_wind=sim_handler.n_wind)
            
            set_poisson_fr(sim_handler.nest, 0., sim_handler.pop_list_to_nest[:self.n_wind], total_time + self.T,
                           self.T, sim_handler.rng, self.resolution, in_spikes, n_wind=sim_handler.n_wind)
            
    def CS(self, sim_handler, yT, trial_time, total_time):
        
        dt = int((self.t_end - self.t_start_MF)/self.n_wind)
        CS_dict = {}
        for i in range(self.n_wind):
            t_start = self.t_start_MF + i*dt
            CS_dict[t_start] = i
        for i_t in CS_dict.keys():
            i_wind = CS_dict[i_t]
            if trial_time >= i_t and trial_time < i_t+dt:
                pops = [ip for ip in range(self.n_wind)]
                pops.remove(i_wind)
                id_gen = [sim_handler.pop_list_to_nest[id_gen_i] for id_gen_i in pops]
                set_poisson_fr(sim_handler.nest, 0., id_gen, total_time + self.T,
                            self.T, sim_handler.rng, self.resolution, in_spikes = "EBCC", n_wind=sim_handler.n_wind)
           
            
        
    def ending_loop(self, sim_handler, trial_time, total_time, in_spikes="active"):
        if trial_time < self.t_start_MF or trial_time >= self.t_end:
            set_poisson_fr(sim_handler.nest, [0., 0.], sim_handler.pop_list_to_nest[:self.n_wind], total_time + self.T,
                           self.T, sim_handler.rng, self.resolution, in_spikes=in_spikes, n_wind=sim_handler.n_wind)
            
    def define_input_glom(self, trial_time,fr):
        n_spk = 1000/fr*self.bins
        resto = trial_time%self.bins
        spk = np.arange(trial_time-resto, trial_time-resto +self.bins ,n_spk)
        sg = trial_time/self.bins
        self.nest_.SetStatus(self.CS_stim[sg : sg + 1], params={"spike_times": spk})

