# PD_BGs_Cereb_loop
Multiarea model of Basal ganglia and Cerebellum. It simulate both healthy and Parkinsonian conditions.

## Required packages
- Nest 2.20.x
- fooof 1.0.0
- cerebnest [cerebnest](https://github.com/marcobiasizzo/cereb-nest/tree/nest2.20.2_python3.8) branch nest2.20.2_python3.8
- bgmodel [bgmodel](https://github.com/marcobiasizzo/bgmodel)

## Initialization procedure
- Clone this repository
```
Follow instruction in README_installation.md
```

## Download the ZENODO archive and run simulations
```
wget https://zenodo.org/records/10970203/files/PD_BG_Cereb_loop.zip
unzip PD_BG_Cereb_loop
cd PD_BG_Cereb_loop
```

here you can:

* Run `main.py` to perform a simulation
TODO: add instructions

* Run `run_script.py` to perform multiple times main.py

## Run 
* Run `main.py` to perform a simulation
    This scripts need to receive the following arguments:
    - simulation_id: id to be added on the folder name identifying the simulation
    - mode: 0 for dopamine depletion only in the BG, 1 for dopamine depletion only in the cerebellum, 2 for dopamine depletion in both regions
    - experiment: 0 for general motor state, 1 for EBCC
    - dopamine_depetion_level: 0..4 indexing this list  [0.,-0.1,-0.2,-0.4,-0.8] where 0. is physiological condition, -0.8 is most sever PD

    e.g. `python3 main.py 0 2 0 4` simulates the network in the general motor state with dopamine depletion in both areas in worst pathological case.

* Run `run_script.py` to perform multiple times main.py (default is 5 times for replicating the results)
    This scripts need to receive the following argument:
    - experiment: 0 for general motor state, 1 for EBCC

    e.g. 
    `python3 run_scripts.py 0` simulates the network in the general motor state with all dopamine depletion levels in all sites (dopamine depletion only in the BG, only in the cerebellum, and in both regions).
    `python3 run_scripts.py 1` simulates the network berforming the EBCC protocol with three dopamine depletion levels (physiological case, medium severity and worse pathological case) in all sites (dopamine depletion only in the BG, only in the cerebellum, and in both regions).

* Run `generate_figures.ipynb` to generate the figures

Simulation data will be automatically saved in `last_result`.

One can also define their own simulations, editing the "user parameters" at the beginning of the main script.
Other useful parameters can be found in the main file, while model parameters are set in the population repository.

## Previous repository
https://github.com/marcobiasizzo/BGs_Cereb_nest_PD
