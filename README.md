# PD_BGs_Cereb_loop
Multiarea model of Basal ganglia and Cerebellum. It simulate both healthy and Parkinsonian conditions.

## Required packages
- Nest 2.20.x
- fooof
- cerebnest [cerebnest](https://github.com/marcobiasizzo/cereb-nest/tree/nest2.20.2_python3.8) branch nest2.20.2_python3.8
- bgmodel [bgmodel](https://github.com/marcobiasizzo/bgmodel)

## Initialization procedure
- Clone this repository
```
$ git clone https://github.com/benedettagambosi/PD_BGs_Cereb_loop.git
```

## Run 
- Run main.py to perform a simulation
- Run run_script.py to perform multiple times main.py

Simulation data will be automatically saved in savings or share_results.

To define your simulation, firsly edit the "user parameters" at the beginning of the main script.
Other useful parameters can be found in the main file, while model parameters are set in the population repository.
