import subprocess
import sys

trials = 5
exp =  int(sys.argv[1]) # indexing this list -> ['active', 'EBCC']

if exp:
    dopa_lvl = [0,3,4] # -> simulates only physio, middle severity and worse pathological case
else:
    dopa_lvl = range(5)
    
for idx in range(trials):
    for mod in range(3): # indexing this list -> ['external_dopa', 'internal_dopa', 'both_dopa'] external: only bg, internal only cerebellum
        for dopa in dopa_lvl: # indexing this list -> [0.,-0.1,-0.2,-0.4,-0.8]
            program = [f'./main.py {str(idx + 1)} {str(mod)} {str(exp)} {str(dopa)}']
            
            subprocess.call(program, shell=True)    # , capture_output=True)
            print("Finished:" + program[0])
