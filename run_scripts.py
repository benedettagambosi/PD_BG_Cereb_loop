import subprocess

trials = 5
exp = 1 # indexing this list -> ['active', 'EBCC']
for idx in range(trials):
    for mod in range(3): # indexing this list -> ['external_dopa', 'internal_dopa', 'both_dopa'] external: only bg, internal only cerebellum
        for dopa in range(5): # indexing this list -> [0.,-0.1,-0.2,-0.4,-0.8]
            program = [f'./main.py {str(idx + 1)} {str(mod)} {str(exp)} {str(dopa)}']

            subprocess.call(program, shell=True)    # , capture_output=True)
            print("Finished:" + program[0])


            
