import subprocess

trials = 5
exp = 1
for idx in range(trials):
    for mod in [1]: #range(3):
        # for dopa in [0,3,4]: #range(5):
        for dopa in range(5):
            program = [f'./main.py {str(idx + 1)} {str(mod)} {str(exp)} {str(dopa)}']

            subprocess.call(program, shell=True)    # , capture_output=True)
            print("Finished:" + program[0])


            
