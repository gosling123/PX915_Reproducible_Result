import os
import time
from utils import read_input



## run_epoch
#
# Runs epoch1d simulations for set intensity 
# @param dir  Directory to store epoch data to and where the input.deck file is
# @param output  Ouput to command line (True) or to run.log file (False)
# @param npro  Number of processors to eun epoch1d on (MPI)    
def run_epoch(dir, output = True, npro = 3):
    """Runs epoch1d simulations for set intensity ."""
    if not isinstance(npro,int) or (npro < 1):
            raise Exception("ERROR: npro argument must be an integer > 0")
    if not isinstance(dir,str):
            raise Exception("ERROR: dir argument must be a string (directory)")
    epoch_path = os.getenv('PX915_RR')
    
    path = f'{epoch_path}/{dir}'
    dir_exist = os.path.exists(path)

    if dir_exist == False:
        print(f'ERROR: Directory {dir} does not exist or is not in this repository.')
        print(f'Please ensure the repository path is added to your .bashrc as PX915_RR and that the directory {dir} is within the repository.')
        return None
    
    input_path = f'{path}/input.deck'
    input_exist = os.path.exists(input_path)

    if input_exist == False:
        print(f'ERROR: Input file not in {dir}.')
        print(f'Please ensure an input file is conatined within {dir} and named input.deck')
        return None

    I = read_input(f'{dir}/', param = 'intensity')
    Ln = read_input(f'{dir}/', param = 'ne_scale_len')
    ppc = read_input(f'{dir}/', param = 'ppc')
    print(f'Output Directory exists at {dir}')
    print(f'running epoch1d for I = {I} W/cm^2 ; Ln = {Ln} m ; PPC = {ppc} ; dir = {dir}')
    start = time.time()
    if output:
        os.system(f'{epoch_path}/epoch.sh ' + str(dir) + ' ' + str(npro) + ' log')
    else:
        os.system(f'{epoch_path}/epoch.sh ' + str(dir) + ' ' + str(npro))
    print(f'{dir} Simulation Complete in {(time.time() - start)/60} minutes')
        

