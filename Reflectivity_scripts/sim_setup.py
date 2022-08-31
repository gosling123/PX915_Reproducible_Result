from utils import *

## create_dir
#
# Creates a directory inside repository to host an epoch sim
# @param dir  Directory name
def create_dir(dir):
    if not isinstance(dir,str):
            raise Exception("ERROR: dir argument must be a string (directory)")
    epoch_path = os.getenv('PX915_RR')
    try:
        os.mkdir(f'{epoch_path}/{dir}')
    except:
        print(f'Directory already exists')

## create_sub_dir
#
# Creates a directory and directory within inside repository to host an epoch sim
# @param dir  Directory name
# @param sub_dir  Sub-Directory name
def create_sub_dir(dir, sub_dir):
    if not isinstance(dir,str):
            raise Exception("ERROR: dir argument must be a string (directory)")
    if not isinstance(sub_dir,str):
            raise Exception("ERROR: sub_dir argument must be a string (sub-directory)")
    epoch_path = os.getenv('PX915_RR')
    try:
        os.mkdir(f'{epoch_path}/{dir}')
    except:
        print(f'{dir} directory already exists')
    try:
        os.mkdir(f'{epoch_path}/{dir}/{sub_dir}')
    except:
        print(f'{dir}/{sub_dir} directory already exists')

## input_deck
#
# Copies a preset input deck into chosen directory setting the values of
# laser intensity (I), density scale length (Ln) and particles per cell (ppc).
# @param I  Laser intensity (W/cm^2)  
# @param Ln  Density scale length (m)
# @param ppc  Paricles per cell
# @param dir  Directory name
# @param input_file  Input file base to copy from input_decks folder
def input_deck(I, Ln, ppc, dir, input_file):
    if not isinstance(ppc,int) or (ppc < 1):
            raise Exception("ERROR: ppc argument must be an integer > 0")
    if not isinstance(dir,str):
            raise Exception("ERROR: dir argument must be a string (directory)")
    if not isinstance(input_file,str):
            raise Exception("ERROR: input_file argument must be a string (.deck file)")
    epoch_path = os.getenv('PX915_RR')  
    try:
        os.system(f'{epoch_path}/{dir}/touch input.deck')
    except:
        print(f'{dir}/input.deck already exists')
    try:
        os.system(f'cp {epoch_path}/{input_file} {epoch_path}/{dir}/input.deck')
    except:
        return print('ERROR: Ensure the input_file name is correct as in the input_decks directory')
    replace_line('intensity_w_cm2 =', f'intensity_w_cm2 = {I}', fname = f'{epoch_path}/{dir}/input.deck')
    replace_line('Ln =', f'Ln = {Ln}', fname = f'{epoch_path}/{dir}/input.deck')
    replace_line('PPC =', f'PPC = {ppc}', fname = f'{epoch_path}/{dir}/input.deck')


## epoch_sim_sub_dirs
#
# Creates epoch sim directories with sub directories and populates it with chosen input.deck format.
# @param dir  Directory name
# @param sub_dir  Sub-Directory name
# @param input_file  Input file base to copy from input_decks folder
# @param I  Laser intensity (W/cm^2)  
# @param Ln  Density scale length (m)
# @param ppc  Paricles per cell
# @param time  Time string in the form of hours:minutes:seconds (hh:mm:ss)
# @param nodes  Number of computational nodes to request
# @param hpc  (Logical) Whether to add hpc (avon) hob script or not
def epoch_sim_sub_dir(dir, sub_dir, input_file, I, Ln, ppc = 100):
    if not isinstance(ppc,int) or (ppc < 1):
            raise Exception("ERROR: ppc argument must be an integer > 0")
    if not isinstance(dir,str):
            raise Exception("ERROR: dir argument must be a string (directory)")
    if not isinstance(sub_dir,str):
            raise Exception("ERROR: sub_dir argument must be a string (sub-directory)")
    if not isinstance(input_file,str):
            raise Exception("ERROR: input_file argument must be a string (.deck file)")
    create_sub_dir(dir, sub_dir)
    dir = f'{dir}/{sub_dir}'
    input_deck(I, Ln, ppc, dir, input_file)
        
    return print(f'created directory {dir} and input.deck')




