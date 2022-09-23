import fileinput
import sys
import json
import numpy as np

## read_input
#
# Read inputs in input.deck in chosen directory
# @param dir : Directory which holds input.deck file you want to read (str)
# @param param : Specific paramter to read from input file ('intensity', 'momentum', 'ppc' or 'ne_scale_len')
def read_input(dir, param):
    """Read inputs in input.deck in chosen directory."""
    line = []
    with open(dir+'input.deck') as f:
            found = False
            while not found:
                line = f.readline()
                words = line.split()
                if len(words) < 1:
                    continue

                if param == 'intensity':
                    if words[0] == "intensity_w_cm2":
                        found = True
                        return float(words[2])

                elif param == 'momentum':
                    if words[0] == "range1":
                        found = True
                        return float(words[2][1:-1]), float(words[3][:-1])
            

                elif param == 'ppc':
                    if words[0] == "PPC":
                        found = True
                        return float(words[2])


                elif param == 'ne_scale_len':
                    if words[0] == "Ln":
                        found = True
                        return float(words[2])

                else:
                    print('Please set param to one of the following as a string \
                           intensity, momentum, ppc or ne_scale_len')
                    break


## replace_line
#
# Function rewrite line in input.deck via python
# @param line_in : original line in input.deck 
# @param line_out : repacement of line_in in input.deck
def replace_line(line_in, line_out, fname):
  """Function rewrite line in input.deck via python."""
  finput = fileinput.input(fname, inplace=1)
  for i, line in enumerate(finput):
    sys.stdout.write(line.replace(line_in, line_out))
  finput.close()

## read_json_file
#
# Function wrapper for reading json file
# @param fname : json file name
def read_json_file(fname):
    """Function wrapper for reading json file."""
    with open(fname, 'r') as f:
            data = json.load(f)
    return np.array(data)