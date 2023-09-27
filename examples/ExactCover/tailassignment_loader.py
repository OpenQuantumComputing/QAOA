import numpy as np
import os, sys

def load_FR_CR(filename):

    if not os.path.isfile(filename):
        print("File not found.")
        raise IOError

    # Matrix in file on format
    # C_1: index of ones in row 1 separated by comma
    # C_2: index of ones in row 2 ...
    # ...
    # C_R: index of ones in row R
    # best state 

    file = np.loadtxt(filename, dtype = str)

    R,F = filename.split('/')[-1].split('_')[1:3]
    R,F = int(R), int(F)

    FR   = np.zeros((F,R))
    CR   = np.zeros(R)
    best = np.zeros(R)
    
    for r in range(R):
        CR[r]   = float(file[r].split(':')[0])    
        indexes = file[r].split(':')[1].split(',')
        for ind in indexes:
            FR[int(ind), r ] = 1
    
    best_str = file[R].split(',')
    best     = np.array([int(i) for i in best_str]) 
            
    return FR, CR, best

def npy_loader(filename):
    """
    Loading the examples saved on npy format.

    Parameters
    ----------
    filename : string
        filename
    
    Returns
    -------
    FR : array
        Constraint matrix
    CR : array
        Weights

    """
    matrix = np.load(filename)

    # The costs are saved in the last column
    CR     = matrix[-1]
    FR     = matrix[:-1]

    return FR, CR
