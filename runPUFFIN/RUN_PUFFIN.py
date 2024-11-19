# MAIN EXECUTABLE FILE

import torch
import numpy as np
import pandas as pd
from score_import import getEnsemble, runEnsemble, runEnsembleMN
from embedpep_import import getPeps
import warnings
import sys


def makePrediction(seqs, model):
    encodedInput = np.stack(getPeps(seqs))
    encodedInput = torch.tensor(encodedInput, dtype = torch.float32)
    results = runEnsemble(model, encodedInput, verbose = False)
    results = np.array(results)

    mean = np.mean(results[:, :, 0], axis = 0)
    vr1 = np.mean(results[:, :, 1] + results[:, :, 2], axis = 0)
    vr2 = np.var(results[:, :, 0], axis = 0)
    vr = vr1 + vr2

    df = pd.DataFrame({"mean" : mean, "variance" : vr}, index = seqs)
    return df


if len(sys.argv) < 2:
    print( "Please provide an input .pep file with 9mers on each line" )
else:
    infile = sys.argv[1]
    with open(infile, 'rt') as fin:
        peptides = fin.read().rstrip('\n').split('\n')
        
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ensemble_401 = getEnsemble(401)
        ensemble_402 = getEnsemble(402)

    output = makePrediction(peptides, ensemble_401)
    print ("DR401")
    print (output)
    print ()
    output = makePrediction(peptides, ensemble_402)
    print ("DR402")
    print (output)
