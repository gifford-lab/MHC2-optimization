import numpy as np
import argparse

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.distributions import Normal

def load_state(allele, trial, rep):
    fbase = "./model{}/checkpoint_trial{}_init{}.pt"
    checkpoint_path = fbase.format(allele, trial, rep)
    return torch.load(checkpoint_path, encoding="utf-8")

def fixNet(mod):
    for submod in mod.modules():
        if 'Conv' in str(type(submod)):
            setattr(submod, 'padding_mode', 'zeros')

def rundata(x, net):
    mn, vr = net(x)
    mn = mn.cpu().numpy()
    vr = vr.cpu().numpy()
    return mn, vr

def verify():
    return "Gaussian Model"

#Set allele to 1 for DR401 and 2 for DR402
def getEnsemble(allele):
    ensemble = []
    for split in range(1,11):
        for rep in (1,2):
            net = load_state(allele, split, rep)
            fixNet(net)
            net.eval()
            for m in net.modules():
                if m.__class__.__name__.startswith('Dropout'):
                    m.train()
            net = net.cuda()
            ensemble.append(net)
    return ensemble

def runEnsemble(ensemble, data, verbose = True):
    results = []
    with torch.no_grad():
        for neti, net in enumerate(ensemble):
            if verbose: print ("Running model {}".format(neti))
            mns = []
            vrs = []
            data = data.cuda() 
            for rnd in range(50):
                if verbose: print ("round {}".format(rnd))
                mn, vr = rundata(data, net)
                mns.append(mn)
                vrs.append(vr)
            vr2 = np.var(mns, axis = 0)
            mns = np.mean(mns, axis = 0)
            vrs = np.mean(vrs, axis = 0)
            summary = [(float(x), float(y), float(z)) for x,y,z in zip(mns, vrs, vr2)]
            results.append(summary)
    return results

def runEnsembleMN(ensemble, data, verbose = True):
    results = []
    data = data.cuda() 
    with torch.no_grad():
        for neti, net in enumerate(ensemble):
            if verbose: print ("Running model {}".format(neti))
            mns = []
            for rnd in range(50):
                if verbose: print ("round {}".format(rnd))
                mn, vr = rundata(data, net)
                mns.append(mn)
            mns = np.mean(mns, axis = 0)
            results.append(mns)
    results = np.mean(results, axis = 0)
    return results



