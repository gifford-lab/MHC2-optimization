from torch.autograd import Variable
import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from torch.distributions import Normal

import sys
from resnet import ResNet, BasicBlock, Bottleneck

class PEPnet(nn.Module):

    def __init__(self, config):
        super(PEPnet, self).__init__()

        block_type = BasicBlock if config['pep_block'] == 'basic' else Bottleneck
        self.resnet = ResNet(
                block_type,
                config['pep_layers'],
                seq_len = config['pep_len'],
                conv_fn = config['pep_conv_fn'],
                embed_size = config['pep_embed_size'],
                )
        self.outlen =  config['pep_conv_fn']*(2**(len(config['pep_layers'])-1)) * block_type(1, 1).expansion
        self.config = config

    def forward(self, x):
        return self.resnet(x)


class Net(nn.Module):
    def __init__(self, config):
        super(Net, self).__init__()

        self.PEPnet = PEPnet(config)
        self.fc1 = nn.Linear(self.PEPnet.outlen, 1)
        self.fc2 = nn.Linear(self.PEPnet.outlen, 1)
        self.do2 = nn.Dropout(0.2)
        self.nl = nn.ReLU()
        self.config = config

    def forward(self, pep):
        out = self.embed(pep)
        out = self.do2(out)
        mean = self.fc1(out)
        std = F.softplus(self.fc2(out))
        return mean, std

    def embed(self, pep):
        return self.PEPnet(pep)
