"""
Implementation of SVG(0) in using pytorch
"""

import argparse
import gym
import numpy as np


import torch
import torch.nn as nn
import torhc.nn.functional as F
import torch.autograd as  autograd
from torhc.autograd import Variable

parser = argparse.ArgumentParser(description='SVG(0) in PyTorch')
parser.add_argument('--seed', type=int, default=123)

args = parser.parse_args()

env = gym.make('Pendulum-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)



class QFunction(nn.Module):
    """
    Value function for estimate state action value
    """
    def __init__(
        self, 
        state_dim, 
        action_dim, 
        hidden_dim, 
        n_hidden_layers):
        super(QFunction, self).__init__()

        self.layers = []
        first_layer = nn.Linear((state_dim + action_dim), hidden_dim)
        last_layer = nn.Linear(hidden_dim, 1)


class Policy(nn.Module):
    def __init__(self):
        super()