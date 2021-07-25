# coding=utf-8
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter

np.random.seed(369)
torch.manual_seed(957)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_dim, output_dim)
        self.a2 = nn.ReLU()

        self.layers = [self.l1, self.a1, self.l2, self.a2]


    def forward(self, x):
        for index, layer in enumerate(self.layers):
            x = layer(x)

        return x
