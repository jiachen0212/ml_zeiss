# coding=utf-8
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter

np.random.seed(369)
torch.manual_seed(957)


class copy_x(nn.Module):
    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: Tensor

    def __init__(self, in_features: int, out_features: int) -> None:
        super(copy_x, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))

    #     self.reset_parameters()
    #
    # def reset_parameters(self) -> None:
    #     pass

    def forward(self, input: Tensor) -> Tensor:
        # return F.linear(input, self.weight)
        return self.weight  # 直接赋值

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}'.format(
            self.in_features, self.out_features
        )


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        # self.bn = nn.BatchNorm1d(num_features=7)
        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_dim, output_dim)
        self.a2 = nn.ReLU()
        # self.l3 = nn.Linear(hidden_dim, output_dim)
        # self.a3 = nn.ReLU()
        # self.l4 = nn.Linear(hidden_dim, output_dim)
        # self.a4 = nn.ReLU()

        self.layers = [self.l1, self.a1, self.l2, self.a2]

    def forward(self, x):
        # print(x[0], 'input')
        for index, layer in enumerate(self.layers):
            x = layer(x)
            # print(x[0], '==')
            # print(index, x.shape)

        return x
