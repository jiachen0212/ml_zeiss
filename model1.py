# coding=utf-8
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.l1 = nn.Linear(input_dim, hidden_dim)
        self.a1 = nn.ReLU()
        self.l2 = nn.Linear(hidden_dim, output_dim)
        self.a2 = nn.ReLU()
        # self.l3 = nn.Linear(hidden_dim, output_dim)
        # self.a3 = nn.ReLU()
        # self.l4 = nn.Linear(hidden_dim, hidden_dim)
        # self.a4 = nn.ReLU()
        # self.l5 = nn.Linear(hidden_dim, output_dim)
        # self.a5 = nn.ReLU()

        # self.layers = [self.l1, self.a1, self.l2, self.a2, self.l3, self.a3, self.l4, self.a4, self.l5, self.a5]
        self.layers = [self.l1, self.a1, self.l2, self.a2]


    def forward(self, x):
        for index, layer in enumerate(self.layers):
            x = layer(x)
        return x