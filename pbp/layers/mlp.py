from functools import partial

import torch.nn as nn

from ..factory import ObjectFactory


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, n_units, n_layers):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_dim, n_units), nn.ReLU(),
            *(nn.Linear(n_units, n_units), nn.ReLU()) * n_layers,
            nn.Linear(n_units, out_dim))

    def forward(self, x):
        return self.model(x)
