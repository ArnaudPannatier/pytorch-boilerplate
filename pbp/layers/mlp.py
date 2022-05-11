import torch.nn as nn


def MLP(in_dim, out_dim, n_units, n_layers):
    return nn.Sequential(
        nn.Linear(in_dim, n_units),
        *(
            nn.Linear(n_units, n_units) if t % 2 else nn.ReLU()
            for t in range(2 * n_layers)
        ),
        nn.ReLU(),
        nn.Linear(n_units, out_dim),
    )
