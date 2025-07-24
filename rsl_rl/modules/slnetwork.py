import torch
import torch.nn as nn


# define supervised learning network for loading model
class SlNetwork(nn.Module):
    """
    Supervised Learning Network

    """
    def __init__(self, input_dims, output_dims, hidden_dims=[256,256]):
        super(SlNetwork, self).__init__()

        layers = []
        activation = nn.ReLU()

        layers.append(nn.Linear(input_dims, hidden_dims[0]))
        layers.append(activation)
        for l in range(len(hidden_dims)):
            if l == len(hidden_dims) - 1:
                layers.append(nn.Linear(hidden_dims[l], output_dims))
            else:
                layers.append(nn.Linear(hidden_dims[l], hidden_dims[l + 1]))
                layers.append(activation)

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = self.net(x)
        return x

