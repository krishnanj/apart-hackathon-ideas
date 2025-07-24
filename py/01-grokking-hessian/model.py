# Simple MLP for modular addition
default_hidden = 128
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=default_hidden, output_dim=97, n_hidden_layers=2):
        super().__init__()
        # Build MLP layers
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        # Forward pass
        return self.net(x)