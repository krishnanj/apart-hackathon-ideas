import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=128, output_dim=97, n_hidden_layers=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers += [nn.Linear(hidden_dim, output_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def create_model_from_params(params, input_dim=None, output_dim=None):
    """Create model based on parameters"""
    hidden_dim = params.get("hidden_dim", 128)
    n_hidden_layers = params.get("depth", 2)
    
    # Determine input/output dimensions
    if input_dim is None:
        input_dim = params.get("input_dim", 2)
    if output_dim is None:
        output_dim = params.get("output_dim", 1)  # Default for regression
    
    return MLP(input_dim=input_dim, hidden_dim=hidden_dim, 
               output_dim=output_dim, n_hidden_layers=n_hidden_layers)