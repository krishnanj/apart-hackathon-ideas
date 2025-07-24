# Dataset utilities for modular addition grokking
import torch

def create_mod_add_dataset(p=97):
    # Create all (x, y) pairs for modular addition
    xs = torch.arange(p)
    ys = torch.arange(p)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='ij')
    X = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    Y = (grid_x + grid_y).flatten() % p
    # Split into train/test
    train_idx = torch.randperm(len(X))[:int(0.1 * len(X))]
    test_idx = torch.tensor([i for i in range(len(X)) if i not in train_idx])
    return X[train_idx], Y[train_idx], X[test_idx], Y[test_idx]