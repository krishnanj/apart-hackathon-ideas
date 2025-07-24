# Dataset utilities for modular addition grokking
import torch

def create_mod_add_dataset(p=97, train_frac=0.1):
    # Create all (x, y) pairs for modular addition
    xs = torch.arange(p)
    ys = torch.arange(p)
    grid_x, grid_y = torch.meshgrid(xs, ys, indexing='ij')
    X = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    Y = (grid_x + grid_y).flatten() % p
    # Split into train/test
    n_train = int(train_frac * len(X))
    idx = torch.randperm(len(X))
    train_idx = idx[:n_train]
    test_idx = idx[n_train:]
    return X[train_idx], Y[train_idx], X[test_idx], Y[test_idx]