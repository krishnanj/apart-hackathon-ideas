import torch
from synthetic_functions import create_synthetic_dataset, create_composite_dataset

def create_mod_add_dataset(p=97, train_frac=0.1, seed=0):
    """Create modular addition dataset"""
    torch.manual_seed(seed)
    all_data = torch.tensor([(i, j) for i in range(p) for j in range(p)])
    all_labels = (all_data[:, 0] + all_data[:, 1]) % p
    num_train = int(train_frac * len(all_data))
    idx = torch.randperm(len(all_data))
    train_idx = idx[:num_train]
    test_idx = idx[num_train:]
    return all_data[train_idx], all_labels[train_idx], all_data[test_idx], all_labels[test_idx]

def create_dataset_from_params(params):
    """Create dataset based on parameters"""
    if params.get("use_synthetic", False):
        # Check if this is a composite function
        if params.get("use_composite", False):
            # Use composite functions
            inner_func = params.get("inner_func", "poly")
            outer_func = params.get("outer_func", "sin")
            train_frac = params.get("train_frac", 0.8)
            seed = params.get("seed", 42)
            
            return create_composite_dataset(inner_func, outer_func, train_frac, seed)
        else:
            # Use regular synthetic functions
            func_type = params.get("func_type", "polynomial")
            complexity = params.get("complexity", 3)
            train_frac = params.get("train_frac", 0.8)
            seed = params.get("seed", 42)
            
            return create_synthetic_dataset(func_type, complexity, train_frac, seed)
    else:
        # Use modular addition (original)
        p = params.get("p", 97)
        train_frac = params.get("train_frac", 0.8)
        seed = params.get("seed", 42)
        
        return create_mod_add_dataset(p, train_frac, seed)

# Add this function to dataset.py
def create_symmetry_dataset(train_frac=0.8, seed=42):
    """Create dataset specifically for symmetry analysis"""
    from synthetic_functions import create_synthetic_dataset
    return create_synthetic_dataset("symmetric", 3, train_frac, seed)