import torch

def create_mod_add_dataset(p=97, train_frac=0.1, seed=0):
    torch.manual_seed(seed)
    all_data = torch.tensor([(i, j) for i in range(p) for j in range(p)])
    all_labels = (all_data[:, 0] + all_data[:, 1]) % p
    num_train = int(train_frac * len(all_data))
    idx = torch.randperm(len(all_data))
    train_idx = idx[:num_train]
    test_idx = idx[num_train:]
    return all_data[train_idx], all_labels[train_idx], all_data[test_idx], all_labels[test_idx]