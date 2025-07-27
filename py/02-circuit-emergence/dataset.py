#!/usr/bin/env python3
"""
Dataset creation with support for both MLP and Transformer formats
"""

import torch
import numpy as np
from synthetic_functions import SyntheticFunctionGenerator

def create_dataset_from_params(params):
    """Create dataset based on parameters"""
    use_synthetic = params.get('use_synthetic', False)
    architecture = params.get('architecture', 'mlp')
    
    print(f"DEBUG: use_synthetic={use_synthetic}, architecture={architecture}")
    
    if use_synthetic:
        return create_synthetic_dataset(params, architecture)
    else:
        return create_mod_add_dataset(params, architecture)

def create_synthetic_dataset(params, architecture):
    """Create synthetic dataset in appropriate format"""
    func_type = params.get('func_type', 'polynomial')
    complexity = params.get('complexity', 3)
    input_range = params.get('input_range', [-5, 5])
    n_samples = params.get('n_samples', 1000)
    use_composite = params.get('use_composite', False)
    train_frac = params.get('train_frac', 0.8)
    
    print(f"DEBUG: func_type={func_type}, complexity={complexity}, architecture={architecture}")
    
    generator = SyntheticFunctionGenerator()
    
    if use_composite:
        inner_func = params.get('inner_func', 'poly')
        outer_func = params.get('outer_func', 'sin')
        # Fix: Use the correct method name
        x, g_x, f_gx = generator.generate_composite(
            inner_func, outer_func, seed=42
        )
        
        if architecture == 'mlp':
            # Split into train/test for composite MLP
            n_train = int(train_frac * len(x))
            indices = torch.randperm(len(x))
            train_idx = indices[:n_train]
            test_idx = indices[n_train:]
            
            train_x = x[train_idx]
            train_g_x = g_x[train_idx]
            train_f_gx = f_gx[train_idx]
            test_x = x[test_idx]
            test_g_x = g_x[test_idx]
            test_f_gx = f_gx[test_idx]
            
            return train_x, train_g_x, train_f_gx, test_x, test_g_x, test_f_gx
        else:  # transformer
            return convert_to_sequence_format(x, g_x, f_gx, params)
    else:
        # Fix: Use the correct method name
        x, y = generator.generate_dataset(
            func_type, complexity, seed=42
        )
        
        print(f"DEBUG: Generated x={x.shape}, y={y.shape}")
        
        if architecture == 'mlp':
            # Split into train/test for simple MLP
            n_train = int(train_frac * len(x))
            indices = torch.randperm(len(x))
            train_idx = indices[:n_train]
            test_idx = indices[n_train:]
            
            train_x = x[train_idx]
            train_y = y[train_idx]
            test_x = x[test_idx]
            test_y = y[test_idx]
            
            print(f"DEBUG: MLP returning: train_x={train_x.shape}, train_y={train_y.shape}")
            return train_x, train_y, test_x, test_y
        else:  # transformer
            print(f"DEBUG: Converting to sequence format...")
            result = convert_to_sequence_format(x, y, None, params)
            print(f"DEBUG: Sequence format result: {type(result)}")
            if result is None:
                print(f"DEBUG: convert_to_sequence_format returned None!")
            return result

def convert_to_sequence_format(x, y, g_y=None, params=None):
    """Convert continuous data to sequence format for transformers"""
    print(f"DEBUG: convert_to_sequence_format called with x={x.shape}, y={y.shape}, g_y={g_y}")
    
    # For synthetic functions, create sequence format
    # Format: [x, op, y, =, result] or [x, op1, g(x), op2, f(g(x))]
    
    if g_y is not None:
        # Composite function: [x, op1, g(x), op2, f(g(x))]
        # Normalize values to reasonable token range
        x_norm = ((x - x.min()) / (x.max() - x.min()) * 50).long()
        g_y_norm = ((g_y - g_y.min()) / (g_y.max() - g_y.min()) * 50).long()
        y_norm = ((y - y.min()) / (y.max() - y.min()) * 50).long()
        
        # Create sequences
        sequences = []
        labels = []
        for i in range(len(x)):
            seq = torch.tensor([
                x_norm[i].item(),  # x
                0,                  # op1 (placeholder)
                g_y_norm[i].item(), # g(x)
                1,                  # op2 (placeholder)
                y_norm[i].item()    # f(g(x))
            ], dtype=torch.long)
            sequences.append(seq)
            labels.append(y_norm[i].item())
        
        sequences = torch.stack(sequences)
        labels = torch.tensor(labels, dtype=torch.long)
        
        # Split into train/test
        n_train = int(0.8 * len(sequences))
        indices = torch.randperm(len(sequences))
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        train_x = sequences[train_idx]
        train_y = labels[train_idx]
        test_x = sequences[test_idx]
        test_y = labels[test_idx]
        
        # For composite functions, we need to return g(x) and f(g(x)) separately
        # Create g(x) and f(g(x)) sequences
        g_sequences = []
        fg_sequences = []
        for i in range(len(x)):
            g_seq = torch.tensor([
                x_norm[i].item(),  # x
                0,                  # op
                g_y_norm[i].item(), # g(x)
            ], dtype=torch.long)
            g_sequences.append(g_seq)
            
            fg_seq = torch.tensor([
                x_norm[i].item(),  # x
                1,                  # op
                y_norm[i].item(),   # f(g(x))
            ], dtype=torch.long)
            fg_sequences.append(fg_seq)
        
        g_sequences = torch.stack(g_sequences)
        fg_sequences = torch.stack(fg_sequences)
        
        train_g_x = g_sequences[train_idx]
        train_f_gx = fg_sequences[train_idx]
        test_g_x = g_sequences[test_idx]
        test_f_gx = fg_sequences[test_idx]
        
        print(f"DEBUG: Composite function returning: train_x={train_x.shape}, train_y={train_y.shape}")
        return train_x, train_g_x, train_f_gx, test_x, test_g_x, test_f_gx
    else:
        # Simple function: [x, op, y, =, result]
        # Normalize values to reasonable token range
        x_norm = ((x - x.min()) / (x.max() - x.min()) * 50).long()
        y_norm = ((y - y.min()) / (y.max() - y.min()) * 50).long()
        
        print(f"DEBUG: Normalized x_norm={x_norm.shape}, y_norm={y_norm.shape}")
        
        # Create sequences
        sequences = []
        labels = []
        for i in range(len(x)):
            seq = torch.tensor([
                x_norm[i].item(),  # x
                0,                  # op (placeholder)
                0,                  # y (placeholder for simple functions)
                1,                  # = (placeholder)
                y_norm[i].item()    # result
            ], dtype=torch.long)
            sequences.append(seq)
            labels.append(y_norm[i].item())
        
        sequences = torch.stack(sequences)
        labels = torch.tensor(labels, dtype=torch.long)
        
        print(f"DEBUG: Created sequences={sequences.shape}, labels={labels.shape}")
        
        # Split into train/test
        n_train = int(0.8 * len(sequences))
        indices = torch.randperm(len(sequences))
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        train_x = sequences[train_idx]
        train_y = labels[train_idx]
        test_x = sequences[test_idx]
        test_y = labels[test_idx]
        
        print(f"DEBUG: Simple function returning: train_x={train_x.shape}, train_y={train_y.shape}")
        return train_x, train_y, test_x, test_y

def create_mod_add_dataset(params, architecture):
    """Create modular addition dataset"""
    p = params.get('p', 97)
    train_frac = params.get('train_frac', 0.8)
    
    if architecture == 'mlp':
        # Original MLP format
        all_data = torch.tensor([(i, j) for i in range(p) for j in range(p)], dtype=torch.float32)
        all_labels = (all_data[:, 0] + all_data[:, 1]) % p
        
        n_train = int(train_frac * len(all_data))
        indices = torch.randperm(len(all_data))
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        train_x = all_data[train_idx]
        train_y = all_labels[train_idx]
        test_x = all_data[test_idx]
        test_y = all_labels[test_idx]
        
        return train_x, train_y, test_x, test_y
    else:
        # Transformer sequence format: [x, +, y, =, (x+y)%p]
        sequences = []
        labels = []
        
        for i in range(p):
            for j in range(p):
                seq = torch.tensor([i, 0, j, 1, (i + j) % p], dtype=torch.long)
                sequences.append(seq)
                labels.append((i + j) % p)
        
        sequences = torch.stack(sequences)
        labels = torch.tensor(labels, dtype=torch.long)
        
        n_train = int(train_frac * len(sequences))
        indices = torch.randperm(len(sequences))
        train_idx = indices[:n_train]
        test_idx = indices[n_train:]
        
        train_x = sequences[train_idx]
        train_y = labels[train_idx]
        test_x = sequences[test_idx]
        test_y = labels[test_idx]
        
        return train_x, train_y, test_x, test_y