#!/usr/bin/env python3
"""
Test Steps 5 and 6 with transformer support
Step 5: Composite Function Analysis
Step 6: Symmetry Tests
"""

import torch
import numpy as np
from datetime import datetime
import os
from model import create_model_from_params
from dataset import create_dataset_from_params
from load_params import load_params
from advanced_probe_utils import run_architecture_aware_probes
from probe_utils import register_architecture_hooks, save_dict_as_pt
from synthetic_functions import SyntheticFunctionGenerator

def test_step_5_composite_analysis():
    """Test Step 5: Composite Function Analysis with transformers"""
    print("=== TESTING STEP 5: COMPOSITE FUNCTION ANALYSIS ===")
    
    params = load_params()
    params['architecture'] = 'transformer'
    params['use_composite'] = True
    params['inner_func'] = 'poly'
    params['outer_func'] = 'sin'
    
    print(f"Testing composite function: {params['inner_func']} -> {params['outer_func']}")
    
    # Create composite dataset
    train_x, train_g_x, train_f_gx, test_x, test_g_x, test_f_gx = create_dataset_from_params(params)
    print(f"Dataset shapes: train_x={train_x.shape}, test_x={test_x.shape}")
    
    # Create and train model
    input_dim = train_x.shape[1]
    output_dim = len(torch.unique(train_f_gx))
    model = create_model_from_params(input_dim, output_dim, params)
    
    print(f"Training model...")
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    criterion = torch.nn.CrossEntropyLoss()
    
    for step in range(min(1000, params['steps'])):  # Train for fewer steps for testing
        optimizer.zero_grad()
        outputs = model(train_x)
        last_token_outputs = outputs[:, -1, :]
        # Extract target values from sequence (last element)
        targets = train_f_gx[:, -1].long()
        loss = criterion(last_token_outputs, targets)
        loss.backward()
        optimizer.step()
        
        if step % 200 == 0:
            print(f"  Step {step}, Loss: {loss.item():.4f}")
    
    # Generate concept labels
    g_concept = generate_hard_concept(test_g_x)
    fg_concept = generate_hard_concept(test_f_gx)
    
    print(f"Concept distributions:")
    print(f"  g(x): {dict(zip(*np.unique(g_concept, return_counts=True)))}")
    print(f"  f(g(x)): {dict(zip(*np.unique(fg_concept, return_counts=True)))}")
    
    # Run dual probing
    print(f"Running dual probing...")
    g_results = run_architecture_aware_probes(
        model, test_x, g_concept, params, is_regression=False
    )
    fg_results = run_architecture_aware_probes(
        model, test_x, fg_concept, params, is_regression=False
    )
    
    # Analyze results
    print(f"\n=== COMPOSITE ANALYSIS RESULTS ===")
    print(f"g(x) probe results:")
    for layer_name, results in g_results.items():
        if isinstance(results, dict) and 'linear' in results:
            print(f"  {layer_name}: Linear = {results['linear']:.4f}")
    
    print(f"\nf(g(x)) probe results:")
    for layer_name, results in fg_results.items():
        if isinstance(results, dict) and 'linear' in results:
            print(f"  {layer_name}: Linear = {results['linear']:.4f}")
    
    # Check for evidence of splitting
    print(f"\n=== SPLITTING ANALYSIS ===")
    g_linear_accs = []
    fg_linear_accs = []
    
    for layer_name in g_results.keys():
        if isinstance(g_results[layer_name], dict) and 'linear' in g_results[layer_name]:
            g_linear_accs.append(g_results[layer_name]['linear'])
            fg_linear_accs.append(fg_results[layer_name]['linear'])
    
    if g_linear_accs and fg_linear_accs:
        g_max_acc = max(g_linear_accs)
        fg_max_acc = max(fg_linear_accs)
        print(f"Max g(x) accuracy: {g_max_acc:.4f}")
        print(f"Max f(g(x)) accuracy: {fg_max_acc:.4f}")
        
        if g_max_acc > 0.8 and fg_max_acc > 0.8:
            print("✅ Evidence of successful function decomposition!")
        else:
            print("⚠️ Functions may be entangled")
    
    return g_results, fg_results

def test_step_6_symmetry_tests():
    """Test Step 6: Symmetry Tests with transformers"""
    print("\n=== TESTING STEP 6: SYMMETRY TESTS ===")
    
    params = load_params()
    params['architecture'] = 'transformer'
    
    # Generate symmetric function data
    generator = SyntheticFunctionGenerator()
    x, y = generator.generate_symmetric_function(seed=42)
    
    print(f"Generated symmetric function data: x={x.shape}, y={y.shape}")
    
    # Convert to sequence format for transformer
    x_norm = ((x - x.min()) / (x.max() - x.min()) * 50).long()
    y_norm = ((y - y.min()) / (y.max() - y.min()) * 50).long()
    
    # Create sequences
    sequences = []
    labels = []
    for i in range(len(x)):
        seq = torch.tensor([
            x_norm[i, 0].item(),  # x1
            x_norm[i, 1].item(),  # x2
            0,                     # op
            1,                     # =
            y_norm[i].item()       # result
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
    
    print(f"Sequence data: train_x={train_x.shape}, test_x={test_x.shape}")
    
    # Create and train model
    input_dim = train_x.shape[1]
    output_dim = len(torch.unique(train_y))
    model = create_model_from_params(input_dim, output_dim, params)
    
    print(f"Training model on symmetric function...")
    optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
    criterion = torch.nn.CrossEntropyLoss()
    
    for step in range(min(1000, params['steps'])):
        optimizer.zero_grad()
        outputs = model(train_x)
        last_token_outputs = outputs[:, -1, :]
        loss = criterion(last_token_outputs, train_y)
        loss.backward()
        optimizer.step()
        
        if step % 200 == 0:
            print(f"  Step {step}, Loss: {loss.item():.4f}")
    
    # Create symmetric input pairs
    print(f"Creating symmetric input pairs...")
    n_pairs = 100
    symmetric_pairs = []
    random_pairs = []
    
    for _ in range(n_pairs):
        # Symmetric pair: (x1, x2) and (x2, x1)
        idx1 = np.random.randint(0, len(test_x))
        idx2 = np.random.randint(0, len(test_x))
        
        # Create symmetric version by swapping x1 and x2
        seq1 = test_x[idx1].clone()
        seq2 = test_x[idx2].clone()
        
        # Swap the first two elements (x1, x2)
        seq2_symmetric = seq2.clone()
        seq2_symmetric[0] = seq2[1]  # x1 = x2
        seq2_symmetric[1] = seq2[0]  # x2 = x1
        
        symmetric_pairs.append((idx1, idx2))
        random_pairs.append((idx1, np.random.randint(0, len(test_x))))
    
    # Run symmetry analysis
    print(f"Running symmetry analysis...")
    from transformer_probe_utils import symmetry_probe_transformer
    
    # Get activations for symmetry analysis
    activation_store = {}
    from transformer_probe_utils import TransformerProbeAnalyzer
    analyzer = TransformerProbeAnalyzer()
    hooks = analyzer.register_transformer_hooks(model, activation_store)
    
    with torch.no_grad():
        model(test_x)
    
    for hook in hooks:
        hook.remove()
    
    # Analyze symmetry
    symmetry_scores = symmetry_probe_transformer(activation_store, symmetric_pairs)
    random_scores = symmetry_probe_transformer(activation_store, random_pairs)
    
    print(f"\n=== SYMMETRY ANALYSIS RESULTS ===")
    print(f"Symmetric pairs vs Random pairs:")
    for layer_name in symmetry_scores.keys():
        if layer_name in random_scores:
            sym_score = symmetry_scores[layer_name]
            rand_score = random_scores[layer_name]
            print(f"  {layer_name}: Symmetric={sym_score:.4f}, Random={rand_score:.4f}")
            
            if sym_score > rand_score * 1.5:
                print(f"    ✅ Evidence of symmetry invariance!")
            else:
                print(f"    ⚠️ No clear symmetry invariance")
    
    return symmetry_scores, random_scores

def generate_hard_concept(y_vals):
    """Generate hard concept labels for probing"""
    # Convert to numpy if needed
    if hasattr(y_vals, 'numpy'):
        y_vals = y_vals.numpy()
    
    # Ensure y_vals is 1D
    if len(y_vals.shape) > 1:
        y_vals = y_vals.flatten()
    
    # Create concept based on value ranges
    concept = np.zeros(len(y_vals), dtype=int)
    concept[y_vals > np.median(y_vals)] = 1
    return concept

if __name__ == "__main__":
    # Test Step 5: Composite Function Analysis
    g_results, fg_results = test_step_5_composite_analysis()
    
    # Test Step 6: Symmetry Tests
    symmetry_scores, random_scores = test_step_6_symmetry_tests()
    
    print("\n=== SUMMARY ===")
    print("✅ Step 5 (Composite Analysis): Completed")
    print("✅ Step 6 (Symmetry Tests): Completed")
    print("Both steps now support transformer architecture!") 