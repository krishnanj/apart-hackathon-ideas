#!/usr/bin/env python3
"""
Run enhanced probe system with both MLP and transformer support
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

def run_enhanced_probes():
    """Run enhanced probe system with current architecture setting."""
    params = load_params()
    architecture = params.get('architecture', 'mlp')
    
    print(f"=== RUNNING ENHANCED PROBE SYSTEM ===")
    print(f"Architecture: {architecture.upper()}")
    
    # Create dataset
    print("Creating dataset...")
    train_x, train_y, test_x, test_y = create_dataset_from_params(params)
    print(f"Dataset shapes: train={train_x.shape}, test={test_x.shape}")
    
    # Create model
    print("Creating model...")
    if architecture == 'transformer':
        input_dim = test_x.shape[1]  # sequence length
        output_dim = params.get('transformer_config', {}).get('num_tokens', 100)
    else:
        input_dim = test_x.shape[1] if len(test_x.shape) > 1 else 1
        output_dim = test_y.shape[1] if len(test_y.shape) > 1 else 1
    
    model = create_model_from_params(input_dim, output_dim, params)
    print(f"Model created: {type(model).__name__}")
    
    # Generate concept labels (customize based on your task)
    print("Generating concept labels...")
    if len(test_y.shape) > 1:
        concept_labels = test_y[:, 0].numpy()
    else:
        concept_labels = test_y.numpy()
    
    # Run probes
    print("Running probes...")
    if architecture == 'transformer':
        # Transformer probing
        probe_results = run_architecture_aware_probes(
            model, test_x, concept_labels, params, is_regression=False
        )
    else:
        # MLP probing with activation capture
        activation_store = {}
        depth = params.get('depth', 5)
        layer_indices = list(range(depth))
        
        hooks = register_architecture_hooks(model, layer_indices, activation_store, params)
        
        with torch.no_grad():
            model(test_x.float())
        
        for hook in hooks:
            hook.remove()
        
        probe_results = run_architecture_aware_probes(
            model, test_x, concept_labels, params, 
            activation_store=activation_store, is_regression=False
        )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"results/{architecture}_enhanced_probe_{timestamp}.pt"
    save_dict_as_pt(probe_results, results_filename)
    print(f"Saved results to: {results_filename}")
    
    # Print summary
    print("\n=== PROBE RESULTS SUMMARY ===")
    for layer_name, results in probe_results.items():
        if isinstance(results, dict):
            if 'linear' in results:
                print(f"  {layer_name}: Linear = {results['linear']:.4f}")
            if 'tree' in results:
                print(f"  {layer_name}: Tree = {results['tree']:.4f}")
            if 'svm' in results:
                print(f"  {layer_name}: SVM = {results['svm']:.4f}")
            if 'mlp' in results:
                print(f"  {layer_name}: MLP = {results['mlp']:.4f}")
        elif layer_name == 'attention_analysis':
            print(f"  {layer_name}: Attention analysis completed")
    
    return probe_results

def run_comparison_probes():
    """Run probes on both architectures for comparison."""
    params = load_params()
    
    print("=== RUNNING COMPARISON PROBES ===")
    
    results = {}
    
    for architecture in ['mlp', 'transformer']:
        print(f"\n--- Testing {architecture.upper()} ---")
        
        # Update params
        test_params = params.copy()
        test_params['architecture'] = architecture
        
        try:
            # Create dataset and model
            train_x, train_y, test_x, test_y = create_dataset_from_params(test_params)
            
            if architecture == 'transformer':
                input_dim = test_x.shape[1]
                output_dim = test_params.get('transformer_config', {}).get('num_tokens', 100)
            else:
                input_dim = test_x.shape[1] if len(test_x.shape) > 1 else 1
                output_dim = test_y.shape[1] if len(test_y.shape) > 1 else 1
            
            model = create_model_from_params(input_dim, output_dim, test_params)
            
            # Generate concept labels
            if len(test_y.shape) > 1:
                concept_labels = test_y[:, 0].numpy()
            else:
                concept_labels = test_y.numpy()
            
            # Run probes
            if architecture == 'transformer':
                probe_results = run_architecture_aware_probes(
                    model, test_x, concept_labels, test_params, is_regression=False
                )
            else:
                activation_store = {}
                depth = test_params.get('depth', 5)
                layer_indices = list(range(depth))
                
                hooks = register_architecture_hooks(model, layer_indices, activation_store, test_params)
                
                with torch.no_grad():
                    model(test_x.float())
                
                for hook in hooks:
                    hook.remove()
                
                probe_results = run_architecture_aware_probes(
                    model, test_x, concept_labels, test_params, 
                    activation_store=activation_store, is_regression=False
                )
            
            results[architecture] = probe_results
            print(f"  ✅ {architecture.upper()} completed successfully")
            
        except Exception as e:
            print(f"  ❌ {architecture.upper()} failed: {e}")
            results[architecture] = {}
    
    # Save comparison results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"results/comparison_probes_{timestamp}.pt"
    save_dict_as_pt(results, results_filename)
    print(f"\nSaved comparison results to: {results_filename}")
    
    return results

if __name__ == "__main__":
    # Run enhanced probes with current architecture
    run_enhanced_probes()
    
    # Optionally run comparison
    # run_comparison_probes() 