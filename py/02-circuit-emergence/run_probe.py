#!/usr/bin/env python3
"""
Probing script with support for both MLP and Transformer
"""

import torch
import numpy as np
from datetime import datetime
import os
from model import create_model_from_params
from dataset import create_dataset_from_params
from probe_utils import register_hooks, register_architecture_hooks, save_dict_as_pt
from advanced_probe_utils import MultiProbeAnalyzer, run_architecture_aware_probes
from load_params import load_params

def run_probe_with_architecture():
    """Run probes with architecture specified in params"""
    params = load_params()
    architecture = params.get('architecture', 'mlp')
    
    print(f"Running probes with {architecture.upper()} architecture...")
    
    # Load trained model
    model = create_model_from_params(2, 97, params)  # Default dimensions
    # Load model weights here...
    
    # Get test data
    train_x, train_y, test_x, test_y = create_dataset_from_params(params)
    
    # Use unified probing system
    run_unified_probes(model, test_x, test_y, params)

def run_unified_probes(model, test_x, test_y, params):
    """Run probes using the unified architecture-aware system."""
    architecture = params.get('architecture', 'mlp')
    
    print(f"Running unified probes for {architecture} architecture...")
    
    # Generate concept labels (you can customize this based on your task)
    concept_labels = test_y.numpy() if hasattr(test_y, 'numpy') else test_y
    
    # Run probes using the architecture-aware system
    probe_results = run_architecture_aware_probes(
        model, test_x, concept_labels, params, is_regression=False
    )
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"results/{architecture}_probe_results_{timestamp}.pt"
    save_dict_as_pt(probe_results, results_filename)
    print(f"Saved probe results to: {results_filename}")
    
    # Print summary
    print("\nProbe Results Summary:")
    for layer_name, results in probe_results.items():
        if isinstance(results, dict) and 'linear' in results:
            print(f"  {layer_name}: Linear probe accuracy = {results['linear']:.4f}")
        elif layer_name == 'attention_analysis':
            print(f"  {layer_name}: Attention analysis completed")
    
    return probe_results

def run_mlp_probes(model, test_x, test_y, params):
    """Run probes on MLP activations"""
    # Existing MLP probing code...
    pass

def run_transformer_probes(model, test_x, test_y, params):
    """Run probes on Transformer attention and activations"""
    # Get attention weights and activations
    attention_weights = {}
    activations = {}
    
    # Register hooks for attention weights
    def attention_hook(module, input, output):
        attention_weights['attention'] = output[1]  # attention weights
    
    # Register hooks for activations
    def activation_hook(module, input, output):
        activations['activation'] = output
    
    # Register hooks on attention layers
    for name, module in model.named_modules():
        if 'attn' in name:
            module.register_forward_hook(attention_hook)
        if 'ln_f' in name:
            module.register_forward_hook(activation_hook)
    
    # Run forward pass
    with torch.no_grad():
        outputs = model(test_x)
    
    # Analyze attention patterns
    analyze_attention_patterns(attention_weights, test_x, test_y, params)
    
    # Run traditional probes on activations
    if activations:
        probe_analyzer = MultiProbeAnalyzer(["linear", "tree", "svm"])
        probe_results = probe_analyzer.run_probes(activations, test_y.numpy(), is_regression=False)
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_filename = f"results/transformer_probe_{timestamp}.pt"
        save_dict_as_pt(probe_results, results_filename)
        print(f"Saved: {results_filename}")

def analyze_attention_patterns(attention_weights, test_x, test_y, params):
    """Analyze attention patterns for concept emergence"""
    if 'attention' not in attention_weights:
        return
    
    attention = attention_weights['attention']
    
    # Analyze attention patterns
    # - Which tokens attend to which other tokens
    # - How attention patterns change across layers
    # - Concept emergence in attention weights
    
    print("Analyzing attention patterns...")
    # Add attention analysis code here...

if __name__ == "__main__":
    run_probe_with_architecture()