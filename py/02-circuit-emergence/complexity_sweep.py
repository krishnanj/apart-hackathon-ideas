#!/usr/bin/env python3
"""
Complexity Sweep: Track concept emergence across complexity levels
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import yaml

from model import create_model_from_params
from dataset import create_dataset_from_params
from probe_utils import register_hooks, register_architecture_hooks, save_dict_as_pt
from advanced_probe_utils import MultiProbeAnalyzer, run_architecture_aware_probes
from load_params import load_params

def run_complexity_sweep():
    """Run comprehensive complexity sweep with harder concepts"""
    params = load_params()
    
    print(f"Starting complexity sweep for {params['complexity_function']} function")
    print(f"Complexity range: {params['complexity_range']}")
    print(f"Model widths: {params['complexity_widths']}")
    
    results = {}
    
    for complexity in params['complexity_range']:
        print(f"\n=== Analyzing complexity {complexity} ===")
        results[complexity] = {}
        
        for width in params['complexity_widths']:
            print(f"  Training model width {width} for complexity {complexity}...")
            
            # Update params for this complexity
            params['complexity'] = complexity
            params['widths'] = [width]
            params['depth'] = params['complexity_depth']
            
            # Create dataset and model
            dataset_result = create_dataset_from_params(params)
            
            # Handle both regular and composite datasets
            if params.get("use_composite", False):
                # Composite dataset returns 6 values
                train_x, train_g, train_fg, test_x, test_g, test_fg = dataset_result
                train_y = train_fg  # Use composite function as target
                test_y = test_fg
            else:
                # Regular dataset returns 4 values
                train_x, train_y, test_x, test_y = dataset_result
            
            # Determine input/output dimensions from data
            input_dim = train_x.shape[1]
            # Handle both 1D and 2D output tensors
            if len(train_y.shape) > 1:
                output_dim = train_y.shape[1]
            else:
                # For 1D tensors (like transformer labels), use the number of unique values
                output_dim = len(torch.unique(train_y))
            
            print(f"    Input shape: {train_x.shape}, Output shape: {train_y.shape}")
            
            # Create model with correct dimensions
            model = create_model_from_params(input_dim, output_dim, params)
            
            # Train model
            optimizer = torch.optim.Adam(model.parameters(), lr=params['lr'])
            
            # Choose appropriate loss function based on architecture
            architecture = params.get('architecture', 'mlp')
            if architecture == 'transformer':
                criterion = torch.nn.CrossEntropyLoss()
            else:
                criterion = torch.nn.MSELoss()
            
            for step in range(params['steps']):
                optimizer.zero_grad()
                outputs = model(train_x)
                
                if architecture == 'transformer':
                    # For transformer, we predict the last token
                    # outputs shape: [batch_size, seq_len, num_tokens]
                    # We want to predict the last token, so take outputs[:, -1, :]
                    last_token_outputs = outputs[:, -1, :]  # [batch_size, num_tokens]
                    loss = criterion(last_token_outputs, train_y.long())
                else:
                    loss = criterion(outputs, train_y)
                
                loss.backward()
                optimizer.step()
                
                if step % 1000 == 0:
                    print(f"    Step {step}, Loss: {loss.item():.4f}")
            
            # Run probes with harder concept
            print(f"  Running probes for width {width}...")
            if params.get("use_composite", False):
                # For composite functions, probe for both g(x) and f(g(x))
                probe_results = run_composite_probes_on_model(model, test_x, test_g, test_fg, params)
            else:
                probe_results = run_probes_on_model(model, test_x, test_y, params)
            
            results[complexity][width] = probe_results
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"results/complexity_sweep_results_{timestamp}.pt"
    save_dict_as_pt(results, results_file)
    print(f"\nSaved complexity sweep results to: {results_file}")
    
    # Analyze and plot
    analyze_phase_changes(results)
    create_complexity_plots(results, timestamp)

def run_composite_probes_on_model(model, test_x, test_g, test_fg, params):
    """Run probes for composite functions - probe both g(x) and f(g(x))"""
    architecture = params.get('architecture', 'mlp')
    
    # Convert tensors to numpy
    x_vals = test_x[:, 0].numpy() if len(test_x.shape) > 1 else test_x.numpy()
    g_vals = test_g[:, 0].numpy() if len(test_g.shape) > 1 else test_g.numpy()
    fg_vals = test_fg[:, 0].numpy() if len(test_fg.shape) > 1 else test_fg.numpy()
    
    # Create concepts for both g(x) and f(g(x))
    g_concept = generate_hard_concept(g_vals)
    fg_concept = generate_hard_concept(fg_vals)
    
    print(f"    g(x) concept distribution: {dict(zip(*np.unique(g_concept, return_counts=True)))}")
    print(f"    f(g(x)) concept distribution: {dict(zip(*np.unique(fg_concept, return_counts=True)))}")
    
    if architecture == 'transformer':
        # Use transformer-specific probing for both concepts
        g_results = run_architecture_aware_probes(
            model, test_x, g_concept, params, is_regression=False
        )
        fg_results = run_architecture_aware_probes(
            model, test_x, fg_concept, params, is_regression=False
        )
    else:
        # Use MLP probing with activation capture
        activations = {}
        depth = params.get('complexity_depth', 5)
        layer_indices = list(range(depth))
        
        hooks = register_architecture_hooks(model, layer_indices, activations, params)
        
        with torch.no_grad():
            model(test_x)
        
        for hook in hooks:
            hook.remove()
        
        g_results = run_architecture_aware_probes(
            model, test_x, g_concept, params, 
            activation_store=activations, is_regression=False
        )
        fg_results = run_architecture_aware_probes(
            model, test_x, fg_concept, params, 
            activation_store=activations, is_regression=False
        )
    
    # Combine results - extract linear probe accuracies
    combined_results = {}
    for layer_name in g_results.keys():
        if isinstance(g_results[layer_name], dict) and 'linear' in g_results[layer_name]:
            g_linear_acc = g_results[layer_name]['linear']
            fg_linear_acc = fg_results[layer_name]['linear']
            
            combined_results[f"g_{layer_name}"] = g_linear_acc
            combined_results[f"fg_{layer_name}"] = fg_linear_acc
    
    # Debug: Print sample probe results
    for layer_name in list(g_results.keys())[:3]:  # Show first 3
        if isinstance(g_results[layer_name], dict) and 'linear' in g_results[layer_name]:
            g_acc = g_results[layer_name]['linear']
            fg_acc = fg_results[layer_name]['linear']
            print(f"    {layer_name} g(x) probe accuracy: {g_acc:.4f}")
            print(f"    {layer_name} f(g(x)) probe accuracy: {fg_acc:.4f}")
    
    return combined_results

def run_probes_on_model(model, test_x, test_y, params):
    """Run probes with architecture-aware system"""
    architecture = params.get('architecture', 'mlp')
    
    # Convert tensors to numpy
    x_vals = test_x[:, 0].numpy() if len(test_x.shape) > 1 else test_x.numpy()
    y_vals = test_y[:, 0].numpy() if len(test_y.shape) > 1 else test_y.numpy()
    
    # Generate hard concept
    concept_labels = generate_hard_concept(y_vals)
    print(f"    Concept distribution: {dict(zip(*np.unique(concept_labels, return_counts=True)))}")
    
    if architecture == 'transformer':
        # Use transformer-specific probing
        probe_results = run_architecture_aware_probes(
            model, test_x, concept_labels, params, is_regression=False
        )
    else:
        # Use MLP probing with activation capture
        activations = {}
        depth = params.get('complexity_depth', 5)
        layer_indices = list(range(depth))
        
        hooks = register_architecture_hooks(model, layer_indices, activations, params)
        
        with torch.no_grad():
            model(test_x)
        
        for hook in hooks:
            hook.remove()
        
        probe_results = run_architecture_aware_probes(
            model, test_x, concept_labels, params, 
            activation_store=activations, is_regression=False
        )
    
    # Extract linear probe accuracies from nested dictionary
    linear_results = {}
    for layer_name, results in probe_results.items():
        if isinstance(results, dict) and 'linear' in results:
            linear_results[layer_name] = results['linear']
    
    # Debug: Print sample probe results
    for layer_name in list(linear_results.keys())[:3]:  # Show first 3
        print(f"    {layer_name} linear probe accuracy: {linear_results[layer_name]:.4f}")
    
    return linear_results

def generate_hard_concept(y_vals):
    """Generate a much harder concept for probing"""
    # Try multiple hard concepts and pick the best one
    concepts = {
        'zero_crossings': generate_zero_crossings_concept(y_vals),
        'oscillation_count': generate_oscillation_concept(y_vals),
        'symmetry_breaking': generate_symmetry_concept(y_vals),
        'complexity_estimate': generate_complexity_concept(y_vals)
    }
    
    # Pick concept with best class distribution
    best_concept = None
    best_score = 0
    
    for name, labels in concepts.items():
        unique_labels, counts = np.unique(labels, return_counts=True)
        if len(unique_labels) == 2:
            # Score based on balance (closer to 50/50 is better)
            balance = min(counts) / max(counts)
            if balance > best_score:
                best_score = balance
                best_concept = labels
    
    if best_concept is None:
        # Fallback to simple concept
        return (y_vals > y_vals.mean()).astype(int)
    
    return best_concept

def generate_zero_crossings_concept(y_vals):
    """Detect if function has multiple zero crossings"""
    zero_crossings = 0
    for i in range(1, len(y_vals)):
        if (y_vals[i-1] < 0 and y_vals[i] >= 0) or (y_vals[i-1] >= 0 and y_vals[i] < 0):
            zero_crossings += 1
    return np.array([1 if zero_crossings > 2 else 0] * len(y_vals))

def generate_oscillation_concept(y_vals):
    """Detect number of oscillations (direction changes)"""
    oscillations = 0
    for i in range(2, len(y_vals)):
        if (y_vals[i] - y_vals[i-1]) * (y_vals[i-1] - y_vals[i-2]) < 0:
            oscillations += 1
    return np.array([1 if oscillations > 3 else 0] * len(y_vals))

def generate_symmetry_concept(y_vals):
    """Detect if function is symmetric around origin"""
    # Check if function has odd symmetry (f(-x) ≈ -f(x))
    mid_point = len(y_vals) // 2
    left_half = y_vals[:mid_point]
    right_half = y_vals[mid_point:][::-1]  # Reverse right half
    
    # Calculate symmetry score
    symmetry_error = np.mean(np.abs(left_half + right_half))
    return np.array([1 if symmetry_error < 0.1 else 0] * len(y_vals))

def generate_complexity_concept(y_vals):
    """Estimate function complexity using variance of differences"""
    if len(y_vals) < 3:
        return np.array([0] * len(y_vals))
    
    # Calculate variance of second differences
    second_diff = np.diff(y_vals, n=2)
    complexity = np.var(second_diff)
    return np.array([1 if complexity > np.median(second_diff) else 0] * len(y_vals))

def analyze_phase_changes(results):
    """Analyze phase changes in concept emergence"""
    print("\nAnalyzing phase changes...")
    
    # Find layers where concept becomes decodable
    for complexity in results:
        print(f"Complexity {complexity}:")
        for width in results[complexity]:
            accuracies = results[complexity][width]
            # Find first layer with accuracy > threshold
            threshold = 0.8
            for layer in sorted(accuracies.keys()):
                if accuracies[layer] > threshold:
                    print(f"  Width {width}: Layer {layer} (acc: {accuracies[layer]:.3f})")
                    break

def create_complexity_plots(results, timestamp):
    """Create comprehensive complexity sweep plots"""
    print("\nCreating complexity sweep plots...")
    
    # Extract data for plotting
    complexities = sorted(results.keys())
    widths = list(results[complexities[0]].keys())
    
    # Check if we have composite results
    sample_result = results[complexities[0]][widths[0]]
    is_composite = any('g_layer_' in key or 'fg_layer_' in key for key in sample_result.keys())
    
    if is_composite:
        create_composite_plots(results, timestamp)
    else:
        create_regular_plots(results, timestamp)

def create_regular_plots(results, timestamp):
    """Create plots for regular (non-composite) results"""
    complexities = sorted(results.keys())
    widths = list(results[complexities[0]].keys())
    layers = [key for key in results[complexities[0]][widths[0]].keys() if 'layer' in key]
    
    # Sort layers by extracting layer number, handling both MLP and transformer naming
    def get_layer_number(layer_name):
        parts = layer_name.split('_')
        # For transformer: layer_0_ln1 -> extract 0
        # For MLP: layer_0 -> extract 0
        for i, part in enumerate(parts):
            if part == 'layer' and i + 1 < len(parts):
                try:
                    return int(parts[i + 1])
                except ValueError:
                    continue
        return 0  # fallback
    
    layers = sorted(layers, key=get_layer_number)
    
    # Create emergence plot
    plt.figure(figsize=(12, 8))
    for i, width in enumerate(widths):
        plt.subplot(2, 2, i+1)
        for layer in layers:
            accuracies = [results[c][width][layer] for c in complexities]
            # Extract layer number for label
            layer_num = get_layer_number(layer)
            plt.plot(complexities, accuracies, marker='o', label=f'Layer {layer_num}')
        plt.xlabel('Complexity')
        plt.ylabel('Probe Accuracy')
        plt.title(f'Width {width}')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'plots/complexity_emergence_{timestamp}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: plots/complexity_emergence_{timestamp}.png")
    
    # Create heatmap
    plt.figure(figsize=(15, 5))
    for i, width in enumerate(widths):
        plt.subplot(1, 3, i+1)
        data = np.array([[results[c][width][layer] for layer in layers] for c in complexities])
        im = plt.imshow(data.T, cmap='viridis', aspect='auto')
        plt.colorbar(im)
        plt.xlabel('Complexity')
        plt.ylabel('Layer')
        plt.title(f'Width {width}')
        plt.xticks(range(len(complexities)), complexities)
        plt.yticks(range(len(layers)), [get_layer_number(layer) for layer in layers])
    
    plt.tight_layout()
    plt.savefig(f'plots/complexity_heatmap_{timestamp}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: plots/complexity_heatmap_{timestamp}.png")

def create_composite_plots(results, timestamp):
    """Create plots for composite function results"""
    complexities = sorted(results.keys())
    widths = list(results[complexities[0]].keys())
    
    # Separate g(x) and f(g(x)) results
    g_layers = [key for key in results[complexities[0]][widths[0]].keys() if 'g_layer_' in key]
    fg_layers = [key for key in results[complexities[0]][widths[0]].keys() if 'fg_layer_' in key]
    
    # Create comparison plots
    plt.figure(figsize=(15, 10))
    
    for i, width in enumerate(widths):
        # g(x) plot
        plt.subplot(2, 3, i+1)
        for layer in g_layers:
            layer_num = layer.split('_')[-1]
            accuracies = [results[c][width][layer] for c in complexities]
            plt.plot(complexities, accuracies, marker='o', label=f'Layer {layer_num}')
        plt.xlabel('Complexity')
        plt.ylabel('g(x) Probe Accuracy')
        plt.title(f'g(x) - Width {width}')
        plt.legend()
        plt.grid(True)
        
        # f(g(x)) plot
        plt.subplot(2, 3, i+4)
        for layer in fg_layers:
            layer_num = layer.split('_')[-1]
            accuracies = [results[c][width][layer] for c in complexities]
            plt.plot(complexities, accuracies, marker='o', label=f'Layer {layer_num}')
        plt.xlabel('Complexity')
        plt.ylabel('f(g(x)) Probe Accuracy')
        plt.title(f'f(g(x)) - Width {width}')
        plt.legend()
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'plots/composite_emergence_{timestamp}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: plots/composite_emergence_{timestamp}.png")

if __name__ == "__main__":
    run_complexity_sweep()
    print("\nComplexity sweep completed!")