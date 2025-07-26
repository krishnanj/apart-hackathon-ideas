#!/usr/bin/env python3
"""
Composite Function Analysis: Dual probing for f(g(x)) vs g(x)
Tests multiple composite function combinations
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os

from model import create_model_from_params
from dataset import create_dataset_from_params
from probe_utils import register_hooks, save_dict_as_pt
from advanced_probe_utils import MultiProbeAnalyzer
from load_params import load_params

def run_composite_analysis(inner_func="poly", outer_func="sin"):
    """Run composite function analysis with dual probing for specific function combination"""
    params = load_params()
    
    # Enable composite functions
    params["use_composite"] = True
    params["inner_func"] = inner_func
    params["outer_func"] = outer_func
    
    # Load parameters
    widths = params["widths"]
    depth = params["depth"]
    seed = params["seed"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Generate timestamp and function prefix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    func_prefix = f"composite_{inner_func}_{outer_func}"
    
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Create composite dataset
    print(f"Creating composite dataset: {inner_func} -> {outer_func}...")
    train_x, train_g_x, train_f_gx, test_x, test_g_x, test_f_gx = create_dataset_from_params(params)
    
    print(f"Dataset shapes:")
    print(f"  train_x: {train_x.shape}, train_g_x: {train_g_x.shape}, train_f_gx: {train_f_gx.shape}")
    print(f"  test_x: {test_x.shape}, test_g_x: {test_g_x.shape}, test_f_gx: {test_f_gx.shape}")
    
    # Train models
    print(f"\nTraining models for {inner_func} -> {outer_func}...")
    train_x = train_x.float().to(device)
    train_f_gx = train_f_gx.float().to(device)  # Train on f(g(x))
    
    for w in widths:
        print(f"Training model with width {w}...")
        
        # Create model
        input_dim = train_x.shape[1]
        output_dim = train_f_gx.shape[1] if len(train_f_gx.shape) > 1 else 1
        model = create_model_from_params(params, input_dim=input_dim, output_dim=output_dim).to(device)
        
        # Train on f(g(x))
        opt = torch.optim.Adam(model.parameters(), lr=params["lr"])
        
        for step in range(params["steps"]):
            logits = model(train_x)
            
            if output_dim == 1:
                logits = logits.squeeze(-1)
            loss = torch.nn.functional.mse_loss(logits, train_f_gx.squeeze(-1))
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            if step % 1000 == 0:
                print(f"  Step {step}, Loss: {loss.item():.4f}")
        
        # Save model
        checkpoint_name = f"checkpoints/{func_prefix}_width{w}_{timestamp}.pt"
        torch.save(model.state_dict(), checkpoint_name)
        print(f"  Saved model: {checkpoint_name}")
    
    # Dual probing analysis
    print(f"\nRunning dual probing analysis for {inner_func} -> {outer_func}...")
    analyzer = MultiProbeAnalyzer()
    
    # Results storage
    g_probe_results = {}  # Probe results for g(x)
    f_probe_results = {}  # Probe results for f(g(x))
    
    for w in widths:
        print(f"Analyzing model with width {w}...")
        
        # Load model
        model = create_model_from_params(params, input_dim=test_x.shape[1], output_dim=1).to(device)
        checkpoint_files = [f for f in os.listdir("checkpoints") 
                          if f.startswith(f"{func_prefix}_width{w}_") and f.endswith(".pt")]
        
        if not checkpoint_files:
            print(f"  No checkpoint found for width {w}")
            continue
            
        latest_checkpoint = sorted(checkpoint_files)[-1]
        ckpt_path = os.path.join("checkpoints", latest_checkpoint)
        model.load_state_dict(torch.load(ckpt_path))
        model.eval()
        
        # Get activations
        activation_store = {}
        hooks = register_hooks(model, layer_indices=list(range(depth + 1)), activation_store=activation_store)
        
        with torch.no_grad():
            _ = model(test_x.float().to(device))
        
        # Dual probing: g(x) vs f(g(x))
        print(f"  Probing for g(x) concept...")
        g_results = analyzer.run_probes(activation_store, test_g_x, is_regression=True)
        g_probe_results[w] = g_results
        
        print(f"  Probing for f(g(x)) concept...")
        f_results = analyzer.run_probes(activation_store, test_f_gx, is_regression=True)
        f_probe_results[w] = f_results
        
        for h in hooks:
            h.remove()
    
    # Save results
    results = {
        'g_probe_results': g_probe_results,
        'f_probe_results': f_probe_results,
        'config': {
            'inner_func': inner_func,
            'outer_func': outer_func,
            'widths': widths,
            'depth': depth
        }
    }
    
    results_filename = f"results/{func_prefix}_dual_probe_results_{timestamp}.pt"
    save_dict_as_pt(results, results_filename)
    print(f"\nSaved results to: {results_filename}")
    
    # Create comparison plots
    print(f"\nCreating comparison plots for {inner_func} -> {outer_func}...")
    create_composite_comparison_plots(results, func_prefix, timestamp)
    
    return results

def run_multiple_composite_analyses():
    """Run composite analysis for multiple function combinations"""
    # Define combinations to test
    combinations = [
        ("poly", "sin"),   # polynomial -> trigonometric
        ("sin", "poly"),   # trigonometric -> polynomial  
        ("relu", "poly"),  # relu -> polynomial
        ("poly", "relu"),  # polynomial -> relu
        ("sin", "relu"),   # trigonometric -> relu
        ("relu", "sin"),   # relu -> trigonometric
    ]
    
    all_results = {}
    
    for inner_func, outer_func in combinations:
        print(f"\n{'='*60}")
        print(f"Testing composite function: {inner_func} -> {outer_func}")
        print(f"{'='*60}")
        
        try:
            results = run_composite_analysis(inner_func, outer_func)
            all_results[f"{inner_func}_{outer_func}"] = results
        except Exception as e:
            print(f"Error with {inner_func} -> {outer_func}: {e}")
            continue
    
    # Create summary comparison
    create_summary_comparison(all_results)
    
    return all_results

def create_summary_comparison(all_results):
    """Create summary comparison across all composite functions"""
    print("\nCreating summary comparison...")
    
    # Extract emergence data for each combination
    summary_data = {}
    
    for combo_name, results in all_results.items():
        g_results = results['g_probe_results']
        f_results = results['f_probe_results']
        
        # Calculate average emergence layer for each concept
        g_emergence_layers = []
        f_emergence_layers = []
        
        for width in g_results.keys():
            # Find g(x) emergence
            for layer in range(len(g_results[width])):
                if g_results[width][layer]['linear'] > 0.8:
                    g_emergence_layers.append(layer)
                    break
            else:
                g_emergence_layers.append(len(g_results[width]))
            
            # Find f(g(x)) emergence
            for layer in range(len(f_results[width])):
                if f_results[width][layer]['linear'] > 0.8:
                    f_emergence_layers.append(layer)
                    break
            else:
                f_emergence_layers.append(len(f_results[width]))
        
        summary_data[combo_name] = {
            'g_avg_emergence': np.mean(g_emergence_layers),
            'f_avg_emergence': np.mean(f_emergence_layers),
            'g_std_emergence': np.std(g_emergence_layers),
            'f_std_emergence': np.std(f_emergence_layers)
        }
    
    # Create summary plot
    plt.figure(figsize=(12, 8))
    
    combos = list(summary_data.keys())
    g_emergence = [summary_data[combo]['g_avg_emergence'] for combo in combos]
    f_emergence = [summary_data[combo]['f_avg_emergence'] for combo in combos]
    
    x = np.arange(len(combos))
    width = 0.35
    
    plt.bar(x - width/2, g_emergence, width, label='g(x) emergence', alpha=0.8)
    plt.bar(x + width/2, f_emergence, width, label='f(g(x)) emergence', alpha=0.8)
    
    plt.xlabel('Composite Function Combinations')
    plt.ylabel('Average Emergence Layer')
    plt.title('Composite Function Emergence Comparison')
    plt.xticks(x, combos, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"plots/composite_summary_comparison_{timestamp}.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved summary comparison: {plot_filename}")
    
    # Print summary table
    print("\nSummary Table:")
    print(f"{'Combination':<15} {'g(x) Avg':<10} {'f(g(x)) Avg':<12} {'Difference':<12}")
    print("-" * 50)
    for combo in combos:
        g_avg = summary_data[combo]['g_avg_emergence']
        f_avg = summary_data[combo]['f_avg_emergence']
        diff = g_avg - f_avg
        print(f"{combo:<15} {g_avg:<10.2f} {f_avg:<12.2f} {diff:<12.2f}")

def create_composite_comparison_plots(results, func_prefix, timestamp):
    """Create comparison plots for g(x) vs f(g(x)) probing"""
    g_results = results['g_probe_results']
    f_results = results['f_probe_results']
    widths = results['config']['widths']
    depth = results['config']['depth']
    
    # Get probe types
    first_width = list(g_results.keys())[0]
    probe_types = list(g_results[first_width][0].keys())
    
    for probe_type in probe_types:
        print(f"Creating plots for {probe_type} probe...")
        
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        
        # Plot g(x) results
        for layer_idx in range(depth + 1):
            xs = []
            ys = []
            for w in widths:
                if w in g_results:
                    xs.append(w)
                    ys.append(g_results[w][layer_idx][probe_type])
            plt.plot(xs, ys, marker='o', linestyle='--', label=f"g(x) - Layer {layer_idx}")
        
        # Plot f(g(x)) results
        for layer_idx in range(depth + 1):
            xs = []
            ys = []
            for w in widths:
                if w in f_results:
                    xs.append(w)
                    ys.append(f_results[w][layer_idx][probe_type])
            plt.plot(xs, ys, marker='s', linestyle='-', label=f"f(g(x)) - Layer {layer_idx}")
        
        plt.xlabel("Model width")
        plt.ylabel(f"Probe accuracy ({probe_type})")
        plt.title(f"Composite Function Analysis: g(x) vs f(g(x)) - {func_prefix}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_filename = f"plots/{func_prefix}_{probe_type}_composite_comparison_{timestamp}.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {plot_filename}")
        
        # Create emergence analysis plot
        create_emergence_analysis_plot(g_results, f_results, probe_type, func_prefix, timestamp)

def create_emergence_analysis_plot(g_results, f_results, probe_type, func_prefix, timestamp):
    """Create emergence analysis plot showing when concepts become decodable"""
    widths = list(g_results.keys())
    depth = len(g_results[widths[0]])
    
    # Find emergence points (first layer where accuracy > 0.8)
    g_emergence = {}
    f_emergence = {}
    
    for w in widths:
        # Find g(x) emergence
        for layer in range(depth):
            if g_results[w][layer][probe_type] > 0.8:
                g_emergence[w] = layer
                break
        else:
            g_emergence[w] = depth  # Never emerged
            
        # Find f(g(x)) emergence
        for layer in range(depth):
            if f_results[w][layer][probe_type] > 0.8:
                f_emergence[w] = layer
                break
        else:
            f_emergence[w] = depth  # Never emerged
    
    # Plot emergence comparison
    plt.figure(figsize=(10, 6))
    
    g_layers = [g_emergence[w] for w in widths]
    f_layers = [f_emergence[w] for w in widths]
    
    plt.plot(widths, g_layers, marker='o', label='g(x) emergence', linewidth=2)
    plt.plot(widths, f_layers, marker='s', label='f(g(x)) emergence', linewidth=2)
    
    plt.xlabel("Model width")
    plt.ylabel("Layer of emergence (accuracy > 0.8)")
    plt.title(f"Concept Emergence Analysis - {func_prefix} ({probe_type})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.5, depth + 0.5)
    
    plot_filename = f"plots/{func_prefix}_{probe_type}_emergence_analysis_{timestamp}.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved emergence analysis: {plot_filename}")

if __name__ == "__main__":
    # Run single composite analysis (original behavior)
    # results = run_composite_analysis()
    
    # Run multiple composite analyses
    all_results = run_multiple_composite_analyses()
    print("\nAll composite function analyses completed!") 