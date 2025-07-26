import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from datetime import datetime
from load_params import load_params

# Load parameters to get function prefix
params = load_params()
use_synthetic = params.get("use_synthetic", False)

if use_synthetic:
    func_type = params.get("func_type", "polynomial")
    complexity = params.get("complexity", 3)
    func_prefix = f"{func_type}_deg{complexity}"
else:
    func_prefix = "mod_add"

# Directory where probe results are saved
RESULTS_DIR = "results"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# List of model widths to plot (should match those in params.yaml)
widths = [16, 32, 64, 128, 256]

# Find the most recent probe results file for this function type
results_files = [f for f in os.listdir(RESULTS_DIR) 
                if f.startswith(f"{func_prefix}_") and f.endswith(".pt")]

if not results_files:
    print(f"No probe results found for {func_prefix}")
    exit()

# Get the most recent results file
latest_results = sorted(results_files)[-1]
results_path = os.path.join(RESULTS_DIR, latest_results)
print(f"Loading probe results from: {results_path}")

# Load probe results
probe_results_all = torch.load(results_path, weights_only=False)

# Check if this is multi-probe results or single-probe results
if isinstance(probe_results_all[list(probe_results_all.keys())[0]], dict):
    # Multi-probe results (new format)
    print("Detected multi-probe results, creating plots for each probe type...")
    
    # Get probe types from the first width's results
    first_width = list(probe_results_all.keys())[0]
    probe_types = list(probe_results_all[first_width][0].keys())
    
    for probe_type in probe_types:
        print(f"Creating plots for {probe_type} probe...")
        
        # Create individual layer plots for this probe type
        for width in widths:
            if width in probe_results_all:
                # Extract results for this width and probe type
                width_results = probe_results_all[width]
                probe_acc = [width_results[layer][probe_type] for layer in sorted(width_results.keys())]
                layers = np.arange(len(probe_acc))
                
                plt.figure(figsize=(8, 6))
                plt.plot(layers, probe_acc, marker='o', linewidth=2, markersize=8)
                plt.title(f"Probe Accuracy vs. Layer (width={width}) - {func_prefix} ({probe_type})")
                plt.xlabel("Layer")
                plt.ylabel("Probe Accuracy")
                
                # Adjust y-axis to show variation better
                min_acc = min(probe_acc)
                max_acc = max(probe_acc)
                if max_acc - min_acc < 0.1:  # If variation is small
                    plt.ylim(min_acc - 0.05, max_acc + 0.05)
                else:
                    plt.ylim(0, 1.1)  # Extend above 1.0 to show perfect accuracy
                
                plt.grid(True, alpha=0.3)
                
                # Add value annotations on points
                for i, (layer, acc) in enumerate(zip(layers, probe_acc)):
                    plt.annotate(f'{acc:.3f}', (layer, acc), 
                                textcoords="offset points", xytext=(0,10), 
                                ha='center', fontsize=9)
                
                # Save with timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                plot_filename = f"{PLOTS_DIR}/{func_prefix}_{probe_type}_probe_accuracy_vs_layer_width{width}_{timestamp}.png"
                plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Saved plot: {plot_filename}")
        
        # Create width comparison plot for this probe type
        plt.figure(figsize=(10, 6))
        for layer_idx in range(len(probe_results_all[first_width])):
            xs = []
            ys = []
            for w in widths:
                if w in probe_results_all:
                    xs.append(w)
                    ys.append(probe_results_all[w][layer_idx][probe_type])
            plt.plot(xs, ys, marker='o', label=f"Layer {layer_idx}")

        plt.xlabel("Model width")
        plt.ylabel(f"Probe accuracy ({probe_type})")
        plt.title(f"Concept Decodability vs. Width - {func_prefix} ({probe_type})")
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1.1)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"{PLOTS_DIR}/{func_prefix}_{probe_type}_probe_accuracy_vs_width_{timestamp}.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved width comparison plot: {plot_filename}")

else:
    # Single-probe results (old format)
    print("Detected single-probe results, creating individual plots...")
    
    # Generate individual plots for each width
    for width in widths:
        if width in probe_results_all:
            probe_acc = probe_results_all[width]
            layers = np.arange(len(probe_acc))
            
            plt.figure(figsize=(8, 6))
            plt.plot(layers, probe_acc, marker='o', linewidth=2, markersize=8)
            plt.title(f"Probe Accuracy vs. Layer (width={width}) - {func_prefix}")
            plt.xlabel("Layer")
            plt.ylabel("Probe Accuracy")
            
            # Adjust y-axis to show variation better
            min_acc = min(probe_acc)
            max_acc = max(probe_acc)
            if max_acc - min_acc < 0.1:  # If variation is small
                plt.ylim(min_acc - 0.05, max_acc + 0.05)
            else:
                plt.ylim(0, 1.1)  # Extend above 1.0 to show perfect accuracy
            
            plt.grid(True, alpha=0.3)
            
            # Add value annotations on points
            for i, (layer, acc) in enumerate(zip(layers, probe_acc)):
                plt.annotate(f'{acc:.3f}', (layer, acc), 
                            textcoords="offset points", xytext=(0,10), 
                            ha='center', fontsize=9)
            
            # Save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"{PLOTS_DIR}/{func_prefix}_probe_accuracy_vs_layer_width{width}_{timestamp}.png"
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved plot: {plot_filename}")

print(f"Saved probe accuracy plots for all widths for {func_prefix}") 