#!/usr/bin/env python3
"""
Generate plots from existing complexity sweep results
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

def get_layer_number(layer_name):
    """Extract layer number from layer name"""
    parts = layer_name.split('_')
    for i, part in enumerate(parts):
        if part == 'layer' and i + 1 < len(parts):
            try:
                return int(parts[i + 1])
            except ValueError:
                continue
    return 0

def create_regular_plots(results, timestamp):
    """Create plots for regular (non-composite) results"""
    complexities = sorted(results.keys())
    widths = list(results[complexities[0]].keys())
    layers = [key for key in results[complexities[0]][widths[0]].keys() if 'layer' in key]
    layers = sorted(layers, key=get_layer_number)
    
    # Create emergence plot
    plt.figure(figsize=(12, 8))
    for i, width in enumerate(widths):
        plt.subplot(2, 2, i+1)
        for layer in layers:
            accuracies = [results[c][width][layer] for c in complexities]
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

def main():
    # Find the most recent results file
    results_dir = "results"
    result_files = [f for f in os.listdir(results_dir) if f.startswith("complexity_sweep_results_")]
    
    if not result_files:
        print("No complexity sweep results found!")
        return
    
    # Get the most recent file
    latest_file = sorted(result_files)[-1]
    results_path = os.path.join(results_dir, latest_file)
    
    print(f"Loading results from: {results_path}")
    results = torch.load(results_path)
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    # Generate plots
    print("Creating complexity sweep plots...")
    create_regular_plots(results, timestamp)
    
    print("Plots generated successfully!")

if __name__ == "__main__":
    main() 