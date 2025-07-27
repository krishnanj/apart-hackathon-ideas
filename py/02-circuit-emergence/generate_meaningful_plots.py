#!/usr/bin/env python3
"""
Generate meaningful plots from complexity sweep results
Focus on layers that show actual variation
"""

import torch
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

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
    
    # Focus on position embeddings which show variation
    complexities = sorted(results.keys())
    widths = list(results[complexities[0]].keys())
    
    # Plot 1: Position embeddings across complexity
    plt.figure(figsize=(12, 8))
    for i, width in enumerate(widths):
        plt.subplot(2, 2, i+1)
        pos_accuracies = [results[c][width]['pos_embeddings'] for c in complexities]
        plt.plot(complexities, pos_accuracies, marker='o', linewidth=2, markersize=8)
        plt.xlabel('Complexity')
        plt.ylabel('Position Embedding Probe Accuracy')
        plt.title(f'Width {width}')
        plt.grid(True, alpha=0.3)
        plt.ylim(0.4, 0.8)
    
    plt.tight_layout()
    plt.savefig(f'plots/pos_embeddings_vs_complexity_{timestamp}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: plots/pos_embeddings_vs_complexity_{timestamp}.png")
    
    # Plot 2: All layers heatmap (focusing on position embeddings)
    plt.figure(figsize=(15, 10))
    
    # Get all layer names
    sample_result = results[complexities[0]][widths[0]]
    layer_names = list(sample_result.keys())
    
    # Create subplots for each width
    for i, width in enumerate(widths):
        plt.subplot(2, 2, i+1)
        
        # Create data matrix
        data = []
        layer_labels = []
        
        for layer in layer_names:
            accuracies = [results[c][width][layer] for c in complexities]
            data.append(accuracies)
            layer_labels.append(layer)
        
        data = np.array(data)
        
        # Create heatmap
        im = plt.imshow(data, cmap='viridis', aspect='auto')
        plt.colorbar(im)
        plt.xlabel('Complexity')
        plt.ylabel('Layer')
        plt.title(f'Width {width} - All Layers')
        plt.xticks(range(len(complexities)), complexities)
        plt.yticks(range(len(layer_labels)), layer_labels, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(f'plots/all_layers_heatmap_{timestamp}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: plots/all_layers_heatmap_{timestamp}.png")
    
    # Plot 3: Comparison of embeddings vs position embeddings
    plt.figure(figsize=(12, 8))
    for i, width in enumerate(widths):
        plt.subplot(2, 2, i+1)
        
        emb_accuracies = [results[c][width]['embeddings'] for c in complexities]
        pos_accuracies = [results[c][width]['pos_embeddings'] for c in complexities]
        
        plt.plot(complexities, emb_accuracies, marker='o', label='Token Embeddings', linewidth=2)
        plt.plot(complexities, pos_accuracies, marker='s', label='Position Embeddings', linewidth=2)
        
        plt.xlabel('Complexity')
        plt.ylabel('Probe Accuracy')
        plt.title(f'Width {width} - Embeddings Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0.4, 1.1)
    
    plt.tight_layout()
    plt.savefig(f'plots/embeddings_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
    print(f"Saved: plots/embeddings_comparison_{timestamp}.png")
    
    print("Meaningful plots generated successfully!")

if __name__ == "__main__":
    main() 