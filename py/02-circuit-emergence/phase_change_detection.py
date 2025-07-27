#!/usr/bin/env python3
"""
Phase Change Detection: Identify qualitative changes in emergence
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from datetime import datetime
import os

def detect_phase_changes(complexity_results, emergence_threshold=0.8):
    """Detect phase changes in emergence patterns"""
    
    complexities = sorted(complexity_results.keys())
    widths = list(complexity_results[complexities[0]].keys())
    probe_types = ["linear", "tree", "svm"]
    
    phase_changes = {}
    
    for width in widths:
        phase_changes[width] = {}
        
        for probe_type in probe_types:
            # Extract emergence layers for this width/probe combination
            emergence_layers = []
            for complexity in complexities:
                # Find first layer where accuracy >= threshold
                layer_found = False
                for layer in sorted(complexity_results[complexity][width].keys()):
                    if complexity_results[complexity][width][layer][probe_type] >= emergence_threshold:
                        emergence_layers.append(layer)
                        layer_found = True
                        break
                
                if not layer_found:
                    emergence_layers.append(None)  # Never emerged
            
            # Detect discontinuities
            discontinuities = detect_discontinuities(emergence_layers, complexities)
            phase_changes[width][probe_type] = discontinuities
    
    return phase_changes

def detect_discontinuities(emergence_layers, complexities):
    """Detect statistical discontinuities in emergence patterns"""
    
    # Filter out None values for analysis
    valid_indices = [i for i, layer in enumerate(emergence_layers) if layer is not None]
    valid_layers = [emergence_layers[i] for i in valid_indices]
    valid_complexities = [complexities[i] for i in valid_indices]
    
    if len(valid_layers) < 3:
        return []  # Not enough data for discontinuity detection
    
    discontinuities = []
    
    # Method 1: Statistical change point detection
    for i in range(1, len(valid_layers) - 1):
        before = valid_layers[:i]
        after = valid_layers[i:]
        
        if len(before) >= 2 and len(after) >= 2:
            # T-test for difference in means
            t_stat, p_value = stats.ttest_ind(before, after)
            
            if p_value < 0.05:  # Significant difference
                discontinuities.append({
                    'complexity': valid_complexities[i],
                    'index': valid_indices[i],
                    'p_value': p_value,
                    't_stat': t_stat,
                    'before_mean': np.mean(before),
                    'after_mean': np.mean(after)
                })
    
    # Method 2: Large jumps in emergence layer
    for i in range(1, len(valid_layers)):
        jump = abs(valid_layers[i] - valid_layers[i-1])
        if jump >= 2:  # Large jump in emergence layer
            discontinuities.append({
                'complexity': valid_complexities[i],
                'index': valid_indices[i],
                'jump_size': jump,
                'type': 'large_jump'
            })
    
    return discontinuities

def create_phase_change_plots(complexity_results, phase_changes, timestamp):
    """Create phase change visualization plots"""
    
    complexities = sorted(complexity_results.keys())
    widths = list(complexity_results[complexities[0]].keys())
    probe_types = ["linear", "tree", "svm"]
    
    # Plot 1: Phase change summary
    plt.figure(figsize=(15, 10))
    
    for i, probe_type in enumerate(probe_types):
        plt.subplot(2, 2, i+1)
        
        for width in widths:
            if width in phase_changes and probe_type in phase_changes[width]:
                changes = phase_changes[width][probe_type]
                
                if changes:
                    change_complexities = [c['complexity'] for c in changes]
                    change_scores = [c.get('p_value', 1.0) for c in changes]
                    
                    plt.scatter(change_complexities, change_scores, 
                              label=f'Width {width}', s=100, alpha=0.7)
        
        plt.xlabel('Complexity at Phase Change')
        plt.ylabel('Statistical Significance (p-value)')
        plt.title(f'{probe_type.capitalize()} Phase Changes')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
    
    # Plot 4: Phase change frequency
    plt.subplot(2, 2, 4)
    
    phase_change_counts = {}
    for width in widths:
        for probe_type in probe_types:
            if width in phase_changes and probe_type in phase_changes[width]:
                count = len(phase_changes[width][probe_type])
                key = f'{width}_{probe_type}'
                phase_change_counts[key] = count
    
    if phase_change_counts:
        labels = list(phase_change_counts.keys())
        counts = list(phase_change_counts.values())
        
        plt.bar(range(len(labels)), counts)
        plt.xticks(range(len(labels)), labels, rotation=45)
        plt.ylabel('Number of Phase Changes')
        plt.title('Phase Change Frequency')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_filename = f"plots/phase_change_analysis_{timestamp}.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_filename}")

def analyze_emergence_patterns(complexity_results):
    """Analyze patterns in emergence behavior"""
    
    patterns = {}
    
    for complexity in complexity_results:
        complexity_data = complexity_results[complexity]
        
        # Analyze emergence consistency across widths
        emergence_consistency = {}
        for probe_type in ["linear", "tree", "svm"]:
            layers = []
            for width in complexity_data:
                # Find emergence layer
                for layer in sorted(complexity_data[width].keys()):
                    if complexity_data[width][layer][probe_type] >= 0.8:
                        layers.append(layer)
                        break
                else:
                    layers.append(None)
            
            # Calculate consistency
            valid_layers = [l for l in layers if l is not None]
            if valid_layers:
                consistency = 1.0 - (np.std(valid_layers) / np.mean(valid_layers)) if np.mean(valid_layers) > 0 else 0
                emergence_consistency[probe_type] = consistency
            else:
                emergence_consistency[probe_type] = 0
        
        patterns[complexity] = emergence_consistency
    
    return patterns

if __name__ == "__main__":
    # This script is typically called from complexity_sweep.py
    print("Phase change detection utilities loaded")