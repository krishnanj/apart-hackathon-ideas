import torch
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import re

def plot_current_composite_results():
    """Plot current composite analysis results from checkpoints"""
    
    # Find all composite checkpoints
    checkpoint_pattern = "checkpoints/composite_*_width*.pt"
    checkpoints = glob.glob(checkpoint_pattern)
    
    print(f"Found {len(checkpoints)} checkpoints:")
    for cp in checkpoints:
        print(f"  {cp}")
    
    # Group by function type
    results = {}
    for checkpoint in checkpoints:
        # Parse checkpoint name properly
        # Format: composite_poly_sin_width128_20250727_194234.pt
        match = re.search(r'composite_(\w+)_(\w+)_width(\d+)_(\d+)_(\d+)\.pt', checkpoint)
        if match:
            inner_func = match.group(1)  # e.g., "poly"
            outer_func = match.group(2)  # e.g., "sin"
            width = int(match.group(3))  # e.g., 128
            func_type = f"{inner_func}_{outer_func}"
            
            if func_type not in results:
                results[func_type] = {}
            
            # Load model and get basic info
            try:
                model = torch.load(checkpoint, map_location='cpu')
                results[func_type][width] = {
                    'checkpoint': checkpoint,
                    'model_state': model,
                    'timestamp': os.path.getctime(checkpoint)
                }
                print(f"  Loaded {func_type} width {width}")
            except Exception as e:
                print(f"  Error loading {checkpoint}: {e}")
    
    # Create summary plot
    if results:
        create_summary_plot(results)
    else:
        print("No valid checkpoints found!")

def create_summary_plot(results):
    """Create a summary plot of current training progress"""
    
    num_funcs = len(results)
    cols = min(3, num_funcs)  # Max 3 columns
    rows = (num_funcs + cols - 1) // cols  # Calculate needed rows
    
    plt.figure(figsize=(5*cols, 4*rows))
    
    # Plot for each function type
    for i, (func_type, widths) in enumerate(results.items()):
        plt.subplot(rows, cols, i+1)
        
        # Plot training progress (simplified)
        widths_list = sorted(widths.keys())
        timestamps = [widths[w]['timestamp'] for w in widths_list]
        
        # Convert timestamps to relative time
        if timestamps:
            base_time = min(timestamps)
            relative_times = [(t - base_time) / 3600 for t in timestamps]  # hours
            
            plt.plot(relative_times, widths_list, 'o-', label=func_type, linewidth=2, markersize=8)
            plt.xlabel('Training time (hours)')
            plt.ylabel('Model width')
            plt.title(f'{func_type.upper()} - Training Progress')
            plt.grid(True, alpha=0.3)
            
            # Add width labels
            for w, t in zip(widths_list, relative_times):
                plt.annotate(f'w{w}', (t, w), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=8)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"plots/current_composite_progress_{timestamp}.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved progress plot: {plot_filename}")
    
    # Create checkpoint status table
    print("\nCheckpoint Status:")
    print("=" * 60)
    for func_type, widths in results.items():
        print(f"\n{func_type.upper()}:")
        for width in sorted(widths.keys()):
            timestamp = datetime.fromtimestamp(widths[width]['timestamp'])
            print(f"  Width {width:3d}: {timestamp.strftime('%H:%M:%S')}")

if __name__ == "__main__":
    plot_current_composite_results() 