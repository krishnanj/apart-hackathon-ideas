import torch
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from composite_analysis import generate_hard_concept
from transformer_probe_utils import TransformerProbeAnalyzer
from model import create_model_from_params
from synthetic_functions import SyntheticFunctionGenerator
import yaml

def quick_probe_checkpoints():
    """Quickly probe existing checkpoints and generate plots"""
    
    # Load params
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Find all composite checkpoints
    checkpoint_pattern = "checkpoints/composite_*_width*.pt"
    checkpoints = glob.glob(checkpoint_pattern)
    
    print(f"Found {len(checkpoints)} checkpoints to probe...")
    
    # Group by function type
    results = {}
    for checkpoint in checkpoints:
        # Parse checkpoint name
        parts = checkpoint.split('_')
        if len(parts) >= 4:
            func_type = f"{parts[1]}_{parts[2]}"  # e.g., "relu_sin"
            width = int(parts[-1].split('.')[0])  # e.g., 16
            
            if func_type not in results:
                results[func_type] = {}
            
            print(f"Probing {func_type} width {width}...")
            
            # Quick probe this checkpoint
            probe_result = quick_probe_single_checkpoint(checkpoint, func_type, width, params)
            if probe_result:
                results[func_type][width] = probe_result
    
    # Generate plots
    if results:
        create_quick_probe_plots(results)
    else:
        print("No valid results to plot!")

def quick_probe_single_checkpoint(checkpoint_path, func_type, width, params):
    """Quickly probe a single checkpoint"""
    try:
        # Load model
        model_state = torch.load(checkpoint_path, map_location='cpu')
        
        # Create model with same architecture
        params['width'] = width
        params['architecture'] = 'transformer'  # Force transformer for composite
        model = create_model_from_params(params)
        model.load_state_dict(model_state)
        model.eval()
        
        # Generate test data using SyntheticFunctionGenerator
        generator = SyntheticFunctionGenerator()
        inner_func, outer_func = func_type.split('_')
        test_x, test_g_x, test_f_gx = generator.generate_composite(
            inner_func, outer_func, seed=42
        )
        
        # Convert to transformer format
        from dataset import convert_to_sequence_format
        test_x = convert_to_sequence_format(test_x, test_g_x, test_f_gx, params)
        test_g_x = convert_to_sequence_format(test_g_x, None, None, params)
        test_f_gx = convert_to_sequence_format(test_f_gx, None, None, params)
        
        # Generate concepts
        g_concept = generate_hard_concept(test_g_x.numpy())
        fg_concept = generate_hard_concept(test_f_gx.numpy())
        
        # Reshape for transformer
        batch_size = test_x.shape[0]
        seq_len = test_x.shape[1]
        g_concept = g_concept.reshape(batch_size, 1).repeat(1, seq_len).flatten()
        fg_concept = fg_concept.reshape(batch_size, 1).repeat(1, seq_len).flatten()
        
        # Quick probe with transformer analyzer
        analyzer = TransformerProbeAnalyzer(model)
        
        # Probe g(x) concept
        g_results = analyzer.run_transformer_probes(test_x, g_concept)
        
        # Probe f(g(x)) concept  
        fg_results = analyzer.run_transformer_probes(test_x, fg_concept)
        
        return {
            'g_results': g_results,
            'fg_results': fg_results,
            'width': width,
            'func_type': func_type
        }
        
    except Exception as e:
        print(f"  Error probing {checkpoint_path}: {e}")
        return None

def create_quick_probe_plots(results):
    """Create quick probe result plots"""
    
    num_funcs = len(results)
    cols = min(3, num_funcs)
    rows = (num_funcs + cols - 1) // cols
    
    plt.figure(figsize=(6*cols, 5*rows))
    
    for i, (func_type, widths) in enumerate(results.items()):
        plt.subplot(rows, cols, i+1)
        
        # Plot g(x) vs f(g(x)) results
        widths_list = sorted(widths.keys())
        g_accuracies = []
        fg_accuracies = []
        
        for width in widths_list:
            result = widths[width]
            
            # Get average accuracy across layers
            g_avg = np.mean([v for v in result['g_results'].values() if isinstance(v, (int, float))])
            fg_avg = np.mean([v for v in result['fg_results'].values() if isinstance(v, (int, float))])
            
            g_accuracies.append(g_avg)
            fg_accuracies.append(fg_avg)
        
        plt.plot(widths_list, g_accuracies, 'o-', label='g(x) concept', linewidth=2)
        plt.plot(widths_list, fg_accuracies, 's-', label='f(g(x)) concept', linewidth=2)
        
        plt.xlabel('Model Width')
        plt.ylabel('Average Probe Accuracy')
        plt.title(f'{func_type.upper()} - Concept Decodability')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.1)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"plots/quick_composite_probe_results_{timestamp}.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved quick probe results: {plot_filename}")
    
    # Print summary
    print("\nQuick Probe Results Summary:")
    print("=" * 50)
    for func_type, widths in results.items():
        print(f"\n{func_type.upper()}:")
        for width in sorted(widths.keys()):
            result = widths[width]
            g_avg = np.mean([v for v in result['g_results'].values() if isinstance(v, (int, float))])
            fg_avg = np.mean([v for v in result['fg_results'].values() if isinstance(v, (int, float))])
            print(f"  Width {width:3d}: g(x)={g_avg:.3f}, f(g(x))={fg_avg:.3f}")

if __name__ == "__main__":
    quick_probe_checkpoints() 