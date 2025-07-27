import torch
import os
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import yaml
from model import create_model_from_params
from synthetic_functions import SyntheticFunctionGenerator
from dataset import convert_to_sequence_format

def analyze_checkpoint_results():
    """Actually analyze the trained models and show real results"""
    
    # Load params
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Find all composite checkpoints
    checkpoint_pattern = "checkpoints/composite_*_width*.pt"
    checkpoints = glob.glob(checkpoint_pattern)
    
    print(f"Found {len(checkpoints)} checkpoints to analyze...")
    
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
            
            print(f"Analyzing {func_type} width {width}...")
            
            # Analyze this checkpoint
            analysis_result = analyze_single_checkpoint(checkpoint, func_type, width, params)
            if analysis_result:
                results[func_type][width] = analysis_result
    
    # Generate meaningful plots
    if results:
        create_meaningful_plots(results)
    else:
        print("No valid results to plot!")

def analyze_single_checkpoint(checkpoint_path, func_type, width, params):
    """Analyze a single checkpoint for actual results"""
    try:
        # Load model
        model_state = torch.load(checkpoint_path, map_location='cpu')
        
        # Create model with same architecture
        params['width'] = width
        params['architecture'] = 'transformer'
        
        # Get transformer config
        transformer_config = params.get('transformer_config', {})
        dim = transformer_config.get('dim', 128)
        num_layers = transformer_config.get('num_layers', 2)
        num_heads = transformer_config.get('num_heads', 4)
        seq_len = transformer_config.get('seq_len', 5)
        num_tokens = transformer_config.get('num_tokens', 100)
        
        # Create model with proper arguments
        model = create_model_from_params(
            input_dim=seq_len,  # For transformer, input_dim is seq_len
            output_dim=num_tokens,  # For transformer, output_dim is num_tokens
            params=params
        )
        model.load_state_dict(model_state)
        model.eval()
        
        # Generate test data
        generator = SyntheticFunctionGenerator()
        inner_func, outer_func = func_type.split('_')
        test_x, test_g_x, test_f_gx = generator.generate_composite(
            inner_func, outer_func, seed=42
        )
        
        # Convert to transformer format
        test_x = convert_to_sequence_format(test_x, test_g_x, test_f_gx, params)
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(test_x)
            predictions = outputs.argmax(dim=-1)
        
        # Calculate accuracy
        # For composite functions, we need to extract the last token as prediction
        last_token_predictions = predictions[:, -1]
        
        # Convert targets to appropriate format
        test_g_x_seq = convert_to_sequence_format(test_g_x, None, None, params)
        test_f_gx_seq = convert_to_sequence_format(test_f_gx, None, None, params)
        
        # Calculate accuracies
        g_accuracy = (last_token_predictions == test_g_x_seq[:, -1]).float().mean().item()
        fg_accuracy = (last_token_predictions == test_f_gx_seq[:, -1]).float().mean().item()
        
        # Get model size info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        return {
            'g_accuracy': g_accuracy,
            'fg_accuracy': fg_accuracy,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'width': width,
            'func_type': func_type,
            'checkpoint_path': checkpoint_path
        }
        
    except Exception as e:
        print(f"  Error analyzing {checkpoint_path}: {e}")
        return None

def create_meaningful_plots(results):
    """Create meaningful plots showing actual results"""
    
    num_funcs = len(results)
    cols = min(3, num_funcs)
    rows = (num_funcs + cols - 1) // cols
    
    plt.figure(figsize=(6*cols, 5*rows))
    
    for i, (func_type, widths) in enumerate(results.items()):
        plt.subplot(rows, cols, i+1)
        
        # Plot actual accuracies
        widths_list = sorted(widths.keys())
        g_accuracies = [widths[w]['g_accuracy'] for w in widths_list]
        fg_accuracies = [widths[w]['fg_accuracy'] for w in widths_list]
        
        plt.plot(widths_list, g_accuracies, 'o-', label='g(x) accuracy', linewidth=2, markersize=8)
        plt.plot(widths_list, fg_accuracies, 's-', label='f(g(x)) accuracy', linewidth=2, markersize=8)
        
        plt.xlabel('Model Width')
        plt.ylabel('Accuracy')
        plt.title(f'{func_type.upper()} - Model Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.1)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"plots/checkpoint_analysis_results_{timestamp}.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved analysis results: {plot_filename}")
    
    # Print detailed results
    print("\nDetailed Checkpoint Analysis:")
    print("=" * 60)
    for func_type, widths in results.items():
        print(f"\n{func_type.upper()}:")
        for width in sorted(widths.keys()):
            result = widths[width]
            print(f"  Width {width:3d}: g(x)={result['g_accuracy']:.3f}, f(g(x))={result['fg_accuracy']:.3f}, params={result['total_params']:,}")

if __name__ == "__main__":
    analyze_checkpoint_results() 