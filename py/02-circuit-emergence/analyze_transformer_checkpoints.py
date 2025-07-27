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

def detect_model_architecture(checkpoint_path):
    """Detect if checkpoint is MLP or Transformer based on state dict keys"""
    try:
        model_state = torch.load(checkpoint_path, map_location='cpu')
        if isinstance(model_state, dict):
            keys = list(model_state.keys())
            if 'token_embeddings.weight' in keys:
                return 'transformer'
            elif 'net.0.weight' in keys:
                return 'mlp'
        return 'unknown'
    except:
        return 'unknown'

def analyze_transformer_checkpoints():
    """Analyze only transformer checkpoints"""
    
    # Load params
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Find all composite checkpoints
    checkpoint_pattern = "checkpoints/composite_*_width*.pt"
    all_checkpoints = glob.glob(checkpoint_pattern)
    
    # Filter for transformer checkpoints only
    transformer_checkpoints = []
    for checkpoint in all_checkpoints:
        if detect_model_architecture(checkpoint) == 'transformer':
            transformer_checkpoints.append(checkpoint)
    
    print(f"Found {len(transformer_checkpoints)} transformer checkpoints out of {len(all_checkpoints)} total")
    
    # Group by function type
    results = {}
    for checkpoint in transformer_checkpoints:
        # Parse checkpoint name
        parts = checkpoint.split('_')
        if len(parts) >= 4:
            func_type = f"{parts[1]}_{parts[2]}"  # e.g., "relu_sin"
            width = int(parts[-1].split('.')[0])  # e.g., 16
            
            if func_type not in results:
                results[func_type] = {}
            
            print(f"Analyzing transformer {func_type} width {width}...")
            
            # Analyze this checkpoint
            analysis_result = analyze_single_transformer_checkpoint(checkpoint, func_type, width, params)
            if analysis_result:
                results[func_type][width] = analysis_result
    
    # Generate meaningful plots
    if results:
        create_meaningful_plots(results)
    else:
        print("No valid transformer results to plot!")

def analyze_single_transformer_checkpoint(checkpoint_path, func_type, width, params):
    """Analyze a single transformer checkpoint for actual results"""
    try:
        # Load model state
        model_state = torch.load(checkpoint_path, map_location='cpu')
        
        # Create transformer model
        params['width'] = width
        params['architecture'] = 'transformer'
        
        # Get transformer config
        transformer_config = params.get('transformer_config', {})
        dim = transformer_config.get('dim', 128)
        num_layers = transformer_config.get('num_layers', 2)
        num_heads = transformer_config.get('num_heads', 4)
        seq_len = transformer_config.get('seq_len', 5)
        num_tokens = transformer_config.get('num_tokens', 100)
        
        # Create transformer model
        model = create_model_from_params(
            input_dim=seq_len,
            output_dim=num_tokens,
            params=params
        )
        
        # Load state dict
        model.load_state_dict(model_state)
        model.eval()
        
        # Generate test data
        generator = SyntheticFunctionGenerator()
        inner_func, outer_func = func_type.split('_')
        test_x, test_g_x, test_f_gx = generator.generate_composite(
            inner_func, outer_func, seed=42
        )
        
        # Convert to transformer format - this returns a tuple of 6 values
        train_x, train_g_x, train_f_gx, test_x, test_g_x, test_f_gx = convert_to_sequence_format(
            test_x, test_f_gx, test_g_x, params
        )
        
        # Get model predictions on test data
        with torch.no_grad():
            outputs = model(test_x)
            predictions = outputs.argmax(dim=-1)
        
        # Calculate accuracy
        last_token_predictions = predictions[:, -1]
        
        # Calculate accuracies
        g_accuracy = (last_token_predictions == test_g_x[:, -1]).float().mean().item()
        fg_accuracy = (last_token_predictions == test_f_gx[:, -1]).float().mean().item()
        
        # Get model size info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"    g(x) accuracy: {g_accuracy:.3f}, f(g(x)) accuracy: {fg_accuracy:.3f}")
        
        return {
            'g_accuracy': g_accuracy,
            'fg_accuracy': fg_accuracy,
            'total_params': total_params,
            'trainable_params': trainable_params,
            'width': width,
            'func_type': func_type,
            'architecture': 'transformer',
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
        plt.title(f'{func_type.upper()} - Transformer Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1.1)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"plots/transformer_checkpoint_analysis_{timestamp}.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved transformer analysis results: {plot_filename}")
    
    # Print detailed results
    print("\nDetailed Transformer Checkpoint Analysis:")
    print("=" * 60)
    for func_type, widths in results.items():
        print(f"\n{func_type.upper()}:")
        for width in sorted(widths.keys()):
            result = widths[width]
            print(f"  Width {width:3d}: g(x)={result['g_accuracy']:.3f}, f(g(x))={result['fg_accuracy']:.3f}, params={result['total_params']:,}")

if __name__ == "__main__":
    analyze_transformer_checkpoints() 