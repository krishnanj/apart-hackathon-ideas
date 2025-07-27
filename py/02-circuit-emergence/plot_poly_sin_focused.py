import torch
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import yaml
from model import create_model_from_params
from synthetic_functions import SyntheticFunctionGenerator
from dataset import convert_to_sequence_format
import numpy as np

def plot_poly_sin_focused():
    """Create a focused plot for POLY_SIN results only"""
    
    # Load params
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Find POLY_SIN transformer checkpoints
    checkpoint_pattern = "checkpoints/composite_poly_sin_width*.pt"
    all_checkpoints = glob.glob(checkpoint_pattern)
    
    # Filter for transformer checkpoints only
    transformer_checkpoints = []
    for checkpoint in all_checkpoints:
        try:
            model_state = torch.load(checkpoint, map_location='cpu')
            if isinstance(model_state, dict) and 'token_embeddings.weight' in model_state.keys():
                transformer_checkpoints.append(checkpoint)
        except:
            continue
    
    print(f"Found {len(transformer_checkpoints)} POLY_SIN transformer checkpoints")
    
    # Analyze each checkpoint
    results = {}
    for checkpoint in transformer_checkpoints:
        try:
            # Parse checkpoint name to get timestamp
            parts = checkpoint.split('_')
            timestamp = int(parts[-1].split('.')[0])  # This is actually a timestamp
            
            # Analyze this checkpoint
            analysis_result = analyze_single_checkpoint(checkpoint, timestamp, params)
            if analysis_result:
                results[timestamp] = analysis_result
                
        except Exception as e:
            print(f"Error analyzing {checkpoint}: {e}")
    
    # Create focused plot
    if results:
        create_focused_plot(results)
    else:
        print("No valid results to plot!")

def analyze_single_checkpoint(checkpoint_path, timestamp, params):
    """Analyze a single transformer checkpoint"""
    try:
        # Load model state
        model_state = torch.load(checkpoint_path, map_location='cpu')
        
        # Create transformer model
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
        test_x, test_g_x, test_f_gx = generator.generate_composite('poly', 'sin', seed=42)
        
        # Convert to transformer format
        train_x, train_g_x, train_f_gx, test_x, test_g_x, test_f_gx = convert_to_sequence_format(
            test_x, test_f_gx, test_g_x, params
        )
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(test_x)
            predictions = outputs.argmax(dim=-1)
        
        # Calculate accuracy
        last_token_predictions = predictions[:, -1]
        
        # Calculate accuracies
        g_accuracy = (last_token_predictions == test_g_x[:, -1]).float().mean().item()
        fg_accuracy = (last_token_predictions == test_f_gx[:, -1]).float().mean().item()
        
        return {
            'g_accuracy': g_accuracy,
            'fg_accuracy': fg_accuracy,
            'timestamp': timestamp
        }
        
    except Exception as e:
        print(f"  Error analyzing {checkpoint_path}: {e}")
        return None

def create_focused_plot(results):
    """Create a focused plot for POLY_SIN results"""
    
    plt.figure(figsize=(10, 6))
    
    # Sort by timestamp
    timestamps = sorted(results.keys())
    g_accuracies = [results[t]['g_accuracy'] for t in timestamps]
    fg_accuracies = [results[t]['fg_accuracy'] for t in timestamps]
    
    # Plot with better styling
    plt.plot(timestamps, g_accuracies, 'o-', label='g(x) accuracy', 
             linewidth=2, markersize=8, color='blue', alpha=0.7)
    plt.plot(timestamps, fg_accuracies, 's-', label='f(g(x)) accuracy', 
             linewidth=2, markersize=8, color='orange', alpha=0.7)
    
    plt.xlabel('Training Timestamp', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('POLY_SIN: Transformer Learning of Composite Functions', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1.1)
    
    # Add text annotations
    plt.text(0.02, 0.98, f'f(g(x)) accuracy: {np.mean(fg_accuracies):.3f}', 
             transform=plt.gca().transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))
    plt.text(0.02, 0.92, f'g(x) accuracy: {np.mean(g_accuracies):.3f}', 
             transform=plt.gca().transAxes, fontsize=10, 
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3))
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f"plots/poly_sin_focused_analysis_{timestamp}.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved focused POLY_SIN analysis: {plot_filename}")
    
    # Print summary statistics
    print(f"\nPOLY_SIN Analysis Summary:")
    print(f"Number of checkpoints: {len(results)}")
    print(f"Average f(g(x)) accuracy: {np.mean(fg_accuracies):.3f} ± {np.std(fg_accuracies):.3f}")
    print(f"Average g(x) accuracy: {np.mean(g_accuracies):.3f} ± {np.std(g_accuracies):.3f}")
    print(f"Accuracy gap: {np.mean(fg_accuracies) - np.mean(g_accuracies):.3f}")

if __name__ == "__main__":
    plot_poly_sin_focused() 