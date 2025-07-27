import torch
import glob
import yaml
from model import create_model_from_params
from synthetic_functions import SyntheticFunctionGenerator
from dataset import convert_to_sequence_format

def debug_transformer_analysis():
    """Debug the transformer analysis step by step"""
    
    # Load params
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    
    # Find transformer checkpoints
    checkpoint_pattern = "checkpoints/composite_*_width*.pt"
    all_checkpoints = glob.glob(checkpoint_pattern)
    
    print(f"Found {len(all_checkpoints)} total checkpoints")
    
    # Test first transformer checkpoint
    for checkpoint in all_checkpoints:
        try:
            model_state = torch.load(checkpoint, map_location='cpu')
            keys = list(model_state.keys())
            
            if 'token_embeddings.weight' in keys:
                print(f"\n--- Testing transformer checkpoint: {checkpoint} ---")
                
                # Parse checkpoint name
                parts = checkpoint.split('_')
                func_type = f"{parts[1]}_{parts[2]}"
                width = int(parts[-1].split('.')[0])
                
                print(f"Function type: {func_type}")
                print(f"Width: {width}")
                
                # Create model
                params['width'] = width
                params['architecture'] = 'transformer'
                
                transformer_config = params.get('transformer_config', {})
                dim = transformer_config.get('dim', 128)
                num_layers = transformer_config.get('num_layers', 2)
                num_heads = transformer_config.get('num_heads', 4)
                seq_len = transformer_config.get('seq_len', 5)
                num_tokens = transformer_config.get('num_tokens', 100)
                
                print(f"Model config: dim={dim}, layers={num_layers}, heads={num_heads}, seq_len={seq_len}, tokens={num_tokens}")
                
                # Create model
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
                
                print(f"Test data shapes: x={test_x.shape}, g_x={test_g_x.shape}, f_gx={test_f_gx.shape}")
                
                # Convert to transformer format
                test_x_seq = convert_to_sequence_format(test_x, test_g_x, test_f_gx, params)
                print(f"Transformer input shape: {test_x_seq.shape}")
                
                # Get predictions
                with torch.no_grad():
                    outputs = model(test_x_seq)
                    predictions = outputs.argmax(dim=-1)
                
                print(f"Model output shape: {outputs.shape}")
                print(f"Predictions shape: {predictions.shape}")
                
                # Calculate accuracy
                last_token_predictions = predictions[:, -1]
                
                # Convert targets
                test_g_x_seq = convert_to_sequence_format(test_g_x, None, None, params)
                test_f_gx_seq = convert_to_sequence_format(test_f_gx, None, None, params)
                
                g_accuracy = (last_token_predictions == test_g_x_seq[:, -1]).float().mean().item()
                fg_accuracy = (last_token_predictions == test_f_gx_seq[:, -1]).float().mean().item()
                
                print(f"Accuracies: g(x)={g_accuracy:.3f}, f(g(x))={fg_accuracy:.3f}")
                
                # Test a few predictions
                print(f"First 5 predictions: {last_token_predictions[:5]}")
                print(f"First 5 g(x) targets: {test_g_x_seq[:5, -1]}")
                print(f"First 5 f(g(x)) targets: {test_f_gx_seq[:5, -1]}")
                
                break  # Just test the first transformer checkpoint
                
        except Exception as e:
            print(f"Error with {checkpoint}: {e}")
            continue

if __name__ == "__main__":
    debug_transformer_analysis() 