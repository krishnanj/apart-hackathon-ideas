import torch
import glob

def debug_new_checkpoints():
    """Debug the new checkpoint files"""
    
    # Find the new checkpoints
    checkpoint_pattern = "checkpoints/composite_*_width*.pt"
    checkpoints = glob.glob(checkpoint_pattern)
    
    print(f"Found {len(checkpoints)} checkpoints")
    
    # Check first few checkpoints
    for i, checkpoint in enumerate(checkpoints[:3]):
        print(f"\n--- Checkpoint {i+1}: {checkpoint} ---")
        
        try:
            # Load checkpoint
            checkpoint_data = torch.load(checkpoint, map_location='cpu')
            
            print(f"Type: {type(checkpoint_data)}")
            
            if isinstance(checkpoint_data, dict):
                print(f"Keys: {list(checkpoint_data.keys())}")
                for key, value in checkpoint_data.items():
                    if isinstance(value, torch.Tensor):
                        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
                    else:
                        print(f"  {key}: {type(value)}")
            elif isinstance(checkpoint_data, torch.Tensor):
                print(f"Shape: {checkpoint_data.shape}")
                print(f"Dtype: {checkpoint_data.dtype}")
            else:
                print(f"Unknown type: {type(checkpoint_data)}")
                
        except Exception as e:
            print(f"Error loading {checkpoint}: {e}")

if __name__ == "__main__":
    debug_new_checkpoints() 