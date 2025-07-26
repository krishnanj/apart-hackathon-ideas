import torch
import torch.nn.functional as F
from model import create_model_from_params
from dataset import create_dataset_from_params
import os
from load_params import load_params
from datetime import datetime
params = load_params()

# Load parameters
p = params["p"]
train_frac = params["train_frac"]
seed = params["seed"]
widths = params["widths"]
depth = params["depth"]
lr = params["lr"]
steps = params["steps"]
use_synthetic = params.get("use_synthetic", False)

# Generate timestamp and function prefix
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

if use_synthetic:
    func_type = params.get("func_type", "polynomial")
    complexity = params.get("complexity", 3)
    func_prefix = f"{func_type}_deg{complexity}"
else:
    func_prefix = "mod_add"

os.makedirs("checkpoints", exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Create dataset
if use_synthetic:
    train_x, train_y, test_x, test_y = create_dataset_from_params(params)
    # For synthetic functions, we use MSE loss (regression)
    loss_fn = F.mse_loss
    is_regression = True
else:
    train_x, train_y, _, _ = create_dataset_from_params(params)
    # For modular addition, we use cross entropy (classification)
    loss_fn = F.cross_entropy
    is_regression = False

train_x = train_x.float().to(device)
train_y = train_y.to(device)

for w in widths:
    # Create model with appropriate dimensions
    if use_synthetic:
        input_dim = train_x.shape[1]
        output_dim = train_y.shape[1] if len(train_y.shape) > 1 else 1
        model = create_model_from_params(params, input_dim=input_dim, output_dim=output_dim).to(device)
    else:
        model = create_model_from_params(params, input_dim=2, output_dim=p).to(device)
    
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(steps):
        logits = model(train_x)
        
        if is_regression:
            # For regression, squeeze output if needed
            if logits.shape[-1] == 1:
                logits = logits.squeeze(-1)
            loss = loss_fn(logits, train_y.squeeze(-1))
        else:
            # For classification
            loss = loss_fn(logits, train_y)
            
        opt.zero_grad()
        loss.backward()
        opt.step()
        
        if step % 1000 == 0:
            print(f"Width {w}, Step {step}, Loss: {loss.item():.4f}")

    # Save with function prefix and timestamp
    checkpoint_name = f"checkpoints/{func_prefix}_width{w}_{timestamp}.pt"
    torch.save(model.state_dict(), checkpoint_name)
    print(f"Saved model for width={w} as {checkpoint_name}")