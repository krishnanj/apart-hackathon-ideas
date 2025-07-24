import torch
import torch.nn.functional as F
from model import MLP
from dataset import create_mod_add_dataset
import os
from load_params import load_params
params = load_params()

p = params["p"]
train_frac = params["train_frac"]
seed = params["seed"]
widths = params["widths"]
depth = params["depth"]
lr = params["lr"]
steps = params["steps"]


os.makedirs("checkpoints", exist_ok=True)
widths = [16, 32, 64, 128, 256]
device = "cuda" if torch.cuda.is_available() else "cpu"

train_x, train_y, _, _ = create_mod_add_dataset(p=97, train_frac=1.0)
train_x = train_x.float().to(device)
train_y = train_y.to(device)

for w in widths:
    model = MLP(hidden_dim=w).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(5000):
        logits = model(train_x)
        loss = F.cross_entropy(logits, train_y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    torch.save(model.state_dict(), f"checkpoints/model_width{w}.pt")
    print(f"Saved model for width={w}")