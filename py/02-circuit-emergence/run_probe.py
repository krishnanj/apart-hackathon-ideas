# run_probe.py
import torch
from model import MLP
from dataset import create_mod_add_dataset
from probe_utils import register_hooks, run_probe, save_dict_as_pt
from load_params import load_params
params = load_params()

import os
import matplotlib.pyplot as plt

p = params["p"]
train_frac = params["train_frac"]
seed = params["seed"]
widths = params["widths"]
depth = params["depth"]
lr = params["lr"]
steps = params["steps"]

concept_key = params["concept"]

if concept_key == "x_mod_2":
    concept_fn = lambda x: (x[:, 0] % 2 == 0).long()
elif concept_key == "x_equals_y":
    concept_fn = lambda x: (x[:, 0] == x[:, 1]).long()
else:
    raise ValueError(f"Unknown concept: {concept_key}")

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)

_, _, test_x, _ = create_mod_add_dataset(p=p, train_frac=train_frac, seed=seed)
concept_labels = concept_fn(test_x)

probe_accuracies_all = {}

for w in widths:
    model = MLP(input_dim=2, hidden_dim=w, output_dim=p, n_hidden_layers=depth).to(device)
    ckpt_path = f"checkpoints/model_width{w}.pt"
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    activation_store = {}
    hooks = register_hooks(model, layer_indices=list(range(depth + 1)), activation_store=activation_store)

    with torch.no_grad():
        _ = model(test_x.float().to(device))

    accs = run_probe(activation_store, concept_labels)
    probe_accuracies_all[w] = accs

    for h in hooks:
        h.remove()

save_dict_as_pt(probe_accuracies_all, "results/probe_accuracies.pt")

# --- PLOT ---
for layer_idx in range(depth + 1):
    xs = []
    ys = []
    for w in widths:
        xs.append(w)
        ys.append(probe_accuracies_all[w][layer_idx])
    plt.plot(xs, ys, marker='o', label=f"Layer {layer_idx}")

plt.xlabel("Model width")
plt.ylabel("Probe accuracy (concept: x % 2 == 0)")
plt.title("Concept Decodability vs. Width")
plt.legend()
plt.grid(True)
plt.savefig("plots/probe_accuracy_vs_width.png")
plt.close()