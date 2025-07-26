# run_probe.py
import torch
from model import create_model_from_params
from dataset import create_dataset_from_params
from probe_utils import register_hooks, save_dict_as_pt
from advanced_probe_utils import MultiProbeAnalyzer, run_single_probe, analyze_probe_complexity
from load_params import load_params
from datetime import datetime
params = load_params()

import os
import matplotlib.pyplot as plt
import numpy as np

p = params["p"]
train_frac = params["train_frac"]
seed = params["seed"]
widths = params["widths"]
depth = params["depth"]
lr = params["lr"]
steps = params["steps"]
use_synthetic = params.get("use_synthetic", False)

# Probe configuration
probe_mode = params.get("probe_mode", "all")
probe_type = params.get("probe_type", "linear")
custom_probes = params.get("custom_probes", ["linear", "tree", "svm"])

# Generate timestamp and function prefix
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

if use_synthetic:
    func_type = params.get("func_type", "polynomial")
    complexity = params.get("complexity", 3)
    func_prefix = f"{func_type}_deg{complexity}"
else:
    func_prefix = "mod_add"

concept_key = params["concept"]

# Define concept functions that work with both 1D and 2D data
def create_concept_fn(concept_key, input_dim):
    """Create concept function based on input dimension"""
    if input_dim == 1:
        # For 1D synthetic functions
        if concept_key == "x_mod_2":
            return lambda x: (x[:, 0] % 2 == 0).long()
        elif concept_key == "x_positive":
            return lambda x: (x[:, 0] > 0).long()
        elif concept_key == "x_squared":
            return lambda x: (x[:, 0] ** 2 > 2).long()
        elif concept_key == "x_range":
            return lambda x: ((x[:, 0] > -2) & (x[:, 0] < 2)).long()
        else:
            raise ValueError(f"Unknown concept for 1D data: {concept_key}")
    else:
        # For 2D data (modular addition, symmetric functions)
        if concept_key == "x_mod_2":
            return lambda x: (x[:, 0] % 2 == 0).long()
        elif concept_key == "x_equals_y":
            return lambda x: (x[:, 0] == x[:, 1]).long()
        elif concept_key == "x_positive":
            return lambda x: (x[:, 0] > 0).long()
        elif concept_key == "symmetry_test":
            return lambda x: ((x[:, 0] + x[:, 1]) % 2 == 0).long()  # Test symmetry property
        else:
            raise ValueError(f"Unknown concept for 2D data: {concept_key}")

device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs("results", exist_ok=True)
os.makedirs("plots", exist_ok=True)

# Create dataset
if use_synthetic:
    _, _, test_x, test_y = create_dataset_from_params(params)
else:
    _, _, test_x, _ = create_dataset_from_params(params)

# Create concept function based on input dimension
input_dim = test_x.shape[1]
concept_fn = create_concept_fn(concept_key, input_dim)
concept_labels = concept_fn(test_x)

# Check if we have at least 2 classes
unique_labels = torch.unique(concept_labels)
print(f"Concept labels: {unique_labels.tolist()}, shape: {concept_labels.shape}")
if len(unique_labels) < 2:
    print(f"Warning: Only {len(unique_labels)} class(es) found. Concept may not be suitable for this data.")
    # Try a different concept for 1D data
    if input_dim == 1:
        print("Trying alternative concept for 1D data...")
        concept_fn = lambda x: (x[:, 0] > 0).long()  # x > 0
        concept_labels = concept_fn(test_x)
        unique_labels = torch.unique(concept_labels)
        print(f"New concept labels: {unique_labels.tolist()}")
        if len(unique_labels) < 2:
            print("Still only one class. Skipping probe analysis.")
            exit()

# Determine if this is a regression task
is_regression = use_synthetic and test_y.shape[1] == 1

probe_results_all = {}

for w in widths:
    # Create model with appropriate dimensions
    if use_synthetic:
        input_dim = test_x.shape[1]
        output_dim = test_y.shape[1] if len(test_y.shape) > 1 else 1
        model = create_model_from_params(params, input_dim=input_dim, output_dim=output_dim).to(device)
    else:
        model = create_model_from_params(params, input_dim=2, output_dim=p).to(device)
    
    # Load the most recent checkpoint for this width and function type
    checkpoint_dir = "checkpoints"
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) 
                       if f.startswith(f"{func_prefix}_width{w}_") and f.endswith(".pt")]
    
    if not checkpoint_files:
        print(f"No checkpoint found for {func_prefix}_width{w}")
        continue
        
    # Get the most recent checkpoint
    latest_checkpoint = sorted(checkpoint_files)[-1]
    ckpt_path = os.path.join(checkpoint_dir, latest_checkpoint)
    
    print(f"Loading checkpoint: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    activation_store = {}
    hooks = register_hooks(model, layer_indices=list(range(depth + 1)), activation_store=activation_store)

    with torch.no_grad():
        _ = model(test_x.float().to(device))

    # Run probes based on configuration
    if probe_mode == "all":
        print(f"Running all probe types for width {w}...")
        analyzer = MultiProbeAnalyzer()
        results = analyzer.run_probes(activation_store, concept_labels, is_regression)
        probe_results_all[w] = results
        
    elif probe_mode == "single":
        print(f"Running {probe_type} probe for width {w}...")
        accs = run_single_probe(activation_store, concept_labels, probe_type, is_regression)
        probe_results_all[w] = accs
        
    elif probe_mode == "custom":
        print(f"Running custom probes {custom_probes} for width {w}...")
        analyzer = MultiProbeAnalyzer(custom_probes)
        results = analyzer.run_probes(activation_store, concept_labels, is_regression)
        probe_results_all[w] = results

    for h in hooks:
        h.remove()

# Save results with function prefix and timestamp
results_filename = f"results/{func_prefix}_{concept_key}_probe_results_{timestamp}.pt"
save_dict_as_pt(probe_results_all, results_filename)

# Generate plots based on probe mode
if probe_mode == "single":
    # Single probe plot (original format)
    for layer_idx in range(depth + 1):
        xs = []
        ys = []
        for w in widths:
            if w in probe_results_all:
                xs.append(w)
                ys.append(probe_results_all[w][layer_idx])
        plt.plot(xs, ys, marker='o', label=f"Layer {layer_idx}")

    plt.xlabel("Model width")
    plt.ylabel(f"Probe accuracy (concept: {concept_key})")
    plt.title(f"Concept Decodability vs. Width - {func_prefix} ({probe_type})")
    plt.legend()
    plt.grid(True)

    # Adjust y-axis to show variation better
    all_accuracies = []
    for w in widths:
        if w in probe_results_all:
            all_accuracies.extend(probe_results_all[w])

    if all_accuracies:
        min_acc = min(all_accuracies)
        max_acc = max(all_accuracies)
        if max_acc - min_acc < 0.1:
            plt.ylim(min_acc - 0.05, max_acc + 0.05)
        else:
            plt.ylim(0, 1.1)

    plot_filename = f"plots/{func_prefix}_{concept_key}_{probe_type}_probe_accuracy_vs_width_{timestamp}.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()

else:
    # Multi-probe plots
    probe_types = list(probe_results_all[list(probe_results_all.keys())[0]][0].keys())
    
    for probe_type in probe_types:
        plt.figure(figsize=(10, 6))
        for layer_idx in range(depth + 1):
            xs = []
            ys = []
            for w in widths:
                if w in probe_results_all:
                    xs.append(w)
                    ys.append(probe_results_all[w][layer_idx][probe_type])
            plt.plot(xs, ys, marker='o', label=f"Layer {layer_idx}")

        plt.xlabel("Model width")
        plt.ylabel(f"Probe accuracy ({probe_type})")
        plt.title(f"Concept Decodability vs. Width - {func_prefix} ({probe_type})")
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1.1)
        
        plot_filename = f"plots/{func_prefix}_{concept_key}_{probe_type}_probe_accuracy_vs_width_{timestamp}.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        plt.close()

print(f"Probe analysis completed and saved to {results_filename}")
print(f"Plots saved to plots/ directory")