import os
import matplotlib.pyplot as plt
import numpy as np

# Directory where probe results are saved (to be produced by run_probe.py)
RESULTS_DIR = "probe_results"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# List of model widths to plot (should match those in params.yaml)
widths = [16, 32, 64, 128, 256]

for width in widths:
    # Placeholder: load probe accuracy per layer for this width
    # Example: probe_acc = np.load(f"{RESULTS_DIR}/probe_acc_width{width}.npy")
    # For now, use random data as a placeholder
    probe_acc = np.random.uniform(0.5, 1.0, size=4)  # 4 layers (input + 3 hidden)
    layers = np.arange(len(probe_acc))
    plt.figure()
    plt.plot(layers, probe_acc, marker='o')
    plt.title(f"Probe Accuracy vs. Layer (width={width})")
    plt.xlabel("Layer")
    plt.ylabel("Probe Accuracy")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.savefig(f"{PLOTS_DIR}/probe_accuracy_vs_layer_width{width}.png")
    plt.close()

print("Saved probe accuracy plots for all widths.") 