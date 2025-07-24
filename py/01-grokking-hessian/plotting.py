# Plotting for grokking Hessian experiment
import torch
import matplotlib.pyplot as plt
import numpy as np

depths = [1, 2, 3, 4, 5, 6, 7]

# Plot Hessian eigenvalue
plt.figure()
for d in depths:
    results = torch.load(f"results_depth{d}.pt", weights_only=False)
    plt.plot(results['hess'], label=f"depth={d}")
    if results.get('grokking_step') is not None:
        step = results['grokking_step']
        idx = step // 1000
        plt.axvline(idx, color=plt.gca().lines[-1].get_color(), linestyle=':', alpha=0.7)
plt.xlabel("Checkpoints (x1000 steps)")
plt.ylabel("Top Hessian Eigenvalue")
plt.title("Hessian curvature vs. training")
plt.legend()
plt.savefig("plots/hessian_vs_depth.png")

# Plot train/test loss (log scale)
plt.figure()
for d in depths:
    results = torch.load(f"results_depth{d}.pt", weights_only=False)
    plt.plot(results['train'], label=f"Train depth={d}", linestyle='--')
    plt.plot(results['test'], label=f"Test depth={d}", linestyle='-')
    if results.get('grokking_step') is not None:
        step = results['grokking_step']
        idx = step // 100
        plt.axvline(idx, color=plt.gca().lines[-1].get_color(), linestyle=':', alpha=0.7)
plt.yscale("log")
plt.xlabel("Checkpoints (x100 steps)")
plt.ylabel("Loss")
plt.title("Train/Test Loss vs. Training Steps")
plt.legend()
plt.savefig("plots/loss_vs_steps.png")

# Plot test accuracy
plt.figure()
for d in depths:
    results = torch.load(f"results_depth{d}.pt", weights_only=False)
    plt.plot(results['test_acc'], label=f"Test Acc depth={d}")
    if results.get('grokking_step') is not None:
        step = results['grokking_step']
        idx = step // 100
        plt.axvline(idx, color=plt.gca().lines[-1].get_color(), linestyle=':', alpha=0.7)
plt.xlabel("Checkpoints (x100 steps)")
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy vs. Training Steps")
plt.legend()
plt.savefig("plots/test_acc_vs_steps.png")

# Plot gradient norm
plt.figure()
for d in depths:
    results = torch.load(f"results_depth{d}.pt", weights_only=False)
    if 'grad_norm' in results:
        plt.plot(results['grad_norm'], label=f"depth={d}")
        if results.get('grokking_step') is not None:
            step = results['grokking_step']
            idx = step // 1000
            plt.axvline(idx, color=plt.gca().lines[-1].get_color(), linestyle=':', alpha=0.7)
plt.xlabel("Checkpoints (x1000 steps)")
plt.ylabel("Gradient Norm")
plt.title("Gradient Norm vs. Training")
plt.legend()
plt.savefig("plots/grad_norm_vs_steps.png")