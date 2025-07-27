#!/usr/bin/env python3
"""
Symmetry analysis with support for both MLP and Transformer
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.manifold import MDS
# import seaborn as sns  # Commented out to avoid dependency issues

from model import create_model_from_params
from dataset import create_dataset_from_params
from probe_utils import register_hooks, save_dict_as_pt
from advanced_probe_utils import MultiProbeAnalyzer
from load_params import load_params

def generate_symmetric_pairs(n_samples=500, input_range=(-5, 5), seed=42):
    """Generate symmetric input pairs for analysis"""
    np.random.seed(seed)
    
    # Generate random 2D inputs
    x1 = np.random.uniform(input_range[0], input_range[1], n_samples)
    x2 = np.random.uniform(input_range[0], input_range[1], n_samples)
    
    # Create symmetric pairs: (x1, x2) and (x2, x1)
    symmetric_pairs = []
    for i in range(n_samples):
        pair1 = torch.tensor([x1[i], x2[i]], dtype=torch.float32)
        pair2 = torch.tensor([x2[i], x1[i]], dtype=torch.float32)
        symmetric_pairs.append((pair1, pair2))
    
    return symmetric_pairs

def generate_random_pairs(n_samples=500, input_range=(-5, 5), seed=42):
    """Generate random input pairs for baseline comparison"""
    np.random.seed(seed + 100)  # Different seed
    
    # Generate random 2D inputs
    x1 = np.random.uniform(input_range[0], input_range[1], n_samples)
    x2 = np.random.uniform(input_range[0], input_range[1], n_samples)
    x3 = np.random.uniform(input_range[0], input_range[1], n_samples)
    x4 = np.random.uniform(input_range[0], input_range[1], n_samples)
    
    # Create random pairs
    random_pairs = []
    for i in range(n_samples):
        pair1 = torch.tensor([x1[i], x2[i]], dtype=torch.float32)
        pair2 = torch.tensor([x3[i], x4[i]], dtype=torch.float32)
        random_pairs.append((pair1, pair2))
    
    return random_pairs

def calculate_activation_distances(model, input_pairs, device="cpu"):
    """Calculate activation distances for input pairs across all layers"""
    model.eval()
    depth = len([m for m in model.modules() if isinstance(m, torch.nn.Linear)]) - 1
    
    # Store activations for each layer
    layer_activations = {i: [] for i in range(depth + 1)}
    
    # Register hooks to capture activations
    activation_store = {}
    hooks = register_hooks(model, list(range(depth + 1)), activation_store)
    
    with torch.no_grad():
        for pair1, pair2 in input_pairs:
            # Get activations for first input
            _ = model(pair1.unsqueeze(0).to(device))
            act1 = {layer: activation_store[layer][0].cpu() for layer in activation_store}
            
            # Get activations for second input
            _ = model(pair2.unsqueeze(0).to(device))
            act2 = {layer: activation_store[layer][0].cpu() for layer in activation_store}
            
            # Store activations
            for layer in range(depth + 1):
                layer_activations[layer].append((act1[layer], act2[layer]))
    
    # Calculate distances
    distances = {}
    for layer in range(depth + 1):
        layer_distances = []
        for act1, act2 in layer_activations[layer]:
            # Flatten activations and calculate Euclidean distance
            dist = torch.norm(act1.flatten() - act2.flatten()).item()
            layer_distances.append(dist)
        distances[layer] = layer_distances
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    return distances

def calculate_symmetry_scores(symmetric_distances, random_distances):
    """Calculate symmetry scores for each layer"""
    symmetry_scores = {}
    
    for layer in symmetric_distances.keys():
        avg_symmetric = np.mean(symmetric_distances[layer])
        avg_random = np.mean(random_distances[layer])
        
        # Symmetry score: 1 - (symmetric_distance / random_distance)
        # Higher score = more symmetric (closer to identical activations)
        symmetry_score = 1 - (avg_symmetric / avg_random)
        symmetry_scores[layer] = max(0, symmetry_score)  # Ensure non-negative
    
    return symmetry_scores

def run_symmetry_analysis():
    """Run comprehensive symmetry analysis"""
    params = load_params()
    
    # Enable symmetric functions
    params["use_synthetic"] = True
    params["func_type"] = "symmetric"
    
    # Load parameters
    widths = params["widths"]
    depth = params["depth"]
    seed = params["seed"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Generate timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    
    # Generate symmetric and random pairs
    print("Generating symmetric and random input pairs...")
    symmetric_pairs = generate_symmetric_pairs(n_samples=500, seed=seed)
    random_pairs = generate_random_pairs(n_samples=500, seed=seed)
    
    print(f"Generated {len(symmetric_pairs)} symmetric pairs and {len(random_pairs)} random pairs")
    
    # Train models if needed
    print("\nChecking for existing models...")
    func_prefix = "symmetric_deg3"
    
    for w in widths:
        checkpoint_files = [f for f in os.listdir("checkpoints") 
                          if f.startswith(f"{func_prefix}_width{w}_") and f.endswith(".pt")]
        
        if not checkpoint_files:
            print(f"Training model with width {w}...")
            
            # Create dataset and train
            train_x, train_y, test_x, test_y = create_dataset_from_params(params)
            train_x = train_x.float().to(device)
            train_y = train_y.float().to(device)
            
            # Create and train model
            model = create_model_from_params(params, input_dim=2, output_dim=1).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=params["lr"])
            
            for step in range(params["steps"]):
                logits = model(train_x)
                loss = torch.nn.functional.mse_loss(logits.squeeze(-1), train_y.squeeze(-1))
                
                opt.zero_grad()
                loss.backward()
                opt.step()
                
                if step % 1000 == 0:
                    print(f"  Step {step}, Loss: {loss.item():.4f}")
            
            # Save model
            checkpoint_name = f"checkpoints/{func_prefix}_width{w}_{timestamp}.pt"
            torch.save(model.state_dict(), checkpoint_name)
            print(f"  Saved model: {checkpoint_name}")
        else:
            print(f"Found existing model for width {w}")
    
    # Run symmetry analysis
    print("\nRunning symmetry analysis...")
    all_symmetry_results = {}
    
    for w in widths:
        print(f"Analyzing symmetry for width {w}...")
        
        # Load model
        model = create_model_from_params(params, input_dim=2, output_dim=1).to(device)
        checkpoint_files = [f for f in os.listdir("checkpoints") 
                          if f.startswith(f"{func_prefix}_width{w}_") and f.endswith(".pt")]
        latest_checkpoint = sorted(checkpoint_files)[-1]
        ckpt_path = os.path.join("checkpoints", latest_checkpoint)
        model.load_state_dict(torch.load(ckpt_path))
        
        # Calculate distances
        print(f"  Calculating symmetric pair distances...")
        symmetric_distances = calculate_activation_distances(model, symmetric_pairs, device)
        
        print(f"  Calculating random pair distances...")
        random_distances = calculate_activation_distances(model, random_pairs, device)
        
        # Calculate symmetry scores
        symmetry_scores = calculate_symmetry_scores(symmetric_distances, random_distances)
        
        all_symmetry_results[w] = {
            'symmetric_distances': symmetric_distances,
            'random_distances': random_distances,
            'symmetry_scores': symmetry_scores
        }
        
        print(f"  Symmetry scores: {symmetry_scores}")
    
    # Save results
    results_filename = f"results/symmetry_analysis_results_{timestamp}.pt"
    save_dict_as_pt(all_symmetry_results, results_filename)
    print(f"\nSaved results to: {results_filename}")
    
    # Create plots
    print("\nCreating symmetry analysis plots...")
    create_symmetry_plots(all_symmetry_results, timestamp)
    
    return all_symmetry_results

def create_symmetry_plots(all_symmetry_results, timestamp):
    """Create comprehensive symmetry analysis plots"""
    widths = list(all_symmetry_results.keys())
    depth = len(all_symmetry_results[widths[0]]['symmetry_scores'])
    
    # Plot 1: Symmetry scores across layers and widths
    plt.figure(figsize=(12, 8))
    
    for w in widths:
        symmetry_scores = [all_symmetry_results[w]['symmetry_scores'][layer] for layer in range(depth)]
        layers = range(depth)
        plt.plot(layers, symmetry_scores, marker='o', label=f'Width {w}', linewidth=2)
    
    plt.xlabel('Layer')
    plt.ylabel('Symmetry Score')
    plt.title('Symmetry Learning Across Layers and Model Widths')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    plot_filename = f"plots/symmetry_scores_{timestamp}.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_filename}")
    
    # Plot 2: Distance comparison (symmetric vs random)
    plt.figure(figsize=(15, 10))
    
    for i, w in enumerate(widths):
        plt.subplot(2, 3, i+1)
        
        symmetric_dists = [np.mean(all_symmetry_results[w]['symmetric_distances'][layer]) 
                          for layer in range(depth)]
        random_dists = [np.mean(all_symmetry_results[w]['random_distances'][layer]) 
                       for layer in range(depth)]
        layers = range(depth)
        
        plt.plot(layers, symmetric_dists, marker='o', label='Symmetric pairs', linewidth=2)
        plt.plot(layers, random_dists, marker='s', label='Random pairs', linewidth=2)
        
        plt.xlabel('Layer')
        plt.ylabel('Average Distance')
        plt.title(f'Width {w}')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_filename = f"plots/distance_comparison_{timestamp}.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_filename}")
    
    # Plot 3: Symmetry emergence analysis
    plt.figure(figsize=(10, 6))
    
    emergence_layers = []
    for w in widths:
        symmetry_scores = [all_symmetry_results[w]['symmetry_scores'][layer] for layer in range(depth)]
        # Find first layer where symmetry score > 0.5
        for layer, score in enumerate(symmetry_scores):
            if score > 0.5:
                emergence_layers.append(layer)
                break
        else:
            emergence_layers.append(depth)  # Never emerged
    
    plt.plot(widths, emergence_layers, marker='o', linewidth=2, markersize=8)
    plt.xlabel('Model Width')
    plt.ylabel('Layer of Symmetry Emergence (score > 0.5)')
    plt.title('Symmetry Emergence vs Model Width')
    plt.grid(True, alpha=0.3)
    plt.ylim(-0.5, depth + 0.5)
    
    plot_filename = f"plots/symmetry_emergence_{timestamp}.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_filename}")

if __name__ == "__main__":
    results = run_symmetry_analysis()
    print("\nSymmetry analysis completed!")