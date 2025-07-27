#!/usr/bin/env python3
"""
Grokking Experiment: Track internal circuit development during grokking
Consolidated version with multiple setups and fixes
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from sklearn.linear_model import LogisticRegression
import yaml

from model import create_model_from_params
from dataset import create_mod_add_dataset
from probe_utils import register_hooks, save_dict_as_pt
from advanced_probe_utils import MultiProbeAnalyzer
from load_params import load_params

def run_grokking_experiment(setup="correct"):
    """Run grokking experiment with different setups"""
    params = load_params()
    
    print(f"Starting grokking experiment with {setup} setup...")
    
    if setup == "correct":
        # CORRECT setup based on original paper - MODULAR ADDITION
        p = 113  # Prime number
        subset_ratio = 0.1  # Only 10% of data for training
        weight_decay = 0.1  # Strong regularization
        n_steps = 50000  # Very long training
        hidden_dim = 128
        depth = 2
        lr = 0.001
        print("Using CORRECT setup: p=113, subset_ratio=0.1, weight_decay=0.1, MODULAR ADDITION")
        
    elif setup == "original":
        # Original paper hyperparameters - MODULAR ADDITION
        p = 113
        subset_ratio = 0.1
        weight_decay = 0.1
        lr = 0.001
        n_steps = 50000
        hidden_dim = 128
        depth = 2
        print("Using ORIGINAL setup: p=113, subset_ratio=0.1, weight_decay=0.1, MODULAR ADDITION")
        
    else:  # "basic"
        # Basic setup (what we had before) - MODULAR ADDITION
        p = 97
        subset_ratio = 0.3
        weight_decay = 0.01
        n_steps = 10000
        hidden_dim = 128
        depth = 3
        lr = 0.001
        print("Using BASIC setup: p=97, subset_ratio=0.3, weight_decay=0.01, MODULAR ADDITION")
    
    checkpoint_interval = 1000  # Save checkpoints every 1000 steps
    
    # Create dataset with subset training - MODULAR ADDITION
    train_x, train_y, test_x, test_y = create_grokking_dataset_modular(p, subset_ratio)
    
    print(f"Training set size: {len(train_x)} ({len(train_x)/(p*p)*100:.1f}%)")
    print(f"Test set size: {len(test_x)} ({len(test_x)/(p*p)*100:.1f}%)")
    print(f"Model: {depth} layers, {hidden_dim} hidden units")
    print(f"Training for {n_steps} steps with weight_decay={weight_decay}")
    print(f"Task: (x + y) mod {p} (multi-class classification with {p} classes)")
    print(f"Random chance accuracy: {1.0/p:.4f} ({1.0/p*100:.2f}%)")
    
    # Create model for MODULAR ADDITION (multi-class)
    model = create_modular_model(hidden_dim, depth, p)
    
    # Training with regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()  # Multi-class classification loss
    
    # Tracking variables
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    grokking_checkpoints = []
    
    print(f"\nTraining with {setup} setup...")
    for step in range(n_steps):
        # Training step
        optimizer.zero_grad()
        outputs = model(train_x.float())
        loss = criterion(outputs, train_y.long())
        loss.backward()
        optimizer.step()
        
        # Track metrics
        if step % 1000 == 0:
            train_loss = loss.item()
            test_loss = criterion(model(test_x.float()), test_y.long()).item()
            
            train_acc = compute_modular_accuracy(model(train_x.float()), train_y.long())
            test_acc = compute_modular_accuracy(model(test_x.float()), test_y.long())
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            
            print(f"Step {step}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
            print(f"         Train Acc: {train_acc:.4f} ({train_acc*100:.2f}%), Test Acc: {test_acc:.4f} ({test_acc*100:.2f}%)")
            
            # Check for grokking
            if test_acc > 0.8:  # Much better than random (0.9%)
                print(f"🎉 GROKKING DETECTED at step {step}!")
            
            # Save checkpoint for probing
            if step % checkpoint_interval == 0:
                checkpoint_data = {
                    'step': step,
                    'model_state': model.state_dict(),
                    'train_acc': train_acc,
                    'test_acc': test_acc,
                    'train_loss': train_loss,
                    'test_loss': test_loss
                }
                grokking_checkpoints.append(checkpoint_data)
    
    # Analyze internal circuit development
    print("\nAnalyzing internal circuit development...")
    circuit_analysis = analyze_circuit_development(grokking_checkpoints, test_x, test_y, p)
    
    # Create grokking plots
    create_grokking_plots(train_losses, test_losses, train_accuracies, test_accuracies, 
                          circuit_analysis, checkpoint_interval, setup)
    
    print(f"{setup.capitalize()} grokking experiment completed!")
    
    return {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'circuit_analysis': circuit_analysis
    }

def create_grokking_dataset_modular(p, subset_ratio):
    """Create dataset with subset training for grokking - MODULAR ADDITION"""
    # Generate full dataset
    all_data = torch.tensor([(i, j) for i in range(p) for j in range(p)], dtype=torch.float32)
    # MODULAR ADDITION TASK: (x + y) mod p
    all_labels = (all_data[:, 0] + all_data[:, 1]) % p
    
    # Create subset for training
    n_train = int(subset_ratio * len(all_data))
    indices = torch.randperm(len(all_data))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    train_x = all_data[train_idx]
    train_y = all_labels[train_idx]
    test_x = all_data[test_idx]
    test_y = all_labels[test_idx]
    
    return train_x, train_y, test_x, test_y

def create_modular_model(hidden_dim, depth, p):
    """Create model for modular addition (multi-class classification)"""
    import torch.nn as nn
    
    class ModularMLP(nn.Module):
        def __init__(self, input_dim=2, hidden_dim=128, depth=2, output_dim=113):
            super().__init__()
            layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
            for _ in range(depth - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
            layers += [nn.Linear(hidden_dim, output_dim)]  # p outputs for multi-class
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)
    
    return ModularMLP(2, hidden_dim, depth, p)

def compute_modular_accuracy(outputs, targets):
    """Compute accuracy for modular addition (multi-class classification)"""
    predictions = torch.argmax(outputs, dim=1)
    correct = (predictions == targets).sum().item()
    return correct / len(targets)

def analyze_circuit_development(checkpoints, test_x, test_y, p):
    """Analyze how internal circuits develop during grokking"""
    circuit_results = {}
    
    # Define concepts to probe for (adapted for different p values)
    concepts = {
        'parity': lambda x: (x[:, 0] + x[:, 1]) % 2,  # Even/odd parity
        'sum_range': lambda x: ((x[:, 0] + x[:, 1]) > p//2).long(),  # Sum in upper half
        'modulo_structure': lambda x: ((x[:, 0] + x[:, 1]) % 10 == 0).long()  # Modulo 10
    }
    
    for i, checkpoint in enumerate(checkpoints):
        step = checkpoint['step']
        print(f"  Analyzing checkpoint {i+1}/{len(checkpoints)} (step {step})")
        
        # Load model state
        model = create_modular_model(128, 2, p)
        model.load_state_dict(checkpoint['model_state'])
        
        # Get activations
        activations = {}
        hooks = register_hooks(model, list(range(2)), activations)  # 2 layers
        
        with torch.no_grad():
            model(test_x.float())
        
        for hook in hooks:
            hook.remove()
        
        # Probe for each concept
        step_results = {}
        for concept_name, concept_fn in concepts.items():
            concept_labels = concept_fn(test_x).numpy()
            
            # Run probes
            probe_analyzer = MultiProbeAnalyzer(["linear"])
            probe_results = probe_analyzer.run_probes(activations, concept_labels, is_regression=False)
            
            # Extract linear probe accuracies
            linear_accuracies = {}
            for layer in probe_results:
                linear_accuracies[layer] = probe_results[layer]['linear']
            
            step_results[concept_name] = linear_accuracies
        
        circuit_results[step] = step_results
    
    return circuit_results

def create_grokking_plots(train_losses, test_losses, train_accuracies, test_accuracies, 
                         circuit_analysis, checkpoint_interval, setup):
    """Create comprehensive grokking analysis plots"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot 1: Loss curves
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    steps = list(range(0, len(train_losses) * 1000, 1000))
    plt.plot(steps, train_losses, label='Train Loss', linewidth=2)
    plt.plot(steps, test_losses, label='Test Loss', linewidth=2)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title(f'Loss During Training ({setup.capitalize()} Setup)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 2: Accuracy curves
    plt.subplot(2, 3, 2)
    plt.plot(steps, train_accuracies, label='Train Accuracy', linewidth=2)
    plt.plot(steps, test_accuracies, label='Test Accuracy', linewidth=2)
    # Add random chance line based on p value
    if setup in ["correct", "original"]:
        random_chance = 1.0/113
    else:
        random_chance = 1.0/97
    plt.axhline(y=random_chance, color='red', linestyle='--', 
                label=f'Random Chance ({random_chance*100:.1f}%)')
    plt.xlabel('Training Steps')
    plt.ylabel('Accuracy')
    plt.title(f'Accuracy During Training ({setup.capitalize()} Setup)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Grokking moment detection
    plt.subplot(2, 3, 3)
    # Find grokking moment (sudden increase in test accuracy)
    test_acc_array = np.array(test_accuracies)
    if len(test_acc_array) > 10:
        # Simple grokking detection: when test accuracy jumps significantly
        acc_diff = np.diff(test_acc_array)
        grokking_indices = np.where(acc_diff > 0.1)[0]
        if len(grokking_indices) > 0:
            grokking_step = steps[grokking_indices[0]]
            plt.axvline(x=grokking_step, color='red', linestyle='--', 
                       label=f'Grokking at step {grokking_step}')
    
    plt.plot(steps, test_accuracies, linewidth=2)
    plt.axhline(y=random_chance, color='red', linestyle='--', label='Random Chance')
    plt.xlabel('Training Steps')
    plt.ylabel('Test Accuracy')
    plt.title(f'Grokking Detection ({setup.capitalize()} Setup)')
    plt.grid(True, alpha=0.3)
    
    # Plot 4-6: Circuit development for each concept
    concepts = ['parity', 'sum_range', 'modulo_structure']
    for i, concept in enumerate(concepts):
        plt.subplot(2, 3, i+4)
        
        checkpoint_steps = list(circuit_analysis.keys())
        for layer in [0, 1]:
            accuracies = []
            for step in checkpoint_steps:
                if concept in circuit_analysis[step]:
                    accuracies.append(circuit_analysis[step][concept][layer])
                else:
                    accuracies.append(0)
            
            plt.plot(checkpoint_steps, accuracies, marker='o', label=f'Layer {layer}')
        
        plt.axhline(y=0.5, color='red', linestyle='--', label='Random Chance (50%)')
        plt.xlabel('Training Steps')
        plt.ylabel('Probe Accuracy')
        plt.title(f'{concept.capitalize()} Concept')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_filename = f"plots/grokking_{setup}_{timestamp}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {plot_filename}")
    
    # Save results
    results = {
        'train_losses': train_losses,
        'test_losses': test_losses,
        'train_accuracies': train_accuracies,
        'test_accuracies': test_accuracies,
        'circuit_analysis': circuit_analysis
    }
    
    results_filename = f"results/grokking_{setup}_{timestamp}.pt"
    save_dict_as_pt(results, results_filename)
    print(f"Saved: {results_filename}")

if __name__ == "__main__":
    # Run with different setups
    setups = ["correct", "original", "basic"]
    
    print("=== GROKKING EXPERIMENT SUITE ===")
    print("Available setups:")
    print("  - correct: p=113, subset_ratio=0.1, weight_decay=0.1, MODULAR ADDITION (recommended)")
    print("  - original: p=113, subset_ratio=0.1, weight_decay=0.1, MODULAR ADDITION")
    print("  - basic: p=97, subset_ratio=0.3, weight_decay=0.01, MODULAR ADDITION")
    
    # Run the correct setup by default
    run_grokking_experiment("correct") 