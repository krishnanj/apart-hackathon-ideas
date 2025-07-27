#!/usr/bin/env python3
"""
Fixed Original Grokking Setup: Address the accuracy and task definition issues
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

def run_original_grokking_fixed():
    """Run grokking experiment with fixes for accuracy issues"""
    
    print("Starting FIXED original grokking experiment...")
    print("Addressing accuracy calculation and task definition issues")
    
    # Original paper hyperparameters
    p = 113  # Different prime (not 97)
    subset_ratio = 0.1  # Much smaller subset (10% instead of 30%)
    weight_decay = 0.1  # Much stronger regularization
    lr = 0.001
    n_steps = 50000  # Much longer training
    checkpoint_interval = 1000  # Checkpoints every 1000 steps
    
    # Model parameters (smaller network)
    hidden_dim = 128
    depth = 2
    input_dim = 2
    output_dim = p
    
    # Create dataset with original paper setup
    train_x, train_y, test_x, test_y = create_original_grokking_dataset_fixed(p, subset_ratio)
    
    print(f"Training set size: {len(train_x)}")
    print(f"Test set size: {len(test_x)}")
    print(f"Model: {depth} layers, {hidden_dim} hidden units")
    print(f"Training for {n_steps} steps with weight_decay={weight_decay}")
    print(f"Task: (x + y) mod {p} classification with {p} classes")
    print(f"Random chance accuracy: {1.0/p:.4f} ({1.0/p*100:.2f}%)")
    
    # Create model (smaller than our previous attempts)
    model = create_original_grokking_model(input_dim, hidden_dim, output_dim, depth)
    
    # Training with original paper setup
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    
    # Tracking variables
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    grokking_checkpoints = []
    
    print("\nTraining with FIXED original paper setup...")
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
            
            train_acc = compute_accuracy_fixed(model(train_x.float()), train_y.long())
            test_acc = compute_accuracy_fixed(model(test_x.float()), test_y.long())
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            
            print(f"Step {step}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
            print(f"         Train Acc: {train_acc:.4f} ({train_acc*100:.2f}%), Test Acc: {test_acc:.4f} ({test_acc*100:.2f}%)")
            
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
    circuit_analysis = analyze_circuit_development_original_fixed(grokking_checkpoints, test_x, test_y)
    
    # Create grokking plots
    create_original_grokking_plots_fixed(train_losses, test_losses, train_accuracies, test_accuracies, 
                                        circuit_analysis, checkpoint_interval)
    
    print("FIXED original grokking experiment completed!")

def create_original_grokking_dataset_fixed(p, subset_ratio):
    """Create dataset with original paper setup - FIXED VERSION"""
    # Generate full dataset
    all_data = torch.tensor([(i, j) for i in range(p) for j in range(p)], dtype=torch.float32)
    all_labels = (all_data[:, 0] + all_data[:, 1]) % p
    
    # Create subset for training (much smaller)
    n_train = int(subset_ratio * len(all_data))
    indices = torch.randperm(len(all_data))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    train_x = all_data[train_idx]
    train_y = all_labels[train_idx]
    test_x = all_data[test_idx]
    test_y = all_labels[test_idx]
    
    print(f"Dataset created: {len(train_x)} train, {len(test_x)} test samples")
    print(f"Label range: {train_y.min().item()} to {train_y.max().item()}")
    print(f"Unique labels in train: {len(torch.unique(train_y))}")
    print(f"Unique labels in test: {len(torch.unique(test_y))}")
    
    return train_x, train_y, test_x, test_y

def create_original_grokking_model(input_dim, hidden_dim, output_dim, depth):
    """Create model with original paper architecture"""
    import torch.nn as nn
    
    class OriginalGrokkingMLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, depth):
            super().__init__()
            layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
            for _ in range(depth - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
            layers += [nn.Linear(hidden_dim, output_dim)]
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)
    
    return OriginalGrokkingMLP(input_dim, hidden_dim, output_dim, depth)

def compute_accuracy_fixed(outputs, targets):
    """Compute accuracy for classification - FIXED VERSION"""
    predictions = torch.argmax(outputs, dim=1)
    correct = (predictions == targets).sum().item()
    accuracy = correct / len(targets)
    
    # Debug info
    if len(torch.unique(targets)) < 10:  # Only print for small number of classes
        print(f"  Debug - Predictions: {predictions[:10]}, Targets: {targets[:10]}")
        print(f"  Debug - Correct: {correct}/{len(targets)} = {accuracy:.4f}")
    
    return accuracy

def analyze_circuit_development_original_fixed(checkpoints, test_x, test_y):
    """Analyze how internal circuits develop during grokking - FIXED VERSION"""
    circuit_results = {}
    
    # Define concepts to probe for (adapted for p=113)
    concepts = {
        'parity': lambda x: (x[:, 0] + x[:, 1]) % 2,  # Even/odd parity
        'sum_range': lambda x: ((x[:, 0] + x[:, 1]) > 56).long(),  # Sum in upper half (p=113)
        'modulo_structure': lambda x: ((x[:, 0] + x[:, 1]) % 10 == 0).long()  # Modulo 10
    }
    
    for i, checkpoint in enumerate(checkpoints):
        step = checkpoint['step']
        print(f"  Analyzing checkpoint {i+1}/{len(checkpoints)} (step {step})")
        
        # Load model state
        model = create_original_grokking_model(2, 128, 113, 2)
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

def create_original_grokking_plots_fixed(train_losses, test_losses, train_accuracies, test_accuracies, 
                                       circuit_analysis, checkpoint_interval):
    """Create comprehensive grokking analysis plots - FIXED VERSION"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Plot 1: Loss curves
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    steps = list(range(0, len(train_losses) * 1000, 1000))
    plt.plot(steps, train_losses, label='Train Loss', linewidth=2)
    plt.plot(steps, test_losses, label='Test Loss', linewidth=2)
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title('Loss During Training (Fixed Setup)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 2: Accuracy curves
    plt.subplot(2, 3, 2)
    plt.plot(steps, train_accuracies, label='Train Accuracy', linewidth=2)
    plt.plot(steps, test_accuracies, label='Test Accuracy', linewidth=2)
    plt.axhline(y=1.0/113, color='red', linestyle='--', label='Random Chance (0.9%)')
    plt.xlabel('Training Steps')
    plt.ylabel('Accuracy')
    plt.title('Accuracy During Training (Fixed Setup)')
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
    plt.axhline(y=1.0/113, color='red', linestyle='--', label='Random Chance')
    plt.xlabel('Training Steps')
    plt.ylabel('Test Accuracy')
    plt.title('Grokking Detection (Fixed Setup)')
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
    plot_filename = f"plots/original_grokking_fixed_{timestamp}.png"
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
    
    results_filename = f"results/original_grokking_fixed_{timestamp}.pt"
    save_dict_as_pt(results, results_filename)
    print(f"Saved: {results_filename}")

if __name__ == "__main__":
    run_original_grokking_fixed() 