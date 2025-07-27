#!/usr/bin/env python3
"""
Transformer Grokking Experiment with Concept Decoding
Adapts teddykoker approach to work with our probing system
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
from tqdm import tqdm
import math

from transformer_probe_utils import TransformerProbeAnalyzer
from advanced_probe_utils import run_architecture_aware_probes
from probe_utils import save_dict_as_pt
from load_params import load_params

class TransformerBlock(nn.Module):
    """Causal transformer block for grokking"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.ln_2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, dim]
        x = self.ln_1(x)
        attn_out, _ = self.attn(x, x, x)
        x = x + attn_out
        x = self.ln_2(x)
        mlp_out = self.mlp(x)
        x = x + mlp_out
        return x

class GrokkingTransformer(nn.Module):
    """Transformer for grokking experiments"""
    def __init__(self, dim=128, num_layers=2, num_heads=4, num_tokens=99, seq_len=5):
        super().__init__()
        self.token_embeddings = nn.Embedding(num_tokens, dim)
        self.position_embeddings = nn.Embedding(seq_len, dim)
        self.ln_f = nn.LayerNorm(dim)
        self.lm_head = nn.Linear(dim, num_tokens, bias=False)
        self.seq_len = seq_len
        
        # Use 'layers' instead of 'blocks' to match probing utilities
        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads) for _ in range(num_layers)
        ])
        
    def forward(self, x):
        # x shape: [batch_size, seq_len]
        b, t = x.shape
        
        # Get embeddings
        tok_emb = self.token_embeddings(x)
        # Ensure position indices don't exceed embedding table size and match actual sequence length
        pos_indices = torch.arange(min(t, self.seq_len), device=x.device)
        pos_emb = self.position_embeddings(pos_indices)
        # Expand pos_emb to match tok_emb shape: [seq_len, dim] -> [batch_size, seq_len, dim]
        pos_emb = pos_emb.unsqueeze(0).expand(b, -1, -1)
        x = tok_emb + pos_emb
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

def modular_inverse(y, p):
    """Compute modular inverse using extended Euclidean algorithm"""
    def extended_gcd(a, b):
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y
    
    gcd, x, _ = extended_gcd(y, p)
    if gcd != 1:
        return None  # No inverse exists
    return (x % p + p) % p

def create_grokking_dataset(p=97, subset_ratio=0.5):
    """Create modular division dataset for grokking"""
    # Tokens for special symbols
    eq_token = p
    op_token = p + 1
    
    # Generate all possible (x, y) pairs
    data = []
    for x in range(p):
        for y in range(1, p):  # y != 0 to avoid division by zero
            # Compute x / y mod p = x * y^(-1) mod p
            y_inv = modular_inverse(y, p)
            if y_inv is not None:
                result = (x * y_inv) % p
                # Create sequence: [x, op, y, eq, result]
                sequence = [x, op_token, y, eq_token, result]
                data.append(sequence)
    
    data = torch.tensor(data, dtype=torch.long)
    
    # Split into train/validation
    n_train = int(subset_ratio * len(data))
    indices = torch.randperm(len(data))
    train_idx = indices[:n_train]
    valid_idx = indices[n_train:]
    
    train_data = data[train_idx].T  # [seq_len, n_train]
    valid_data = data[valid_idx].T   # [seq_len, n_valid]
    
    return train_data, valid_data, p, op_token, eq_token

def generate_concept_labels(data, p, concept_type="parity"):
    """Generate concept labels for probing"""
    # data shape: [seq_len, batch_size]
    x_vals = data[0]  # First token (x values)
    y_vals = data[2]  # Third token (y values)
    results = data[4]  # Fifth token (results)
    
    if concept_type == "parity":
        # Even/odd parity of x
        return (x_vals % 2).numpy().flatten()
    elif concept_type == "range":
        # Whether x is in upper half
        return (x_vals > p//2).numpy().flatten()
    elif concept_type == "y_parity":
        # Even/odd parity of y
        return (y_vals % 2).numpy().flatten()
    elif concept_type == "result_range":
        # Whether result is in upper half
        return (results > p//2).numpy().flatten()
    elif concept_type == "divisibility":
        # Whether x is divisible by 2
        return (x_vals % 2 == 0).numpy().flatten()
    elif concept_type == "zero_crossings":
        # Detect if results have multiple crossings of p/2 (modular zero crossings)
        results_np = results.numpy()
        crossings = 0
        for i in range(1, len(results_np)):
            prev_crossed = results_np[i-1] < p//2
            curr_crossed = results_np[i] < p//2
            if prev_crossed != curr_crossed:
                crossings += 1
        return np.array([1 if crossings > 2 else 0] * len(results_np))
    elif concept_type == "oscillation_count":
        # Detect number of oscillations in modular space
        results_np = results.numpy()
        oscillations = 0
        for i in range(2, len(results_np)):
            # Check if direction changes in modular arithmetic
            diff1 = (results_np[i-1] - results_np[i-2]) % p
            diff2 = (results_np[i] - results_np[i-1]) % p
            if diff1 > p//2:
                diff1 = diff1 - p
            if diff2 > p//2:
                diff2 = diff2 - p
            if diff1 * diff2 < 0:
                oscillations += 1
        return np.array([1 if oscillations > 3 else 0] * len(results_np))
    elif concept_type == "symmetry_breaking":
        # Detect if results are symmetric around p/2
        results_np = results.numpy()
        mid_point = len(results_np) // 2
        left_half = results_np[:mid_point]
        right_half = results_np[mid_point:][::-1]  # Reverse right half
        # Check symmetry around p/2
        symmetry_error = np.mean(np.abs((left_half - p//2) + (right_half - p//2)))
        return np.array([1 if symmetry_error < 5 else 0] * len(results_np))
    elif concept_type == "complexity_estimate":
        # Estimate complexity using variance of modular differences
        results_np = results.numpy()
        if len(results_np) < 3:
            return np.array([0] * len(results_np))
        # Calculate modular differences
        diff1 = np.diff(results_np)
        diff1 = np.minimum(diff1, p - diff1)  # Take smaller difference
        complexity = np.var(diff1)
        return np.array([1 if complexity > np.median(diff1) else 0] * len(results_np))
    elif concept_type == "modular_remainder":
        # Whether result has specific remainder (e.g., divisible by 3)
        results_np = results.numpy()
        return (results_np % 3 == 0).astype(int)
    elif concept_type == "input_output_parity":
        # Whether x and result have same parity
        x_vals_np = x_vals.numpy()
        results_np = results.numpy()
        return ((x_vals_np % 2) == (results_np % 2)).astype(int)
    elif concept_type == "modular_symmetry":
        # Whether result is symmetric around p/2
        results_np = results.numpy()
        return (np.abs(results_np - p//2) < p//4).astype(int)
    elif concept_type == "alternating":
        # Whether results alternate between high/low values
        results_np = results.numpy()
        if len(results_np) < 2:
            return np.array([0] * len(results_np))
        alternating_count = 0
        for i in range(1, len(results_np)):
            prev_high = results_np[i-1] > p//2
            curr_high = results_np[i] > p//2
            if prev_high != curr_high:
                alternating_count += 1
        return np.array([1 if alternating_count > len(results_np)//3 else 0] * len(results_np))
    elif concept_type == "monotonic":
        # Whether results are monotonically increasing
        results_np = results.numpy()
        if len(results_np) < 2:
            return np.array([0] * len(results_np))
        increasing_count = 0
        for i in range(1, len(results_np)):
            if results_np[i] > results_np[i-1]:
                increasing_count += 1
        return np.array([1 if increasing_count > len(results_np)//2 else 0] * len(results_np))
    elif concept_type == "gcd_pattern":
        # Whether gcd(x,y) affects result pattern
        x_vals_np = x_vals.numpy()
        y_vals_np = y_vals.numpy()
        results_np = results.numpy()
        # Check if results with gcd=1 have different patterns
        gcd_1_results = results_np[(x_vals_np % 2 == 1) & (y_vals_np % 2 == 1)]
        if len(gcd_1_results) > 0:
            avg_gcd_1 = np.mean(gcd_1_results)
            return (results_np > avg_gcd_1).astype(int)
        else:
            return np.array([0] * len(results_np))
    elif concept_type == "prime_factor":
        # Whether result has specific prime factors
        results_np = results.numpy()
        return (results_np % 2 == 0).astype(int)  # Even results
    elif concept_type == "modular_inverse_prop":
        # Properties of modular inverse relationships
        x_vals_np = x_vals.numpy()
        y_vals_np = y_vals.numpy()
        results_np = results.numpy()
        # Check if x*y ≡ 1 (mod p) patterns
        return ((x_vals_np * y_vals_np) % p == 1).astype(int)
    else:
        raise ValueError(f"Unknown concept type: {concept_type}")

def run_transformer_grokking_experiment():
    """Run transformer grokking experiment with concept decoding"""
    
    print("=== TRANSFORMER GROKKING EXPERIMENT ===")
    print("Task: Modular Division (x / y mod p)")
    print("Architecture: Transformer with attention")
    
    # Parameters
    p = 97
    subset_ratio = 0.5  # Train on 50% of data
    budget = 30000  # Reduced for testing
    batch_size = 512
    lr = 1e-3
    weight_decay = 0
    
    # Model parameters
    dim = 128
    num_layers = 2
    num_heads = 4
    num_tokens = p + 2  # p tokens + op_token + eq_token
    seq_len = 5
    
    print(f"Parameters: p={p}, subset_ratio={subset_ratio}, budget={budget}")
    print(f"Model: {num_layers} layers, {dim} dim, {num_heads} heads")
    print(f"Random chance accuracy: {1.0/p:.4f} ({1.0/p*100:.2f}%)")
    
    # Create dataset
    train_data, valid_data, p, op_token, eq_token = create_grokking_dataset(p, subset_ratio)
    
    print(f"Training set size: {train_data.shape[1]}")
    print(f"Validation set size: {valid_data.shape[1]}")
    
    # Create model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GrokkingTransformer(dim, num_layers, num_heads, num_tokens, seq_len).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(0.9, 0.98),
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: 1 if update > 10 else update / 10
    )
    
    # Training tracking
    train_acc, val_acc, train_loss, val_loss = [], [], [], []
    checkpoint_interval = 1000
    grokking_checkpoints = []
    
    # Concept types to probe for (simple + complex + mathematical)
    concept_types = [
        # Simple concepts (baseline - should show instant learning)
        "parity", "range", "y_parity", "result_range", "divisibility",
        # Complex concepts (should show gradual emergence)
        "modular_remainder", "input_output_parity", "modular_symmetry", 
        "alternating", "monotonic",
        # Mathematical relationship concepts
        "gcd_pattern", "prime_factor", "modular_inverse_prop"
    ]
    
    print(f"\nTraining with concept probing every {checkpoint_interval} steps...")
    
    steps_per_epoch = math.ceil(train_data.shape[1] / batch_size)
    total_epochs = int(budget // steps_per_epoch)
    
    for e in tqdm(range(total_epochs)):
        # Randomly shuffle train data
        train_data = train_data[:, torch.randperm(train_data.shape[1])]
        
        for data, is_train in [(train_data, True), (valid_data, False)]:
            model.train(is_train)
            total_loss = 0
            total_acc = 0
            
            # Process in batches
            dl = torch.split(data, batch_size, dim=1)
            for input in dl:
                input = input.to(device)
                
                with torch.set_grad_enabled(is_train):
                    # input has shape [seq_len, batch_size], we want [batch_size, seq_len-1]
                    input_seq = input[:-1].T  # Transpose to get [batch_size, seq_len-1]
                    logits = model(input_seq)
                    # Calculate loss only on the answer part (last element)
                    loss = F.cross_entropy(logits[:, -1], input[-1])
                    total_loss += loss.item() * input.shape[-1]
                
                if is_train:
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                
                acc = (logits[:, -1].argmax(-1) == input[-1]).float().mean()
                total_acc += acc.item() * input.shape[-1]
            
            if is_train:
                train_acc.append(total_acc / train_data.shape[-1])
                train_loss.append(total_loss / train_data.shape[-1])
            else:
                val_acc.append(total_acc / valid_data.shape[-1])
                val_loss.append(total_loss / valid_data.shape[-1])
        
        # Save checkpoint for probing
        if (e + 1) % (checkpoint_interval // steps_per_epoch) == 0:
            step = (e + 1) * steps_per_epoch
            print(f"\nStep {step}: Train Acc: {train_acc[-1]:.4f}, Val Acc: {val_acc[-1]:.4f}")
            
            # Save checkpoint
            checkpoint_data = {
                'step': step,
                'model_state': model.state_dict(),
                'train_acc': train_acc[-1],
                'val_acc': val_acc[-1],
                'train_loss': train_loss[-1],
                'val_loss': val_loss[-1]
            }
            grokking_checkpoints.append(checkpoint_data)
            
            # Run concept probing
            print(f"  Running concept probing...")
            concept_results = run_concept_probing(
                model, valid_data, concept_types, p, step
            )
            checkpoint_data['concept_results'] = concept_results
            
            # Check for grokking
            if val_acc[-1] > 0.8:
                print(f"🎉 GROKKING DETECTED at step {step}!")
    
    # Analyze concept development
    print(f"\nAnalyzing concept development during grokking...")
    analyze_concept_development(grokking_checkpoints, concept_types)
    
    # Create plots
    create_grokking_plots(train_acc, val_acc, train_loss, val_loss, 
                          grokking_checkpoints, steps_per_epoch)
    
    print("Transformer grokking experiment completed!")
    return grokking_checkpoints

def run_concept_probing(model, data, concept_types, p, step):
    """Run concept probing on transformer model"""
    model.eval()
    
    # Prepare data for probing
    # Use validation data for probing
    probe_data = data.T  # [batch_size, seq_len]
    
    concept_results = {}
    
    for concept_type in concept_types:
        try:
            # Generate concept labels
            concept_labels = generate_concept_labels(data, p, concept_type)
            

            
            # Run transformer-specific probing
            params = {'architecture': 'transformer'}
            probe_results = run_architecture_aware_probes(
                model, probe_data, concept_labels, params, is_regression=False
            )
            
            # Extract linear probe accuracies
            linear_accuracies = {}
            for layer_name, results in probe_results.items():
                if isinstance(results, dict) and 'linear' in results:
                    linear_accuracies[layer_name] = results['linear']
            
            concept_results[concept_type] = linear_accuracies
            
        except Exception as e:
            print(f"    Error probing {concept_type}: {e}")
            concept_results[concept_type] = {}
    
    return concept_results

def analyze_concept_development(checkpoints, concept_types):
    """Analyze how concepts develop during grokking"""
    print(f"\n=== CONCEPT DEVELOPMENT ANALYSIS ===")
    
    for concept_type in concept_types:
        print(f"\n{concept_type.upper()} concept development:")
        
        # Track concept accuracy over time
        concept_accuracies = []
        steps = []
        
        for checkpoint in checkpoints:
            step = checkpoint['step']
            concept_results = checkpoint.get('concept_results', {})
            
            if concept_type in concept_results:
                # Get max accuracy across all layers
                accuracies = list(concept_results[concept_type].values())
                if accuracies:
                    max_acc = max(accuracies)
                    concept_accuracies.append(max_acc)
                    steps.append(step)
        
        if concept_accuracies:
            # Plot concept development
            plt.figure(figsize=(10, 6))
            plt.plot(steps, concept_accuracies, 'o-', linewidth=2, markersize=4)
            plt.title(f'{concept_type.title()} Concept Development During Grokking')
            plt.xlabel('Training Steps')
            plt.ylabel('Max Linear Probe Accuracy')
            plt.grid(True, alpha=0.3)
            plt.ylim(0, 1)
            
            # Save plot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_filename = f"plots/concept_development_{concept_type}_{timestamp}.png"
            os.makedirs("plots", exist_ok=True)
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved: {plot_filename}")
            
            # Print summary
            max_acc = max(concept_accuracies)
            final_acc = concept_accuracies[-1]
            print(f"  Max accuracy: {max_acc:.4f}")
            print(f"  Final accuracy: {final_acc:.4f}")

def create_grokking_plots(train_acc, val_acc, train_loss, val_loss, checkpoints, steps_per_epoch):
    """Create grokking plots with concept development"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("plots", exist_ok=True)
    
    # Convert epochs to steps
    steps = torch.arange(len(train_acc)).numpy() * steps_per_epoch
    
    # Create comprehensive plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Accuracy plot
    axes[0, 0].plot(steps, train_acc, label="train", linewidth=2)
    axes[0, 0].plot(steps, val_acc, label="val", linewidth=2)
    axes[0, 0].legend()
    axes[0, 0].set_title("Transformer Grokking: Accuracy")
    axes[0, 0].set_xlabel("Optimization Steps")
    axes[0, 0].set_ylabel("Accuracy")
    axes[0, 0].set_xscale("log", base=10)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss plot
    axes[0, 1].plot(steps, train_loss, label="train", linewidth=2)
    axes[0, 1].plot(steps, val_loss, label="val", linewidth=2)
    axes[0, 1].legend()
    axes[0, 1].set_title("Transformer Grokking: Loss")
    axes[0, 1].set_xlabel("Optimization Steps")
    axes[0, 1].set_ylabel("Loss")
    axes[0, 1].set_xscale("log", base=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Concept development heatmap
    if checkpoints:
        concept_types = ["parity", "range", "y_parity", "result_range", "divisibility"]
        checkpoint_steps = [cp['step'] for cp in checkpoints]
        
        # Create heatmap data
        heatmap_data = []
        for concept_type in concept_types:
            concept_accuracies = []
            for checkpoint in checkpoints:
                concept_results = checkpoint.get('concept_results', {})
                if concept_type in concept_results:
                    accuracies = list(concept_results[concept_type].values())
                    max_acc = max(accuracies) if accuracies else 0
                    concept_accuracies.append(max_acc)
                else:
                    concept_accuracies.append(0)
            heatmap_data.append(concept_accuracies)
        
        if heatmap_data:
            im = axes[1, 0].imshow(heatmap_data, cmap='viridis', aspect='auto')
            axes[1, 0].set_title("Concept Development Heatmap")
            axes[1, 0].set_xlabel("Checkpoint")
            axes[1, 0].set_ylabel("Concept Type")
            axes[1, 0].set_yticks(range(len(concept_types)))
            axes[1, 0].set_yticklabels(concept_types)
            plt.colorbar(im, ax=axes[1, 0])
    
    # Attention analysis placeholder
    axes[1, 1].text(0.5, 0.5, "Attention Analysis\n(To be implemented)", 
                     ha='center', va='center', transform=axes[1, 1].transAxes)
    axes[1, 1].set_title("Attention Pattern Analysis")
    
    plt.tight_layout()
    plot_filename = f"plots/transformer_grokking_comprehensive_{timestamp}.png"
    plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comprehensive plot: {plot_filename}")

if __name__ == "__main__":
    # Run transformer grokking experiment
    checkpoints = run_transformer_grokking_experiment()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"results/transformer_grokking_results_{timestamp}.pt"
    save_dict_as_pt({'checkpoints': checkpoints}, results_filename)
    print(f"Saved results: {results_filename}") 