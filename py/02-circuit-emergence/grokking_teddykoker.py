#!/usr/bin/env python3
"""
Replicate teddykoker/grokking implementation exactly
Modular Division with Transformer Architecture
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
import os
import numpy as np
from tqdm import tqdm

class Block(nn.Module):
    """Causal transformer block"""
    def __init__(self, dim, num_heads):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.ln_2 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        attn_mask = torch.full(
            (len(x), len(x)), -float("Inf"), device=x.device, dtype=x.dtype
        )
        attn_mask = torch.triu(attn_mask, diagonal=1)

        x = self.ln_1(x)
        a, _ = self.attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x

class Decoder(nn.Module):
    """Causal Transformer decoder"""
    def __init__(self, dim=128, num_layers=2, num_heads=4, num_tokens=97, seq_len=5):
        super().__init__()
        self.token_embeddings = nn.Embedding(num_tokens, dim)
        self.position_embeddings = nn.Embedding(seq_len, dim)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(Block(dim, num_heads))
        self.ln_f = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_tokens, bias=False)

    def forward(self, x):
        h = self.token_embeddings(x)
        positions = torch.arange(x.shape[0], device=x.device).unsqueeze(-1)
        h = h + self.position_embeddings(positions).expand_as(h)
        for layer in self.layers:
            h = layer(h)
        h = self.ln_f(h)
        logits = self.head(h)
        return logits

def modular_inverse(y, p):
    """Compute y^(-1) mod p using Fermat's little theorem"""
    # Convert to numpy for pow() function
    y_np = y.numpy()
    p_np = p
    result = np.zeros_like(y_np)
    
    for i in range(len(y_np)):
        if y_np[i] != 0:
            result[i] = pow(int(y_np[i]), p_np - 2, p_np)
        else:
            result[i] = 0
    
    return torch.tensor(result, dtype=torch.long)

def division_mod_p_data(p, eq_token, op_token):
    """x◦y = x/y (mod p) for 0 ≤ x < p, 0 < y < p"""
    x = torch.arange(p)
    y = torch.arange(1, p)  # y cannot be 0 for division
    x, y = torch.cartesian_prod(x, y).T
    
    eq = torch.ones_like(x) * eq_token
    op = torch.ones_like(x) * op_token
    
    # Compute x/y mod p using modular inverse
    # For prime p: x/y mod p = x * y^(-1) mod p
    y_inv = modular_inverse(y, p)
    result = (x * y_inv) % p
    
    # Format: [x, op, y, eq, result]
    return torch.stack([x, op, y, eq, result])

def run_teddykoker_grokking():
    """Run the exact teddykoker implementation"""
    
    # Parameters from teddykoker
    p = 97
    budget = 300000  # 3e5 steps
    batch_size = 512
    lr = 1e-3
    beta1 = 0.9
    beta2 = 0.98
    weight_decay = 0  # No regularization!
    
    print("=== TEDDYKOKER GROKKING REPLICATION ===")
    print(f"Task: Modular Division (x/y mod {p})")
    print(f"Model: Transformer (2 layers, 128 dim, 4 heads)")
    print(f"Training: {budget} steps, batch_size={batch_size}, lr={lr}")
    print(f"Weight Decay: {weight_decay} (no regularization)")
    
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Tokens for <op> and <=>
    eq_token = p
    op_token = p + 1
    
    # Create model exactly as teddykoker
    model = Decoder(
        dim=128, num_layers=2, num_heads=4, num_tokens=p + 2, seq_len=5
    ).to(device)
    
    # Create dataset: division mod p
    data = division_mod_p_data(p, eq_token, op_token)
    train_idx, valid_idx = torch.randperm(data.shape[1]).split(data.shape[1] // 2)
    train_data, valid_data = data[:, train_idx], data[:, valid_idx]
    
    print(f"Training set size: {train_data.shape[1]}")
    print(f"Validation set size: {valid_data.shape[1]}")
    
    # Optimizer exactly as teddykoker
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: 1 if update > 10 else update / 10
    )
    
    steps_per_epoch = math.ceil(train_data.shape[1] / batch_size)
    total_epochs = int(budget // steps_per_epoch)
    
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total epochs: {total_epochs}")
    
    train_acc, val_acc, train_loss, val_loss = [], [], [], []
    
    # Training loop exactly as teddykoker
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
                    logits = model(input[:-1])
                    # Calculate loss only on the answer part (last element)
                    loss = F.cross_entropy(logits[-1], input[-1])
                    total_loss += loss.item() * input.shape[-1]
                
                if is_train:
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                
                acc = (logits[-1].argmax(-1) == input[-1]).float().mean()
                total_acc += acc.item() * input.shape[-1]
            
            if is_train:
                train_acc.append(total_acc / train_data.shape[-1])
                train_loss.append(total_loss / train_data.shape[-1])
            else:
                val_acc.append(total_acc / valid_data.shape[-1])
                val_loss.append(total_loss / valid_data.shape[-1])
        
        # Save plots every 100 epochs
        if (e + 1) % 100 == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create plots directory
            os.makedirs("plots", exist_ok=True)
            
            # Accuracy plot
            steps = torch.arange(len(train_acc)).numpy() * steps_per_epoch
            plt.figure(figsize=(12, 5))
            
            plt.subplot(1, 2, 1)
            plt.plot(steps, train_acc, label="train", linewidth=2)
            plt.plot(steps, val_acc, label="val", linewidth=2)
            plt.legend()
            plt.title("Modular Division (training on 50% of data)")
            plt.xlabel("Optimization Steps")
            plt.ylabel("Accuracy")
            plt.xscale("log", base=10)
            plt.grid(True, alpha=0.3)
            
            # Loss plot
            plt.subplot(1, 2, 2)
            plt.plot(steps, train_loss, label="train", linewidth=2)
            plt.plot(steps, val_loss, label="val", linewidth=2)
            plt.legend()
            plt.title("Modular Division (training on 50% of data)")
            plt.xlabel("Optimization Steps")
            plt.ylabel("Loss")
            plt.xscale("log", base=10)
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plot_filename = f"plots/teddykoker_grokking_{timestamp}.png"
            plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved: {plot_filename}")
            
            # Print current metrics
            print(f"Epoch {e+1}: Train Acc: {train_acc[-1]:.4f}, Val Acc: {val_acc[-1]:.4f}")
            print(f"         Train Loss: {train_loss[-1]:.4f}, Val Loss: {val_loss[-1]:.4f}")
            
            # Check for grokking
            if val_acc[-1] > 0.8:
                print(f"🎉 GROKKING DETECTED at epoch {e+1}!")
    
    print("Training completed!")
    return {
        'train_acc': train_acc,
        'val_acc': val_acc,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'steps': torch.arange(len(train_acc)).numpy() * steps_per_epoch
    }

if __name__ == "__main__":
    results = run_teddykoker_grokking()
    print("Teddykoker replication completed!") 