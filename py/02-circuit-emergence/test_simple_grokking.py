#!/usr/bin/env python3
"""
Test simple grokking setup to verify the issue
"""

import torch
import torch.nn as nn
import numpy as np

def test_simple_grokking():
    """Test with a much simpler setup to see if grokking works"""
    
    print("=== TESTING SIMPLE GROKKING SETUP ===")
    
    # Use a much smaller p to make the task easier
    p = 7  # Only 7 classes instead of 113
    subset_ratio = 0.3  # 30% training data
    hidden_dim = 64  # Smaller model
    depth = 2
    
    print(f"Testing with p={p}, hidden_dim={hidden_dim}, depth={depth}")
    print(f"Random chance: {1.0/p:.4f} ({1.0/p*100:.2f}%)")
    
    # Create simple dataset
    all_data = torch.tensor([(i, j) for i in range(p) for j in range(p)], dtype=torch.float32)
    all_labels = (all_data[:, 0] + all_data[:, 1]) % p
    
    # Split data
    n_train = int(subset_ratio * len(all_data))
    indices = torch.randperm(len(all_data))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    train_x = all_data[train_idx]
    train_y = all_labels[train_idx]
    test_x = all_data[test_idx]
    test_y = all_labels[test_idx]
    
    print(f"Train size: {len(train_x)}, Test size: {len(test_x)}")
    
    # Create simple model
    class SimpleMLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim, depth):
            super().__init__()
            layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
            for _ in range(depth - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
            layers += [nn.Linear(hidden_dim, output_dim)]
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)
    
    model = SimpleMLP(2, hidden_dim, p, depth)
    
    # Training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    
    print("\nTraining simple model...")
    for step in range(5000):  # Shorter training
        # Training step
        optimizer.zero_grad()
        outputs = model(train_x.float())
        loss = criterion(outputs, train_y.long())
        loss.backward()
        optimizer.step()
        
        # Track metrics
        if step % 500 == 0:
            train_loss = loss.item()
            test_loss = criterion(model(test_x.float()), test_y.long()).item()
            
            train_acc = (torch.argmax(model(train_x.float()), dim=1) == train_y.long()).float().mean().item()
            test_acc = (torch.argmax(model(test_x.float()), dim=1) == test_y.long()).float().mean().item()
            
            train_losses.append(train_loss)
            test_losses.append(test_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)
            
            print(f"Step {step}: Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")
            print(f"         Train Acc: {train_acc:.4f} ({train_acc*100:.2f}%), Test Acc: {test_acc:.4f} ({test_acc*100:.2f}%)")
    
    # Final results
    final_train_acc = train_accuracies[-1]
    final_test_acc = test_accuracies[-1]
    
    print(f"\n=== FINAL RESULTS ===")
    print(f"Final Train Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
    print(f"Final Test Accuracy: {final_test_acc:.4f} ({final_test_acc*100:.2f}%)")
    print(f"Random chance: {1.0/p:.4f} ({1.0/p*100:.2f}%)")
    
    if final_test_acc > 0.5:  # Much better than random
        print("✅ SUCCESS: Model learned the task!")
        return True
    else:
        print("❌ FAILURE: Model did not learn the task")
        return False

if __name__ == "__main__":
    test_simple_grokking() 