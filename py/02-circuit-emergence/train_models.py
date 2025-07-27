#!/usr/bin/env python3
"""
Training script with support for both MLP and Transformer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import os
from model import create_model_from_params
from dataset import create_dataset_from_params
from load_params import load_params

def train_model_with_architecture():
    """Train model with architecture specified in params"""
    params = load_params()
    architecture = params.get('architecture', 'mlp')
    
    print(f"=== STEP 2: NEURAL NETWORK TRAINING ===")
    print(f"Training with {architecture.upper()} architecture...")
    
    # Create dataset
    train_x, train_y, test_x, test_y = create_dataset_from_params(params)
    
    print(f"Dataset shapes:")
    print(f"  Train: {train_x.shape}, {train_y.shape}")
    print(f"  Test: {test_x.shape}, {test_y.shape}")
    
    # Determine input/output dimensions
    if architecture == 'mlp':
        input_dim = train_x.shape[1] if len(train_x.shape) > 1 else 1
        output_dim = train_y.shape[1] if len(train_y.shape) > 1 else 1
    else:  # transformer
        input_dim = train_x.shape[1]  # sequence length
        output_dim = params.get('transformer_config', {}).get('num_tokens', 100)
    
    print(f"Model dimensions: input_dim={input_dim}, output_dim={output_dim}")
    
    # Create model
    model = create_model_from_params(input_dim, output_dim, params)
    print(f"Model created: {type(model).__name__}")
    
    # Training parameters
    lr = params.get('lr', 0.001)
    n_epochs = params.get('n_epochs', 50)  # Reduced for testing
    batch_size = params.get('batch_size', 32)
    
    print(f"Training parameters: lr={lr}, n_epochs={n_epochs}, batch_size={batch_size}")
    
    # Loss and optimizer
    if architecture == 'mlp':
        criterion = nn.MSELoss() if output_dim == 1 else nn.CrossEntropyLoss()
    else:  # transformer
        criterion = nn.CrossEntropyLoss()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print(f"Loss function: {type(criterion).__name__}")
    print(f"Optimizer: {type(optimizer).__name__}")
    
    # Training loop
    train_losses = []
    test_losses = []
    
    print(f"\nStarting training...")
    for epoch in range(n_epochs):
        model.train()
        
        if architecture == 'mlp':
            # MLP training
            optimizer.zero_grad()
            outputs = model(train_x.float())
            loss = criterion(outputs, train_y.float() if output_dim == 1 else train_y.long())
            loss.backward()
            optimizer.step()
            
            # Test
            model.eval()
            with torch.no_grad():
                test_outputs = model(test_x.float())
                test_loss = criterion(test_outputs, test_y.float() if output_dim == 1 else test_y.long())
        else:
            # Transformer training
            optimizer.zero_grad()
            # For transformers, we predict the last token
            outputs = model(train_x)
            loss = criterion(outputs[:, -1, :], train_y.long())
            loss.backward()
            optimizer.step()
            
            # Test
            model.eval()
            with torch.no_grad():
                test_outputs = model(test_x)
                test_loss = criterion(test_outputs[:, -1, :], test_y.long())
        
        train_losses.append(loss.item())
        test_losses.append(test_loss.item())
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"checkpoints/model_{architecture}_{timestamp}.pt"
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), model_filename)
    print(f"Saved: {model_filename}")
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        if architecture == 'mlp':
            train_outputs = model(train_x.float())
            test_outputs = model(test_x.float())
        else:
            train_outputs = model(train_x)
            test_outputs = model(test_x)
    
    print(f"\nFinal Results:")
    print(f"  Train Loss: {train_losses[-1]:.4f}")
    print(f"  Test Loss: {test_losses[-1]:.4f}")
    print(f"  Model saved: {model_filename}")
    
    return model, train_losses, test_losses

def test_step2_architecture_switch():
    """Test Step 2 with both MLP and Transformer"""
    print(f"\n=== TESTING STEP 2 ARCHITECTURE SWITCH ===")
    
    params = load_params()
    
    # Test MLP
    print(f"\n1. Testing MLP training...")
    params['architecture'] = 'mlp'
    params['use_composite'] = False
    try:
        model, train_losses, test_losses = train_model_with_architecture()
        print(f"   ✅ MLP training successful")
        print(f"   Final train loss: {train_losses[-1]:.4f}")
        print(f"   Final test loss: {test_losses[-1]:.4f}")
    except Exception as e:
        print(f"   ❌ MLP training failed: {e}")
        return False
    
    # Test Transformer
    print(f"\n2. Testing Transformer training...")
    params['architecture'] = 'transformer'
    try:
        model, train_losses, test_losses = train_model_with_architecture()
        print(f"   ✅ Transformer training successful")
        print(f"   Final train loss: {train_losses[-1]:.4f}")
        print(f"   Final test loss: {test_losses[-1]:.4f}")
    except Exception as e:
        print(f"   ❌ Transformer training failed: {e}")
        return False
    
    print(f"\n✅ Step 2 (Neural Network Training) with both architectures works!")
    return True

if __name__ == "__main__":
    # Test single architecture training
    success1 = train_model_with_architecture()
    
    # Test architecture switching
    success2 = test_step2_architecture_switch()
    
    if success1 and success2:
        print(f"\n🎉 Step 2 (Neural Network Training) with Transformer support is ready!")
    else:
        print(f"\n❌ Step 2 tests failed. Check the errors above.")