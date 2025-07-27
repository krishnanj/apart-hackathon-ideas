#!/usr/bin/env python3
"""
Test script to verify transformer support works for all steps
"""

import torch
import numpy as np
from datetime import datetime
import os
from model import create_model_from_params
from dataset import create_dataset_from_params
from load_params import load_params

def test_transformer_support():
    """Test transformer support for all components"""
    print("=== TESTING TRANSFORMER SUPPORT ===")
    
    # Load params
    params = load_params()
    
    # Test 1: Dataset creation
    print("\n1. Testing dataset creation...")
    try:
        train_x, train_y, test_x, test_y = create_dataset_from_params(params)
        print(f"   ✅ Dataset created successfully")
        print(f"   Train shape: {train_x.shape}, {train_y.shape}")
        print(f"   Test shape: {test_x.shape}, {test_y.shape}")
    except Exception as e:
        print(f"   ❌ Dataset creation failed: {e}")
        return False
    
    # Test 2: Model creation
    print("\n2. Testing model creation...")
    try:
        if params.get('architecture') == 'transformer':
            input_dim = train_x.shape[1]  # sequence length
            output_dim = params.get('transformer_config', {}).get('num_tokens', 100)
        else:
            input_dim = train_x.shape[1] if len(train_x.shape) > 1 else 1
            output_dim = train_y.shape[1] if len(train_y.shape) > 1 else 1
        
        model = create_model_from_params(input_dim, output_dim, params)
        print(f"   ✅ Model created successfully")
        print(f"   Model type: {type(model).__name__}")
    except Exception as e:
        print(f"   ❌ Model creation failed: {e}")
        return False
    
    # Test 3: Forward pass
    print("\n3. Testing forward pass...")
    try:
        model.eval()
        with torch.no_grad():
            if params.get('architecture') == 'transformer':
                # For transformer, input should be [batch_size, seq_len]
                test_input = train_x[:10]  # [batch_size, seq_len]
                outputs = model(test_input)
                print(f"   ✅ Transformer forward pass successful")
                print(f"   Output shape: {outputs.shape}")
            else:
                outputs = model(train_x[:10].float())
                print(f"   ✅ MLP forward pass successful")
                print(f"   Output shape: {outputs.shape}")
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        return False
    
    # Test 4: Training step
    print("\n4. Testing training step...")
    try:
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = torch.nn.CrossEntropyLoss()
        
        if params.get('architecture') == 'transformer':
            # For transformer, input should be [batch_size, seq_len]
            train_input = train_x[:10]  # [batch_size, seq_len]
            outputs = model(train_input)  # [batch_size, seq_len, num_tokens]
            
            # For transformer, we predict the last token
            # outputs shape: [batch_size, seq_len, num_tokens]
            # We want to predict the last token, so take outputs[:, -1, :]
            last_token_outputs = outputs[:, -1, :]  # [batch_size, num_tokens]
            targets = train_y[:10].long()  # [batch_size] - ensure it's long dtype
            
            print(f"   Debug - last_token_outputs shape: {last_token_outputs.shape}")
            print(f"   Debug - targets shape: {targets.shape}")
            
            loss = criterion(last_token_outputs, targets)
        else:
            outputs = model(train_x[:10].float())
            loss = criterion(outputs, train_y[:10].long())
        
        loss.backward()
        optimizer.step()
        print(f"   ✅ Training step successful")
        print(f"   Loss: {loss.item():.4f}")
    except Exception as e:
        print(f"   ❌ Training step failed: {e}")
        print(f"   Error details: {type(e).__name__}: {str(e)}")
        return False
    
    print("\n✅ All transformer support tests passed!")
    return True

def test_architecture_switch():
    """Test switching between MLP and Transformer"""
    print("\n=== TESTING ARCHITECTURE SWITCH ===")
    
    params = load_params()
    
    # Test MLP
    print("\n1. Testing MLP architecture...")
    params['architecture'] = 'mlp'
    params['use_composite'] = False  # Disable composite for MLP test
    try:
        train_x, train_y, test_x, test_y = create_dataset_from_params(params)
        model = create_model_from_params(2, 97, params)
        print(f"   ✅ MLP works")
    except Exception as e:
        print(f"   ❌ MLP failed: {e}")
        return False
    
    # Test Transformer
    print("\n2. Testing Transformer architecture...")
    params['architecture'] = 'transformer'
    params['use_composite'] = False  # Disable composite for transformer test
    try:
        train_x, train_y, test_x, test_y = create_dataset_from_params(params)
        model = create_model_from_params(5, 100, params)  # seq_len=5, num_tokens=100
        print(f"   ✅ Transformer works")
    except Exception as e:
        print(f"   ❌ Transformer failed: {e}")
        return False
    
    print("\n✅ Architecture switch works!")
    return True

if __name__ == "__main__":
    success1 = test_transformer_support()
    success2 = test_architecture_switch()
    
    if success1 and success2:
        print("\n🎉 All tests passed! Transformer support is ready.")
    else:
        print("\n❌ Some tests failed. Check the errors above.") 