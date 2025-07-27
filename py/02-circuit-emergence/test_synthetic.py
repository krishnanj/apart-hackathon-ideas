#!/usr/bin/env python3
"""
Test script for synthetic function generation and dataset creation.
"""

import torch
from synthetic_functions import SyntheticFunctionGenerator, create_synthetic_dataset
from dataset import create_dataset_from_params
from load_params import load_params
import matplotlib.pyplot as plt

def test_synthetic_functions():
    """Test the synthetic function generator"""
    print("Testing synthetic function generator...")
    
    generator = SyntheticFunctionGenerator()
    
    # Test different function types
    funcs = ["polynomial", "trigonometric", "relu_tree", "symmetric"]
    complexities = [2, 3, 4]
    
    for func in funcs:
        for comp in complexities:
            try:
                x, y = generator.generate_dataset(func, comp, seed=42)
                print(f"✓ {func} (complexity={comp}): x shape {x.shape}, y shape {y.shape}")
            except Exception as e:
                print(f"✗ Error with {func} (complexity={comp}): {e}")

def test_dataset_creation():
    """Test dataset creation with parameters"""
    print("\nTesting dataset creation...")
    
    # Test modular addition (MLP format)
    params = {"use_synthetic": False, "p": 97, "train_frac": 0.8, "seed": 42, "architecture": "mlp"}
    train_x, train_y, test_x, test_y = create_dataset_from_params(params)
    print(f"✓ Modular addition (MLP): train {train_x.shape}, test {test_x.shape}")
    
    # Test modular addition (Transformer format)
    params = {"use_synthetic": False, "p": 97, "train_frac": 0.8, "seed": 42, "architecture": "transformer"}
    train_x, train_y, test_x, test_y = create_dataset_from_params(params)
    print(f"✓ Modular addition (Transformer): train {train_x.shape}, test {test_x.shape}")
    
    # Test synthetic polynomial (MLP format)
    params = {"use_synthetic": True, "func_type": "polynomial", "complexity": 3, 
              "train_frac": 0.8, "seed": 42, "architecture": "mlp"}
    train_x, train_y, test_x, test_y = create_dataset_from_params(params)
    print(f"✓ Synthetic polynomial (MLP): train {train_x.shape}, test {test_x.shape}")
    
    # Test synthetic polynomial (Transformer format)
    params = {"use_synthetic": True, "func_type": "polynomial", "complexity": 3, 
              "train_frac": 0.8, "seed": 42, "architecture": "transformer"}
    train_x, train_y, test_x, test_y = create_dataset_from_params(params)
    print(f"✓ Synthetic polynomial (Transformer): train {train_x.shape}, test {test_x.shape}")

def test_function_visualization():
    """Create a simple visualization of synthetic functions"""
    print("\nCreating function visualizations...")
    
    generator = SyntheticFunctionGenerator(n_samples=200)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()
    
    # Polynomial
    x, y = generator.generate_polynomial(degree=3, seed=42)
    axes[0].scatter(x, y, alpha=0.6, s=10)
    axes[0].set_title("Polynomial (degree=3)")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")
    
    # Trigonometric
    x, y = generator.generate_trigonometric(n_components=2, seed=42)
    axes[1].scatter(x, y, alpha=0.6, s=10)
    axes[1].set_title("Trigonometric (2 components)")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("y")
    
    # ReLU Tree
    x, y = generator.generate_relu_tree(depth=3, seed=42)
    axes[2].scatter(x, y, alpha=0.6, s=10)
    axes[2].set_title("ReLU Tree (depth=3)")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("y")
    
    # Symmetric
    x, y = generator.generate_symmetric_function(seed=42)
    axes[3].scatter(x[:, 0], x[:, 1], c=y.flatten(), alpha=0.6, s=10)
    axes[3].set_title("Symmetric Function")
    axes[3].set_xlabel("x1")
    axes[3].set_ylabel("x2")
    
    plt.tight_layout()
    plt.savefig("synthetic_functions_test.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✓ Saved function visualization to synthetic_functions_test.png")

if __name__ == "__main__":
    test_synthetic_functions()
    test_dataset_creation()
    test_function_visualization()
    print("\nAll tests completed!") 