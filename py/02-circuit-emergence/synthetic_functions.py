import torch
import numpy as np
from typing import Callable, Tuple, List
import random

class SyntheticFunctionGenerator:
    """Generates synthetic functions of varying complexity for circuit emergence studies."""
    
    def __init__(self, input_range: Tuple[float, float] = (-5, 5), n_samples: int = 1000):
        self.input_range = input_range
        self.n_samples = n_samples
        
    def generate_polynomial(self, degree: int, seed: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate polynomial function f(x) = a_n*x^n + ... + a_0"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Generate random coefficients with scaling to prevent overflow
        coeffs = np.random.uniform(-2, 2, degree + 1)
        
        # Scale coefficients to prevent numerical overflow for high degrees
        for i in range(degree + 1):
            coeffs[i] = coeffs[i] / (2.0 ** i)  # More aggressive scaling
        
        # Generate inputs
        x = torch.linspace(self.input_range[0], self.input_range[1], self.n_samples)
        
        # Compute polynomial
        y = torch.zeros_like(x)
        for i, coeff in enumerate(coeffs):
            y += coeff * (x ** i)
            
        return x.unsqueeze(1), y.unsqueeze(1)
    
    def generate_trigonometric(self, n_components: int, seed: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate trigonometric function f(x) = sum(a_i*sin(w_i*x) + b_i*cos(w_i*x))"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Generate random amplitudes and frequencies
        a_coeffs = np.random.uniform(-1, 1, n_components)
        b_coeffs = np.random.uniform(-1, 1, n_components)
        freqs = np.random.uniform(0.5, 2.0, n_components)
        
        x = torch.linspace(self.input_range[0], self.input_range[1], self.n_samples)
        y = torch.zeros_like(x)
        
        for i in range(n_components):
            y += a_coeffs[i] * torch.sin(freqs[i] * x) + b_coeffs[i] * torch.cos(freqs[i] * x)
            
        return x.unsqueeze(1), y.unsqueeze(1)
    
    def generate_relu_tree(self, depth: int, seed: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate nested ReLU function: f(x) = ReLU(ReLU(...(x)))"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        x = torch.linspace(self.input_range[0], self.input_range[1], self.n_samples)
        y = x.clone()
        
        # Apply nested ReLU operations
        for _ in range(depth):
            # Random linear transformation + ReLU
            weight = np.random.uniform(-2, 2)
            bias = np.random.uniform(-1, 1)
            y = torch.relu(weight * y + bias)
            
        return x.unsqueeze(1), y.unsqueeze(1)
    
    def generate_composite(self, inner_func: str, outer_func: str, seed: int = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate composite function f(g(x)) and return both g(x) and f(g(x))"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        x = torch.linspace(self.input_range[0], self.input_range[1], self.n_samples)
        
        # Generate inner function g(x)
        if inner_func == "poly":
            g_x, _ = self.generate_polynomial(degree=2, seed=seed)
        elif inner_func == "sin":
            g_x, _ = self.generate_trigonometric(n_components=1, seed=seed)
        elif inner_func == "relu":
            g_x, _ = self.generate_relu_tree(depth=2, seed=seed)
        else:
            raise ValueError(f"Unknown inner function: {inner_func}")
            
        # Generate outer function f(x)
        if outer_func == "poly":
            _, f_gx = self.generate_polynomial(degree=3, seed=seed+100)
        elif outer_func == "sin":
            _, f_gx = self.generate_trigonometric(n_components=2, seed=seed+100)
        elif outer_func == "relu":
            _, f_gx = self.generate_relu_tree(depth=3, seed=seed+100)
        else:
            raise ValueError(f"Unknown outer function: {outer_func}")
            
        return x.unsqueeze(1), g_x, f_gx
    
    def generate_symmetric_function(self, seed: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate function with known symmetry: f(x1, x2) = f(x2, x1)"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        # Generate 2D inputs
        x1 = torch.linspace(self.input_range[0], self.input_range[1], int(np.sqrt(self.n_samples)))
        x2 = torch.linspace(self.input_range[0], self.input_range[1], int(np.sqrt(self.n_samples)))
        X1, X2 = torch.meshgrid(x1, x2, indexing='ij')
        
        # Symmetric function: f(x1, x2) = (x1 + x2)^2 + sin(x1*x2)
        y = (X1 + X2)**2 + torch.sin(X1 * X2)
        
        # Flatten for dataset
        x = torch.stack([X1.flatten(), X2.flatten()], dim=1)
        y = y.flatten().unsqueeze(1)
        
        return x, y
    
    def generate_dataset(self, func_type: str, complexity: int, seed: int = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate dataset for specified function type and complexity"""
        if func_type == "polynomial":
            return self.generate_polynomial(degree=complexity, seed=seed)
        elif func_type == "trigonometric":
            return self.generate_trigonometric(n_components=complexity, seed=seed)
        elif func_type == "relu_tree":
            return self.generate_relu_tree(depth=complexity, seed=seed)
        elif func_type == "symmetric":
            return self.generate_symmetric_function(seed=seed)
        else:
            raise ValueError(f"Unknown function type: {func_type}")

def create_synthetic_dataset(func_type: str, complexity: int, train_frac: float = 0.8, seed: int = 42):
    """Create train/test split for synthetic function dataset"""
    generator = SyntheticFunctionGenerator()
    x, y = generator.generate_dataset(func_type, complexity, seed)
    
    # Split into train/test
    n_train = int(train_frac * len(x))
    indices = torch.randperm(len(x))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    return x[train_idx], y[train_idx], x[test_idx], y[test_idx]

def create_composite_dataset(inner_func: str, outer_func: str, train_frac: float = 0.8, seed: int = 42):
    """Create train/test split for composite function dataset with dual concepts"""
    generator = SyntheticFunctionGenerator()
    x, g_x, f_gx = generator.generate_composite(inner_func, outer_func, seed)
    
    # Split into train/test
    n_train = int(train_frac * len(x))
    indices = torch.randperm(len(x))
    train_idx = indices[:n_train]
    test_idx = indices[n_train:]
    
    return (x[train_idx], g_x[train_idx], f_gx[train_idx], 
            x[test_idx], g_x[test_idx], f_gx[test_idx])

if __name__ == "__main__":
    # Test the function generator
    generator = SyntheticFunctionGenerator()
    
    # Test different function types
    funcs = ["polynomial", "trigonometric", "relu_tree", "symmetric"]
    complexities = [2, 3, 4, 5]
    
    for func in funcs:
        for comp in complexities:
            try:
                x, y = generator.generate_dataset(func, comp, seed=42)
                print(f"{func} (complexity={comp}): x shape {x.shape}, y shape {y.shape}")
            except Exception as e:
                print(f"Error with {func} (complexity={comp}): {e}")
    
    # Test composite functions
    print("\nTesting composite functions:")
    composite_configs = [("poly", "sin"), ("sin", "poly"), ("relu", "poly")]
    for inner, outer in composite_configs:
        try:
            x, g_x, f_gx = generator.generate_composite(inner, outer, seed=42)
            print(f"Composite {inner}->{outer}: x shape {x.shape}, g(x) shape {g_x.shape}, f(g(x)) shape {f_gx.shape}")
        except Exception as e:
            print(f"Error with composite {inner}->{outer}: {e}")

# Add these functions to synthetic_functions.py after the existing methods

def generate_complexity_variants(self, func_type: str, complexity_range: List[int], seed: int = None):
    """Generate functions with varying complexity levels"""
    variants = {}
    
    for complexity in complexity_range:
        try:
            x, y = self.generate_dataset(func_type, complexity, seed)
            variants[complexity] = (x, y)
        except Exception as e:
            print(f"Warning: Failed to generate {func_type} with complexity {complexity}: {e}")
    
    return variants

def generate_grokking_dataset(self, func_type: str, subset_ratio: float = 0.3, seed: int = None):
    """Generate subset dataset for grokking experiments"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Generate full dataset
    x, y = self.generate_dataset(func_type, 3, seed)  # Use complexity 3 as default
    
    # Create subset
    n_subset = int(subset_ratio * len(x))
    indices = torch.randperm(len(x))
    subset_idx = indices[:n_subset]
    remaining_idx = indices[n_subset:]
    
    return x[subset_idx], y[subset_idx], x[remaining_idx], y[remaining_idx]

def generate_robustness_test_data(self, func_type: str, complexity: int, noise_levels: List[float], seed: int = None):
    """Generate data with varying noise levels for robustness testing"""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Generate base dataset
    x, y = self.generate_dataset(func_type, complexity, seed)
    
    # Add noise at different levels
    noisy_datasets = {}
    for noise_level in noise_levels:
        noise = torch.randn_like(y) * noise_level
        y_noisy = y + noise
        noisy_datasets[noise_level] = (x, y_noisy)
    
    return noisy_datasets 