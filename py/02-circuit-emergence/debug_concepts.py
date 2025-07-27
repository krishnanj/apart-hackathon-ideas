#!/usr/bin/env python3
"""
Debug script to understand why all concept labels are the same
"""

import torch
import numpy as np
from synthetic_functions import SyntheticFunctionGenerator

def test_concept_generation():
    """Test concept generation for different polynomial degrees"""
    generator = SyntheticFunctionGenerator(n_samples=200)
    
    for degree in [1, 2, 3, 4, 5, 6]:
        print(f"\n=== Testing degree {degree} ===")
        
        # Generate polynomial
        x, y = generator.generate_polynomial(degree=degree, seed=42)
        x_vals = x[:, 0]
        y_vals = y[:, 0]
        
        print(f"X range: [{x_vals.min():.3f}, {x_vals.max():.3f}]")
        print(f"Y range: [{y_vals.min():.3f}, {y_vals.max():.3f}]")
        
        # Test Concept 1: Number of roots in [0, 1]
        mask_01 = (x_vals >= 0) & (x_vals <= 1)
        if mask_01.sum() > 5:
            x_01 = x_vals[mask_01]
            y_01 = y_vals[mask_01]
            sign_changes = np.sum(np.diff(np.sign(y_01)) != 0)
            concept1 = 1 if sign_changes > 1 else 0
            print(f"Concept 1 (roots in [0,1]): {concept1} (sign_changes: {sign_changes})")
        else:
            print("Concept 1: Not enough points in [0,1]")
        
        # Test Concept 2: Local maxima
        local_maxima = []
        for i in range(1, len(x_vals) - 1):
            if y_vals[i] > y_vals[i-1] and y_vals[i] > y_vals[i+1]:
                local_maxima.append(1)
            else:
                local_maxima.append(0)
        concept2 = 1 if sum(local_maxima) > 0 else 0
        print(f"Concept 2 (local maxima): {concept2} (count: {sum(local_maxima)})")
        
        # Test Concept 3: Strictly increasing
        strictly_increasing = np.all(np.diff(y_vals) > 0)
        concept3 = 1 if strictly_increasing else 0
        print(f"Concept 3 (strictly increasing): {concept3}")
        
        # Test Concept 4: Positive leading coefficient
        if len(x_vals) > 5:
            last_points = y_vals[-5:]
            growth_rate = np.polyfit(range(5), last_points, 1)[0]
            concept4 = 1 if growth_rate > 0 else 0
            print(f"Concept 4 (positive growth): {concept4} (growth_rate: {growth_rate:.3f})")
        else:
            concept4 = 1 if y_vals[-1] > y_vals[0] else 0
            print(f"Concept 4 (simple trend): {concept4}")

if __name__ == "__main__":
    test_concept_generation() 