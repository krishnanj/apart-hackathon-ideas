#!/usr/bin/env python3
"""
Quick test to verify new concepts work properly
"""

import torch
import numpy as np
from transformer_grokking_experiment import generate_concept_labels

def test_new_concepts():
    """Test the new concept generation functions"""
    print("=== TESTING NEW CONCEPTS ===")
    
    # Create sample data
    p = 97
    batch_size = 100
    seq_len = 5
    
    # Simulate data shape: [seq_len, batch_size]
    data = torch.randint(0, p, (seq_len, batch_size))
    
    # Test each new concept
    new_concepts = [
        "modular_remainder", "input_output_parity", "modular_symmetry",
        "alternating", "monotonic", "gcd_pattern", "prime_factor", "modular_inverse_prop"
    ]
    
    for concept in new_concepts:
        try:
            labels = generate_concept_labels(data, p, concept)
            unique_labels, counts = np.unique(labels, return_counts=True)
            
            print(f"✅ {concept}:")
            print(f"   Labels: {unique_labels}")
            print(f"   Counts: {counts}")
            print(f"   Balance: {min(counts)/max(counts):.3f}")
            
            # Check if we have at least 2 classes
            if len(unique_labels) >= 2:
                print(f"   ✅ Good: {len(unique_labels)} classes")
            else:
                print(f"   ❌ Bad: Only {len(unique_labels)} class")
                
        except Exception as e:
            print(f"❌ {concept}: Error - {e}")
        
        print()

if __name__ == "__main__":
    test_new_concepts() 