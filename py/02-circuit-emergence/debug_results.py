#!/usr/bin/env python3
"""
Debug script to inspect complexity sweep results
"""

import torch
import os

def main():
    # Find the most recent results file
    results_dir = "results"
    result_files = [f for f in os.listdir(results_dir) if f.startswith("complexity_sweep_results_")]
    
    if not result_files:
        print("No complexity sweep results found!")
        return
    
    # Get the most recent file
    latest_file = sorted(result_files)[-1]
    results_path = os.path.join(results_dir, latest_file)
    
    print(f"Loading results from: {results_path}")
    results = torch.load(results_path)
    
    print(f"\nResults structure:")
    print(f"Keys: {list(results.keys())}")
    
    for complexity in sorted(results.keys()):
        print(f"\nComplexity {complexity}:")
        for width in sorted(results[complexity].keys()):
            print(f"  Width {width}:")
            for layer, acc in results[complexity][width].items():
                print(f"    {layer}: {acc}")

if __name__ == "__main__":
    main() 