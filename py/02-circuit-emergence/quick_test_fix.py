#!/usr/bin/env python3
"""
Quick test to verify transformer probing fix works
"""

import torch
import numpy as np
from composite_analysis import run_composite_analysis
from load_params import load_params

def quick_test():
    """Test the transformer probing fix with just one model"""
    print("=== QUICK TEST: Transformer Probing Fix ===")
    
    # Load params and force transformer
    params = load_params()
    params['architecture'] = 'transformer'
    params['widths'] = [16]  # Just test one width
    params['steps'] = 100  # Quick training
    
    print("Testing composite analysis with transformer...")
    try:
        results = run_composite_analysis('poly', 'sin')
        print("✅ SUCCESS: Transformer probing works!")
        return True
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

if __name__ == "__main__":
    success = quick_test()
    if success:
        print("\n🎉 The fix works! You can now run the full test.")
    else:
        print("\n💥 The fix didn't work. Need to debug further.") 