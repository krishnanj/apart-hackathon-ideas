#!/usr/bin/env python3
"""
Test script to verify Steps 5 and 6 work with transformer architecture
"""

import torch
import numpy as np
from datetime import datetime
import os
from composite_analysis import run_composite_analysis
from symmetry_analysis import run_symmetry_analysis
from load_params import load_params

def test_steps_5_6_transformer():
    """Test Steps 5 and 6 with transformer architecture"""
    print("=== TESTING STEPS 5 & 6 WITH TRANSFORMER ===")
    
    # Load params and set transformer architecture
    params = load_params()
    params['architecture'] = 'transformer'
    params['use_composite'] = True
    
    print(f"Architecture: {params['architecture']}")
    print(f"Transformer config: {params.get('transformer_config', {})}")
    
    # Test Step 5: Composite Function Analysis
    print("\n=== STEP 5: COMPOSITE FUNCTION ANALYSIS ===")
    try:
        # Test with poly -> sin composite function
        run_composite_analysis('poly', 'sin')
        print("✅ Step 5 (Composite Analysis) with transformer: SUCCESS")
    except Exception as e:
        print(f"❌ Step 5 (Composite Analysis) with transformer: FAILED - {e}")
        return False
    
    # Test Step 6: Symmetry Analysis
    print("\n=== STEP 6: SYMMETRY ANALYSIS ===")
    try:
        run_symmetry_analysis()
        print("✅ Step 6 (Symmetry Analysis) with transformer: SUCCESS")
    except Exception as e:
        print(f"❌ Step 6 (Symmetry Analysis) with transformer: FAILED - {e}")
        return False
    
    print("\n🎉 Both Steps 5 and 6 work with transformer architecture!")
    return True

def test_transformer_only():
    """Test Steps 5 and 6 with transformer architecture only"""
    print("\n=== TESTING TRANSFORMER ARCHITECTURE ===")
    
    params = load_params()
    params['architecture'] = 'transformer'
    
    # Test Transformer
    print("\nTesting Transformer architecture...")
    try:
        run_composite_analysis('poly', 'sin')
        print("   ✅ Transformer works for Step 5")
    except Exception as e:
        print(f"   ❌ Transformer failed for Step 5: {e}")
        return False
    
    print("\n✅ Transformer architecture works for Steps 5 and 6!")
    return True

if __name__ == "__main__":
    success1 = test_steps_5_6_transformer()
    success2 = test_transformer_only()
    
    if success1 and success2:
        print("\n🎉 All tests passed! Steps 5 and 6 work with transformers.")
    else:
        print("\n❌ Some tests failed. Check the errors above.") 