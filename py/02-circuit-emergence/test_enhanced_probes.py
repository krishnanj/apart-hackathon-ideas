#!/usr/bin/env python3
"""
Test script for enhanced probe system with transformer support
"""

import torch
import numpy as np
from datetime import datetime
import os
from model import create_model_from_params
from dataset import create_dataset_from_params
from load_params import load_params
from advanced_probe_utils import run_architecture_aware_probes
from probe_utils import register_architecture_hooks

def test_enhanced_probes():
    """Test the enhanced probe system with both architectures."""
    print("=== TESTING ENHANCED PROBE SYSTEM ===")
    
    # Load params
    params = load_params()
    
    # Test both architectures
    architectures = ['mlp', 'transformer']
    
    for architecture in architectures:
        print(f"\n--- Testing {architecture.upper()} Architecture ---")
        
        # Update params for this architecture
        test_params = params.copy()
        test_params['architecture'] = architecture
        
        try:
            # Create dataset
            train_x, train_y, test_x, test_y = create_dataset_from_params(test_params)
            print(f"  Dataset created: {test_x.shape}, {test_y.shape}")
            
            # Create model
            if architecture == 'transformer':
                input_dim = test_x.shape[1]  # sequence length
                output_dim = test_params.get('transformer_config', {}).get('num_tokens', 100)
            else:
                input_dim = test_x.shape[1] if len(test_x.shape) > 1 else 1
                output_dim = test_y.shape[1] if len(test_y.shape) > 1 else 1
            
            model = create_model_from_params(input_dim, output_dim, test_params)
            print(f"  Model created: {type(model).__name__}")
            
            # Test forward pass
            with torch.no_grad():
                if architecture == 'transformer':
                    outputs = model(test_x[:10])  # Test with small batch
                    print(f"  Forward pass successful: {outputs.shape}")
                else:
                    outputs = model(test_x[:10].float())
                    print(f"  Forward pass successful: {outputs.shape}")
            
            # Test probe system
            concept_labels = test_y[:10].numpy() if hasattr(test_y, 'numpy') else test_y[:10]
            
            if architecture == 'transformer':
                # For transformer, we can run probes directly
                probe_results = run_architecture_aware_probes(
                    model, test_x[:10], concept_labels, test_params, is_regression=False
                )
            else:
                # For MLP, we need to capture activations first
                activation_store = {}
                hooks = register_architecture_hooks(model, list(range(3)), activation_store, test_params)
                
                with torch.no_grad():
                    model(test_x[:10].float())
                
                for hook in hooks:
                    hook.remove()
                
                probe_results = run_architecture_aware_probes(
                    model, test_x[:10], concept_labels, test_params, 
                    activation_store=activation_store, is_regression=False
                )
            
            print(f"  Probe results: {len(probe_results)} layers analyzed")
            
            # Print sample results
            for layer_name, results in list(probe_results.items())[:3]:  # Show first 3 layers
                if isinstance(results, dict) and 'linear' in results:
                    print(f"    {layer_name}: Linear accuracy = {results['linear']:.4f}")
            
            print(f"  ✅ {architecture.upper()} probe test PASSED")
            
        except Exception as e:
            print(f"  ❌ {architecture.upper()} probe test FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n=== ENHANCED PROBE SYSTEM TEST COMPLETE ===")

def test_transformer_attention_analysis():
    """Test transformer-specific attention analysis."""
    print("\n=== TESTING TRANSFORMER ATTENTION ANALYSIS ===")
    
    params = load_params()
    params['architecture'] = 'transformer'
    
    try:
        # Create transformer model and data
        train_x, train_y, test_x, test_y = create_dataset_from_params(params)
        model = create_model_from_params(test_x.shape[1], 100, params)
        
        # Test attention analysis
        from transformer_probe_utils import TransformerProbeAnalyzer
        
        analyzer = TransformerProbeAnalyzer(['linear'])
        activation_store = {}
        attention_store = {}
        
        hooks = analyzer.register_transformer_hooks(model, activation_store, attention_store)
        
        with torch.no_grad():
            model(test_x[:5])  # Small batch
        
        for hook in hooks:
            hook.remove()
        
        # Analyze attention patterns
        attention_analysis = analyzer.analyze_attention_patterns(
            attention_store, test_x[:5], test_y[:5], params
        )
        
        print(f"  Attention analysis completed: {len(attention_analysis)} metrics")
        for metric_name in attention_analysis.keys():
            print(f"    {metric_name}: Available")
        
        print("  ✅ Transformer attention analysis PASSED")
        
    except Exception as e:
        print(f"  ❌ Transformer attention analysis FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_probes()
    test_transformer_attention_analysis() 