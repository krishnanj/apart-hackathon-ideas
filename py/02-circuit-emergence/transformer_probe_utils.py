#!/usr/bin/env python3
"""
Transformer-specific probe utilities for analyzing attention and activations
"""

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

class TransformerProbeAnalyzer:
    """Specialized analyzer for transformer architectures with attention analysis."""
    
    def __init__(self, probe_types=None):
        self.probe_types = probe_types or ["linear", "tree", "svm", "mlp"]
        self.probes = self._initialize_probes()
    
    def _initialize_probes(self):
        """Initialize different probe types."""
        probes = {}
        
        if "linear" in self.probe_types:
            probes["linear"] = LogisticRegression(max_iter=1000, random_state=42)
        
        if "tree" in self.probe_types:
            probes["tree"] = RandomForestClassifier(n_estimators=50, random_state=42)
        
        if "svm" in self.probe_types:
            probes["svm"] = SVC(kernel='rbf', gamma='scale', random_state=42)
        
        if "mlp" in self.probe_types:
            probes["mlp"] = MLPClassifier(hidden_layer_sizes=(32,), max_iter=1000, random_state=42)
        
        return probes
    
    def register_transformer_hooks(self, model, activation_store, attention_store=None):
        """Register hooks for transformer model to capture activations and attention."""
        hooks = []
        
        # Hook for token embeddings
        def embedding_hook(module, input, output):
            activation_store['embeddings'] = output.detach().cpu()
        
        # Hook for final layer norm
        def final_ln_hook(module, input, output):
            activation_store['final_ln'] = output.detach().cpu()
        
        # Hook for attention weights
        def attention_hook(module, input, output):
            if attention_store is not None:
                attention_store['attention'] = output[1].detach().cpu()  # attention weights
        
        # Register hooks on transformer components
        for name, module in model.named_modules():
            if 'token_embeddings' in name:
                hooks.append(module.register_forward_hook(embedding_hook))
            elif 'ln_f' in name:
                hooks.append(module.register_forward_hook(final_ln_hook))
            elif 'attn' in name and hasattr(module, 'forward'):
                hooks.append(module.register_forward_hook(attention_hook))
        
        # Register hooks for each transformer layer
        for i, layer in enumerate(model.layers):
            # Hook for layer norm before attention
            def ln1_hook(module, input, output, layer_idx=i):
                activation_store[f'layer_{layer_idx}_ln1'] = output.detach().cpu()
            
            # Hook for layer norm before MLP
            def ln2_hook(module, input, output, layer_idx=i):
                activation_store[f'layer_{layer_idx}_ln2'] = output.detach().cpu()
            
            # Hook for MLP output
            def mlp_hook(module, input, output, layer_idx=i):
                activation_store[f'layer_{layer_idx}_mlp'] = output.detach().cpu()
            
            # Hook for residual connections
            def residual_hook(module, input, output, layer_idx=i):
                activation_store[f'layer_{layer_idx}_residual'] = output.detach().cpu()
            
            hooks.append(layer.ln_1.register_forward_hook(ln1_hook))
            hooks.append(layer.ln_2.register_forward_hook(ln2_hook))
            hooks.append(layer.mlp.register_forward_hook(mlp_hook))
            
            # Note: residual connections are handled in the forward pass
            # We'll capture them by hooking the layer output
        
        return hooks
    
    def run_transformer_probes(self, activation_store, concept_labels, is_regression=False):
        """Run probes on transformer activations."""
        results = {}
        
        for layer_name, X in activation_store.items():
            y = concept_labels
            
            if X.shape[0] == 0 or y.shape[0] == 0:
                print(f"Warning: No samples for {layer_name}, skipping probes.")
                results[layer_name] = {probe_name: float('nan') for probe_name in self.probes.keys()}
                continue
            
            # Flatten activations if needed (for sequence data)
            if len(X.shape) > 2:
                X = X.reshape(X.shape[0], -1)
            
            # Check if shapes are compatible AFTER flattening
            if X.shape[0] != y.shape[0]:
                print(f"Warning: Shape mismatch for {layer_name}: X={X.shape}, y={y.shape}, skipping.")
                results[layer_name] = {probe_name: float('nan') for probe_name in self.probes.keys()}
                continue
            
            layer_results = {}
            
            for probe_name, probe in self.probes.items():
                try:
                    if is_regression:
                        # For regression tasks, use R² score
                        if isinstance(probe, LogisticRegression):
                            reg_probe = Ridge(alpha=1.0)
                        elif isinstance(probe, RandomForestClassifier):
                            reg_probe = RandomForestRegressor(n_estimators=50, random_state=42)
                        elif isinstance(probe, SVC):
                            reg_probe = SVR(kernel='rbf', gamma='scale')
                        elif isinstance(probe, MLPClassifier):
                            reg_probe = MLPRegressor(hidden_layer_sizes=(32,), max_iter=1000, random_state=42)
                        else:
                            reg_probe = probe
                        
                        reg_probe.fit(X, y.float())
                        score = reg_probe.score(X, y.float())
                    else:
                        # For classification tasks, use accuracy
                        probe.fit(X, y)
                        score = probe.score(X, y)
                    
                    layer_results[probe_name] = score
                    
                except Exception as e:
                    print(f"Error with {probe_name} probe on {layer_name}: {e}")
                    layer_results[probe_name] = float('nan')
            
            results[layer_name] = layer_results
        
        return results
    
    def analyze_attention_patterns(self, attention_store, test_x, test_y, params):
        """Analyze attention patterns for concept emergence."""
        if 'attention' not in attention_store:
            print("No attention weights found in attention_store")
            return {}
        
        attention = attention_store['attention']
        print(f"Attention shape: {attention.shape}")
        
        # Analyze attention patterns
        attention_analysis = {}
        
        # 1. Average attention weights per layer
        if len(attention.shape) == 4:  # [batch, heads, seq_len, seq_len]
            avg_attention = attention.mean(dim=0)  # Average over batch
            attention_analysis['avg_attention'] = avg_attention.numpy()
        
        # 2. Attention entropy (measure of focus)
        attention_probs = torch.softmax(attention, dim=-1)
        entropy = -torch.sum(attention_probs * torch.log(attention_probs + 1e-8), dim=-1)
        attention_analysis['attention_entropy'] = entropy.mean(dim=0).numpy()
        
        # 3. Cross-attention analysis (if applicable)
        if attention.shape[-1] != attention.shape[-2]:
            # This might be cross-attention
            attention_analysis['cross_attention'] = True
        
        return attention_analysis

def run_transformer_single_probe(activation_store, concept_labels, probe_type="linear", is_regression=False):
    """Run a single probe type on transformer activations."""
    analyzer = TransformerProbeAnalyzer([probe_type])
    results = analyzer.run_transformer_probes(activation_store, concept_labels, is_regression)
    
    # Convert to simple format for compatibility
    accs = []
    for layer_name in sorted(results.keys()):
        accs.append(results[layer_name][probe_type])
    
    return accs

def analyze_transformer_complexity(activation_store, concept_labels, is_regression=False):
    """Analyze how probe complexity affects decodability in transformers."""
    probe_types = ["linear", "tree", "svm", "mlp"]
    analyzer = TransformerProbeAnalyzer(probe_types)
    results = analyzer.run_transformer_probes(activation_store, concept_labels, is_regression)
    
    # Create complexity analysis
    complexity_analysis = {}
    for layer_name in results.keys():
        layer_results = results[layer_name]
        complexity_analysis[layer_name] = {
            'linear': layer_results.get('linear', float('nan')),
            'non_linear': max(layer_results.get('tree', 0), 
                             layer_results.get('svm', 0), 
                             layer_results.get('mlp', 0))
        }
    
    return complexity_analysis

def symmetry_probe_transformer(activation_store, input_pairs):
    """Specialized probe for detecting symmetry in transformer activations."""
    symmetry_scores = {}
    
    for layer_name, activations in activation_store.items():
        distances = []
        for (idx1, idx2) in input_pairs:
            if idx1 < len(activations) and idx2 < len(activations):
                # Handle sequence data
                if len(activations.shape) > 2:
                    # For sequence data, compare the last token representation
                    act1 = activations[idx1, -1]  # Last token
                    act2 = activations[idx2, -1]  # Last token
                else:
                    act1 = activations[idx1]
                    act2 = activations[idx2]
                
                dist = torch.norm(act1 - act2)
                distances.append(dist.item())
        
        if distances:
            # Lower distance = more symmetric
            avg_distance = np.mean(distances)
            symmetry_scores[layer_name] = 1.0 / (1.0 + avg_distance)  # Convert to similarity score
        else:
            symmetry_scores[layer_name] = float('nan')
    
    return symmetry_scores 