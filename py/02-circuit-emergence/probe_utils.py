# probe_utils.py
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import os

# Import transformer support if available
try:
    from transformer_probe_utils import TransformerProbeAnalyzer
    TRANSFORMER_SUPPORT = True
except ImportError:
    TRANSFORMER_SUPPORT = False

def register_hooks(model, layer_indices, activation_store):
    hooks = []
    for idx in layer_indices:
        def hook_fn(module, input, output, idx=idx):
            activation_store[idx] = output.detach().cpu()
        hooks.append(model.net[idx * 2].register_forward_hook(hook_fn))
    return hooks

def run_probe(activation_store, concept_labels):
    accs = []
    for layer, X in activation_store.items():
        y = concept_labels
        print(f"Layer {layer}: X shape = {X.shape}, y shape = {y.shape}")
        if X.shape[0] == 0 or y.shape[0] == 0:
            print(f"Warning: No samples for layer {layer}, skipping probe.")
            accs.append(float('nan'))
            continue
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X, y)
        acc = clf.score(X, y)
        accs.append(acc)
    return accs

def register_transformer_hooks(model, activation_store, attention_store=None):
    """Register hooks for transformer model (if transformer support is available)."""
    if not TRANSFORMER_SUPPORT:
        print("Warning: Transformer support not available")
        return []
    
    transformer_analyzer = TransformerProbeAnalyzer()
    return transformer_analyzer.register_transformer_hooks(model, activation_store, attention_store)

def register_architecture_hooks(model, layer_indices, activation_store, params, attention_store=None):
    """Register hooks based on architecture type."""
    architecture = params.get('architecture', 'mlp')
    
    if architecture == 'transformer':
        return register_transformer_hooks(model, activation_store, attention_store)
    else:
        return register_hooks(model, layer_indices, activation_store)

def save_dict_as_pt(dict_obj, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(dict_obj, filepath)