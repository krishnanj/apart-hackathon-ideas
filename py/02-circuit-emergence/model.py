#!/usr/bin/env python3
"""
Model definitions for both MLP and Transformer architectures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

def create_model_from_params(input_dim, output_dim, params):
    """Create model based on architecture parameter"""
    architecture = params.get('architecture', 'mlp')
    
    if architecture == 'mlp':
        return create_mlp_model(input_dim, output_dim, params)
    elif architecture == 'transformer':
        return create_transformer_model(input_dim, output_dim, params)
    else:
        raise ValueError(f"Unknown architecture: {architecture}")

def create_mlp_model(input_dim, output_dim, params):
    """Create MLP model (existing functionality)"""
    hidden_dim = params.get('hidden_dim', 128)
    depth = params.get('depth', 3)
    
    class MLP(nn.Module):
        def __init__(self, input_dim, hidden_dim, depth, output_dim):
            super().__init__()
            layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
            for _ in range(depth - 1):
                layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
            layers += [nn.Linear(hidden_dim, output_dim)]
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)
    
    return MLP(input_dim, hidden_dim, depth, output_dim)

def create_transformer_model(input_dim, output_dim, params):
    """Create Transformer model for sequence tasks"""
    transformer_config = params.get('transformer_config', {})
    dim = transformer_config.get('dim', 128)
    num_layers = transformer_config.get('num_layers', 2)
    num_heads = transformer_config.get('num_heads', 4)
    seq_len = transformer_config.get('seq_len', 5)
    num_tokens = transformer_config.get('num_tokens', 100)
    
    class TransformerBlock(nn.Module):
        def __init__(self, dim, num_heads):
            super().__init__()
            self.ln_1 = nn.LayerNorm(dim)
            self.ln_2 = nn.LayerNorm(dim)
            self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
            self.mlp = nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim),
            )

        def forward(self, x):
            # x shape: [batch_size, seq_len, dim]
            x = self.ln_1(x)
            # Use batch_first=True for MultiheadAttention
            attn_out, _ = self.attn(x, x, x)
            x = x + attn_out
            x = self.ln_2(x)
            mlp_out = self.mlp(x)
            x = x + mlp_out
            return x

    class Transformer(nn.Module):
        def __init__(self, dim, num_layers, num_heads, num_tokens, seq_len):
            super().__init__()
            self.token_embeddings = nn.Embedding(num_tokens, dim)
            self.position_embeddings = nn.Embedding(seq_len, dim)
            self.layers = nn.ModuleList()
            for _ in range(num_layers):
                self.layers.append(TransformerBlock(dim, num_heads))
            self.ln_f = nn.LayerNorm(dim)
            self.head = nn.Linear(dim, num_tokens, bias=False)

        def forward(self, x):
            # x shape: [batch_size, seq_len]
            batch_size, seq_len = x.shape
            
            # Get embeddings
            h = self.token_embeddings(x)  # [batch_size, seq_len, dim]
            positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
            h = h + self.position_embeddings(positions)
            
            # Apply transformer layers
            for layer in self.layers:
                h = layer(h)
            
            h = self.ln_f(h)
            logits = self.head(h)  # [batch_size, seq_len, num_tokens]
            return logits
    
    return Transformer(dim, num_layers, num_heads, num_tokens, seq_len)