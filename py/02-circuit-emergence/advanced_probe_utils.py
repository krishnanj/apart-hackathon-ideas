import torch
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, r2_score
import warnings
warnings.filterwarnings('ignore')

class MultiProbeAnalyzer:
    """Analyzer that can run multiple types of probes on layer activations."""
    
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
    
    def run_probes(self, activation_store, concept_labels, is_regression=False):
        """Run multiple probes on layer activations."""
        results = {}
        
        for layer, X in activation_store.items():
            y = concept_labels
            
            if X.shape[0] == 0 or y.shape[0] == 0:
                print(f"Warning: No samples for layer {layer}, skipping probes.")
                results[layer] = {probe_name: float('nan') for probe_name in self.probes.keys()}
                continue
            
            layer_results = {}
            
            for probe_name, probe in self.probes.items():
                try:
                    if is_regression:
                        # For regression tasks, use R² score
                        if isinstance(probe, LogisticRegression):
                            # Convert to regression equivalent
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
                    print(f"Error with {probe_name} probe on layer {layer}: {e}")
                    layer_results[probe_name] = float('nan')
            
            results[layer] = layer_results
        
        return results

def run_single_probe(activation_store, concept_labels, probe_type="linear", is_regression=False):
    """Run a single probe type (backward compatibility)."""
    analyzer = MultiProbeAnalyzer([probe_type])
    results = analyzer.run_probes(activation_store, concept_labels, is_regression)
    
    # Convert to old format for compatibility
    accs = []
    for layer in sorted(results.keys()):
        accs.append(results[layer][probe_type])
    
    return accs

def analyze_probe_complexity(activation_store, concept_labels, is_regression=False):
    """Analyze how probe complexity affects decodability."""
    probe_types = ["linear", "tree", "svm", "mlp"]
    analyzer = MultiProbeAnalyzer(probe_types)
    results = analyzer.run_probes(activation_store, concept_labels, is_regression)
    
    # Create complexity analysis
    complexity_analysis = {}
    for layer in results.keys():
        layer_results = results[layer]
        complexity_analysis[layer] = {
            'linear': layer_results.get('linear', float('nan')),
            'non_linear': max(layer_results.get('tree', 0), 
                             layer_results.get('svm', 0), 
                             layer_results.get('mlp', 0))
        }
    
    return complexity_analysis

def symmetry_probe(activation_store, input_pairs):
    """Specialized probe for detecting symmetry in activations."""
    symmetry_scores = {}
    
    for layer, activations in activation_store.items():
        distances = []
        for (idx1, idx2) in input_pairs:
            if idx1 < len(activations) and idx2 < len(activations):
                dist = torch.norm(activations[idx1] - activations[idx2])
                distances.append(dist.item())
        
        if distances:
            # Lower distance = more symmetric
            avg_distance = np.mean(distances)
            symmetry_scores[layer] = 1.0 / (1.0 + avg_distance)  # Convert to similarity score
        else:
            symmetry_scores[layer] = float('nan')
    
    return symmetry_scores 